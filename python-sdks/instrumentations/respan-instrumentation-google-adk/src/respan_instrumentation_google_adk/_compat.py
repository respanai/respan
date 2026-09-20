"""Bridge legacy ADK's direct, cross-task advancement of async generators."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, copy_context
from functools import wraps
from importlib.metadata import version
from typing import Any, Callable

from packaging.version import Version

_owned_llm_generators: ContextVar[list | None] = ContextVar(
    "respan_adk_owned_llm_generators", default=None
)


async def _drive_iterator(source, requests):
    # OI instruments __aiter__, whereas legacy ParallelAgent (and custom
    # agents importing _merge_agent_run) calls __anext__ directly.
    iterator = source.__aiter__()
    llm_generators = []
    token = _owned_llm_generators.set(llm_generators)
    try:
        while True:
            method, args, result = await requests.get()
            try:
                value = await getattr(iterator, method)(*args)
            except BaseException as exc:
                if not result.done():
                    result.set_exception(exc)
                return
            else:
                if not result.done():
                    result.set_result(value)
    finally:
        try:
            # ADK 1.5 leaves _call_llm_async suspended after yielding a response.
            # Close those inner span scopes before the outer agent scope, in
            # the task/context where their context tokens were created.
            for generator in reversed(llm_generators):
                await generator.aclose()
        finally:
            try:
                await iterator.aclose()
            finally:
                try:
                    if source is not iterator:
                        await source.aclose()
                finally:
                    _owned_llm_generators.reset(token)


class _ContextPreservingIterator:
    def __init__(self, source: Any) -> None:
        self._source = source
        self._context = copy_context()
        self._requests: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._closed = False
        self._advancing = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._advance("__anext__")

    async def asend(self, value):
        return await self._advance("asend", value)

    async def athrow(self, *args):
        return await self._advance("athrow", *args)

    async def _advance(self, method, *args):
        if self._closed:
            raise StopAsyncIteration
        if self._advancing:
            raise RuntimeError("Agent async generator is already running")
        if self._worker is None:
            self._requests = asyncio.Queue()
            # Keep a demand-driven worker alive between events. Besides keeping
            # context tokens in their original context, this closes generators
            # in that context when asyncio.run cancels tasks during shutdown.
            self._worker = asyncio.create_task(
                _drive_iterator(self._source, self._requests), context=self._context
            )
        self._advancing = True
        result = asyncio.get_running_loop().create_future()
        self._requests.put_nowait((method, args, result))
        try:
            return await result
        except BaseException:
            await self.aclose()
            raise
        finally:
            self._advancing = False

    async def aclose(self):
        self._closed = True
        if self._worker is not None:
            if not self._worker.done():
                self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        else:
            await self._source.aclose()

    def __del__(self):
        # The worker holds the source, not this adapter, so dropping an iterator
        # also closes it without scheduling an async-generator finalizer race.
        worker = self._worker
        if (
            worker is not None
            and not worker.done()
            and not worker.get_loop().is_closed()
        ):
            worker.cancel()


def patch_legacy_agent_iterator() -> Callable[[], None] | None:
    """Return an undo callback; modern ADK keeps its existing upstream path."""
    if Version(version("google-adk")) >= Version("1.17.0"):
        return None

    from google.adk import Runner
    from google.adk.agents import BaseAgent
    from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

    originals = []

    def wrap(original):
        @wraps(original)
        def run_async(self, *args, **kwargs):
            bound = original.__get__(self, type(self))
            return _ContextPreservingIterator(bound(*args, **kwargs))

        return run_async

    for cls in (BaseAgent, Runner):
        original = cls.run_async
        replacement = wrap(original)
        cls.run_async = replacement
        originals.append((cls, "run_async", original, replacement))

    original_llm_call = BaseLlmFlow._call_llm_async

    @wraps(original_llm_call)
    def call_llm_async(self, *args, **kwargs):
        generator = original_llm_call(self, *args, **kwargs)
        if (owned := _owned_llm_generators.get()) is not None:
            owned.append(generator)
        return generator

    BaseLlmFlow._call_llm_async = call_llm_async
    originals.append(
        (BaseLlmFlow, "_call_llm_async", original_llm_call, call_llm_async)
    )

    def undo():
        for cls, name, original, replacement in reversed(originals):
            if getattr(cls, name) is replacement:
                setattr(cls, name, original)

    return undo
