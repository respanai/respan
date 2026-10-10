"""Owned native Mirascope Model and Toolkit wrappers with detached streams."""

from __future__ import annotations

import asyncio
import functools
import importlib
import inspect
import logging
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry import context, trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_MESSAGE,
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.http_attributes import (
    HTTP_RESPONSE_STATUS_CODE,
)
from opentelemetry.semconv.trace import SpanAttributes as OTel
from opentelemetry.semconv_ai import SpanAttributes as A
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer

from . import _translation as translation
from ._policy import Policy, allowed, key, suppressed
from ._serialization import safe_exception_text

logger = logging.getLogger(__name__)
MIRASCOPE_INSTRUMENTATION_NAME = "mirascope"
_LOCK = threading.RLock()
_RUNTIME = None
_CALL = ContextVar("respan_mirascope_call", default=None)


def safe(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - telemetry preserves native behavior.
        logger.debug("Mirascope observation skipped: %s", type(error).__name__)
        return None


def _is_respan_tracing_enabled():
    instance = getattr(RespanTracer, "_instance", None)
    return instance is None or getattr(instance, "is_enabled", True)


def _status_code(value):
    pending, seen = [value], set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        code = translation.get(candidate, "status_code")
        if type(code) is int and 100 <= code <= 599:
            return code
        for name in (
            "response",
            "raw_response",
            "original_exception",
            "tool_exception",
            "__cause__",
            "__context__",
        ):
            nested = translation.get(candidate, name)
            if nested is not None:
                pending.append(nested)
    return None


def _content_key(name):
    return name.startswith((f"{A.LLM_PROMPTS}.", f"{A.LLM_COMPLETIONS}.")) or name in (
        A.TRACELOOP_ENTITY_INPUT,
        A.TRACELOOP_ENTITY_OUTPUT,
        A.LLM_REQUEST_FUNCTIONS,
        ERROR_MESSAGE,
        OTel.EXCEPTION_MESSAGE,
        OTel.EXCEPTION_STACKTRACE,
    )


class Call:
    def __init__(self, runtime, item, args, kwargs, *, tool=False, stream=False):
        self.runtime, self.item, self.kwargs = runtime, item, dict(kwargs)
        self.tool, self.stream = tool, stream
        self.content = args[1] if len(args) > 1 else kwargs.get("content")
        self.parent = _CALL.get()
        self.lock = threading.RLock()
        self.finished = False
        self.observed_content = False
        self.usage_seen, self.usage, self.http_status = False, {}, None
        self.response = None
        self.partial_calls = {}
        self.finalizer = None
        self.span = None
        self.capture = False

    def start(self):
        initial = allowed(self.runtime.capture) and (
            self.parent is None or self.parent.capture
        )
        self.span = self.runtime.tracer.start_span("tool" if self.tool else "llm")
        self.capture = bool(
            initial
            and self.runtime.registered
            and self.span.is_recording()
            and self.runtime.policy.ancestors(key(self.span))
        )
        self.runtime.calls.add(self)
        if not self.runtime.active:
            self.capture = False
            self.finish()
            return self
        if self.span.is_recording():
            safe(
                self.span.set_attribute,
                RESPAN_LOG_TYPE,
                LOG_TYPE_TOOL if self.tool else LOG_TYPE_CHAT,
            )
            attributes = safe(translation.prepare, self)
            if attributes is not None:
                safe(self.span.set_attributes, attributes)
            else:
                self.capture = False
                safe(self.clear_content)
        return self

    def veto(self):
        try:
            permitted = (
                allowed(self.runtime.capture)
                and self.runtime.policy.ancestors(key(self.span))
                and (self.parent is None or self.parent.capture)
            )
        except Exception:  # noqa: BLE001 - a policy fault never enables capture.
            permitted = False
        if not permitted:
            self.capture = False
            self.partial_calls.clear()
            safe(self.runtime.policy.veto, self.span)
            safe(self.clear_content)

    @contextmanager
    def scope(self):
        if self.finished:
            yield
            return
        call_token = _CALL.set(self)
        span_token = context.attach(trace.set_span_in_context(self.span))
        try:
            yield
        finally:
            try:
                self.veto()
            finally:
                try:
                    context.detach(span_token)
                finally:
                    _CALL.reset(call_token)

    def clear_content(self):
        if not self.span or not self.span.is_recording():
            return
        attributes = self.span._attributes
        for name in tuple(attributes):
            if _content_key(name):
                attributes.pop(name, None)
        if self.span.status.status_code is StatusCode.ERROR:
            self.span._status = Status(StatusCode.ERROR)
        self.span._events = BoundedList(0)

    def observe_chunk(self, chunk):
        kind = translation.get(chunk, "type")
        if kind in {
            "text_start_chunk",
            "text_chunk",
            "tool_call_start_chunk",
            "tool_call_chunk",
            "thought_start_chunk",
            "thought_chunk",
        }:
            self.observed_content = True
        if not self.capture:
            return
        if kind == "tool_call_start_chunk":
            identifier, name = (
                translation.get(chunk, "id"),
                translation.get(chunk, "name"),
            )
            if isinstance(identifier, str) and isinstance(name, str):
                self.partial_calls[identifier] = {
                    "id": identifier,
                    "name": name,
                    "args": "",
                }
        elif kind == "tool_call_chunk":
            identifier, delta = (
                translation.get(chunk, "id"),
                translation.get(chunk, "delta"),
            )
            if identifier in self.partial_calls and isinstance(delta, str):
                self.partial_calls[identifier]["args"] += delta

    def error(self, error):
        self.span.set_status(Status(StatusCode.ERROR))
        self.span.set_attribute(ERROR_TYPE, type(error).__name__)
        if code := _status_code(error):
            self.span.set_attribute(HTTP_RESPONSE_STATUS_CODE, code)
        if self.capture and (message := safe_exception_text(error)):
            self.span.set_attribute(ERROR_MESSAGE, message)
            self.span.add_event(
                "exception",
                {
                    OTel.EXCEPTION_TYPE: type(error).__name__,
                    OTel.EXCEPTION_MESSAGE: message,
                    OTel.EXCEPTION_ESCAPED: not self.tool,
                },
            )

    def finish(self, response=None, error=None):
        with self.lock:
            if self.finished:
                return
            self.veto()
            self.finished = True
            try:
                if error is None and self.tool:
                    error = translation.get(response, "error")
                if isinstance(error, BaseException):
                    safe(self.error, error)
                elif self.http_status is not None:
                    safe(
                        self.span.set_attribute,
                        HTTP_RESPONSE_STATUS_CODE,
                        self.http_status,
                    )
                attributes = safe(translation.finish, self, response)
                if attributes is not None:
                    safe(self.span.set_attributes, attributes)
                if self.capture and self.stream and self.partial_calls:
                    completed = {
                        translation.get(c, "id")
                        for c in translation.get(response, "tool_calls", [])
                    }
                    pending = [
                        c
                        for cid, c in self.partial_calls.items()
                        if cid not in completed
                    ]
                    if pending:
                        safe(
                            self.span.set_attribute,
                            f"{A.LLM_COMPLETIONS}.0.tool_calls",
                            translation.json_string(
                                translation.calls(pending), complete=True
                            ),
                        )
                self.veto()
                if not self.capture:
                    safe(self.clear_content)
                if self.tool:
                    # Common-only tool spans never inherit model/request/usage.
                    for name in tuple(self.span._attributes or {}):
                        if name.startswith(
                            (
                                "gen_ai.request.",
                                "gen_ai.response.",
                                "gen_ai.usage.",
                                "llm.usage.",
                                "llm.request.",
                                f"{A.LLM_PROMPTS}.",
                                f"{A.LLM_COMPLETIONS}.",
                            )
                        ) or name in (
                            A.LLM_SYSTEM,
                            G.GEN_AI_PROVIDER_NAME,
                            A.GEN_AI_IS_STREAMING,
                        ):
                            self.span._attributes.pop(name, None)
            finally:
                safe(self.span.end)
                self.runtime.calls.discard(self)
                self.item = self.content = self.response = self.parent = None
                self.kwargs.clear()
                self.partial_calls.clear()
                self.usage.clear()
                if self.finalizer and self.finalizer.alive:
                    self.finalizer.detach()


class Stream:
    _native = frozenset({"send", "throw", "close", "__enter__", "__exit__"})

    def __init__(self, source, response, state):
        self.source, self.state = source, state
        self.response = weakref.ref(response)

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, "_native") and not hasattr(
            object.__getattribute__(self, "source"), name
        ):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __iter__(self):
        return self

    def advance(self, method, *args, closing=False):
        try:
            with self.state.scope():
                result = method(*args)
                if not closing:
                    safe(self.state.observe_chunk, result)
        except StopIteration:
            safe(self.state.finish, self.response())
            raise
        except BaseException as error:
            safe(self.state.finish, self.response(), error)
            raise
        if closing:
            safe(self.state.finish, self.response())
        return result

    def __next__(self):
        return self.advance(self.source.__next__)

    def send(self, value):
        return self.advance(self.source.send, value)

    def throw(self, *args):
        return self.advance(self.source.throw, *args)

    def close(self):
        return self.advance(self.source.close, closing=True)

    def _enter(self):
        try:
            result = self.source.__enter__()
        except BaseException as error:
            safe(self.state.finish, self.response(), error)
            raise
        finally:
            self.state.veto()
        return self if result is self.source else result

    def _exit(self, *args):
        try:
            return self.source.__exit__(*args)
        except BaseException as error:
            safe(self.state.finish, self.response(), error)
            raise
        finally:
            self.state.veto()
            safe(self.state.finish, self.response())


class AsyncStream:
    _native = frozenset({"asend", "athrow", "aclose", "__aenter__", "__aexit__"})

    def __init__(self, source, response, state):
        self.source, self.state = source, state
        self.response = weakref.ref(response)

    def __getattribute__(self, name):
        if name in object.__getattribute__(self, "_native") and not hasattr(
            object.__getattribute__(self, "source"), name
        ):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __aiter__(self):
        return self

    async def advance(self, method, *args, closing=False):
        try:
            with self.state.scope():
                result = await method(*args)
                if not closing:
                    safe(self.state.observe_chunk, result)
        except StopAsyncIteration:
            safe(self.state.finish, self.response())
            raise
        except BaseException as error:
            safe(self.state.finish, self.response(), error)
            raise
        if closing:
            safe(self.state.finish, self.response())
        return result

    async def __anext__(self):
        return await self.advance(self.source.__anext__)

    async def asend(self, value):
        return await self.advance(self.source.asend, value)

    async def athrow(self, *args):
        return await self.advance(self.source.athrow, *args)

    async def aclose(self):
        return await self.advance(self.source.aclose, closing=True)

    async def _aenter(self):
        try:
            result = await self.source.__aenter__()
        except BaseException as error:
            safe(self.state.finish, self.response(), error)
            raise
        finally:
            self.state.veto()
        return self if result is self.source else result

    async def _aexit(self, *args):
        try:
            return await self.source.__aexit__(*args)
        except BaseException as error:
            safe(self.state.finish, self.response(), error)
            raise
        finally:
            self.state.veto()
            safe(self.state.finish, self.response())


class CMStream(Stream):
    __enter__ = Stream._enter
    __exit__ = Stream._exit


class AsyncCMStream(AsyncStream):
    __aenter__ = AsyncStream._aenter
    __aexit__ = AsyncStream._aexit


def abandon(source, state, asynchronous):
    state.capture = False
    if not asynchronous:
        close = getattr(source, "close", None)
        if close:
            safe(close)
        safe(state.finish)
        return

    async def cleanup():
        close = getattr(source, "aclose", None)
        if close:
            try:
                await close()
            except Exception:  # noqa: BLE001, S110 - finalization preserves callers.
                pass
        safe(state.finish)

    try:
        asyncio.get_running_loop().create_task(cleanup())
    except RuntimeError:
        safe(asyncio.run, cleanup())


class Runtime:
    def __init__(self, capture, provider):
        self.capture, self.provider = capture, provider
        self.active = True
        self.owners = 0
        self.calls, self.patches = set(), []
        self.policy = Policy(capture)
        self.tracer = trace.get_tracer(
            MIRASCOPE_INSTRUMENTATION_NAME, tracer_provider=provider
        )
        self.registered = False

    def begin(self, item, args, kwargs, tool, stream):
        with _LOCK:
            if not self.active or not _is_respan_tracing_enabled() or suppressed():
                return None
            state = Call(self, item, args, kwargs, tool=tool, stream=stream)
            try:
                state.start()
                return None if state.finished else state
            except Exception:  # noqa: BLE001 - preserve native requests on start failure.
                if state.span is not None:
                    state.capture = False
                    safe(state.finish)
                return None

    def factory(self, original, *, tool=False, stream=False, asynchronous=False):
        def item(args, kwargs):
            return (
                kwargs.get("tool_call", args[-1] if len(args) > 1 else None)
                if tool
                else args[0]
                if args
                else kwargs.get("self")
            )

        def finalize(result, state, asynchronous):
            if state.finished:
                return result
            if not stream:
                safe(state.finish, result)
            else:
                source = translation.get(result, "_chunk_iterator")
                if source is not None:
                    if asynchronous:
                        cls = (
                            AsyncCMStream
                            if getattr(type(source), "__aenter__", None)
                            and getattr(type(source), "__aexit__", None)
                            else AsyncStream
                        )
                    else:
                        cls = (
                            CMStream
                            if getattr(type(source), "__enter__", None)
                            and getattr(type(source), "__exit__", None)
                            else Stream
                        )
                    result._chunk_iterator = cls(source, result, state)
                    state.finalizer = weakref.finalize(
                        result, abandon, source, state, asynchronous
                    )
                else:
                    safe(state.finish, result)
            return result

        if asynchronous or inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(*args, **kwargs):
                state = safe(self.begin, item(args, kwargs), args, kwargs, tool, stream)
                if state is None:
                    return await original(*args, **kwargs)
                try:
                    with state.scope():
                        result = await original(*args, **kwargs)
                except BaseException as error:
                    safe(state.finish, None, error)
                    raise
                observed = safe(finalize, result, state, True)
                if observed is None:
                    state.capture = False
                    safe(state.finish)
                return result
        else:

            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                state = safe(self.begin, item(args, kwargs), args, kwargs, tool, stream)
                if state is None:
                    return original(*args, **kwargs)
                try:
                    with state.scope():
                        result = original(*args, **kwargs)
                except BaseException as error:
                    safe(state.finish, None, error)
                    raise
                observed = safe(finalize, result, state, False)
                if observed is None:
                    state.capture = False
                    safe(state.finish)
                return result

        wrapper.__respan_mirascope_wrapper__ = True
        return wrapper

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self.patches.append((owner, name, original, replacement))
        setattr(owner, name, replacement)

    def install(self):
        if hasattr(self.provider, "add_span_processor"):
            self.registered = True
            self.provider.add_span_processor(self.policy)
        self.policy.install()
        model = importlib.import_module("mirascope.llm.models.models").Model
        for name in (
            "call",
            "context_call",
            "call_async",
            "context_call_async",
            "stream",
            "context_stream",
            "stream_async",
            "context_stream_async",
        ):
            self.patch(
                model,
                name,
                self.factory(
                    getattr(model, name),
                    stream="stream" in name,
                    asynchronous=name.endswith("_async"),
                ),
            )
        toolkit = importlib.import_module("mirascope.llm.tools.toolkit")
        for name in (
            "Toolkit",
            "ContextToolkit",
            "AsyncToolkit",
            "AsyncContextToolkit",
        ):
            owner = getattr(toolkit, name)
            self.patch(
                owner,
                "execute",
                self.factory(
                    owner.execute, tool=True, asynchronous=name.startswith("Async")
                ),
            )
        # These scoped source observations precede OpenAI's bool→int DTO coercion.
        try:
            from openai._legacy_response import LegacyAPIResponse
            from openai._response import APIResponse, AsyncAPIResponse
            from openai._streaming import ServerSentEvent
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("openai"):
                return  # OpenAI is an optional Mirascope provider extra.
            raise

        def observe_response(response):
            state = _CALL.get()
            if not self.active or state is None or state.runtime is not self:
                return
            raw = translation.get(response, "http_response")
            status = translation.get(raw, "status_code")
            if type(status) is int and 100 <= status <= 599:
                state.http_status = status
            if "json" in translation.get(
                translation.get(raw, "headers", {}), "content-type", ""
            ):
                text = translation.get(raw, "text")
                if isinstance(text, str):
                    state.usage = translation.raw_usage(text)
                    state.usage_seen = True

        def parsing(original):
            if inspect.iscoroutinefunction(original):

                @functools.wraps(original)
                async def async_parse(response, *args, **kwargs):
                    result = await original(response, *args, **kwargs)
                    safe(observe_response, response)
                    return result

                return async_parse

            @functools.wraps(original)
            def parse(response, *args, **kwargs):
                result = original(response, *args, **kwargs)
                safe(observe_response, response)
                return result

            return parse

        for response_class in (LegacyAPIResponse, APIResponse, AsyncAPIResponse):
            self.patch(response_class, "parse", parsing(response_class.parse))
        original_json = ServerSentEvent.json

        @functools.wraps(original_json)
        def event_json(event, *args, **kwargs):
            result = original_json(event, *args, **kwargs)
            state = _CALL.get()
            if self.active and state is not None and state.runtime is self:

                def observe():
                    source = (
                        translation.get(result, "response")
                        if translation.get(result, "type") == "response.completed"
                        else result
                    )
                    if translation.get(source, "usage") is not None:
                        state.usage = translation.raw_usage(source)
                        state.usage_seen = True

                safe(observe)
            return result

        self.patch(ServerSentEvent, "json", event_json)

    def close(self):
        self.active = False
        for state in tuple(self.calls):
            state.capture = False
            safe(state.finish)
        for owner, name, original, replacement in reversed(self.patches):
            if getattr(owner, name) is replacement:
                setattr(owner, name, original)
        self.patches.clear()
        self.policy.shutdown()
        active = translation.get(self.provider, "_active_span_processor")
        if (
            self.registered
            and active is not None
            and hasattr(active, "_span_processors")
        ):
            active._span_processors = tuple(
                p for p in active._span_processors if p is not self.policy
            )


class MirascopeInstrumentor:
    """Instrument the native Mirascope2.x Model and Toolkit execution surfaces."""

    name = MIRASCOPE_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content=True, tracer_provider=None):
        self._capture_content = capture_content
        self.provider = tracer_provider
        self.runtime = None
        self._is_instrumented = False

    def activate(self):
        global _RUNTIME
        with _LOCK:
            if self.runtime is not None or not _is_respan_tracing_enabled():
                return
            provider = self.provider or trace.get_tracer_provider()
            if _RUNTIME is None:
                runtime = Runtime(self._capture_content, provider)
                try:
                    runtime.install()
                except Exception as error:  # noqa: BLE001 - rollback is best effort.
                    runtime.close()
                    logger.warning(
                        "Mirascope instrumentation unavailable: %s",
                        type(error).__name__,
                    )
                    return
                _RUNTIME = runtime
            elif (
                _RUNTIME.capture != self._capture_content
                or _RUNTIME.provider is not provider
            ):
                raise ValueError(
                    "Mirascope owners must use the same capture policy and tracer provider"
                )
            self.runtime = _RUNTIME
            self.runtime.owners += 1
            self._is_instrumented = True

    def deactivate(self):
        global _RUNTIME
        with _LOCK:
            if self.runtime is None:
                return
            runtime = self.runtime
            self.runtime = None
            self._is_instrumented = False
            runtime.owners -= 1
            if runtime.owners == 0:
                runtime.close()
                if _RUNTIME is runtime:
                    _RUNTIME = None
