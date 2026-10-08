"""Owned, shared hooks into LangChain callback manager configuration."""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from langchain_core.callbacks.base import BaseCallbackManager
from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from ._callback import RespanCallbackHandler, _with_respan_callback

_MISSING = object()
_LOCK = RLock()
_RUNTIME = None


@dataclass
class _Runtime:
    handler: RespanCallbackHandler
    provider: Any
    settings: tuple
    owned_handler: bool
    owners: int = 1
    active: bool = True
    patches: list = field(default_factory=list)

    def patch(self, owner, name, value):
        original = owner.__dict__.get(name, _MISSING)
        self.patches.append((owner, name, original, value))
        setattr(owner, name, value)

    def restore(self):
        self.active = False
        try:
            for owner, name, original, patched in reversed(self.patches):
                if owner.__dict__.get(name) is patched:
                    if original is _MISSING:
                        delattr(owner, name)
                    else:
                        setattr(owner, name, original)
        finally:
            self.patches.clear()
            if self.owned_handler:
                self.handler.shutdown()

    def install(self):
        module = importlib.import_module("langchain_core.callbacks.manager")
        for manager in (module.CallbackManager, module.AsyncCallbackManager):
            descriptor = manager.configure
            original = descriptor.__func__

            @functools.wraps(original)
            def configure(
                cls,
                inheritable_callbacks=None,
                local_callbacks=None,
                *args,
                __original=original,
                **kwargs,
            ):
                if self.active:
                    local = (
                        local_callbacks.handlers
                        if isinstance(local_callbacks, BaseCallbackManager)
                        else local_callbacks
                        if isinstance(local_callbacks, (list, tuple))
                        else ()
                    )
                    if not any(
                        isinstance(callback, RespanCallbackHandler)
                        for callback in local
                    ):
                        inheritable_callbacks = _with_respan_callback(
                            inheritable_callbacks, self.handler
                        )
                return __original(
                    cls, inheritable_callbacks, local_callbacks, *args, **kwargs
                )

            self.patch(manager, "configure", classmethod(configure))
        # Older graph releases exposed additional config helpers. Current graph
        # execution already uses LangChain's callback managers.
        try:
            module = importlib.import_module("langgraph.callbacks")
        except ImportError:
            return
        for name in (
            "get_sync_graph_callback_manager_for_config",
            "get_async_graph_callback_manager_for_config",
        ):
            original = getattr(module, name, None)
            if original is None:
                continue

            @functools.wraps(original)
            def configure(config, *args, __original=original, **kwargs):
                if self.active:
                    config = dict(config or {})
                    config["callbacks"] = _with_respan_callback(
                        config.get("callbacks"), self.handler
                    )
                return __original(config, *args, **kwargs)

            self.patch(module, name, configure)


class LangChainInstrumentor:
    """Install one native callback handler shared by compatible owners."""

    name = "langchain"

    def __init__(
        self, *, callback_handler=None, include_content=True, include_metadata=True
    ):
        self._handler = callback_handler or RespanCallbackHandler(
            include_content=include_content, include_metadata=include_metadata
        )
        self._provided = callback_handler is not None
        self._settings = (
            id(callback_handler) if callback_handler is not None else None,
            self._handler.include_content,
            self._handler.include_metadata,
        )
        self._is_instrumented = False
        self._runtime = None

    @property
    def callback_handler(self):
        return self._runtime.handler if self._runtime is not None else self._handler

    def activate(self):
        global _RUNTIME
        with _LOCK:
            if self._is_instrumented:
                return
            tracer = getattr(RespanTracer, "_instance", None)
            if tracer is not None and not getattr(tracer, "is_enabled", True):
                return
            provider = trace.get_tracer_provider()
            if _RUNTIME is not None:
                if _RUNTIME.provider is not provider:
                    raise RuntimeError(
                        "LangChain instrumentation is active on another provider"
                    )
                if _RUNTIME.settings != self._settings:
                    raise ValueError(
                        "LangChain instrumentation is active with different settings"
                    )
                _RUNTIME.owners += 1
                self._runtime = _RUNTIME
                self._is_instrumented = True
                return
            self._handler._enabled = True
            runtime = _Runtime(
                self._handler, provider, self._settings, not self._provided
            )
            try:
                runtime.install()
            except BaseException:
                runtime.restore()
                raise
            _RUNTIME = runtime
            self._runtime = runtime
            self._is_instrumented = True

    def deactivate(self):
        global _RUNTIME
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            runtime, self._runtime = self._runtime, None
            runtime.owners -= 1
            if runtime.owners:
                return
            _RUNTIME = None
            runtime.restore()
