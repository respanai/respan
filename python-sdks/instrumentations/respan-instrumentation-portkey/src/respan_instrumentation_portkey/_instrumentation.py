"""Reference-counted ownership of Portkey's native OpenInference delegation."""

from __future__ import annotations

import importlib
import inspect
import threading
from typing import Any

from opentelemetry import trace
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_portkey._streaming import Runtime

_LOCK = threading.RLock()
_SHARED = None
_NATIVE_SPECS = [
    ("chat_complete", "Completions", "_original_completions_create"),
    ("chat_complete", "AsyncCompletions", "_original_async_completions_create"),
    ("generation", "Completions", "_original_prompt_completions_create"),
    ("generation", "AsyncCompletions", "_original_async_prompt_completions_create"),
]
_MISSING = object()


def _load_openinference_portkey_class():
    return importlib.import_module(
        "openinference.instrumentation.portkey"
    ).PortkeyInstrumentor


def _native_methods():
    result = []
    for module, name, attribute in _NATIVE_SPECS:
        cls = getattr(
            importlib.import_module("portkey_ai.api_resources.apis." + module), name
        )
        result.append((cls, attribute, inspect.getattr_static(cls, "create")))
    return result


def _restore_native(
    delegate, before, installed, owned, fields_before, fields_installed
):
    if not owned:
        return
    # Upstream uninstrument assigns blindly. Hand it the current foreign method
    # where ownership was replaced so its native teardown preserves that owner.
    current_fields = {name: getattr(delegate, name, _MISSING) for name in fields_before}
    temporary = {}
    for (cls, attribute, original), (_, _, wrapper) in zip(
        before, installed, strict=True
    ):
        current = inspect.getattr_static(cls, "create")
        temporary[attribute] = original if current is wrapper else current
        setattr(delegate, attribute, temporary[attribute])
    try:
        if delegate.is_instrumented_by_opentelemetry:
            delegate.uninstrument()
        else:
            delegate._uninstrument()
    finally:
        for (cls, attribute, original), (_, _, wrapper) in zip(
            before, installed, strict=True
        ):
            if inspect.getattr_static(cls, "create") is wrapper:
                cls.create = original
        for attribute, original in fields_before.items():
            expected = temporary.get(
                attribute, fields_installed.get(attribute, _MISSING)
            )
            if getattr(delegate, attribute, _MISSING) is not expected:
                continue
            if current_fields[attribute] is not fields_installed.get(
                attribute, _MISSING
            ):
                original = current_fields[attribute]
            if original is _MISSING:
                try:
                    delattr(delegate, attribute)
                except AttributeError:
                    pass
            else:
                setattr(delegate, attribute, original)
        delegate._is_instrumented_by_opentelemetry = False


class PortkeyInstrumentor:
    """Keep upstream chat/prompt instrumentation; add canonical SDK inference data."""

    name = "portkey"

    def __init__(
        self, *, capture_content=True, tracer_provider=None, **instrumentor_kwargs: Any
    ):
        self.capture_content = bool(capture_content)
        self.provider = tracer_provider
        self.kwargs = dict(instrumentor_kwargs)
        self._is_instrumented = False
        self._delegate = None

    def activate(self):
        global _SHARED
        if self._is_instrumented:
            return
        current = getattr(RespanTracer, "_instance", None)
        if current is not None and not current.is_enabled:
            return
        provider = self.provider or trace.get_tracer_provider()
        config = (provider, self.capture_content, self.kwargs)
        with _LOCK:
            if self._is_instrumented:
                return
            if _SHARED is not None:
                if _SHARED["config"] != config:
                    raise ValueError("Portkey owners must share provider and settings")
                _SHARED["count"] += 1
                self._delegate = _SHARED["delegate"]
                self._is_instrumented = True
                return
            delegate = _load_openinference_portkey_class()()
            owned = not delegate.is_instrumented_by_opentelemetry
            before = _native_methods()
            fields_before = {a: getattr(delegate, a, _MISSING) for _, a, _ in before}
            fields_before["_tracer"] = getattr(delegate, "_tracer", _MISSING)
            runtime = Runtime(provider, self.capture_content, self.kwargs.get("config"))
            runtime.native_owned = owned
            installed = before
            fields_installed = fields_before
            try:
                try:
                    if owned:
                        delegate.instrument(tracer_provider=provider, **self.kwargs)
                finally:
                    installed = _native_methods()
                    fields_installed = {
                        name: getattr(delegate, name, _MISSING)
                        for name in fields_before
                    }
                if runtime.config is None:
                    runtime.config = delegate._tracer._self_config
                runtime.install()
            except BaseException:
                runtime.close()
                _restore_native(
                    delegate, before, installed, owned, fields_before, fields_installed
                )
                raise
            _SHARED = {
                "config": config,
                "count": 1,
                "delegate": delegate,
                "runtime": runtime,
                "before": before,
                "installed": installed,
                "owned": owned,
                "fields_before": fields_before,
                "fields_installed": fields_installed,
            }
            self._delegate = delegate
            self._is_instrumented = True

    def deactivate(self):
        global _SHARED
        with _LOCK:
            if not self._is_instrumented:
                return
            self._is_instrumented = False
            self._delegate = None
            if _SHARED is None:
                return
            _SHARED["count"] -= 1
            if _SHARED["count"]:
                return
            shared = _SHARED
            _SHARED = None
            try:
                shared["runtime"].close()
            finally:
                _restore_native(
                    shared["delegate"],
                    shared["before"],
                    shared["installed"],
                    shared["owned"],
                    shared["fields_before"],
                    shared["fields_installed"],
                )

    def instrument(self):
        self.activate()

    def uninstrument(self):
        self.deactivate()
