"""Small reusable base for native client-library instrumentation plugins."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_MAX_ITEMS = 12
_MAX_CHARS = 16_000
_SENSITIVE_PARTS = ("api_key", "authorization", "password", "secret", "token")


@dataclass(frozen=True)
class PatchSpec:
    """A vendor class and the public operations that should be wrapped."""

    module: str
    class_name: str
    methods: tuple[str, ...] | None = None
    is_async: bool = False
    label: str = "client"
    exclude: frozenset[str] = field(default_factory=frozenset)


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                result["__truncated__"] = max(len(value) - _MAX_ITEMS, 0)
                break
            key_text = str(key)
            result[key_text] = (
                "<redacted>"
                if any(part in key_text.lower() for part in _SENSITIVE_PARTS)
                else _jsonable(item, depth=depth + 1)
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [_jsonable(item, depth=depth + 1) for item in value[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            return {"count": len(value), "items": items, "truncated": True}
        return items
    for method_name in ("model_dump", "to_dict", "dict", "to_pylist"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _jsonable(method(), depth=depth + 1)
            except Exception:
                pass
    return repr(value)


def _json_dumps(value: Any) -> str:
    text = json.dumps(_jsonable(value), default=str, sort_keys=True)
    if len(text) <= _MAX_CHARS:
        return text
    return json.dumps({"preview": text[:_MAX_CHARS], "truncated": True})


def _call_input(
    wrapped: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(wrapped).bind_partial(*args, **kwargs)
        return {key: value for key, value in bound.arguments.items() if key != "self"}
    except Exception:
        return {"args": list(args), "kwargs": kwargs}


def _instance_identity(instance: Any) -> dict[str, str]:
    identity: dict[str, str] = {}
    for key in (
        "name",
        "_name",
        "table_name",
        "_table_name",
        "index_name",
        "_index_name",
        "collection_name",
        "_collection_name",
        "uri",
        "_uri",
    ):
        value = getattr(instance, key, None)
        if isinstance(value, (str, int)):
            identity[key.lstrip("_")] = str(value)
    config = getattr(instance, "_config", None)
    host = getattr(config, "host", None)
    if isinstance(host, str):
        identity["host"] = host
    return identity


class NativeClientInstrumentor:
    """Base lifecycle and span mapping for vendor client adapters."""

    name = "native-client"
    vendor = "native-client"
    patches: tuple[PatchSpec, ...] = ()
    _patches_applied = False
    _activation_count = 0
    _patched_targets: list[tuple[type, str]] = []
    _active_call: ContextVar[bool] = ContextVar(
        "respan_native_client_active",
        default=False,
    )

    def __init__(self) -> None:
        self._is_instrumented = False

    @staticmethod
    def _tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    @classmethod
    def _operation_name(cls, patch: PatchSpec, method: str) -> str:
        return f"{patch.label}.{method}" if patch.label else method

    @classmethod
    def _span_name(cls, operation: str) -> str:
        return f"{cls.vendor}.{operation}"

    @classmethod
    def _set_start_attributes(
        cls,
        span: Any,
        operation: str,
        instance: Any,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        entity_name = cls._span_name(operation)
        payload = {
            "operation": operation,
            **_instance_identity(instance),
            **_call_input(wrapped, args, kwargs),
        }
        span.set_attribute(RESPAN_LOG_TYPE, "task")
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            _json_dumps(payload),
        )
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, cls.vendor)
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, operation)

    @staticmethod
    def _set_error(span: Any, exc: Exception) -> None:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_dumps({"error": type(exc).__name__, "message": str(exc)}),
        )

    def _trace_sync(
        self,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            tracer = trace.get_tracer(type(self).__module__)
            with tracer.start_as_current_span(
                self._span_name(operation),
                kind=SpanKind.CLIENT,
            ) as span:
                self._set_start_attributes(
                    span,
                    operation,
                    instance,
                    wrapped,
                    args,
                    kwargs,
                )
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    _json_dumps(result),
                )
                return result
        finally:
            active_call.reset(token)

    async def _trace_async(
        self,
        operation: str,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_call = type(self)._active_call
        if active_call.get():
            return await wrapped(*args, **kwargs)
        token = active_call.set(True)
        try:
            tracer = trace.get_tracer(type(self).__module__)
            with tracer.start_as_current_span(
                self._span_name(operation),
                kind=SpanKind.CLIENT,
            ) as span:
                self._set_start_attributes(
                    span,
                    operation,
                    instance,
                    wrapped,
                    args,
                    kwargs,
                )
                try:
                    result = await wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                    _json_dumps(result),
                )
                return result
        finally:
            active_call.reset(token)

    @staticmethod
    def _methods_for(target_class: type, patch: PatchSpec) -> tuple[str, ...]:
        if patch.methods is not None:
            return patch.methods
        return tuple(
            name
            for name in dir(target_class)
            if not name.startswith("_")
            and name not in patch.exclude
            and callable(getattr(target_class, name, None))
        )

    def activate(self) -> None:
        cls = type(self)
        if self._is_instrumented or not self._tracing_enabled():
            return
        if cls._patches_applied:
            cls._activation_count += 1
            self._is_instrumented = True
            return

        patched_targets: list[tuple[type, str]] = []
        for patch in cls.patches:
            try:
                module = importlib.import_module(patch.module)
                target_class = getattr(module, patch.class_name)
            except (ImportError, AttributeError):
                continue
            for method in self._methods_for(target_class, patch):
                if not hasattr(target_class, method):
                    continue
                operation = self._operation_name(patch, method)

                def traced(
                    wrapped: Any,
                    instance: Any,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    _operation: str = operation,
                    _async: bool = patch.is_async,
                ) -> Any:
                    if _async:
                        return self._trace_async(
                            _operation,
                            wrapped,
                            instance,
                            args,
                            kwargs,
                        )
                    return self._trace_sync(
                        _operation,
                        wrapped,
                        instance,
                        args,
                        kwargs,
                    )

                target = f"{patch.class_name}.{method}"
                wrap_function_wrapper(patch.module, target, traced)
                patched_targets.append((target_class, method))

        self._is_instrumented = bool(patched_targets)
        cls._patches_applied = self._is_instrumented
        cls._activation_count = int(self._is_instrumented)
        cls._patched_targets = patched_targets
        if not self._is_instrumented:
            logger.warning(
                "%s instrumentation found no supported methods",
                cls.vendor,
            )

    def deactivate(self) -> None:
        if not self._is_instrumented:
            return
        self._is_instrumented = False
        cls = type(self)
        cls._activation_count = max(cls._activation_count - 1, 0)
        if cls._activation_count:
            return

        for target_class, method in reversed(cls._patched_targets):
            try:
                unwrap(target_class, method)
            except Exception:
                logger.debug(
                    "Failed to unwrap %s.%s.%s",
                    target_class.__module__,
                    target_class.__qualname__,
                    method,
                    exc_info=True,
                )
        cls._patched_targets = []
        cls._patches_applied = False
