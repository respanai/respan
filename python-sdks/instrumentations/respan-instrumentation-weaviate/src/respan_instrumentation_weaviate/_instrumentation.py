"""Native Weaviate v4 collection instrumentation for Respan."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from enum import Enum
from numbers import Real
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_instrumentation_weaviate._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    SENSITIVE_KEY_PARTS,
    WEAVIATE_INSTRUMENTATION_NAME,
    WEAVIATE_PATCH_SPECS,
    PatchSpec,
)
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)


_VECTOR_KEY_PARTS = ("embedding", "vector")


def _is_numeric_vector(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and bool(value)
        and all(isinstance(item, Real) and not isinstance(item, bool) for item in value)
    )


def _is_vector_key(value: str) -> bool:
    normalized = value.lower()
    return any(part in normalized for part in _VECTOR_KEY_PARTS)


def _jsonable(
    value: Any,
    *,
    depth: int = 0,
    vector_context: bool = False,
    preserved_vector: list[bool] | None = None,
) -> Any:
    preserved_vector = preserved_vector if preserved_vector is not None else [False]
    if depth > 7 and not (vector_context or _is_numeric_vector(value)):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Enum):
        return _jsonable(
            value.value,
            depth=depth + 1,
            vector_context=vector_context,
            preserved_vector=preserved_vector,
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(
            asdict(value),
            depth=depth + 1,
            vector_context=vector_context,
            preserved_vector=preserved_vector,
        )
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        omitted = 0
        regular_items = 0
        for key, item in value.items():
            key_text = str(key)
            item_is_vector = vector_context or _is_vector_key(key_text)
            if not item_is_vector and regular_items >= MAX_PREVIEW_ITEMS:
                omitted += 1
                continue
            regular_items += int(not item_is_vector)
            result[key_text] = (
                "<redacted>"
                if any(part in key_text.lower() for part in SENSITIVE_KEY_PARTS)
                else _jsonable(
                    item,
                    depth=depth + 1,
                    vector_context=item_is_vector,
                    preserved_vector=preserved_vector,
                )
            )
        if omitted:
            result["__truncated__"] = omitted
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        preserve_all = vector_context or _is_numeric_vector(value)
        if preserve_all:
            preserved_vector[0] = True
        source = value if preserve_all else value[:MAX_PREVIEW_ITEMS]
        items = [
            _jsonable(
                item,
                depth=depth + 1,
                vector_context=vector_context,
                preserved_vector=preserved_vector,
            )
            for item in source
        ]
        if preserve_all:
            return items
        if len(value) > MAX_PREVIEW_ITEMS:
            return {"count": len(value), "items": items, "truncated": True}
        return items
    for method_name in ("model_dump", "to_dict", "dict", "to_json", "tolist"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            return _jsonable(
                method(),
                depth=depth + 1,
                vector_context=vector_context,
                preserved_vector=preserved_vector,
            )
        except Exception:
            continue
    public_values = getattr(value, "__dict__", None)
    if isinstance(public_values, dict):
        return _jsonable(
            {
                key: item
                for key, item in public_values.items()
                if not key.startswith("_") and not callable(item)
            },
            depth=depth + 1,
            vector_context=vector_context,
            preserved_vector=preserved_vector,
        )
    return repr(value)


def _json_dumps(value: Any) -> str:
    preserved_vector = [False]
    text = json.dumps(
        _jsonable(value, preserved_vector=preserved_vector),
        default=str,
        sort_keys=True,
    )
    if preserved_vector[0] or len(text) <= MAX_ATTRIBUTE_CHARS:
        return text
    return json.dumps(
        {"preview": text[:MAX_ATTRIBUTE_CHARS], "truncated": True},
        sort_keys=True,
    )


def _instance_identity(instance: Any) -> dict[str, str]:
    identity: dict[str, str] = {}
    for source, target in (
        ("name", "collection_name"),
        ("_name", "collection_name"),
        ("tenant", "tenant"),
        ("_tenant", "tenant"),
    ):
        value = getattr(instance, source, None)
        if value is not None and target not in identity:
            identity[target] = str(value)
    return identity


def _call_input(
    label: str,
    operation: str,
    instance: Any,
    wrapped: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(wrapped).bind_partial(*args, **kwargs)
        arguments = {
            key: value for key, value in bound.arguments.items() if key != "self"
        }
    except Exception:
        arguments = {"args": list(args), "kwargs": kwargs}
    return {
        "operation": f"{label}.{operation}",
        **_instance_identity(instance),
        **arguments,
    }


class WeaviateInstrumentor:
    """Trace Weaviate v4 sync and async collection operations."""

    name = WEAVIATE_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _patched_targets: list[tuple[type, str]] = []
    _active_call: ContextVar[bool] = ContextVar(
        "respan_weaviate_active_call",
        default=False,
    )

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    @staticmethod
    def _tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        return tracer is None or bool(getattr(tracer, "is_enabled", True))

    def _set_start_attributes(
        self,
        span: Any,
        label: str,
        operation: str,
        instance: Any,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        operation_name = f"{label}.{operation}"
        entity_name = f"weaviate.{operation_name}"
        span.set_attribute(RESPAN_LOG_TYPE, LOG_TYPE_TASK)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
        span.set_attribute(OTelSpanAttributes.DB_SYSTEM, "weaviate")
        span.set_attribute(OTelSpanAttributes.DB_OPERATION, operation_name)
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                _json_dumps(
                    _call_input(
                        label,
                        operation,
                        instance,
                        wrapped,
                        args,
                        kwargs,
                    )
                ),
            )

    def _set_error(self, span: Any, exc: Exception) -> None:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute("status_code", 500)
        span.set_attribute(ERROR_MESSAGE_ATTR, str(exc))
        if self._capture_content:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_dumps({"error": type(exc).__name__, "message": str(exc)}),
            )

    def _trace_sync(
        self,
        label: str,
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
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"weaviate.{label}.{operation}",
                kind=SpanKind.CLIENT,
            ) as span:
                self._set_start_attributes(
                    span, label, operation, instance, wrapped, args, kwargs
                )
                try:
                    result = wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_dumps(result),
                    )
                return result
        finally:
            active_call.reset(token)

    async def _trace_async(
        self,
        label: str,
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
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"weaviate.{label}.{operation}",
                kind=SpanKind.CLIENT,
            ) as span:
                self._set_start_attributes(
                    span, label, operation, instance, wrapped, args, kwargs
                )
                try:
                    result = await wrapped(*args, **kwargs)
                except Exception as exc:
                    self._set_error(span, exc)
                    raise
                span.set_status(Status(StatusCode.OK))
                if self._capture_content:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        _json_dumps(result),
                    )
                return result
        finally:
            active_call.reset(token)

    def _patch_spec(self, spec: PatchSpec) -> list[tuple[type, str]]:
        try:
            module = importlib.import_module(spec.module)
            target_class = getattr(module, spec.class_name)
        except (ImportError, AttributeError):
            return []

        patched: list[tuple[type, str]] = []
        for operation in spec.methods:
            if not callable(getattr(target_class, operation, None)):
                continue

            def traced(
                wrapped: Any,
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                _label: str = spec.label,
                _operation: str = operation,
                _is_async: bool = spec.is_async,
            ) -> Any:
                if _is_async:
                    return self._trace_async(
                        _label,
                        _operation,
                        wrapped,
                        instance,
                        args,
                        kwargs,
                    )
                return self._trace_sync(
                    _label,
                    _operation,
                    wrapped,
                    instance,
                    args,
                    kwargs,
                )

            # Use the already imported module object. Importing a dotted private
            # Weaviate module a second time can re-enter the package's broad
            # ``weaviate.__init__`` import graph during startup.
            wrap_function_wrapper(
                module,
                f"{spec.class_name}.{operation}",
                traced,
            )
            patched.append((target_class, operation))
        return patched

    def activate(self) -> None:
        """Patch supported Weaviate v4 managers."""
        cls = type(self)
        if self._is_instrumented or not self._tracing_enabled():
            return
        if cls._patches_applied:
            cls._activation_count += 1
            self._is_instrumented = True
            return

        patched_targets: list[tuple[type, str]] = []
        for spec in WEAVIATE_PATCH_SPECS:
            patched_targets.extend(self._patch_spec(spec))

        self._is_instrumented = bool(patched_targets)
        cls._patches_applied = self._is_instrumented
        cls._activation_count = int(self._is_instrumented)
        cls._patched_targets = patched_targets
        if not self._is_instrumented:
            logger.warning("Weaviate instrumentation found no supported v4 methods")

    def deactivate(self) -> None:
        """Remove Weaviate patches after the final active instance stops."""
        if not self._is_instrumented:
            return
        self._is_instrumented = False
        cls = type(self)
        cls._activation_count = max(cls._activation_count - 1, 0)
        if cls._activation_count:
            return
        for target_class, operation in reversed(cls._patched_targets):
            try:
                unwrap(target_class, operation)
            except Exception:
                logger.debug(
                    "Failed to unwrap Weaviate %s.%s",
                    target_class.__name__,
                    operation,
                    exc_info=True,
                )
        cls._patched_targets = []
        cls._patches_applied = False

    def instrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`activate`."""
        self.activate()

    def uninstrument(self) -> None:
        """OpenTelemetry-style alias for :meth:`deactivate`."""
        self.deactivate()
