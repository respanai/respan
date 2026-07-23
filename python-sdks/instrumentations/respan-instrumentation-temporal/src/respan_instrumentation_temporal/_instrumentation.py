"""Temporal's official tracing interceptor adapted to the Respan contract."""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_temporal._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_COLLECTION_ITEMS,
    MAX_SERIALIZATION_DEPTH,
    TASK_LOG_TYPE,
    TEMPORAL_CAPTURED_INPUT,
    TEMPORAL_CLIENT_CONNECT_TARGET,
    TEMPORAL_CLIENT_MODULE,
    TEMPORAL_INSTRUMENTATION_NAME,
    TEMPORAL_OTEL_MODULE,
    TEMPORAL_RAW_ATTRIBUTE_KEYS,
    WORKFLOW_LOG_TYPE,
    WORKFLOW_OPERATION_PREFIXES,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_SAFE_DETAIL = re.compile(r"[^a-zA-Z0-9_.-]+")
_MISSING = object()


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_SERIALIZATION_DEPTH:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        items = list(value.items())
        payload = {
            str(key): _to_jsonable(item, depth=depth + 1)
            for key, item in items[:MAX_COLLECTION_ITEMS]
            if str(key).lower() not in {"headers", "rpc_metadata"}
        }
        if len(items) > MAX_COLLECTION_ITEMS:
            payload["_respan_truncated_items"] = len(items) - MAX_COLLECTION_ITEMS
        return payload
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        payload = [
            _to_jsonable(item, depth=depth + 1) for item in items[:MAX_COLLECTION_ITEMS]
        ]
        if len(items) > MAX_COLLECTION_ITEMS:
            payload.append(
                {"_respan_truncated_items": len(items) - MAX_COLLECTION_ITEMS}
            )
        return payload
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(model_dump(mode="json"), depth=depth + 1)
        except TypeError:
            return _to_jsonable(model_dump(), depth=depth + 1)
        except Exception:
            return repr(value)
    if hasattr(value, "__dict__"):
        return {
            key: _to_jsonable(item, depth=depth + 1)
            for key, item in vars(value).items()
            if not key.startswith("_")
            and key.lower() not in {"headers", "rpc_metadata"}
            and not callable(item)
        }
    return repr(value)


def _json_dumps(value: Any, *, max_chars: int = MAX_ATTRIBUTE_CHARS) -> str:
    serialized = json.dumps(
        _to_jsonable(value),
        default=str,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(serialized) <= max_chars:
        return serialized
    return json.dumps(
        {
            "truncated": True,
            "original_characters": len(serialized),
            "preview": serialized[:max_chars],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _snake_case(value: str) -> str:
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", value)
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value)
    return value.replace("-", "_").lower()


def _span_parts(name: str) -> tuple[str, str | None]:
    operation, separator, detail = name.partition(":")
    return operation, detail if separator and detail else None


def _safe_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    cleaned = _SAFE_DETAIL.sub("_", detail).strip("_.-")
    return cleaned[:120] or None


def _extract_temporal_input(value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in (
        "args",
        "arg",
        "id",
        "workflow",
        "workflow_type",
        "activity",
        "activity_type",
        "query",
        "signal",
        "update",
        "update_id",
        "task_queue",
    ):
        item = getattr(value, field_name, _MISSING)
        if item is not _MISSING and item is not None and not callable(item):
            payload[field_name] = item
    return payload


def _canonical_attributes(
    name: str,
    attributes: Mapping[str, Any] | None,
    *,
    capture_content: bool,
    max_attribute_chars: int,
) -> dict[str, Any]:
    source = dict(attributes or {})
    captured_input = source.pop(TEMPORAL_CAPTURED_INPUT, None)
    temporal_attributes = {
        key: source.pop(key)
        for key in tuple(source)
        if key in TEMPORAL_RAW_ATTRIBUTE_KEYS
    }
    operation, detail = _span_parts(name)
    safe_detail = _safe_detail(detail)
    operation_name = _snake_case(operation)
    entity_name = f"temporal.{operation_name}"
    if safe_detail:
        entity_name = f"{entity_name}.{safe_detail}"
    log_type = (
        WORKFLOW_LOG_TYPE if operation in WORKFLOW_OPERATION_PREFIXES else TASK_LOG_TYPE
    )

    input_payload: dict[str, Any] = {
        "operation": operation_name,
        "detail": detail,
        "content_captured": capture_content,
    }
    if capture_content:
        if temporal_attributes:
            input_payload["temporal"] = temporal_attributes
        if captured_input:
            input_payload["input"] = captured_input

    source[RESPAN_LOG_TYPE] = log_type
    source[SpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    source[SpanAttributes.TRACELOOP_ENTITY_PATH] = entity_name
    source[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _json_dumps(
        input_payload, max_chars=max_attribute_chars
    )
    source[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_dumps(
        {"status": "completed", "content_captured": capture_content},
        max_chars=max_attribute_chars,
    )
    return source


def _set_success_output(span: Any, *, capture_content: bool, max_chars: int) -> None:
    span.set_attribute("status_code", 200)
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        _json_dumps(
            {"status": "completed", "content_captured": capture_content},
            max_chars=max_chars,
        ),
    )


def _set_error_output(
    span: Any,
    exc: BaseException,
    *,
    capture_content: bool,
    max_chars: int,
    record_exception: bool,
) -> None:
    message = str(exc) if capture_content else type(exc).__name__
    if capture_content and record_exception and isinstance(exc, Exception):
        span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, message))
    span.set_attribute("status_code", 500)
    span.set_attribute("error.message", message)
    span.set_attribute(
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        _json_dumps(
            {
                "status": "error",
                "error": type(exc).__name__,
                "message": message,
                "content_captured": capture_content,
            },
            max_chars=max_chars,
        ),
    )


class _CanonicalSpanProxy:
    def __init__(
        self, span: Any, *, capture_content: bool, max_attribute_chars: int
    ) -> None:
        self._span = span
        self._capture_content = capture_content
        self._max_attribute_chars = max_attribute_chars
        self._has_error = False

    def record_exception(self, exception: Exception, *args: Any, **kwargs: Any) -> None:
        self._has_error = True
        _set_error_output(
            self._span,
            exception,
            capture_content=self._capture_content,
            max_chars=self._max_attribute_chars,
            record_exception=False,
        )
        if self._capture_content:
            self._span.record_exception(exception, *args, **kwargs)

    def end(self, *args: Any, **kwargs: Any) -> None:
        if not self._has_error:
            _set_success_output(
                self._span,
                capture_content=self._capture_content,
                max_chars=self._max_attribute_chars,
            )
        self._span.end(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)


class _CanonicalTracer:
    def __init__(
        self, tracer: Any, *, capture_content: bool, max_attribute_chars: int
    ) -> None:
        self._tracer = tracer
        self._capture_content = capture_content
        self._max_attribute_chars = max_attribute_chars

    @contextmanager
    def start_as_current_span(self, name: str, *args: Any, **kwargs: Any):
        kwargs["attributes"] = _canonical_attributes(
            name,
            kwargs.get("attributes"),
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
        )
        with self._tracer.start_as_current_span(name, *args, **kwargs) as span:
            try:
                yield span
            except BaseException as exc:
                _set_error_output(
                    span,
                    exc,
                    capture_content=self._capture_content,
                    max_chars=self._max_attribute_chars,
                    record_exception=True,
                )
                raise
            else:
                _set_success_output(
                    span,
                    capture_content=self._capture_content,
                    max_chars=self._max_attribute_chars,
                )

    def start_span(self, name: str, *args: Any, **kwargs: Any) -> _CanonicalSpanProxy:
        kwargs["attributes"] = _canonical_attributes(
            name,
            kwargs.get("attributes"),
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
        )
        span = self._tracer.start_span(name, *args, **kwargs)
        return _CanonicalSpanProxy(
            span,
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracer, name)


def _build_interceptor(
    base_class: type,
    *,
    tracer: Any,
    capture_content: bool,
    max_attribute_chars: int,
    always_create_workflow_spans: bool,
) -> Any:
    canonical_tracer = _CanonicalTracer(
        tracer,
        capture_content=capture_content,
        max_attribute_chars=max_attribute_chars,
    )

    class RespanTemporalTracingInterceptor(base_class):
        @contextmanager
        def _start_as_current_span(
            self,
            name: str,
            *,
            attributes: Mapping[str, Any] | None,
            input_with_headers: Any = None,
            input_with_ctx: Any = None,
            kind: Any,
            context: Any = None,
        ):
            enriched = dict(attributes or {})
            if capture_content:
                captured: dict[str, Any] = {}
                if input_with_headers is not None:
                    captured.update(_extract_temporal_input(input_with_headers))
                if input_with_ctx is not None:
                    captured.update(_extract_temporal_input(input_with_ctx))
                if captured:
                    enriched[TEMPORAL_CAPTURED_INPUT] = captured
            with super()._start_as_current_span(
                name,
                attributes=enriched,
                input_with_headers=input_with_headers,
                input_with_ctx=input_with_ctx,
                kind=kind,
                context=context,
            ):
                yield None

    RespanTemporalTracingInterceptor.__name__ = "RespanTemporalTracingInterceptor"
    return RespanTemporalTracingInterceptor(
        tracer=canonical_tracer,
        always_create_workflow_spans=always_create_workflow_spans,
    )


class TemporalInstrumentor:
    """Inject a canonicalized official Temporal tracing interceptor."""

    name = TEMPORAL_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _patched_targets: list[tuple[str, str]] = []

    def __init__(
        self,
        *,
        capture_content: bool = True,
        always_create_workflow_spans: bool = False,
        max_attribute_chars: int = MAX_ATTRIBUTE_CHARS,
    ) -> None:
        self._capture_content = capture_content
        self._always_create_workflow_spans = always_create_workflow_spans
        self._max_attribute_chars = max(512, int(max_attribute_chars))
        self._is_instrumented = False
        self._interceptor: Any = None
        self._base_interceptor_class: type | None = None

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _ensure_interceptor(self) -> Any:
        if self._interceptor is not None:
            return self._interceptor
        otel_module = importlib.import_module(TEMPORAL_OTEL_MODULE)
        self._base_interceptor_class = getattr(otel_module, "TracingInterceptor")
        self._interceptor = _build_interceptor(
            self._base_interceptor_class,
            tracer=trace.get_tracer(__name__),
            capture_content=self._capture_content,
            max_attribute_chars=self._max_attribute_chars,
            always_create_workflow_spans=self._always_create_workflow_spans,
        )
        return self._interceptor

    @property
    def interceptor(self) -> Any:
        """The interceptor for explicit Temporal client/test-environment wiring."""
        return self._ensure_interceptor()

    async def _connect(
        self,
        wrapped: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        interceptor = self._ensure_interceptor()
        connect_kwargs = dict(kwargs)
        interceptors = list(connect_kwargs.get("interceptors") or ())
        base_class = self._base_interceptor_class
        has_temporal_tracing = bool(
            base_class is not None
            and any(isinstance(candidate, base_class) for candidate in interceptors)
        )
        if not has_temporal_tracing:
            interceptors.append(interceptor)
            connect_kwargs["interceptors"] = interceptors
        return await wrapped(*args, **connect_kwargs)

    def activate(self) -> None:
        """Patch `Client.connect` to inject the Respan Temporal interceptor."""
        cls = type(self)
        if self._is_instrumented:
            return
        if not self._is_respan_tracing_enabled():
            logger.info(
                "Temporal instrumentation skipped because Respan tracing is disabled"
            )
            return
        if cls._patches_applied:
            cls._activation_count += 1
            self._is_instrumented = True
            return
        try:
            client_module = importlib.import_module(TEMPORAL_CLIENT_MODULE)
            client_class = getattr(client_module, "Client", None)
            if client_class is None or not hasattr(client_class, "connect"):
                logger.warning("Temporal Client.connect is unavailable")
                return
            self._ensure_interceptor()

            async def traced_connect(
                wrapped: Any,
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> Any:
                return await self._connect(wrapped, args, kwargs)

            wrap_function_wrapper(
                TEMPORAL_CLIENT_MODULE,
                TEMPORAL_CLIENT_CONNECT_TARGET,
                traced_connect,
            )
            cls._patched_targets.append(
                (TEMPORAL_CLIENT_MODULE, TEMPORAL_CLIENT_CONNECT_TARGET)
            )
        except ImportError as exc:
            logger.warning(
                "Failed to activate Temporal instrumentation - missing dependency: %s",
                exc,
            )
            return
        except Exception:
            logger.exception("Failed to activate Temporal instrumentation")
            self.deactivate()
            return
        cls._patches_applied = True
        cls._activation_count = 1
        self._is_instrumented = True
        logger.info("Temporal instrumentation activated")

    def deactivate(self) -> None:
        """Restore `Client.connect`; existing clients retain their interceptor."""
        cls = type(self)
        if not self._is_instrumented:
            if cls._patches_applied or not cls._patched_targets:
                return
        else:
            self._is_instrumented = False
            cls._activation_count = max(cls._activation_count - 1, 0)
            if cls._activation_count:
                return
        for module_path, target in reversed(cls._patched_targets):
            try:
                unwrap(module_path, target)
            except Exception:
                logger.debug(
                    "Failed to unwrap %s.%s", module_path, target, exc_info=True
                )
        cls._patched_targets.clear()
        cls._patches_applied = False
        cls._activation_count = 0
        logger.info("Temporal instrumentation deactivated")
