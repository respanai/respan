"""Native sync/async Elasticsearch transport instrumentation for Respan."""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode
from respan_instrumentation_elasticsearch._constants import (
    ELASTICSEARCH_INSTRUMENTATION_NAME,
    ELASTICSEARCH_METHOD,
    ELASTICSEARCH_STATUS_CODE,
    ELASTICSEARCH_TARGET,
    ELASTIC_TRANSPORT_MODULE,
    ELASTIC_TRANSPORT_TARGETS,
    MAX_ATTRIBUTE_CHARS,
    MAX_COLLECTION_ITEMS,
    MAX_SERIALIZATION_DEPTH,
    TASK_LOG_TYPE,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_DOCUMENT_ID_RE = re.compile(r"(?P<prefix>/(?:_doc|_update|_source)/)[^/?]+")
_TASK_ID_RE = re.compile(r"(?P<prefix>/_tasks/)[^/?]+")


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_SERIALIZATION_DEPTH:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        items = list(value.items())
        payload = {
            str(key): _to_jsonable(item, depth=depth + 1)
            for key, item in items[:MAX_COLLECTION_ITEMS]
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
    body = getattr(value, "body", None)
    if body is not None and body is not value:
        return _to_jsonable(body, depth=depth + 1)
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


def _request_parts(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, str, Any]:
    method = kwargs.get("method")
    target = kwargs.get("target")
    if method is None and args:
        method = args[0]
    if target is None and len(args) > 1:
        target = args[1]
    return str(method or "REQUEST").upper(), str(target or "/"), kwargs.get("body")


def _sanitize_target(target: str) -> str:
    sanitized = target.split("?", 1)[0]
    sanitized = _DOCUMENT_ID_RE.sub(
        lambda match: f"{match.group('prefix')}:id", sanitized
    )
    sanitized = _TASK_ID_RE.sub(lambda match: f"{match.group('prefix')}:id", sanitized)
    return sanitized


def _operation_name(method: str, target: str) -> str:
    path = target.split("?", 1)[0]
    segments = [segment for segment in path.split("/") if segment]
    segment_set = set(segments)

    if "_search" in segment_set or "_msearch" in segment_set:
        return "search"
    if "_bulk" in segment_set:
        return "bulk"
    if "_update_by_query" in segment_set:
        return "update_by_query"
    if "_delete_by_query" in segment_set:
        return "delete_by_query"
    if "_update" in segment_set:
        return "update"
    if "_doc" in segment_set or "_source" in segment_set:
        return {
            "GET": "get",
            "HEAD": "exists",
            "DELETE": "delete",
        }.get(method, "index")
    if "_cluster" in segment_set:
        suffix = segments[segments.index("_cluster") + 1 :]
        return "cluster_" + (suffix[0] if suffix else "request")
    if "_indices" in segment_set:
        return "indices_request"
    if "_refresh" in segment_set:
        return "refresh"
    if "_count" in segment_set:
        return "count"
    if "_cat" in segment_set:
        suffix = segments[segments.index("_cat") + 1 :]
        return "cat_" + (suffix[0] if suffix else "request")
    if not segments:
        return "info"
    return "request"


def _response_body(response: Any) -> Any:
    return getattr(response, "body", response)


def _status_code(value: Any, default: int = 200) -> int:
    meta = getattr(value, "meta", None)
    status = getattr(meta, "status", None)
    if isinstance(status, int):
        return status
    if isinstance(value, Mapping):
        raw_status = value.get("status") or value.get("status_code")
        if isinstance(raw_status, int):
            return raw_status
    return default


def _error_status_code(exc: BaseException) -> int:
    status = _status_code(exc, default=500)
    return status if status >= 400 else 500


def _error_message(body: Any, fallback: str) -> str:
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            return str(error.get("reason") or error.get("type") or fallback)
        if error is not None:
            return str(error)
    return fallback


def _active_elasticsearch_span() -> Any | None:
    """Return the official client span so it can be normalized in place."""
    span = trace.get_current_span()
    is_recording = getattr(span, "is_recording", None)
    if callable(is_recording) and not is_recording():
        return None
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_info", None
    )
    scope_name = str(getattr(scope, "name", "") or "")
    if scope_name.startswith(("elasticsearch", "elastic_transport")):
        return span
    return None


class ElasticsearchInstrumentor:
    """Trace all official Elasticsearch sync and async transport requests."""

    name = ELASTICSEARCH_INSTRUMENTATION_NAME
    _patches_applied = False
    _activation_count = 0
    _patched_targets: list[tuple[str, str]] = []

    def __init__(
        self,
        *,
        capture_content: bool = True,
        max_attribute_chars: int = MAX_ATTRIBUTE_CHARS,
        request_hook: Callable[[Any, str, str, dict[str, Any]], None] | None = None,
        response_hook: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self._capture_content = capture_content
        self._max_attribute_chars = max(512, int(max_attribute_chars))
        self._request_hook = request_hook
        self._response_hook = response_hook
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _set_request_attributes(
        self,
        span: Any,
        *,
        method: str,
        target: str,
        body: Any,
        kwargs: dict[str, Any],
    ) -> str:
        operation = _operation_name(method, target)
        entity_name = f"elasticsearch.{operation}"
        sanitized_target = _sanitize_target(target)

        span.set_attribute(RESPAN_LOG_TYPE, TASK_LOG_TYPE)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_name)
        span.set_attribute(ELASTICSEARCH_METHOD, method)
        span.set_attribute(ELASTICSEARCH_TARGET, sanitized_target)

        input_payload: dict[str, Any] = {
            "operation": operation,
            "method": method,
            "target": target if self._capture_content else sanitized_target,
            "content_captured": self._capture_content,
        }
        if self._capture_content:
            if body is not None:
                input_payload["body"] = body
            for key in ("params", "request_timeout", "max_retries"):
                value = kwargs.get(key)
                if value is not None:
                    input_payload[key] = value
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            _json_dumps(input_payload, max_chars=self._max_attribute_chars),
        )
        return operation

    def _set_response_attributes(self, span: Any, response: Any) -> None:
        body = _response_body(response)
        status_code = _status_code(response)
        span.set_attribute(ELASTICSEARCH_STATUS_CODE, status_code)
        span.set_attribute("status_code", status_code)
        output = (
            {"status_code": status_code, "body": body}
            if self._capture_content
            else {"status_code": status_code, "content_captured": False}
        )
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_dumps(output, max_chars=self._max_attribute_chars),
        )
        if status_code >= 400:
            message = (
                _error_message(body, f"Elasticsearch returned HTTP {status_code}")
                if self._capture_content
                else f"Elasticsearch returned HTTP {status_code}"
            )
            span.set_attribute("error.message", message)
            span.set_status(Status(StatusCode.ERROR, message))

    def _set_error_attributes(self, span: Any, exc: BaseException) -> None:
        status_code = _error_status_code(exc)
        message = str(exc) if self._capture_content else type(exc).__name__
        if self._capture_content:
            span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, message))
        span.set_attribute(ELASTICSEARCH_STATUS_CODE, status_code)
        span.set_attribute("status_code", status_code)
        span.set_attribute("error.message", message)
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_dumps(
                {
                    "error": type(exc).__name__,
                    "message": message,
                    "status_code": status_code,
                    "content_captured": self._capture_content,
                },
                max_chars=self._max_attribute_chars,
            ),
        )

    def _trace_sync(
        self,
        wrapped: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        method, target, body = _request_parts(args, kwargs)
        operation = _operation_name(method, target)
        active_span = _active_elasticsearch_span()
        span_context = (
            nullcontext(active_span)
            if active_span is not None
            else trace.get_tracer(__name__).start_as_current_span(
                f"elasticsearch.{operation}", kind=SpanKind.CLIENT
            )
        )
        with span_context as span:
            self._set_request_attributes(
                span, method=method, target=target, body=body, kwargs=kwargs
            )
            if self._request_hook is not None:
                self._request_hook(span, method, target, kwargs)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exc:
                self._set_error_attributes(span, exc)
                raise
            self._set_response_attributes(span, response)
            if self._response_hook is not None:
                self._response_hook(span, _response_body(response))
            return response

    async def _trace_async(
        self,
        wrapped: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        method, target, body = _request_parts(args, kwargs)
        operation = _operation_name(method, target)
        active_span = _active_elasticsearch_span()
        span_context = (
            nullcontext(active_span)
            if active_span is not None
            else trace.get_tracer(__name__).start_as_current_span(
                f"elasticsearch.{operation}", kind=SpanKind.CLIENT
            )
        )
        with span_context as span:
            self._set_request_attributes(
                span, method=method, target=target, body=body, kwargs=kwargs
            )
            if self._request_hook is not None:
                self._request_hook(span, method, target, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exc:
                self._set_error_attributes(span, exc)
                raise
            self._set_response_attributes(span, response)
            if self._response_hook is not None:
                self._response_hook(span, _response_body(response))
            return response

    def _patch_transport(self, target: str, *, asynchronous: bool) -> bool:
        module = importlib.import_module(ELASTIC_TRANSPORT_MODULE)
        class_name, method_name = target.split(".", 1)
        target_class = getattr(module, class_name, None)
        if target_class is None or not hasattr(target_class, method_name):
            return False

        if asynchronous:

            async def traced(
                wrapped: Callable[..., Any],
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> Any:
                return await self._trace_async(wrapped, args, kwargs)

        else:

            def traced(
                wrapped: Callable[..., Any],
                instance: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> Any:
                return self._trace_sync(wrapped, args, kwargs)

        wrap_function_wrapper(ELASTIC_TRANSPORT_MODULE, target, traced)
        type(self)._patched_targets.append((ELASTIC_TRANSPORT_MODULE, target))
        return True

    def activate(self) -> None:
        """Patch sync and async Elasticsearch transport methods."""
        cls = type(self)
        if self._is_instrumented:
            return
        if not self._is_respan_tracing_enabled():
            logger.info(
                "Elasticsearch instrumentation skipped because Respan tracing is disabled"
            )
            return
        if cls._patches_applied:
            cls._activation_count += 1
            self._is_instrumented = True
            return
        try:
            patched = False
            for target, asynchronous in ELASTIC_TRANSPORT_TARGETS:
                patched = (
                    self._patch_transport(target, asynchronous=asynchronous) or patched
                )
        except ImportError as exc:
            logger.warning(
                "Failed to activate Elasticsearch instrumentation - missing dependency: %s",
                exc,
            )
            return
        except Exception:
            logger.exception("Failed to activate Elasticsearch instrumentation")
            self.deactivate()
            return
        if not patched:
            return
        cls._patches_applied = True
        cls._activation_count = 1
        self._is_instrumented = True
        logger.info("Elasticsearch instrumentation activated")

    def deactivate(self) -> None:
        """Restore original transport methods."""
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
        logger.info("Elasticsearch instrumentation deactivated")
