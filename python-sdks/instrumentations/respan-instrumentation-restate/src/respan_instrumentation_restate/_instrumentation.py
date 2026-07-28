"""Instrument Restate handlers through the SDK's invocation context managers."""

from __future__ import annotations

import importlib
import json
import logging
import threading
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Status, StatusCode
from opentelemetry.semconv_ai import SpanAttributes
from wrapt import wrap_function_wrapper

from respan_instrumentation_restate._constants import (
    RESTATE_CONTEXT_MANAGER_MARKER,
    RESTATE_INSTRUMENTATION_NAME,
    RESTATE_REGISTRATION_TARGETS,
)
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_PATCHED_TARGETS: list[tuple[str, str]] = []
_ENABLED = False
_CAPTURE_CONTENT = True


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, Sequence):
        return [_jsonable(item, depth=depth + 1) for item in value]
    try:
        return serialize_value(value=value)
    except Exception:
        return repr(value)


def _json_string(value: Any) -> str:
    return json.dumps(_jsonable(value), default=str, ensure_ascii=False)


def _deserialize_input(context: Any) -> Any:
    invocation = context.invocation
    handler_io = context.handler.handler_io
    try:
        return handler_io.input_serde.deserialize(invocation.input_buffer)
    except Exception:
        return _jsonable(invocation.input_buffer)


def _invocation_details(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    handler = context.handler
    service_tag = handler.service_tag
    invocation = context.invocation
    server_context = importlib.import_module("restate.server_context")
    replaying_var = getattr(server_context, "restate_context_is_replaying", None)
    is_replaying = bool(replaying_var.get()) if replaying_var is not None else False

    metadata = {
        "service_kind": service_tag.kind,
        "service_name": service_tag.name,
        "handler_name": handler.name,
        "handler_kind": handler.kind,
        "invocation_id": invocation.invocation_id,
        "replaying": is_replaying,
    }
    for name in ("key", "scope", "limit_key", "idempotency_key"):
        value = getattr(invocation, name, None)
        if value:
            metadata[name] = value
    if service_tag.metadata:
        metadata["service_metadata"] = dict(service_tag.metadata)
    if handler.metadata:
        metadata["handler_metadata"] = dict(handler.metadata)

    input_payload = dict(metadata)
    if _CAPTURE_CONTENT:
        input_payload["input"] = _deserialize_input(context)
    return metadata, input_payload


def _log_type(context: Any) -> str:
    service_kind = str(context.handler.service_tag.kind or "")
    handler_kind = str(context.handler.kind or "")
    return (
        "workflow"
        if service_kind == "workflow" and handler_kind == "workflow"
        else "task"
    )


def _span_attributes(
    context: Any,
    *,
    metadata: dict[str, Any],
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    handler = context.handler
    invocation = context.invocation
    entity_name = f"{handler.service_tag.name}.{handler.name}"
    attrs: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: _log_type(context),
        RESPAN_TRACE_GROUP_ID: str(invocation.invocation_id),
        RESPAN_METADATA: _json_string({"restate": metadata}),
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _json_string(input_payload),
    }
    key = getattr(invocation, "key", None)
    if key:
        attrs[RESPAN_THREADS_ID] = str(key)
    return attrs


@asynccontextmanager
async def _invocation_context():
    """Create one span around a Restate handler invocation attempt."""

    if not _ENABLED:
        yield
        return

    server_context = importlib.import_module("restate.server_context")
    context = server_context.current_context()
    if context is None:
        yield
        return

    metadata, input_payload = _invocation_details(context)
    attrs = _span_attributes(
        context,
        metadata=metadata,
        input_payload=input_payload,
    )
    tracer = trace.get_tracer(__name__)
    span_name = (
        f"restate.{context.handler.service_tag.kind}."
        f"{context.handler.service_tag.name}.{context.handler.name}"
    )
    with tracer.start_as_current_span(
        span_name,
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield
        except BaseException as exc:
            message = str(exc) or type(exc).__name__
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, message))
            span.set_attribute(
                "status_code",
                int(getattr(exc, "status_code", 500) or 500),
            )
            span.set_attribute(ERROR_MESSAGE_ATTR, message)
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_string(
                    {
                        "status": "error",
                        "error": type(exc).__name__,
                        "message": message if _CAPTURE_CONTENT else type(exc).__name__,
                    }
                ),
            )
            raise
        else:
            span.set_status(Status(StatusCode.OK))
            span.set_attribute("status_code", 200)
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                _json_string({"status": "completed"}),
            )


setattr(_invocation_context, RESTATE_CONTEXT_MANAGER_MARKER, True)


def _ensure_context_manager(instance: Any) -> None:
    managers = list(getattr(instance, "context_managers", None) or ())
    if not any(
        bool(getattr(manager, RESTATE_CONTEXT_MANAGER_MARKER, False))
        for manager in managers
    ):
        managers.append(_invocation_context)
        instance.context_managers = managers


def _registration_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    _ensure_context_manager(instance)
    return wrapped(*args, **kwargs)


class RestateInstrumentor:
    """Inject canonical Respan spans into Restate handler invocations."""

    name = RESTATE_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        """Patch Restate handler registration to add an invocation context."""
        global _ACTIVATION_COUNT, _CAPTURE_CONTENT, _ENABLED

        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module("restate")
        except ImportError as exc:
            logger.warning("Restate instrumentation unavailable: %s", exc)
            return

        with _LOCK:
            if _ACTIVATION_COUNT == 0:
                _CAPTURE_CONTENT = self._capture_content
                try:
                    for module_path, target in RESTATE_REGISTRATION_TARGETS:
                        wrap_function_wrapper(
                            module_path,
                            target,
                            _registration_wrapper,
                        )
                        _PATCHED_TARGETS.append((module_path, target))
                except Exception:
                    for module_path, target in reversed(_PATCHED_TARGETS):
                        try:
                            unwrap(module_path, target)
                        except Exception:
                            logger.debug(
                                "Failed to roll back %s.%s",
                                module_path,
                                target,
                                exc_info=True,
                            )
                    _PATCHED_TARGETS.clear()
                    raise
                _ENABLED = True
            elif _CAPTURE_CONTENT != self._capture_content:
                logger.warning(
                    "Restate is already active; the first capture_content setting wins"
                )
            _ACTIVATION_COUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        """Restore Restate registration methods and disable injected contexts."""
        global _ACTIVATION_COUNT, _ENABLED

        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _ACTIVATION_COUNT = max(0, _ACTIVATION_COUNT - 1)
            if _ACTIVATION_COUNT:
                return
            _ENABLED = False
            for module_path, target in reversed(_PATCHED_TARGETS):
                try:
                    unwrap(module_path, target)
                except Exception:
                    logger.debug(
                        "Failed to unwrap %s.%s",
                        module_path,
                        target,
                        exc_info=True,
                    )
            _PATCHED_TARGETS.clear()
