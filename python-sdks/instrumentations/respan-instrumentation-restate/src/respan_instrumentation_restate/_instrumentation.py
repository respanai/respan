"""Observe native Restate invocation attempts without adding durable work."""
# ruff: noqa: BLE001, S110 -- telemetry isolation must not log native payloads.

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import logging
import threading
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.util import BoundedList
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import read_propagated_attributes
from wrapt import FunctionWrapper

from respan_instrumentation_restate._constants import (
    RESTATE_CONTEXT_MANAGER_MARKER,
    RESTATE_INSTRUMENTATION_NAME,
    RESTATE_REGISTRATION_TARGETS,
)
from respan_instrumentation_restate._policy import AncestorPolicy, suppressed
from respan_instrumentation_restate._serialization import (
    exception_message,
    exception_status,
    json_string,
    json_value,
    safe_text,
    sensitive_key,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_ENABLED = False
_CAPTURE_CONTENT = True
_POLICIES = weakref.WeakKeyDictionary()


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    replacement: Any


_PATCHED_TARGETS: list[_Patch] = []


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    return tracer is None or bool(getattr(tracer, "is_enabled", True))


def _policy():
    provider = trace.get_tracer_provider()
    with _LOCK:
        policy = _POLICIES.get(provider)
        if policy is None:
            if not callable(getattr(provider, "add_span_processor", None)):
                return None
            policy = AncestorPolicy(_CAPTURE_CONTENT)
            provider.add_span_processor(policy)
            _POLICIES[provider] = policy
        policy.enabled = True
        policy.setting = _CAPTURE_CONTENT
        return policy


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _deserialize_input(context):
    # Restate invokes user serde once. Never invoke it again for telemetry.
    serde = importlib.import_module("restate.serde")
    handler_io = _safe_attr(_safe_attr(context, "handler"), "handler_io")
    input_serde = _safe_attr(handler_io, "input_serde")
    native_json = (
        serde.DefaultSerde,
        serde.JsonSerde,
        serde.PydanticJsonSerde,
        serde.MsgspecJsonSerde,
    )
    if (
        type(input_serde) not in native_json
        or _safe_attr(context, "journal_codec") is not None
    ):
        return None, False
    buffer = _safe_attr(_safe_attr(context, "invocation"), "input_buffer")
    if not isinstance(buffer, bytes):
        return None, False
    try:
        return json.loads(buffer) if buffer else None, True
    except (ValueError, UnicodeError):
        return None, False


def _invocation_details(context):
    handler = _safe_attr(context, "handler")
    service = _safe_attr(handler, "service_tag")
    invocation = _safe_attr(context, "invocation")
    replaying = importlib.import_module(
        "restate.server_context"
    ).restate_context_is_replaying.get()
    metadata = {
        "service_kind": safe_text(_safe_attr(service, "kind")),
        "service_name": safe_text(_safe_attr(service, "name")),
        "handler_name": safe_text(_safe_attr(handler, "name")),
        "handler_kind": safe_text(_safe_attr(handler, "kind")),
        "invocation_id": safe_text(_safe_attr(invocation, "invocation_id")),
        "replaying": bool(replaying),
    }
    for name in ("key", "scope", "limit_key", "idempotency_key"):
        value = _safe_attr(invocation, name)
        if value:
            metadata[name] = safe_text(value)
    for name, owner in (("service_metadata", service), ("handler_metadata", handler)):
        value = _safe_attr(owner, "metadata")
        if value:
            metadata[name] = json_value(value)
    payload = dict(metadata)
    value, available = _deserialize_input(context)
    if available:
        payload["input"] = value
    return metadata, payload


def _span_attributes(context, *, metadata, input_payload):
    handler = context.handler
    invocation = context.invocation
    entity = f"{safe_text(handler.service_tag.name)}.{safe_text(handler.name)}"
    parent = trace.get_current_span().get_span_context().is_valid
    attrs = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: "workflow"
        if handler.service_tag.kind == "workflow" and handler.kind == "workflow"
        else "task",
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity,
        SpanAttributes.TRACELOOP_ENTITY_PATH: entity if parent else "",
    }
    if metadata is not None:
        attrs[RESPAN_TRACE_GROUP_ID] = safe_text(invocation.invocation_id)
        if invocation.key:
            attrs[RESPAN_THREADS_ID] = safe_text(invocation.key)
        aggregate = {"restate": metadata}
        for key, value in read_propagated_attributes().items():
            if key.startswith(f"{RESPAN_METADATA}."):
                name = key.removeprefix(f"{RESPAN_METADATA}.")
                safe = "[REDACTED]" if sensitive_key(name) else json_value(value)
                aggregate[name] = safe
                attrs[key] = (
                    safe
                    if isinstance(safe, str | bool | int | float)
                    else json_string(safe)
                )
            elif key in {RESPAN_TRACE_GROUP_ID, RESPAN_THREADS_ID}:
                attrs[key] = safe_text(value)
        attrs[RESPAN_METADATA] = json_string(aggregate)
        if "input" in input_payload:
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_string(input_payload)
    return attrs


def _clear_content(span):
    # SDK mutability is necessary when content is vetoed after span start.
    attrs = getattr(span, "_attributes", None)
    if attrs is not None:
        for key in list(attrs):
            if key not in {
                RESPAN_LOG_METHOD,
                RESPAN_LOG_TYPE,
                SpanAttributes.TRACELOOP_ENTITY_NAME,
                SpanAttributes.TRACELOOP_ENTITY_PATH,
                "status_code",
            }:
                del attrs[key]
    events = getattr(span, "_events", None)
    if events is not None:
        span._events = BoundedList(0)
    status = getattr(span, "status", None)
    if status is not None:
        span._status = Status(status.status_code)


def _recording(span):
    try:
        return span is not None and span.is_recording()
    except Exception:
        return False


@asynccontextmanager
async def _invocation_context():
    span = token = policy = None
    try:
        if not _ENABLED or not _is_respan_tracing_enabled() or suppressed():
            enabled = False
        else:
            native = importlib.import_module("restate.server_context").current_context()
            enabled = native is not None
        if enabled:
            policy = _policy()
            version = importlib.metadata.version("respan-instrumentation-restate")
            span = trace.get_tracer(RESTATE_INSTRUMENTATION_NAME, version).start_span(
                "restate.invocation"
            )
            if span.is_recording():
                allowed = policy is not None and policy.observe(span)
                metadata, payload = (
                    _invocation_details(native) if allowed else (None, {})
                )
                attrs = _span_attributes(
                    native, metadata=metadata, input_payload=payload
                )
                for key, value in attrs.items():
                    span.set_attribute(key, value)
                span.update_name(
                    f"restate.{native.handler.service_tag.kind}.{safe_text(native.handler.service_tag.name)}.{safe_text(native.handler.name)}"
                )
                # Inspect content before attaching the child so parent/path remain native.
                token = otel_context.attach(trace.set_span_in_context(span))
    except Exception:
        if span is not None:
            try:
                _clear_content(span)
            except Exception:
                pass
    try:
        yield
    except BaseException as exc:
        if _recording(span):
            try:
                allowed = policy is not None and policy.observe(span)
                message = exception_message(exc) if allowed else None
                span.set_status(Status(StatusCode.ERROR, message))
                # Only real vendor status fields are mapped; generic failures have no invented HTTP code.
                status = exception_status(exc, default=None)
                if status is not None:
                    span.set_attribute("status_code", status)
                if allowed:
                    span.set_attribute(ERROR_MESSAGE_ATTR, message)
                    span.add_event(
                        "exception",
                        {
                            "exception.type": f"{type(exc).__module__}.{type(exc).__name__}",
                            "exception.message": message,
                        },
                    )
            except Exception:
                try:
                    _clear_content(span)
                except Exception:
                    pass
        raise
    else:
        if _recording(span):
            try:
                span.set_status(Status(StatusCode.OK))
            except Exception:
                pass
    finally:
        if span is not None:
            try:
                if policy is None or not policy.observe(span):
                    _clear_content(span)
            except Exception:
                try:
                    _clear_content(span)
                except Exception:
                    pass
            try:
                if token is not None:
                    otel_context.detach(token)
            except Exception:
                try:
                    otel_context._RUNTIME_CONTEXT.detach(token)
                except Exception:
                    pass
            try:
                span.end()
            except Exception:
                pass


setattr(_invocation_context, RESTATE_CONTEXT_MANAGER_MARKER, True)


def _ensure_context_manager(instance):
    managers = list(getattr(instance, "context_managers", None) or ())
    if _invocation_context not in managers:
        managers.append(_invocation_context)
        instance.context_managers = managers


def _registration_wrapper(wrapped, instance, args, kwargs):
    # A foreign wrapper can retain our wrapper after deactivation.
    if _ENABLED:
        try:
            _ensure_context_manager(instance)
        except Exception:
            pass
    return wrapped(*args, **kwargs)


def _install_patches():
    for module_path, target in RESTATE_REGISTRATION_TARGETS:
        module = importlib.import_module(module_path)
        owner_path, name = target.rsplit(".", 1)
        owner = module
        for component in owner_path.split("."):
            owner = getattr(owner, component)
        original = inspect.getattr_static(owner, name)
        replacement = FunctionWrapper(original, _registration_wrapper)
        setattr(owner, name, replacement)
        _PATCHED_TARGETS.append(_Patch(owner, name, original, replacement))


def _remove_patches():
    for patch in reversed(_PATCHED_TARGETS):
        try:
            if (
                inspect.getattr_static(patch.owner, patch.name, None)
                is patch.replacement
            ):
                setattr(patch.owner, patch.name, patch.original)
        except Exception:
            pass
    _PATCHED_TARGETS.clear()


class RestateInstrumentor:
    """Add one observational span for each registered native invocation attempt."""

    name = RESTATE_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content=True):
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self):
        global _ACTIVATION_COUNT, _CAPTURE_CONTENT, _ENABLED
        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module("restate")
        except ImportError:
            logger.warning("Restate instrumentation unavailable")
            return
        with _LOCK:
            if _ACTIVATION_COUNT == 0:
                _CAPTURE_CONTENT = self._capture_content
                try:
                    _install_patches()
                    _policy()
                except Exception:
                    _remove_patches()
                    for policy in _POLICIES.values():
                        policy.enabled = False
                        policy.clear()
                    raise
                _ENABLED = True
            elif _CAPTURE_CONTENT != self._capture_content:
                raise ValueError(
                    "Restate capture_content must match the active instrumentor"
                )
            _ACTIVATION_COUNT += 1
            self._is_instrumented = True

    def deactivate(self):
        global _ACTIVATION_COUNT, _ENABLED
        with _LOCK:
            if not self._is_instrumented:
                return
            _ACTIVATION_COUNT = max(0, _ACTIVATION_COUNT - 1)
            self._is_instrumented = False
            if _ACTIVATION_COUNT == 0:
                _ENABLED = False
                _remove_patches()
                for policy in _POLICIES.values():
                    policy.enabled = False
                    policy.clear()
