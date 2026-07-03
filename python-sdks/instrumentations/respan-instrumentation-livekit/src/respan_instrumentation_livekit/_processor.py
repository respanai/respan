"""Span processor that normalizes LiveKit native LLM spans."""

from __future__ import annotations

import json
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_livekit._constants import (
    ATTR_LLM_METRICS,
    LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
    LIVEKIT_SCOPE_NAME,
)
from respan_instrumentation_livekit._otel_emitter import (
    register_livekit_tool_parent_context,
)
from respan_instrumentation_livekit._translator import (
    build_livekit_llm_attrs,
    is_livekit_llm_span,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_CUSTOM_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)


def _mutable_attrs(span: ReadableSpan) -> dict[str, Any] | None:
    attrs = getattr(span, "_attributes", None)
    if attrs is None:
        return None
    return dict(attrs)


def _span_context_ids(span: ReadableSpan) -> tuple[str | None, str | None]:
    context = getattr(span, "context", None)
    if context is None and hasattr(span, "get_span_context"):
        context = span.get_span_context()
    trace_id = getattr(context, "trace_id", 0) or 0
    span_id = getattr(context, "span_id", 0) or 0
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _tool_call_ids(attrs: dict[str, Any]) -> list[str]:
    value = attrs.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls")
    if not value:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    call_ids: list[str] = []
    for tool_call in parsed:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id") or tool_call.get("call_id")
        if call_id:
            call_ids.append(str(call_id))
    return call_ids


def _register_tool_parent_contexts(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    trace_id, parent_id = _span_context_ids(span)
    if not trace_id or not parent_id:
        return
    custom_identifier = attrs.get(RESPAN_SPAN_CUSTOM_ID)
    for call_id in _tool_call_ids(attrs):
        register_livekit_tool_parent_context(
            call_id=call_id,
            trace_id=trace_id,
            parent_id=parent_id,
            custom_identifier=str(custom_identifier) if custom_identifier else None,
        )


def _workflow_span_name(attrs: dict[str, Any]) -> str | None:
    workflow_name = attrs.get(RESPAN_TRACE_GROUP_ID) or attrs.get(
        SpanAttributes.TRACELOOP_WORKFLOW_NAME
    )
    return str(workflow_name) if workflow_name else None


def _rename_span_to_workflow(span: ReadableSpan, attrs: dict[str, Any]) -> None:
    workflow_span_name = _workflow_span_name(attrs)
    if workflow_span_name:
        span._name = workflow_span_name


def _scope_name(span: ReadableSpan) -> str | None:
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None)


class LiveKitSpanProcessor(SpanProcessor):
    """Translate LiveKit ``llm_request`` spans before Respan export."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        attrs = _mutable_attrs(span)
        if attrs is None:
            return
        scope_name = _scope_name(span)
        if (
            scope_name not in (None, LIVEKIT_SCOPE_NAME)
            and ATTR_LLM_METRICS not in attrs
            and LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR not in attrs
        ):
            return
        if not is_livekit_llm_span(span_name=span.name, attrs=attrs):
            return

        translated = build_livekit_llm_attrs(
            span_name=span.name,
            attrs=attrs,
            events=tuple(getattr(span, "events", ()) or ()),
        )
        attrs.update(translated)
        _rename_span_to_workflow(span=span, attrs=attrs)
        _register_tool_parent_contexts(span=span, attrs=attrs)
        span._attributes = attrs

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
