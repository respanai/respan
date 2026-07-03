"""Emit LiveKit tool executions as Respan OTEL spans."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_livekit._translator import build_tool_span_attrs
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_CUSTOM_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import (
    build_readable_span,
    inject_span,
    read_propagated_attributes,
)

logger = logging.getLogger(__name__)

_TOOL_PARENT_CONTEXTS: dict[tuple[str | None, str], tuple[str, str]] = {}
_MAX_TOOL_PARENT_CONTEXTS = 2048


def _tool_parent_key(
    call_id: str,
    custom_identifier: str | None,
) -> tuple[str | None, str]:
    return custom_identifier, call_id


def register_livekit_tool_parent_context(
    *,
    call_id: str,
    trace_id: str,
    parent_id: str,
    custom_identifier: str | None = None,
) -> None:
    """Remember the LiveKit LLM span that announced a future tool call."""
    if len(_TOOL_PARENT_CONTEXTS) >= _MAX_TOOL_PARENT_CONTEXTS:
        _TOOL_PARENT_CONTEXTS.clear()
    _TOOL_PARENT_CONTEXTS[_tool_parent_key(call_id, custom_identifier)] = (
        trace_id,
        parent_id,
    )


def _pop_livekit_tool_parent_context(
    call_id: str | None,
    custom_identifier: str | None,
) -> tuple[str | None, str | None]:
    if not call_id:
        return None, None
    key = _tool_parent_key(call_id, custom_identifier)
    parent = _TOOL_PARENT_CONTEXTS.pop(key, None)
    if parent is not None:
        return parent

    fallback_key = _tool_parent_key(call_id, None)
    if fallback_key == key:
        return None, None
    return _TOOL_PARENT_CONTEXTS.pop(fallback_key, (None, None))


def _tool_span_name(tool_name: str, workflow_name: object | None) -> str:
    if workflow_name:
        return f"{workflow_name}.{tool_name}"
    return tool_name


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
    except Exception:
        return None, None

    trace_id = getattr(span_context, "trace_id", 0) or 0
    span_id = getattr(span_context, "span_id", 0) or 0
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def emit_livekit_tool_span(
    *,
    tool_name: str,
    arguments: Any,
    output: Any,
    call_id: str | None,
    start_time_ns: int,
    error: BaseException | None = None,
) -> None:
    """Build and inject a canonical Respan tool span for a LiveKit tool call."""
    try:
        attrs = build_tool_span_attrs(
            tool_name=tool_name,
            arguments=arguments,
            output={"error": str(error)} if error is not None else output,
            call_id=call_id,
        )
        if error is not None:
            attrs.setdefault("error.message", str(error))
            attrs.setdefault("status_code", 500)

        propagated_attrs = read_propagated_attributes()
        workflow_name = propagated_attrs.get(RESPAN_TRACE_GROUP_ID)
        if workflow_name:
            attrs.setdefault(SpanAttributes.TRACELOOP_WORKFLOW_NAME, str(workflow_name))
        span_name = _tool_span_name(tool_name=tool_name, workflow_name=workflow_name)

        trace_id, parent_id = _pop_livekit_tool_parent_context(
            call_id=str(call_id) if call_id else None,
            custom_identifier=propagated_attrs.get(RESPAN_SPAN_CUSTOM_ID),
        )
        if not trace_id or not parent_id:
            trace_id, parent_id = _current_trace_parent_ids()

        span = build_readable_span(
            name=span_name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_time_ns,
            end_time_ns=time.time_ns(),
            attributes=attrs,
            status_code=500 if error is not None else 200,
            error_message=str(error) if error is not None else None,
        )
        inject_span(span=span)
    except Exception:
        logger.debug("Failed to emit LiveKit tool span", exc_info=True)
