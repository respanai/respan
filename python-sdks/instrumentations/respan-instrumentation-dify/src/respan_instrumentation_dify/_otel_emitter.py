"""Build and inject Dify spans into the active Respan OTEL pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import (
    build_readable_span,
    inject_span,
    read_propagated_attributes,
)

from respan_instrumentation_dify._context import read_respan_params
from respan_instrumentation_dify._translator import build_dify_span_data


@dataclass
class DifyCallContext:
    method: str
    endpoint: str
    request_json: Any = None
    request_params: Any = None
    request_data: Any = None
    files: Any = None
    stream: bool = False
    start_time_ns: int = field(default_factory=time.time_ns)
    trace_id: str | None = None
    parent_id: str | None = None
    workflow_name: str | None = None
    propagated_attributes: dict[str, Any] = field(default_factory=dict)
    respan_params: dict[str, Any] = field(default_factory=dict)


def _current_otel_parent() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    try:
        span_context = current_span.get_span_context()
    except Exception:  # noqa: BLE001 -- defensive OTEL provider boundary
        return None, None

    trace_id = getattr(span_context, "trace_id", 0)
    span_id = getattr(span_context, "span_id", 0)
    if not isinstance(trace_id, int) or not isinstance(span_id, int):
        return None, None
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def capture_call_context(
    *,
    method: str,
    endpoint: str,
    request_json: Any = None,
    request_params: Any = None,
    request_data: Any = None,
    files: Any = None,
    stream: bool = False,
) -> DifyCallContext:
    trace_id, parent_id = _current_otel_parent()
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    return DifyCallContext(
        method=method,
        endpoint=endpoint,
        request_json=request_json,
        request_params=request_params,
        request_data=request_data,
        files=files,
        stream=stream,
        trace_id=trace_id,
        parent_id=parent_id,
        workflow_name=str(workflow_name) if workflow_name else None,
        propagated_attributes=read_propagated_attributes(),
        respan_params=read_respan_params(),
    )


def emit_dify_span(
    *,
    call_context: DifyCallContext,
    response: Any = None,
    stream_events: list[Any] | None = None,
    error: Exception | None = None,
    include_content: bool = True,
) -> None:
    span_name, attributes = build_dify_span_data(
        method=call_context.method,
        endpoint=call_context.endpoint,
        request_json=call_context.request_json,
        request_params=call_context.request_params,
        request_data=call_context.request_data,
        files=call_context.files,
        response=response,
        stream_events=stream_events,
        error=error,
        include_content=include_content,
        respan_params=call_context.respan_params,
        propagated_attributes=call_context.propagated_attributes,
        current_workflow_name=call_context.workflow_name,
        parent_id=call_context.parent_id,
    )
    status_code = getattr(response, "status_code", 200)
    if not isinstance(status_code, int):
        status_code = 200
    if error is not None:
        status_code = 500
    span = build_readable_span(
        name=span_name,
        trace_id=call_context.trace_id,
        parent_id=call_context.parent_id,
        start_time_ns=call_context.start_time_ns,
        end_time_ns=time.time_ns(),
        attributes=attributes,
        status_code=status_code,
        error_message=str(error) if error is not None else None,
        merge_propagated=False,
    )
    inject_span(span=span)
