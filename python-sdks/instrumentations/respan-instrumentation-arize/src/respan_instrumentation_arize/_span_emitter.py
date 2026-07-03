"""Emit Arize SDK operations as Respan OTEL spans."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_arize._constants import (
    ARIZE_INSTRUMENTATION_NAME,
    ARIZE_METADATA_INTEGRATION,
    ARIZE_METADATA_OPERATION,
    ARIZE_METADATA_RESOURCE,
    ARIZE_METADATA_STATUS_CODE,
)
from respan_instrumentation_arize._serialization import safe_json_dumps
from respan_sdk.constants.llm_logging import LOG_TYPE_TASK, LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.data_processing.id_processing import format_span_id, format_trace_id
from respan_tracing.utils.span_factory import (
    build_readable_span,
    inject_span,
    read_propagated_attributes,
)

logger = logging.getLogger(__name__)


def _current_trace_context() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    if current_span is None:
        return None, None

    span_context = current_span.get_span_context()
    if not span_context or not span_context.is_valid:
        return None, None

    return (
        format_trace_id(span_context.trace_id),
        format_span_id(span_context.span_id),
    )



def _workflow_name_from_context() -> str | None:
    propagated = read_propagated_attributes()
    trace_group = propagated.get(RESPAN_TRACE_GROUP_ID)
    if isinstance(trace_group, str) and trace_group:
        return trace_group

    metadata_workflow_name = propagated.get(f"{RESPAN_METADATA}.workflow_name")
    if isinstance(metadata_workflow_name, str) and metadata_workflow_name:
        return metadata_workflow_name
    return None


def _status_code_from_result(result: Any) -> int:
    status_code = getattr(result, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return 200


def build_arize_span_attributes(
    *,
    resource: str,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Build canonical Respan attributes for one Arize SDK operation."""
    operation_name = f"arize.{resource}.{method_name}"
    output = (
        {"error": str(error), "error_type": type(error).__name__}
        if error is not None
        else result
    )
    attrs: dict[str, Any] = {
        RESPAN_LOG_TYPE: LOG_TYPE_TASK,
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: operation_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: operation_name,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: safe_json_dumps(
            {
                "args": args,
                "kwargs": kwargs,
            }
        ),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: safe_json_dumps(output),
        ARIZE_METADATA_INTEGRATION: ARIZE_INSTRUMENTATION_NAME,
        ARIZE_METADATA_RESOURCE: resource,
        ARIZE_METADATA_OPERATION: method_name,
    }

    status_code = _status_code_from_result(result)
    if status_code != 200:
        attrs[ARIZE_METADATA_STATUS_CODE] = status_code

    workflow_name = _workflow_name_from_context()
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name

    return attrs


def emit_arize_span(
    *,
    resource: str,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    start_time_ns: int,
    end_time_ns: int,
    error: BaseException | None = None,
) -> bool:
    """Emit one completed Arize SDK operation into the active OTEL pipeline."""
    try:
        trace_id, parent_id = _current_trace_context()
        attrs = build_arize_span_attributes(
            resource=resource,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            result=result,
            error=error,
        )
        span = build_readable_span(
            name=f"arize.{resource}.{method_name}",
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            attributes=attrs,
            status_code=500 if error is not None else _status_code_from_result(result),
            error_message=str(error) if error is not None else None,
        )
        return inject_span(span)
    except Exception:
        logger.debug("Failed to emit Arize span", exc_info=True)
        return False
