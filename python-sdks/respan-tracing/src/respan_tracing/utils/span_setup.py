"""Shared span setup and cleanup logic.

Used by both the decorator API (decorators/base.py) and the imperative
context-manager API (core/client.py) to avoid code duplication.
"""
import json
from typing import List, Optional, Union, Callable, Tuple

from opentelemetry import trace, context as context_api
from opentelemetry.trace.span import Span
from opentelemetry.semconv_ai import TraceloopSpanKindValues, SpanAttributes
from respan_sdk import FilterParamDict
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.respan_types.span_types import RespanSpanAttributes, SpanLink

from ..contexts.span import span_link_to_otel, consume_span_links
from ..constants.tracing import EXPORT_FILTER_ATTR, PROCESSORS_ATTR, SAMPLE_RATE_ATTR

LinksParam = Optional[Union[List[SpanLink], Callable[[], List[SpanLink]]]]

# Span kinds that propagate entity name to children via context
_ENTITY_NAME_KINDS = frozenset([
    TraceloopSpanKindValues.WORKFLOW.value,
    TraceloopSpanKindValues.AGENT.value,
])

# Span kinds that append to entity path
_ENTITY_PATH_KINDS = frozenset([
    TraceloopSpanKindValues.TASK.value,
    TraceloopSpanKindValues.TOOL.value,
])


def setup_span(
    entity_name: str,
    span_kind: str,
    version: Optional[int] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
) -> Tuple[Span, object, Optional[object], Optional[object]]:
    """Create and configure an OpenTelemetry span with Respan metadata.

    Returns:
        Tuple of (span, ctx_token, entity_name_token, entity_path_token).
        The caller MUST call cleanup_span() with these values in a finally block.
    """
    # Normalize kind to string (accepts enum or str)
    span_kind_str = span_kind.value if hasattr(span_kind, "value") else str(span_kind)
    entity_path = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_PATH) or ""

    entity_name_token = None
    entity_path_token = None

    # Propagate entity name for workflow/agent spans (children inherit it
    # as TRACELOOP_WORKFLOW_NAME via RespanSpanProcessor.on_start)
    if span_kind_str in _ENTITY_NAME_KINDS:
        entity_name_token = context_api.attach(
            context_api.set_value(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        )

    # Append to entity path for task/tool spans
    if span_kind_str in _ENTITY_PATH_KINDS:
        entity_path = f"{entity_path}.{entity_name}" if entity_path else entity_name
        entity_path_token = context_api.attach(
            context_api.set_value(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)
        )

    # Resolve span links: explicit param + context-attached
    otel_links: List[trace.Link] = []
    explicit_links = links() if callable(links) else (links or [])
    for link in explicit_links:
        otel_links.append(span_link_to_otel(link))
    otel_links.extend(consume_span_links())

    # Build initial attributes — processors must be set BEFORE start_span()
    # so that on_start's inheritance guard can distinguish explicit processors
    # from spans that should inherit from their parent.
    from ..core.tracer import RespanTracer
    tracer = RespanTracer().get_tracer()
    span_name = f"{entity_name}.{span_kind_str}"
    initial_attributes = {}
    if processors:
        processors_list = [processors] if isinstance(processors, str) else processors
        initial_attributes[PROCESSORS_ATTR] = ",".join(processors_list)

    span = tracer.start_span(
        span_name, attributes=initial_attributes, links=otel_links or None,
    )

    # Set standard Respan attributes
    span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, span_kind_str)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_PATH, entity_path)
    span.set_attribute(
        RespanSpanAttributes.LOG_METHOD.value,
        LogMethodChoices.PYTHON_TRACING.value,
    )
    if version is not None:
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_VERSION, version)

    # Set export filter
    if export_filter is not None:
        try:
            span.set_attribute(EXPORT_FILTER_ATTR, json.dumps(export_filter))
        except (TypeError, ValueError):
            pass

    # Set sample rate
    if sample_rate is not None:
        span.set_attribute(SAMPLE_RATE_ATTR, sample_rate)

    # Activate span in context
    ctx = trace.set_span_in_context(span)
    ctx_token = context_api.attach(ctx)

    return span, ctx_token, entity_name_token, entity_path_token


def cleanup_span(
    span: Span,
    ctx_token: object,
    entity_name_token: Optional[object] = None,
    entity_path_token: Optional[object] = None,
) -> None:
    """End span and detach all context tokens. Must be called in a finally block."""
    span.end()
    context_api.detach(ctx_token)
    if entity_path_token is not None:
        context_api.detach(entity_path_token)
    if entity_name_token is not None:
        context_api.detach(entity_name_token)
