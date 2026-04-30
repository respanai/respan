"""Shared span setup and cleanup logic.

Used by both the decorator API (decorators/base.py) and the imperative
context-manager API (core/client.py) to avoid code duplication.
"""
import json
from typing import List, Optional, Union, Callable, Tuple

from opentelemetry import trace, context as context_api
from opentelemetry.context import Context
from opentelemetry.trace.span import Span
from opentelemetry.semconv_ai import TraceloopSpanKindValues, SpanAttributes
from respan_sdk import FilterParamDict
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD
from respan_sdk.respan_types.span_types import SpanLink

from respan_tracing.contexts.span import span_link_to_otel, consume_span_links
from respan_tracing.constants.tracing import EXPORT_FILTER_ATTR, PROCESSORS_ATTR, SAMPLE_RATE_ATTR
from respan_tracing.processors.base import _active_span_buffer

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

# Span kinds that act as trace entry points. They detach any inherited OTel
# context and start a fresh root trace. The only way to attach as a child of
# a parent trace is to enter a SpanBuffer with explicit
# parent_trace_id+parent_span_id (the SDK's single, explicit continuation
# mechanism — see RespanClient.get_span_buffer).
_ROOT_DEFAULT_KINDS = frozenset([
    TraceloopSpanKindValues.WORKFLOW.value,
    TraceloopSpanKindValues.AGENT.value,
])


def setup_span(
    entity_name: str,
    span_kind: str,
    version: Optional[int] = None,
    processors: Optional[Union[str, List[str]]] = None,
    export_filter: Optional[FilterParamDict] = None,
    links: LinksParam = None,
    sample_rate: Optional[float] = None,
) -> Tuple[Span, object, Optional[object], Optional[object], Optional[object]]:
    """Create and configure an OpenTelemetry span with Respan metadata.

    Trace-root behavior (workflow/agent kinds only):

        Workflow and agent spans always start a **fresh root** trace: the
        inherited OTel context is detached before span creation so OTel
        allocates a new trace_id with no parent. This is the right default
        at every entry point in our system (Celery tasks, Pulsar consumer
        batch handlers, gunicorn views, signal receivers) — relying on the
        caller to leave behind no active span is fragile and was the source
        of the 2026-04-30 "55 runs collapsed into 3 traces" bug.

        Task and tool kinds always inherit (they are sub-steps by definition).

        Continuation across pause/resume or any other "this trace is part
        of another trace" use case is exclusively handled by SpanBuffer
        (RespanClient.get_span_buffer with explicit parent_trace_id +
        parent_span_id). When a SpanBuffer is active, the fresh-root default
        is suppressed and decorators inside it inherit the buffer's parent
        context. There is no per-decorator continuation flag — every parent
        relationship is explicit and named.

    Returns:
        Tuple of (span, ctx_token, entity_name_token, entity_path_token,
        root_ctx_token). root_ctx_token is non-None only when this span
        attached an empty Context to become a fresh root. The caller MUST
        call cleanup_span() with these values in a finally block.
    """
    # Normalize kind to string (accepts enum or str)
    span_kind_str = span_kind.value if hasattr(span_kind, "value") else str(span_kind)

    # SpanBuffer (continuation or trace_id-injection mode) deliberately sets
    # up a parent OTel context. When a buffer is active, respect it — do not
    # detach to a fresh root. This is the SDK's single, explicit continuation
    # mechanism (see RespanClient.get_span_buffer).
    is_root_kind = span_kind_str in _ROOT_DEFAULT_KINDS
    is_inside_span_buffer = _active_span_buffer.get(None) is not None
    is_fresh_root = is_root_kind and not is_inside_span_buffer

    root_ctx_token = None
    if is_fresh_root:
        # Attach an empty Context so tracer.start_span() finds no active parent
        # and creates a fresh root with a new trace_id. Detached last in
        # cleanup_span() to restore the caller's original context.
        root_ctx_token = context_api.attach(Context())
        entity_path = ""
    else:
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
        RESPAN_LOG_METHOD,
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

    return span, ctx_token, entity_name_token, entity_path_token, root_ctx_token


def cleanup_span(
    span: Span,
    ctx_token: object,
    entity_name_token: Optional[object] = None,
    entity_path_token: Optional[object] = None,
    root_ctx_token: Optional[object] = None,
) -> None:
    """End span and detach all context tokens. Must be called in a finally block.

    Tokens are detached in reverse-attach order (LIFO) to preserve OTel context
    invariants. root_ctx_token (from is_new_trace_root=True) is detached last.
    """
    span.end()
    context_api.detach(ctx_token)
    if entity_path_token is not None:
        context_api.detach(entity_path_token)
    if entity_name_token is not None:
        context_api.detach(entity_name_token)
    if root_ctx_token is not None:
        context_api.detach(root_ctx_token)
