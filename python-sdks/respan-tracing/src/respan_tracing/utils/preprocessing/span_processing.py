from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv_ai import SpanAttributes
import logging

logger = logging.getLogger(__name__)


def is_processable_span(span: ReadableSpan) -> bool:
    """
    Determine if a span should be processed based on Respan/Traceloop attributes.

    Logic:
    - If span has TRACELOOP_SPAN_KIND: it's a user-decorated span → process
    - If span has TRACELOOP_ENTITY_PATH: it's a child span within entity context → process
    - If span has LLM_REQUEST_TYPE: it's an auto-instrumented LLM call → process
    - If span has none of the above: it's auto-instrumentation noise → filter out

    GAP: The LLM_REQUEST_TYPE check is a duck-tape fix for standalone auto-instrumented
    LLM spans. It won't cover non-LLM instrumentors (vector DB, retrieval, tool-use, etc.)
    that also lack Traceloop decorator context. The proper fix is an allowlist of recognized
    instrumentation scope names (e.g. "opentelemetry.instrumentation.openai") so we can
    accept any span from a known instrumentor without requiring decorator context or
    checking for provider-specific attributes.

    Args:
        span: The span to evaluate

    Returns:
        bool: True if span should be processed, False if it should be filtered out
    """
    span_kind = span.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND)
    entity_path = span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH, "")

    # User-decorated span (has TRACELOOP_SPAN_KIND)
    if span_kind:
        logger.debug(
            f"[Respan Debug] Processing user-decorated span: {span.name} (kind: {span_kind})"
        )
        return True

    # Child span within entity context (has TRACELOOP_ENTITY_PATH)
    if entity_path and entity_path != "":
        logger.debug(
            f"[Respan Debug] Processing child span within entity context: {span.name} (entityPath: {entity_path})"
        )
        return True

    # Standalone auto-instrumented LLM span (has llm.request.type, e.g. "chat")
    # This covers OpenAI/Anthropic/etc. calls made outside @workflow/@task decorators
    if span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE):
        logger.debug(
            f"[Respan Debug] Processing standalone LLM span: {span.name} "
            f"(llm.request.type: {span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE)})"
        )
        return True

    # Auto-instrumentation noise (HTTP, DB, etc.) - filter out
    logger.debug(
        f"[Respan Debug] Filtering out auto-instrumentation span: {span.name} (no TRACELOOP_SPAN_KIND, entityPath, or llm.request.type)"
    )
    return False


def is_root_span_candidate(span: ReadableSpan) -> bool:
    """
    Determine if a span should be converted to a root span.

    Logic:
    - User-decorated span (TRACELOOP_SPAN_KIND) without entity path should become root
    - Standalone LLM span (LLM_REQUEST_TYPE) without entity path should become root

    Args:
        span: The span to evaluate

    Returns:
        bool: True if span should be made a root span
    """
    span_kind = span.attributes.get(SpanAttributes.TRACELOOP_SPAN_KIND)
    entity_path = span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_PATH, "")
    llm_request_type = span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE)

    has_no_entity_path = not entity_path or entity_path == ""

    # User-decorated span without entity path should become root
    if span_kind is not None and has_no_entity_path:
        logger.debug(f"[Respan Debug] Span is root candidate (user-decorated): {span.name}")
        return True

    # Standalone LLM span without entity path should become root
    if llm_request_type and span_kind is None and has_no_entity_path:
        logger.debug(f"[Respan Debug] Span is root candidate (standalone LLM): {span.name}")
        return True

    return False
