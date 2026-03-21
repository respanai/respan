from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv_ai import SpanAttributes
import logging

logger = logging.getLogger(__name__)

# Instrumentation scope names that identify Google ADK spans
# NOTE: This duplicates detection logic in respan_instrumentation_google_adk.utils.
# We cannot import from that package here because respan_tracing is a core dependency
# of respan_instrumentation_google_adk — importing the other direction would create a
# circular dependency. Consider moving to respan_sdk if this grows.
_ADK_SCOPE_NAMES = {"gcp.vertex.agent", "google_adk", "google-adk"}

# Span names that indicate a Google ADK instrumented span
_ADK_SPAN_NAMES = {"invocation", "agent_run", "call_llm", "execute_tool", "invoke_agent"}


def _is_adk_span(span: ReadableSpan) -> bool:
    """Check if a span originates from Google ADK instrumentation."""
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_library", None
    )
    scope_name = getattr(scope, "name", "") or ""
    if scope_name in _ADK_SCOPE_NAMES:
        return True
    # Fallback: check span name + gen_ai attributes
    span_name = (getattr(span, "name", "") or "").split(" ")[0]
    if span_name in _ADK_SPAN_NAMES:
        attributes = getattr(span, "attributes", None) or {}
        if any(key.startswith("gen_ai.") for key in attributes):
            return True
    return False


def is_processable_span(span: ReadableSpan) -> bool:
    """
    Determine if a span should be processed based on Respan/Traceloop attributes.

    Logic:
    - If span is from a known instrumentation plugin (e.g. Google ADK) → process
    - If span has TRACELOOP_SPAN_KIND: it's a user-decorated span → process
    - If span has TRACELOOP_ENTITY_PATH: it's a child span within entity context → process
    - If span has LLM_REQUEST_TYPE: it's an auto-instrumented LLM call → process
    - If span has none of the above: it's auto-instrumentation noise → filter out

    Args:
        span: The span to evaluate

    Returns:
        bool: True if span should be processed, False if it should be filtered out
    """
    # Instrumentation plugin spans (e.g. Google ADK) — these are handled by their
    # own exporters, so let them through the filter
    if _is_adk_span(span):
        logger.debug(
            "[Respan Debug] Processing Google ADK span: %s", span.name
        )
        return True

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
