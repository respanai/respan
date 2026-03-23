from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv_ai import SpanAttributes
import logging

from respan_sdk.constants.span_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_AGENT_NAME,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    PYDANTIC_AI_AGENT_NAME,
    PYDANTIC_AI_TOOL_ARGUMENTS,
    PYDANTIC_AI_TOOL_RESPONSE,
    RESPAN_LOG_TYPE,
)

logger = logging.getLogger(__name__)

# Attribute names that indicate a GenAI span (OTEL incubating + Pydantic AI vendor attrs)
_GENAI_INDICATOR_ATTRS = (
    GEN_AI_OPERATION_NAME,
    SpanAttributes.LLM_SYSTEM,
    GEN_AI_AGENT_NAME,
    PYDANTIC_AI_AGENT_NAME,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    PYDANTIC_AI_TOOL_ARGUMENTS,
    PYDANTIC_AI_TOOL_RESPONSE,
)


def _is_genai_span(span: ReadableSpan) -> bool:
    attributes = span.attributes or {}
    return any(
        attributes.get(attr_name) is not None
        for attr_name in _GENAI_INDICATOR_ATTRS
    )


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

    # Standalone GenAI span (has gen_ai.system, e.g. "openai")
    # This covers spans from OTEL instrumentors that don't set llm.request.type,
    # such as the OpenAI Responses API instrumentor.
    if span.attributes.get(SpanAttributes.LLM_SYSTEM):
        logger.debug(
            f"[Respan Debug] Processing standalone GenAI span: {span.name} "
            f"(gen_ai.system: {span.attributes.get(SpanAttributes.LLM_SYSTEM)})"
        )
        return True

    # GenAI native spans can be model, agent, or tool spans.
    if _is_genai_span(span):
        logger.debug(
            f"[Respan Debug] Processing GenAI native span: {span.name} "
            f"(gen_ai.operation.name: {span.attributes.get(GEN_AI_OPERATION_NAME)})"
        )
        return True

    # Enriched Respan span (has respan.entity.log_type set by an exporter plugin).
    if span.attributes.get(RESPAN_LOG_TYPE):
        logger.debug(
            f"[Respan Debug] Processing enriched Respan span: {span.name} "
            f"(log_type: {span.attributes.get(RESPAN_LOG_TYPE)})"
        )
        return True

    # Auto-instrumentation noise (HTTP, DB, etc.) - filter out
    logger.debug(
        f"[Respan Debug] Filtering out auto-instrumentation span: {span.name} (no TRACELOOP_SPAN_KIND, entityPath, llm.request.type, or gen_ai.*)"
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

    # Standalone GenAI span (gen_ai.system) without entity path should become root
    gen_ai_system = span.attributes.get(SpanAttributes.LLM_SYSTEM)
    if gen_ai_system and span_kind is None and not llm_request_type and has_no_entity_path:
        logger.debug(f"[Respan Debug] Span is root candidate (standalone GenAI): {span.name}")
        return True

    # GenAI native span without entity path should become root
    is_genai = _is_genai_span(span)
    if is_genai and span_kind is None and has_no_entity_path:
        logger.debug(f"[Respan Debug] Span is root candidate (GenAI native): {span.name}")
        return True

    # Enriched Respan span without entity path should become root
    if span.attributes.get(RESPAN_LOG_TYPE) and span_kind is None and has_no_entity_path:
        logger.debug(f"[Respan Debug] Span is root candidate (enriched Respan): {span.name}")
        return True

    return False
