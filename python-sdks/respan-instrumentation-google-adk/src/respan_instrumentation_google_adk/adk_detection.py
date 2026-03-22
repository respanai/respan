"""Detection utilities for Google ADK spans.

Why this exists instead of using is_processable_span() from respan-tracing:

Raw ADK spans lack the attributes that is_processable_span() checks for
(traceloop.span.kind, traceloop.entity.path, llm.request.type, gen_ai.system).
They carry gen_ai.agent.name and gcp.vertex.agent.* instead, so they'd be
dropped as auto-instrumentation noise.  We need framework-specific detection
to (1) bypass that filter and (2) route spans into the two-pass ADK enrichment.

Once span_processing.py gets an instrumentation scope allowlist (see the GAP
comment there), the bypass half could be removed and this module would only
serve the enrichment routing purpose.
"""

# Instrumentation scope names that identify Google ADK spans
ADK_SCOPE_NAMES = {"gcp.vertex.agent", "google_adk", "google-adk"}

# Span names that indicate a Google ADK instrumented span
ADK_SPAN_NAMES = {
    "invocation", "agent_run", "call_llm",
    "execute_tool", "invoke_agent", "generate_content",
}


def is_adk_span(span: object) -> bool:
    """Check if a span originates from Google ADK instrumentation.

    Works with any span-like object that has instrumentation_scope/name/attributes.
    Uses ``object`` type hint to avoid requiring opentelemetry-sdk as a
    dependency of this module.
    """
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_library", None
    )
    scope_name = getattr(scope, "name", "") or ""
    if scope_name in ADK_SCOPE_NAMES:
        return True
    # Fallback: check span name prefix + gen_ai attributes
    span_name = (getattr(span, "name", "") or "").split(" ")[0]
    if span_name in ADK_SPAN_NAMES:
        attributes = getattr(span, "attributes", None) or {}
        if any(key.startswith("gen_ai.") for key in attributes):
            return True
    return False
