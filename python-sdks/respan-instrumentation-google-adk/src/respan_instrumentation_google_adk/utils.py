"""Utility functions for Respan Google ADK instrumentation."""
from datetime import datetime, timezone
from typing import Optional


def ns_to_datetime(value: Optional[int]) -> Optional[datetime]:
    """Convert nanoseconds timestamp to datetime."""
    if not value:
        return None
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)


def format_trace_id(trace_id: int) -> str:
    """Format trace ID as 32-char hex string."""
    return format(trace_id, "032x")


def format_span_id(span_id: int) -> str:
    """Format span ID as 16-char hex string."""
    return format(span_id, "016x")


_ADK_SCOPE_NAMES = {"gcp.vertex.agent", "google_adk", "google-adk"}
_ADK_SPAN_NAMES = {"invocation", "agent_run", "call_llm", "execute_tool", "invoke_agent", "generate_content"}


def is_adk_span(span: object) -> bool:
    """Check if span is from Google ADK instrumentation."""
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_library", None
    )
    scope_name = getattr(scope, "name", "") or ""
    if scope_name in _ADK_SCOPE_NAMES:
        return True
    # Fallback: check span name prefix + gen_ai attributes
    span_name = (getattr(span, "name", "") or "").split(" ")[0]
    if span_name in _ADK_SPAN_NAMES:
        attributes = getattr(span, "attributes", None) or {}
        if any(key.startswith("gen_ai.") for key in attributes):
            return True
    return False
