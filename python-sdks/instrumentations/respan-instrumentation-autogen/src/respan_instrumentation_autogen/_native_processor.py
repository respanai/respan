"""Normalize native AutoGen OTel spans before Respan export."""

from __future__ import annotations

from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
)
from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
)

AUTOGEN_CORE_SCOPE_NAME = "autogen-core"
AUTOGEN_OPERATION_INVOKE_AGENT = "invoke_agent"
AUTOGEN_OPERATION_EXECUTE_TOOL = "execute_tool"

_AUTOGEN_OPERATION_LOG_TYPES = {
    AUTOGEN_OPERATION_INVOKE_AGENT: LOG_TYPE_AGENT,
    AUTOGEN_OPERATION_EXECUTE_TOOL: LOG_TYPE_TOOL,
}


def _get_mutable_attributes(span: ReadableSpan) -> dict[str, Any] | None:
    # An ended span's ``_attributes`` is a ``BoundedAttributes`` that is frozen
    # (``_immutable``) on opentelemetry-sdk once the span has ended, so writing
    # to it in place raises ``TypeError``. Return a mutable copy; the caller
    # reassigns ``span._attributes`` after normalizing.
    attrs = getattr(span, "_attributes", None)
    if attrs is None:
        return None
    return dict(attrs)


def _get_scope_name(span: ReadableSpan) -> str | None:
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None)


def _get_entity_name(attrs: dict[str, Any], span_name: str) -> str:
    return (
        attrs.get(GEN_AI_AGENT_NAME)
        or attrs.get(GEN_AI_TOOL_NAME)
        or span_name
    )


class AutoGenNativeSpanProcessor(SpanProcessor):
    """Map AutoGen's native GenAI spans to non-LLM Respan log types.

    AutoGen AgentChat emits native spans from the ``autogen-core`` scope with
    ``gen_ai.system=autogen`` for agent and tool operations. Respan's generic
    GenAI exporter treats GenAI system spans without a request type as chat
    spans, so this processor marks native agent/tool spans explicitly and
    removes the LLM-only marker before export.
    """

    def on_start(self, span, parent_context=None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        if _get_scope_name(span) != AUTOGEN_CORE_SCOPE_NAME:
            return

        attrs = _get_mutable_attributes(span)
        if attrs is None:
            return

        log_type = _AUTOGEN_OPERATION_LOG_TYPES.get(attrs.get(GEN_AI_OPERATION_NAME))
        if log_type is None:
            return

        attrs[RESPAN_LOG_TYPE] = log_type
        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            _get_entity_name(attrs, span.name),
        )
        attrs.pop(GEN_AI_SYSTEM, None)
        attrs.pop(SpanAttributes.LLM_REQUEST_TYPE, None)
        # Reassign the normalized copy; the original frozen BoundedAttributes
        # cannot be mutated in place after the span has ended.
        span._attributes = attrs

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
