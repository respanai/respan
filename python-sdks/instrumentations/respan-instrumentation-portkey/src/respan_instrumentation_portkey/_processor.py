"""Portkey-specific span cleanup for the Respan OTLP pipeline."""

from __future__ import annotations

import json
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_instrumentation_portkey._constants import OPENINFERENCE_PORTKEY_MODULE

OTEL_SCOPE_NAME = "otel.scope.name"
GEN_AI_COMPLETION_TOOL_CALLS_ATTR = f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
LLM_REQUEST_FUNCTIONS_ATTR = TLSpanAttributes.LLM_REQUEST_FUNCTIONS

_OFF_CONTRACT_ALIAS_ATTRS = frozenset(
    {
        "completion_tokens",
        "has_tool_calls",
        "model",
        "parallel_tool_calls",
        "prompt_tokens",
        "span_tools",
        "tool_calls",
        "tools",
        "total_request_tokens",
        RESPAN_SPAN_HANDOFFS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_TOOLS,
    }
)
_OPENAI_OMIT_VALUE_PREFIX = "<openai.Omit object"


def _is_portkey_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    if attrs.get(OTEL_SCOPE_NAME) == OPENINFERENCE_PORTKEY_MODULE:
        return True

    instrumentation_scope = getattr(span, "instrumentation_scope", None)
    return getattr(instrumentation_scope, "name", None) == OPENINFERENCE_PORTKEY_MODULE


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _strip_placeholder_values(attrs: dict[str, Any]) -> None:
    for key, value in list(attrs.items()):
        if isinstance(value, str) and value.startswith(_OPENAI_OMIT_VALUE_PREFIX):
            attrs.pop(key, None)


def _normalize_structured_contract_attrs(attrs: dict[str, Any]) -> None:
    tools_json = _json_string(attrs.get(LLM_REQUEST_FUNCTIONS_ATTR))
    if tools_json:
        attrs[LLM_REQUEST_FUNCTIONS_ATTR] = tools_json
    else:
        helper_tools_json = _json_string(attrs.get(RESPAN_SPAN_TOOLS))
        if helper_tools_json:
            attrs[LLM_REQUEST_FUNCTIONS_ATTR] = helper_tools_json

    tool_calls_json = _json_string(attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR))
    if tool_calls_json:
        attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = tool_calls_json
    else:
        helper_tool_calls_json = _json_string(attrs.get(RESPAN_SPAN_TOOL_CALLS))
        if helper_tool_calls_json:
            attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = helper_tool_calls_json


class PortkeySpanContractProcessor(SpanProcessor):
    """Normalize Portkey spans after OpenInference translation."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if not _is_portkey_span(span=span, attrs=attrs):
            return

        _normalize_structured_contract_attrs(attrs)
        _strip_placeholder_values(attrs)

        for alias_attr in _OFF_CONTRACT_ALIAS_ATTRS:
            attrs.pop(alias_attr, None)

        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
