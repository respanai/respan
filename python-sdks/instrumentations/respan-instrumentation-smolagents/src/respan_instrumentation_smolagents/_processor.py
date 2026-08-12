"""smolagents-specific span cleanup for the Respan OTLP pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_sdk.constants.span_attributes import (
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)
from respan_instrumentation_smolagents._constants import (
    ASSISTANT_ROLE,
    SPAN_ALIAS_COMPLETION_TOKENS,
    SPAN_ALIAS_MODEL,
    SPAN_ALIAS_PROMPT_TOKENS,
    SPAN_ALIAS_TOOL_CALLS,
    SPAN_ALIAS_TOOLS,
    SPAN_ALIAS_TOTAL_REQUEST_TOKENS,
    GEN_AI_COMPLETION_CONTENT_ATTR,
    GEN_AI_COMPLETION_ROLE_ATTR,
    GEN_AI_COMPLETION_TOOL_CALLS_ATTR,
    LLM_REQUEST_FUNCTIONS_ATTR,
    OPENINFERENCE_INPUT_MESSAGES_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_ATTR,
    OPENINFERENCE_MESSAGE_CONTENTS_ATTR,
    OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR,
    OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
    OPENINFERENCE_SMOLAGENTS_MODULE,
    OTEL_SCOPE_NAME,
    SMOLAGENTS_FINAL_ANSWER_ARGUMENT,
    SMOLAGENTS_FINAL_ANSWER_TOOL_NAME,
    TOOL_CALL_FUNCTION_ARGUMENTS_FIELD,
    TOOL_CALL_FUNCTION_FIELD,
    TOOL_CALL_FUNCTION_NAME_FIELD,
)

_TOOL_CALLS_ATTR_RE = re.compile(
    rf"^{re.escape(TLSpanAttributes.LLM_COMPLETIONS)}\.\d+\.tool_calls$"
    rf"|^{re.escape(TLSpanAttributes.LLM_PROMPTS)}\.\d+\.tool_calls$"
)

_OFF_CONTRACT_ALIAS_ATTRS = frozenset(
    {
        SPAN_ALIAS_MODEL,
        SPAN_ALIAS_PROMPT_TOKENS,
        SPAN_ALIAS_COMPLETION_TOKENS,
        SPAN_ALIAS_TOTAL_REQUEST_TOKENS,
        SPAN_ALIAS_TOOLS,
        SPAN_ALIAS_TOOL_CALLS,
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
    }
)


def _is_smolagents_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    if attrs.get(OTEL_SCOPE_NAME) == OPENINFERENCE_SMOLAGENTS_MODULE:
        return True

    instrumentation_scope = getattr(span, "instrumentation_scope", None)
    if (
        getattr(instrumentation_scope, "name", None)
        == OPENINFERENCE_SMOLAGENTS_MODULE
    ):
        return True

    return False


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _json_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _json_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _extract_final_answer_content(tool_calls: list[Any]) -> str | None:
    if len(tool_calls) != 1:
        return None

    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get(TOOL_CALL_FUNCTION_FIELD)
    if not isinstance(function, dict):
        return None
    if function.get(TOOL_CALL_FUNCTION_NAME_FIELD) != SMOLAGENTS_FINAL_ANSWER_TOOL_NAME:
        return None

    arguments = _json_dict(function.get(TOOL_CALL_FUNCTION_ARGUMENTS_FIELD))
    if arguments is None or SMOLAGENTS_FINAL_ANSWER_ARGUMENT not in arguments:
        return None

    return _stringify_content(arguments[SMOLAGENTS_FINAL_ANSWER_ARGUMENT])


def _normalize_structured_contract_attrs(attrs: dict[str, Any]) -> None:
    if attrs.get(LLM_REQUEST_FUNCTIONS_ATTR) is None:
        helper_tools = _json_string(attrs.get(RESPAN_SPAN_TOOLS))
        if helper_tools:
            attrs[LLM_REQUEST_FUNCTIONS_ATTR] = helper_tools
    else:
        tools = _json_string(attrs[LLM_REQUEST_FUNCTIONS_ATTR])
        if tools:
            attrs[LLM_REQUEST_FUNCTIONS_ATTR] = tools

    if attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR) is None:
        helper_tool_calls = _json_list(attrs.get(RESPAN_SPAN_TOOL_CALLS))
        if helper_tool_calls:
            attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = helper_tool_calls

    tool_call_roots: set[str] = set()
    for key, value in list(attrs.items()):
        if _TOOL_CALLS_ATTR_RE.match(key):
            normalized = _json_list(value)
            if normalized:
                attrs[key] = normalized
                tool_call_roots.add(key)

    for root in tool_call_roots:
        for key in list(attrs):
            if key.startswith(f"{root}."):
                attrs.pop(key, None)

    completion_tool_calls = _json_list(attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR))
    if completion_tool_calls:
        final_answer_content = _extract_final_answer_content(completion_tool_calls)
        if final_answer_content not in {None, ""}:
            attrs[GEN_AI_COMPLETION_CONTENT_ATTR] = final_answer_content
            attrs.pop(GEN_AI_COMPLETION_TOOL_CALLS_ATTR, None)
        else:
            attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = completion_tool_calls

    if attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR):
        attrs.setdefault(GEN_AI_COMPLETION_ROLE_ATTR, ASSISTANT_ROLE)
        attrs.setdefault(GEN_AI_COMPLETION_CONTENT_ATTR, "")
    elif attrs.get(GEN_AI_COMPLETION_CONTENT_ATTR) not in {None, ""}:
        attrs.setdefault(GEN_AI_COMPLETION_ROLE_ATTR, ASSISTANT_ROLE)


def _flatten_openinference_message_content(attrs: dict[str, Any]) -> None:
    for prefix in (
        OPENINFERENCE_INPUT_MESSAGES_ATTR,
        OPENINFERENCE_OUTPUT_MESSAGES_ATTR,
    ):
        idx_to_texts: dict[int, list[str]] = {}
        for key, value in attrs.items():
            attr_prefix = f"{prefix}."
            if not key.startswith(attr_prefix):
                continue

            rest = key[len(attr_prefix) :]
            idx_part, _, field = rest.partition(".")
            if not idx_part.isdigit():
                continue
            if not field.endswith(f".{OPENINFERENCE_MESSAGE_CONTENT_TEXT_ATTR}"):
                continue
            if f"{OPENINFERENCE_MESSAGE_CONTENTS_ATTR}." not in field:
                continue
            if value in {None, ""}:
                continue
            idx_to_texts.setdefault(int(idx_part), []).append(str(value))

        for idx, texts in idx_to_texts.items():
            content_key = f"{prefix}.{idx}.{OPENINFERENCE_MESSAGE_CONTENT_ATTR}"
            attrs.setdefault(content_key, "\n".join(texts))


class SmolagentsSpanContentProcessor(SpanProcessor):
    """Flatten smolagents structured OI message content before translation."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if not _is_smolagents_span(span=span, attrs=attrs):
            return

        _flatten_openinference_message_content(attrs)
        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class SmolagentsSpanContractProcessor(SpanProcessor):
    """Normalize smolagents spans after OpenInference translation.

    The shared OpenInference translator still emits legacy convenience aliases
    and in-process structured values for existing integrations. smolagents keeps
    the cleanup local so it exports backend-compatible canonical fields.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if not _is_smolagents_span(span=span, attrs=attrs):
            return

        _normalize_structured_contract_attrs(attrs)

        for alias_attr in _OFF_CONTRACT_ALIAS_ATTRS:
            attrs.pop(alias_attr, None)

        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
