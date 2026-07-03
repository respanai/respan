"""OpenRouter span normalization for the Respan OTLP pipeline."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_instrumentation_openrouter._constants import (
    OPENAI_INSTRUMENTATION_SCOPE_FRAGMENT,
    OPENROUTER_HOST_MARKERS,
    OPENROUTER_SYSTEM_NAME,
    OPENROUTER_URL_ATTRIBUTE_KEYS,
)

OTEL_SCOPE_NAME_ATTR = "otel.scope.name"

GEN_AI_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
GEN_AI_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
GEN_AI_TOOL_CALLS_SUFFIX = ".tool_calls"
GEN_AI_TOOL_CALLS_INDEX_FRAGMENT = ".tool_calls."
GEN_AI_COMPLETION_TOOL_CALLS_ATTR = (
    f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
)
GEN_AI_OUTPUT_MESSAGES_ATTR = getattr(
    TLSpanAttributes,
    "GEN_AI_OUTPUT_MESSAGES",
    "gen_ai.output.messages",
)
GEN_AI_TOOL_DEFINITIONS_ATTR = getattr(
    TLSpanAttributes,
    "GEN_AI_TOOL_DEFINITIONS",
    "gen_ai.tool.definitions",
)

_OFF_CONTRACT_ALIAS_ATTRIBUTES = frozenset(
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


def _span_scope_name(span: ReadableSpan, attrs: dict[str, Any]) -> str | None:
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span,
        "_instrumentation_scope",
        None,
    )
    scope_name = getattr(scope, "name", None)
    if scope_name:
        return scope_name
    attr_scope_name = attrs.get(OTEL_SCOPE_NAME_ATTR)
    return attr_scope_name if isinstance(attr_scope_name, str) else None


def _is_openai_span(span: ReadableSpan, attrs: dict[str, Any]) -> bool:
    scope_name = _span_scope_name(span, attrs)
    if (
        isinstance(scope_name, str)
        and OPENAI_INSTRUMENTATION_SCOPE_FRAGMENT in scope_name.lower()
    ):
        return True

    system = attrs.get(TLSpanAttributes.LLM_SYSTEM) or attrs.get("llm.system")
    return isinstance(system, str) and system.lower() == "openai"


def _has_openrouter_url_marker(attrs: dict[str, Any]) -> bool:
    for key in OPENROUTER_URL_ATTRIBUTE_KEYS:
        value = attrs.get(key)
        if value is None:
            continue
        normalized_value = str(value).lower()
        if any(marker in normalized_value for marker in OPENROUTER_HOST_MARKERS):
            return True
    return False


def _drop_attribute(attrs: dict[str, Any], key: str) -> None:
    attrs.pop(key, None)


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _parse_json_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _is_openai_omit(value: Any) -> bool:
    if isinstance(value, str) and value.startswith(_OPENAI_OMIT_VALUE_PREFIX):
        return True
    return repr(value).startswith(_OPENAI_OMIT_VALUE_PREFIX)


def _set_nested_value(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = target
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def _is_gen_ai_message_tool_call_key(key: str) -> bool:
    return (
        key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
        and GEN_AI_TOOL_CALLS_INDEX_FRAGMENT in key
    )


def _is_gen_ai_message_tool_call_aggregate_key(key: str) -> bool:
    return (
        key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
        and key.endswith(GEN_AI_TOOL_CALLS_SUFFIX)
    )


def _tool_calls_content(value: Any) -> str | None:
    tool_calls = _parse_json_if_string(value)
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    descriptions: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments")
        if arguments in {None, ""}:
            descriptions.append(name)
        elif isinstance(arguments, str):
            descriptions.append(f"{name}({arguments})")
        else:
            descriptions.append(f"{name}({json.dumps(arguments, default=str)})")

    if not descriptions:
        return None
    prefix = "Tool call" if len(descriptions) == 1 else "Tool calls"
    return f"{prefix}: {', '.join(descriptions)}"


def _normalize_gen_ai_tool_calls(attrs: dict[str, Any]) -> None:
    indexed_calls: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

    for key in tuple(attrs):
        if not _is_gen_ai_message_tool_call_key(key):
            continue
        message_key, rest = key.split(GEN_AI_TOOL_CALLS_INDEX_FRAGMENT, 1)
        aggregate_key = f"{message_key}{GEN_AI_TOOL_CALLS_SUFFIX}"
        parts = rest.split(".", 1)
        if not parts[0].isdigit() or len(parts) == 1:
            continue
        tool_call_index = int(parts[0])
        tool_call = indexed_calls[aggregate_key].setdefault(tool_call_index, {})
        _set_nested_value(tool_call, parts[1], attrs[key])

    for aggregate_key, tool_calls_by_index in indexed_calls.items():
        if aggregate_key not in attrs:
            attrs[aggregate_key] = [
                tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
            ]

    for key in tuple(attrs):
        if _is_gen_ai_message_tool_call_aggregate_key(key):
            structured_tool_calls = _parse_json_if_string(attrs[key])
            attrs[key] = _json_string(structured_tool_calls) or "[]"
            message_key = key[: -len(GEN_AI_TOOL_CALLS_SUFFIX)]
            content_key = f"{message_key}.content"
            if attrs.get(content_key) in {None, ""}:
                attrs[content_key] = _tool_calls_content(structured_tool_calls) or ""
            role_key = f"{message_key}.role"
            if attrs.get(role_key) is None:
                attrs[role_key] = "assistant"
        elif _is_gen_ai_message_tool_call_key(key):
            _drop_attribute(attrs, key)


def _canonical_tool_definition(tool_definition: Any) -> Any:
    if not isinstance(tool_definition, dict):
        return tool_definition
    if tool_definition.get("type") != "function":
        return tool_definition
    if "function" in tool_definition:
        return tool_definition

    function = {"name": tool_definition.get("name")}
    if tool_definition.get("description") is not None:
        function["description"] = tool_definition.get("description")
    if tool_definition.get("parameters") is not None:
        function["parameters"] = tool_definition.get("parameters")
    return {"type": "function", "function": function}


def _canonical_tool_definitions(value: Any) -> Any:
    tool_definitions = _parse_json_if_string(value)
    if not isinstance(tool_definitions, list):
        return tool_definitions
    return [_canonical_tool_definition(tool) for tool in tool_definitions]


def _tool_call_from_output_part(part: dict[str, Any]) -> dict[str, Any] | None:
    name = part.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = part.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, default=str)
    tool_call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    if part.get("id") is not None:
        tool_call["id"] = part.get("id")
    return tool_call


def _normalize_gen_ai_output_messages(attrs: dict[str, Any]) -> None:
    output_messages = _parse_json_if_string(attrs.get(GEN_AI_OUTPUT_MESSAGES_ATTR))
    if not isinstance(output_messages, list):
        return

    for message_index, message in enumerate(output_messages):
        if not isinstance(message, dict):
            continue
        prefix = f"{TLSpanAttributes.LLM_COMPLETIONS}.{message_index}"
        role = message.get("role")
        if isinstance(role, str) and role:
            attrs.setdefault(f"{prefix}.role", role)

        parts = message.get("parts") or []
        if not isinstance(parts, list):
            continue

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text" and part.get("content") is not None:
                content_parts.append(str(part.get("content")))
            elif part_type == "tool_call":
                tool_call = _tool_call_from_output_part(part)
                if tool_call is not None:
                    tool_calls.append(tool_call)

        content = "".join(content_parts)
        if content:
            attrs.setdefault(f"{prefix}.content", content)
        if tool_calls:
            attrs.setdefault(f"{prefix}.tool_calls", json.dumps(tool_calls))
            if not attrs.get(f"{prefix}.content"):
                attrs[f"{prefix}.content"] = _tool_calls_content(tool_calls) or ""


def _normalize_structured_contract_attrs(attrs: dict[str, Any]) -> None:
    tools_value = attrs.get(TLSpanAttributes.LLM_REQUEST_FUNCTIONS)
    if tools_value is None:
        tools_value = attrs.get(RESPAN_SPAN_TOOLS)
    if tools_value is None:
        tools_value = attrs.get(GEN_AI_TOOL_DEFINITIONS_ATTR)
    tools_json = _json_string(_canonical_tool_definitions(tools_value))
    if tools_json:
        attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = tools_json

    tool_calls_value = attrs.get(GEN_AI_COMPLETION_TOOL_CALLS_ATTR)
    if tool_calls_value is None:
        tool_calls_value = attrs.get(RESPAN_SPAN_TOOL_CALLS) or attrs.get(
            "tool_calls"
        )
    tool_calls_json = _json_string(_parse_json_if_string(tool_calls_value))
    if tool_calls_json:
        attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = tool_calls_json


def _has_llm_message_attrs(attrs: dict[str, Any]) -> bool:
    return any(
        key.startswith((GEN_AI_PROMPT_PREFIX, GEN_AI_COMPLETION_PREFIX))
        for key in attrs
    )


class OpenRouterSpanProcessor(SpanProcessor):
    """Normalize OpenAI-compatible OpenRouter spans before Respan export."""

    def __init__(self, *, normalize_all_openai_spans: bool = True) -> None:
        self._normalize_all_openai_spans = normalize_all_openai_spans

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def _is_openrouter_span(self, span: ReadableSpan, attrs: dict[str, Any]) -> bool:
        system = attrs.get(TLSpanAttributes.LLM_SYSTEM)
        if isinstance(system, str) and system.lower() == OPENROUTER_SYSTEM_NAME:
            return True
        if not _is_openai_span(span, attrs):
            return False
        return self._normalize_all_openai_spans or _has_openrouter_url_marker(attrs)

    def on_end(self, span: ReadableSpan) -> None:
        original_attrs = getattr(span, "_attributes", None)
        if original_attrs is None:
            return

        attrs = dict(original_attrs)
        if not self._is_openrouter_span(span, attrs):
            return

        attrs[TLSpanAttributes.LLM_SYSTEM] = OPENROUTER_SYSTEM_NAME

        if _has_llm_message_attrs(attrs) or attrs.get(TLSpanAttributes.LLM_REQUEST_MODEL):
            attrs.setdefault(
                TLSpanAttributes.LLM_REQUEST_TYPE,
                LLMRequestTypeValues.CHAT.value,
            )
            attrs.setdefault(RESPAN_LOG_TYPE, LOG_TYPE_CHAT)

        _normalize_gen_ai_output_messages(attrs)
        _normalize_gen_ai_tool_calls(attrs)
        _normalize_structured_contract_attrs(attrs)

        for key, value in list(attrs.items()):
            if key in _OFF_CONTRACT_ALIAS_ATTRIBUTES or _is_openai_omit(value):
                attrs.pop(key, None)

        span._attributes = attrs

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
