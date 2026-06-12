"""Groq span cleanup for Respan's OpenInference bridge."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

GROQ_OPENINFERENCE_SCOPE_NAME = "openinference.instrumentation.groq"
GEN_AI_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
GEN_AI_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."
GEN_AI_TOOL_CALLS_SUFFIX = ".tool_calls"
GEN_AI_TOOL_CALLS_INDEX_FRAGMENT = ".tool_calls."

_OFF_CONTRACT_ALIAS_ATTRIBUTES = frozenset(
    {
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "tools",
        "tool_calls",
        "span_tools",
        "has_tool_calls",
        "parallel_tool_calls",
        "respan.span.tools",
        "respan.span.tool_calls",
        "respan.span.handoffs",
    }
)


def _span_scope_name(span: ReadableSpan) -> str | None:
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span,
        "_instrumentation_scope",
        None,
    )
    return getattr(scope, "name", None)


def _is_groq_omit(value: Any) -> bool:
    if isinstance(value, str) and value.startswith("<groq.Omit object at "):
        return True
    value_type = type(value)
    if value_type.__module__.startswith("groq") and value_type.__name__ == "Omit":
        return True
    return repr(value).startswith("<groq.Omit object at ")


def _drop_attribute(attrs: Any, key: str) -> None:
    try:
        attrs.pop(key, None)
    except AttributeError:
        try:
            del attrs[key]
        except (KeyError, TypeError):
            return


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


def _structured_tool_calls(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
    return value


def _tool_calls_content(value: Any) -> str | None:
    tool_calls = _structured_tool_calls(value)
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


def _normalize_gen_ai_tool_calls(attrs: Any) -> None:
    """Keep tool calls in the contract shape the backend parses."""
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
            attrs[key] = _structured_tool_calls(attrs[key])
            message_key = key[: -len(GEN_AI_TOOL_CALLS_SUFFIX)]
            content_key = f"{message_key}.content"
            if attrs.get(content_key) in {None, ""}:
                attrs[content_key] = _tool_calls_content(attrs[key]) or ""
            role_key = f"{message_key}.role"
            if attrs.get(role_key) is None:
                attrs[role_key] = "assistant"
        elif _is_gen_ai_message_tool_call_key(key):
            _drop_attribute(attrs, key)


def _is_groq_span(span: ReadableSpan, attrs: Any) -> bool:
    if _span_scope_name(span) == GROQ_OPENINFERENCE_SCOPE_NAME:
        return True
    if any(_is_groq_omit(value) for value in attrs.values()):
        return True
    system = attrs.get("gen_ai.system") or attrs.get("llm.system")
    return isinstance(system, str) and system.lower() == "groq"


class GroqSpanProcessor(SpanProcessor):
    """Remove Groq/OpenInference bridge artifacts before export."""

    def on_start(self, span, parent_context=None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        attrs = getattr(span, "_attributes", None)
        if attrs is None:
            return
        if not _is_groq_span(span, attrs):
            return

        _normalize_gen_ai_tool_calls(attrs)
        for key in tuple(attrs):
            if key in _OFF_CONTRACT_ALIAS_ATTRIBUTES or _is_groq_omit(attrs[key]):
                _drop_attribute(attrs, key)
