"""Translate LiveKit native OTEL span data into Respan span fields."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_livekit._constants import (
    ASSISTANT_ROLE,
    ATTR_LLM_METRICS,
    ATTR_PROVIDER_REQUEST_IDS,
    EVENT_GEN_AI_ASSISTANT_MESSAGE,
    EVENT_GEN_AI_CHOICE,
    EVENT_GEN_AI_SYSTEM_MESSAGE,
    EVENT_GEN_AI_TOOL_MESSAGE,
    EVENT_GEN_AI_USER_MESSAGE,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    ID_KEY,
    LIVEKIT_CHAT_SPAN_NAME,
    LIVEKIT_LLM_REQUEST_SPAN_NAME,
    LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR,
    NAME_KEY,
    ROLE_KEY,
    TOOL_ROLE,
    TYPE_KEY,
    USER_ROLE,
)
from respan_instrumentation_livekit._serialization import (
    get_value,
    parse_jsonish,
    safe_json,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TOOL, LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_TRACE_GROUP_ID,
)

_PROMPT_EVENT_ROLES = {
    EVENT_GEN_AI_SYSTEM_MESSAGE: "system",
    EVENT_GEN_AI_USER_MESSAGE: USER_ROLE,
    EVENT_GEN_AI_ASSISTANT_MESSAGE: ASSISTANT_ROLE,
    EVENT_GEN_AI_TOOL_MESSAGE: TOOL_ROLE,
}


def _event_name(event: Any) -> str | None:
    return getattr(event, "name", None)


def _event_attrs(event: Any) -> Mapping[str, Any]:
    attrs = getattr(event, "attributes", None)
    return attrs if isinstance(attrs, Mapping) else {}


def _content_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json(value)


def _tool_calls_from_event(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        value = [value.decode() if isinstance(value, bytes) else value]
    elif isinstance(value, Mapping):
        value = [value]
    elif not isinstance(value, Sequence):
        value = [value]

    tool_calls: list[dict[str, Any]] = []
    for item in value:
        parsed = parse_jsonish(item)
        if not isinstance(parsed, Mapping):
            continue
        function = parsed.get(FUNCTION_KEY)
        if not isinstance(function, Mapping):
            continue
        function_name = function.get(NAME_KEY)
        if not function_name:
            continue
        normalized = {
            TYPE_KEY: str(parsed.get(TYPE_KEY) or FUNCTION_TOOL_TYPE),
            FUNCTION_KEY: {
                NAME_KEY: str(function_name),
                "arguments": (
                    function.get("arguments")
                    if isinstance(function.get("arguments"), str)
                    else safe_json(function.get("arguments") or {})
                ),
            },
        }
        call_id = parsed.get(ID_KEY)
        if call_id:
            normalized[ID_KEY] = str(call_id)
        tool_calls.append(normalized)
    return tool_calls


def _metrics(attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    value = attrs.get(ATTR_LLM_METRICS)
    if not isinstance(value, str):
        return {}
    try:
        loaded = json.loads(value)
    except Exception:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _provider_name(attrs: Mapping[str, Any]) -> str:
    provider = attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) or attrs.get(
        SpanAttributes.LLM_SYSTEM
    )
    if not provider:
        return "livekit"
    provider_name = str(provider).lower()
    if provider_name == "unknown":
        return "livekit"
    return provider_name


def _apply_prompt_events(
    translated: dict[str, Any],
    events: Sequence[Any],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    for event in events:
        event_name = _event_name(event)
        role = _PROMPT_EVENT_ROLES.get(event_name)
        if role is None:
            continue
        attrs = _event_attrs(event)
        message: dict[str, Any] = {ROLE_KEY: str(attrs.get(ROLE_KEY) or role)}
        if event_name == EVENT_GEN_AI_TOOL_MESSAGE:
            message[ROLE_KEY] = TOOL_ROLE

        content = attrs.get("content")
        if content is not None:
            message["content"] = _content_to_str(content)

        tool_calls = _tool_calls_from_event(attrs.get("tool_calls"))
        if tool_calls:
            message["tool_calls"] = tool_calls

        if role == TOOL_ROLE and attrs.get(NAME_KEY):
            message[NAME_KEY] = str(attrs[NAME_KEY])
        messages.append(message)

    for index, message in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{index}"
        translated[f"{prefix}.role"] = str(message.get(ROLE_KEY) or USER_ROLE)
        if "content" in message:
            translated[f"{prefix}.content"] = _content_to_str(message["content"])
        if message.get("tool_calls"):
            translated[f"{prefix}.tool_calls"] = safe_json(message["tool_calls"])

    return messages


def _apply_completion_event(
    translated: dict[str, Any],
    events: Sequence[Any],
) -> dict[str, Any]:
    completion: dict[str, Any] = {}
    for event in events:
        if _event_name(event) != EVENT_GEN_AI_CHOICE:
            continue
        attrs = _event_attrs(event)
        content = attrs.get("content")
        if content is not None:
            completion["content"] = _content_to_str(content)
        tool_calls = _tool_calls_from_event(attrs.get("tool_calls"))
        if tool_calls:
            completion["tool_calls"] = tool_calls

    if completion:
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
        translated[f"{prefix}.role"] = ASSISTANT_ROLE
        if "content" in completion:
            translated[f"{prefix}.content"] = completion["content"]
        if completion.get("tool_calls"):
            translated[f"{prefix}.tool_calls"] = safe_json(completion["tool_calls"])
    return completion


def _apply_usage(translated: dict[str, Any], attrs: Mapping[str, Any]) -> None:
    metrics = _metrics(attrs)
    input_tokens = _int_value(
        attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, metrics.get("prompt_tokens"))
    )
    output_tokens = _int_value(
        attrs.get(
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            metrics.get("completion_tokens"),
        )
    )
    total_tokens = _int_value(metrics.get("total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    if input_tokens is not None:
        translated[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        translated[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
    if output_tokens is not None:
        translated[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        translated[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
    if total_tokens is not None:
        translated[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens

    cached_tokens = _int_value(
        metrics.get("prompt_cached_tokens")
        or metrics.get("cache_read_tokens")
        or metrics.get("cache_read_input_tokens")
    )
    cache_read_attr = getattr(SpanAttributes, "LLM_USAGE_CACHE_READ_INPUT_TOKENS", None)
    if cached_tokens is not None and cache_read_attr:
        translated[cache_read_attr] = cached_tokens


def _apply_livekit_metadata(translated: dict[str, Any], attrs: Mapping[str, Any]) -> None:
    metrics = attrs.get(ATTR_LLM_METRICS)
    if isinstance(metrics, str) and metrics:
        translated[f"{RESPAN_METADATA}.livekit_llm_metrics"] = metrics

    provider_request_ids = attrs.get(ATTR_PROVIDER_REQUEST_IDS)
    if provider_request_ids:
        translated[f"{RESPAN_METADATA}.livekit_provider_request_ids"] = (
            provider_request_ids
            if isinstance(provider_request_ids, str)
            else safe_json(provider_request_ids)
        )


def _apply_tool_definitions(translated: dict[str, Any], attrs: Mapping[str, Any]) -> None:
    value = attrs.get(LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR)
    parsed = parse_jsonish(value)
    if parsed:
        translated[SpanAttributes.LLM_REQUEST_FUNCTIONS] = (
            parsed if isinstance(parsed, str) else safe_json(parsed)
        )


def is_livekit_llm_span(span_name: str, attrs: Mapping[str, Any]) -> bool:
    return attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat" and (
        span_name == LIVEKIT_LLM_REQUEST_SPAN_NAME
        or attrs.get(ATTR_LLM_METRICS) is not None
        or attrs.get(LIVEKIT_RESPAN_TOOL_DEFINITIONS_ATTR) is not None
    )


def build_livekit_llm_attrs(
    *,
    span_name: str,
    attrs: Mapping[str, Any],
    events: Sequence[Any],
) -> dict[str, Any]:
    """Build canonical Respan chat attributes from a LiveKit LLM span."""
    translated: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        SpanAttributes.LLM_SYSTEM: _provider_name(attrs),
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: LIVEKIT_CHAT_SPAN_NAME,
        SpanAttributes.TRACELOOP_ENTITY_PATH: LIVEKIT_CHAT_SPAN_NAME,
    }

    model = attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) or attrs.get(
        SpanAttributes.LLM_REQUEST_MODEL
    )
    if model:
        translated[SpanAttributes.LLM_REQUEST_MODEL] = str(model)

    workflow_name = attrs.get(RESPAN_TRACE_GROUP_ID)
    if workflow_name:
        translated[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = str(workflow_name)

    messages = _apply_prompt_events(translated=translated, events=events)
    completion = _apply_completion_event(translated=translated, events=events)
    _apply_usage(translated=translated, attrs=attrs)
    _apply_tool_definitions(translated=translated, attrs=attrs)
    _apply_livekit_metadata(translated=translated, attrs=attrs)

    if messages:
        translated[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(messages)
    if completion:
        translated[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(completion)

    return translated


def normalize_livekit_tools(tools: Any) -> list[dict[str, Any]]:
    """Normalize LiveKit function tools into OpenAI-compatible tool schemas."""
    if not tools:
        return []
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        tools = [tools]

    try:
        from livekit.agents.llm.tool_context import ToolContext

        parsed = ToolContext(tools).parse_function_tools("openai")
        if isinstance(parsed, list):
            return [tool for tool in parsed if isinstance(tool, dict)]
    except Exception:
        pass

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        info = get_value(tool, "info")
        raw_schema = get_value(info, "raw_schema")
        if isinstance(raw_schema, Mapping) and raw_schema.get(NAME_KEY):
            normalized.append({TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: dict(raw_schema)})
            continue

        name = get_value(info, NAME_KEY) or get_value(tool, ID_KEY)
        if not name:
            continue
        function: dict[str, Any] = {
            NAME_KEY: str(name),
            "parameters": {"type": "object", "properties": {}},
        }
        description = get_value(info, "description")
        if description:
            function["description"] = str(description)
        normalized.append({TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function})
    return normalized


def build_tool_span_attrs(
    *,
    tool_name: str,
    arguments: Any,
    output: Any,
    call_id: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
        SpanAttributes.TRACELOOP_ENTITY_NAME: tool_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: tool_name,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: safe_json(
            {
                NAME_KEY: tool_name,
                "arguments": parse_jsonish(arguments),
            }
        ),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: safe_json(output),
    }
    if call_id:
        attrs[f"{RESPAN_METADATA}.livekit_tool_call_id"] = str(call_id)
    return attrs
