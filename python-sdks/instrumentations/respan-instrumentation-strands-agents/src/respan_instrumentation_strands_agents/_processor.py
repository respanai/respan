"""Normalize Strands Agents OTEL spans for the Respan OTLP pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from threading import RLock
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Status, StatusCode
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

from respan_instrumentation_strands_agents._constants import (
    STRANDS_EVENT_CHOICE,
    STRANDS_EVENT_MESSAGE_PREFIX,
    STRANDS_EVENT_MESSAGE_SUFFIX,
    STRANDS_EVENT_OPERATION_DETAILS,
    STRANDS_EVENT_SYSTEM_MESSAGE,
    STRANDS_EVENT_TOOL_MESSAGE,
    STRANDS_NON_LLM_ATTRS_TO_STRIP,
    STRANDS_OPERATION_CHAT,
    STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE,
    STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT,
    STRANDS_OPERATION_EXECUTE_TOOL,
    STRANDS_OPERATION_INVOKE_AGENT,
    STRANDS_OPERATION_INVOKE_PREFIX,
    STRANDS_RAW_ATTR_PREFIXES_TO_STRIP,
    STRANDS_RAW_ATTRS_TO_STRIP,
    STRANDS_SYSTEM_NAME,
    STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
)
from respan_instrumentation_strands_agents._policy import (
    content_enabled,
    filter_attributes,
)
from respan_instrumentation_strands_agents._serialization import (
    json_dumps,
    safe_text,
    to_jsonable,
)

logger = logging.getLogger(__name__)

_GEN_AI_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_GEN_AI_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_OFF_CONTRACT_ALIAS_ATTRS = (
    frozenset(
        {
            RESPAN_SPAN_TOOLS,
            RESPAN_SPAN_TOOL_CALLS,
            RESPAN_SPAN_HANDOFFS,
        }
    )
    | STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP
)


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    return json_dumps(value)


def _set_if_missing(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if attrs.get(key) in (None, "", (), []):
        attrs[key] = value


def _set_if_present(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    attrs[key] = value


def _get_events(span: ReadableSpan) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for event in getattr(span, "events", ()) or ():
        event_name = getattr(event, "name", "")
        event_attributes = getattr(event, "attributes", None) or {}
        if isinstance(event_name, str) and isinstance(event_attributes, Mapping):
            yield event_name, event_attributes


def _is_strands_system(attrs: Mapping[str, Any]) -> bool:
    return (
        attrs.get(SpanAttributes.LLM_SYSTEM) == STRANDS_SYSTEM_NAME
        or attrs.get(gen_ai.GEN_AI_PROVIDER_NAME) == STRANDS_SYSTEM_NAME
    )


def is_strands_agents_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None:
        return scope.name == "strands.telemetry.tracer"
    return _is_strands_system(attrs)


def _extract_log_type(span: ReadableSpan, attrs: Mapping[str, Any]) -> str | None:
    if isinstance(attrs.get(gen_ai.GEN_AI_TOOL_NAME), str):
        return LOG_TYPE_TOOL

    operation_name = attrs.get(gen_ai.GEN_AI_OPERATION_NAME)
    if operation_name == STRANDS_OPERATION_CHAT:
        return LOG_TYPE_CHAT
    if operation_name == STRANDS_OPERATION_EXECUTE_TOOL:
        return LOG_TYPE_TOOL
    if operation_name in {
        STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE,
        STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT,
    }:
        return LOG_TYPE_TASK
    if isinstance(operation_name, str) and operation_name.startswith("memory."):
        return LOG_TYPE_TASK
    if operation_name == STRANDS_OPERATION_INVOKE_AGENT:
        return LOG_TYPE_AGENT
    if isinstance(operation_name, str) and operation_name.startswith(
        STRANDS_OPERATION_INVOKE_PREFIX
    ):
        return LOG_TYPE_AGENT

    if span.name.startswith(f"{STRANDS_OPERATION_EXECUTE_TOOL} "):
        return LOG_TYPE_TOOL
    if span.name.startswith(f"{STRANDS_OPERATION_INVOKE_AGENT} "):
        return LOG_TYPE_AGENT
    return None


def _span_suffix_name(span_name: str, prefix: str, fallback: str) -> str:
    if span_name.startswith(f"{prefix} "):
        suffix = span_name[len(prefix) + 1 :].strip()
        if suffix:
            return suffix
    return fallback


def _extract_agent_name(span: ReadableSpan, attrs: Mapping[str, Any]) -> str:
    agent_name = attrs.get(gen_ai.GEN_AI_AGENT_NAME)
    if isinstance(agent_name, str) and agent_name:
        return safe_text(agent_name)
    return safe_text(
        _span_suffix_name(
            span_name=span.name,
            prefix=STRANDS_OPERATION_INVOKE_AGENT,
            fallback=safe_text(
                attrs.get(gen_ai.GEN_AI_OPERATION_NAME)
                or STRANDS_OPERATION_INVOKE_AGENT
            ).removeprefix(STRANDS_OPERATION_INVOKE_PREFIX),
        )
    )


def _extract_tool_name(span: ReadableSpan, attrs: Mapping[str, Any]) -> str:
    tool_name = attrs.get(gen_ai.GEN_AI_TOOL_NAME)
    if isinstance(tool_name, str) and tool_name:
        return safe_text(tool_name)
    return safe_text(
        _span_suffix_name(
            span_name=span.name,
            prefix=STRANDS_OPERATION_EXECUTE_TOOL,
            fallback=STRANDS_OPERATION_EXECUTE_TOOL,
        )
    )


def _extract_text_from_content(content: Any) -> str | None:
    parsed_content = _safe_json_loads(content)
    if isinstance(parsed_content, str):
        return parsed_content
    if isinstance(parsed_content, Mapping):
        text = parsed_content.get("text")
        if isinstance(text, str):
            return text
        return None
    if not isinstance(parsed_content, list):
        return None

    text_parts = []
    for item in parsed_content:
        if not isinstance(item, Mapping):
            return None
        text = item.get("text")
        if not isinstance(text, str):
            return None
        text_parts.append(text)
    return "\n".join(text_parts)


def _content_for_message(content: Any) -> Any:
    parsed_content = _safe_json_loads(content)
    text = _extract_text_from_content(parsed_content)
    if text is not None:
        return text
    return to_jsonable(parsed_content)


def _parts_to_content(parts: Any) -> Any:
    parsed_parts = _safe_json_loads(parts)
    if not isinstance(parsed_parts, list):
        return to_jsonable(parsed_parts)

    content_blocks: list[Any] = []
    for part in parsed_parts:
        if not isinstance(part, Mapping):
            content_blocks.append(part)
            continue
        part_type = part.get("type")
        if part_type == "text":
            content_blocks.append({"text": part.get("content", "")})
        elif part_type == "tool_call":
            content_blocks.append(
                {
                    "toolUse": {
                        "name": part.get("name", ""),
                        "toolUseId": part.get("id", ""),
                        "input": part.get("arguments", {}),
                    }
                }
            )
        elif part_type == "tool_call_response":
            content_blocks.append(
                {
                    "toolResult": {
                        "toolUseId": part.get("id", ""),
                        "content": part.get("response", ""),
                    }
                }
            )
        else:
            content_blocks.append(to_jsonable(part))

    text = _extract_text_from_content(content_blocks)
    if text is not None:
        return text
    return content_blocks


def _normalize_message(raw_message: Any, default_role: str) -> dict[str, Any] | None:
    parsed_message = _safe_json_loads(raw_message)
    if not isinstance(parsed_message, Mapping):
        content = _content_for_message(parsed_message)
        if content in (None, "", [], {}):
            return None
        return {"role": default_role, "content": content}

    role = parsed_message.get("role") or default_role
    content = parsed_message.get("content")
    if content is None and "parts" in parsed_message:
        content = _parts_to_content(parsed_message.get("parts"))
    normalized_content = _content_for_message(content)
    return {"role": safe_text(role), "content": normalized_content}


def _normalize_messages(value: Any, default_role: str) -> list[dict[str, Any]] | None:
    parsed_value = _safe_json_loads(value)
    if isinstance(parsed_value, list):
        messages = [
            normalized_message
            for item in parsed_value
            if (
                normalized_message := _normalize_message(
                    raw_message=item,
                    default_role=default_role,
                )
            )
            is not None
        ]
        return messages or None

    normalized_message = _normalize_message(
        raw_message=parsed_value,
        default_role=default_role,
    )
    if normalized_message is None:
        return None
    return [normalized_message]


def _operation_detail_messages(
    span: ReadableSpan,
    attr_name: str,
    default_role: str,
) -> list[dict[str, Any]] | None:
    messages: list[dict[str, Any]] = []
    for event_name, event_attributes in _get_events(span):
        if event_name != STRANDS_EVENT_OPERATION_DETAILS:
            continue
        normalized_messages = _normalize_messages(
            value=event_attributes.get(attr_name),
            default_role=default_role,
        )
        if normalized_messages:
            messages.extend(normalized_messages)
    return messages or None


def _legacy_input_messages(span: ReadableSpan) -> list[dict[str, Any]] | None:
    messages: list[dict[str, Any]] = []
    for event_name, event_attributes in _get_events(span):
        if event_name == STRANDS_EVENT_SYSTEM_MESSAGE:
            normalized_message = _normalize_message(
                raw_message={
                    "role": "system",
                    "content": event_attributes.get("content"),
                },
                default_role="system",
            )
        elif event_name.startswith(
            STRANDS_EVENT_MESSAGE_PREFIX
        ) and event_name.endswith(STRANDS_EVENT_MESSAGE_SUFFIX):
            role = event_name[
                len(STRANDS_EVENT_MESSAGE_PREFIX) : -len(STRANDS_EVENT_MESSAGE_SUFFIX)
            ]
            normalized_message = _normalize_message(
                raw_message={
                    "role": event_attributes.get("role") or role,
                    "content": event_attributes.get("content"),
                },
                default_role=role,
            )
        else:
            normalized_message = None

        if normalized_message is not None:
            messages.append(normalized_message)
    return messages or None


def _legacy_output_messages(
    span: ReadableSpan,
    default_role: str,
) -> list[dict[str, Any]] | None:
    messages: list[dict[str, Any]] = []
    for event_name, event_attributes in _get_events(span):
        if event_name != STRANDS_EVENT_CHOICE:
            continue
        normalized_message = _normalize_message(
            raw_message={
                "role": event_attributes.get("role") or default_role,
                "content": event_attributes.get("message"),
            },
            default_role=default_role,
        )
        if normalized_message is not None:
            messages.append(normalized_message)
    return messages or None


def _extract_input_messages(
    span: ReadableSpan, attrs: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    messages = (
        _normalize_messages(attrs.get(gen_ai.GEN_AI_INPUT_MESSAGES), "user")
        or _operation_detail_messages(span, gen_ai.GEN_AI_INPUT_MESSAGES, "user")
        or _legacy_input_messages(span)
        or []
    )
    system_instructions = attrs.get(gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS)
    if not system_instructions:
        for _, event_attrs in _get_events(span):
            if event_attrs.get(gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS):
                system_instructions = event_attrs[gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS]
                break
    if system_instructions and not any(
        message["role"] == "system" for message in messages
    ):
        messages.insert(
            0, {"role": "system", "content": _content_for_message(system_instructions)}
        )
    return messages or None


def _extract_output_messages(
    span: ReadableSpan,
    attrs: Mapping[str, Any],
    default_role: str = "assistant",
) -> list[dict[str, Any]] | None:
    messages = _normalize_messages(
        value=attrs.get(gen_ai.GEN_AI_OUTPUT_MESSAGES),
        default_role=default_role,
    )
    if messages:
        return messages

    messages = _operation_detail_messages(
        span=span,
        attr_name=gen_ai.GEN_AI_OUTPUT_MESSAGES,
        default_role=default_role,
    )
    if messages:
        return messages

    return _legacy_output_messages(span=span, default_role=default_role)


def _message_content_attr_value(content: Any) -> str:
    text = _extract_text_from_content(content)
    if text is not None:
        return text
    json_value = _json_string(content)
    return json_value if json_value is not None else ""


def _tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
    parsed_content = _safe_json_loads(content)
    if not isinstance(parsed_content, list):
        return []

    tool_calls: list[dict[str, Any]] = []
    for item in parsed_content:
        if not isinstance(item, Mapping):
            continue

        tool_payload = item.get("toolUse")
        if isinstance(tool_payload, Mapping):
            tool_calls.append(
                _normalize_tool_call(
                    name=tool_payload.get("name"),
                    tool_call_id=tool_payload.get("toolUseId"),
                    arguments=tool_payload.get("input"),
                )
            )
            continue

        if item.get("type") == "tool_call":
            tool_calls.append(
                _normalize_tool_call(
                    name=item.get("name"),
                    tool_call_id=item.get("id"),
                    arguments=item.get("arguments"),
                )
            )

    return [tool_call for tool_call in tool_calls if tool_call["function"]["name"]]


def _normalize_tool_call(
    name: Any,
    tool_call_id: Any,
    arguments: Any,
) -> dict[str, Any]:
    arguments_string = _json_string(arguments)
    return {
        "id": safe_text(tool_call_id),
        "type": "function",
        "function": {
            "name": safe_text(name),
            "arguments": arguments_string or "",
        },
    }


def _set_indexed_messages(
    attrs: dict[str, Any],
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for message_index, message in enumerate(messages):
        indexed_prefix = f"{prefix}{message_index}"
        attrs[f"{indexed_prefix}.role"] = safe_text(message.get("role"))
        content = message.get("content")
        attrs[f"{indexed_prefix}.content"] = _message_content_attr_value(content)
        tool_calls = _tool_calls_from_content(content)
        if tool_calls:
            attrs[f"{indexed_prefix}.tool_calls"] = json_dumps(
                tool_calls, complete=True
            )


def _normalize_tool_definition(
    tool_definition: Mapping[str, Any],
) -> dict[str, Any] | None:
    tool_name = tool_definition.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None

    function_payload: dict[str, Any] = {"name": tool_name}
    description = tool_definition.get("description")
    if description:
        function_payload["description"] = description
    input_schema = tool_definition.get("inputSchema") or tool_definition.get(
        "parameters"
    )
    if isinstance(input_schema, Mapping) and isinstance(
        input_schema.get("json"), Mapping
    ):
        input_schema = input_schema["json"]
    if input_schema:
        function_payload["parameters"] = input_schema

    return {"type": "function", "function": function_payload}


def _extract_tool_definitions(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    raw_tool_definitions = _safe_json_loads(attrs.get(gen_ai.GEN_AI_TOOL_DEFINITIONS))
    if not isinstance(raw_tool_definitions, list):
        return None

    normalized_tools = []
    for tool_definition in raw_tool_definitions:
        if isinstance(tool_definition, Mapping):
            normalized_tool = _normalize_tool_definition(tool_definition)
        elif isinstance(tool_definition, str):
            normalized_tool = {
                "type": "function",
                "function": {"name": tool_definition},
            }
        else:
            normalized_tool = None
        if normalized_tool is not None:
            normalized_tools.append(normalized_tool)
    return normalized_tools or None


def _extract_usage(
    attrs: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None, int | None]:
    prompt_tokens = attrs.get(gen_ai.GEN_AI_USAGE_INPUT_TOKENS)
    if not isinstance(prompt_tokens, int):
        prompt_tokens = attrs.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS)

    completion_tokens = attrs.get(gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS)
    if not isinstance(completion_tokens, int):
        completion_tokens = attrs.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS)

    total_tokens = attrs.get(SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS)
    if not isinstance(total_tokens, int):
        total_tokens = attrs.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    if not isinstance(total_tokens, int) and (
        type(prompt_tokens) is int
        and prompt_tokens >= 0
        and type(completion_tokens) is int
        and completion_tokens >= 0
    ):
        total_tokens = (prompt_tokens if isinstance(prompt_tokens, int) else 0) + (
            completion_tokens if isinstance(completion_tokens, int) else 0
        )

    cache_read_tokens = attrs.get(
        SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        attrs.get(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS),
    )

    return (
        prompt_tokens if type(prompt_tokens) is int and prompt_tokens >= 0 else None,
        completion_tokens
        if type(completion_tokens) is int and completion_tokens >= 0
        else None,
        total_tokens if type(total_tokens) is int and total_tokens >= 0 else None,
        cache_read_tokens
        if type(cache_read_tokens) is int and cache_read_tokens >= 0
        else None,
    )


def _set_usage_attrs(attrs: dict[str, Any]) -> None:
    prompt_tokens, completion_tokens, total_tokens, cache_read_tokens = _extract_usage(
        attrs=attrs
    )
    for key in (
        gen_ai.GEN_AI_USAGE_INPUT_TOKENS,
        gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
    ):
        value = attrs.get(key)
        if value is not None and (type(value) is not int or value < 0):
            attrs.pop(key, None)
    if prompt_tokens is not None:
        attrs[gen_ai.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attrs[gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if total_tokens is not None:
        attrs[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens
    if cache_read_tokens is not None:
        attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read_tokens
        attrs[SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read_tokens
    cache_write = attrs.get(
        SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        attrs.get("gen_ai.usage.cache_write_input_tokens"),
    )
    if type(cache_write) is int and cache_write >= 0:
        attrs[SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = cache_write
        attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = cache_write


def _set_common_attrs(
    attrs: dict[str, Any],
    *,
    log_type: str,
    entity_name: str,
    entity_path: str,
) -> None:
    attrs[RESPAN_LOG_METHOD] = LogMethodChoices.TRACING_INTEGRATION.value
    attrs[RESPAN_LOG_TYPE] = log_type
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = entity_name
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = entity_path
    attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)


def _set_input_output_attrs(
    attrs: dict[str, Any],
    *,
    input_messages: list[dict[str, Any]] | None,
    output_messages: list[dict[str, Any]] | None,
) -> None:
    if input_messages:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(input_messages)
    if output_messages:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(output_messages)


def _enrich_agent_span(
    span: ReadableSpan,
    attrs: dict[str, Any],
) -> None:
    agent_name = _extract_agent_name(span=span, attrs=attrs)
    _set_common_attrs(
        attrs=attrs,
        log_type=LOG_TYPE_AGENT,
        entity_name=agent_name,
        entity_path=agent_name,
    )
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = agent_name
    _set_input_output_attrs(
        attrs=attrs,
        input_messages=_extract_input_messages(span=span, attrs=attrs),
        output_messages=_extract_output_messages(span=span, attrs=attrs),
    )


def _enrich_task_span(
    span: ReadableSpan,
    attrs: dict[str, Any],
) -> None:
    operation_name = attrs.get(gen_ai.GEN_AI_OPERATION_NAME)
    entity_name = safe_text(
        operation_name if isinstance(operation_name, str) else span.name
    )
    _set_common_attrs(
        attrs=attrs,
        log_type=LOG_TYPE_TASK,
        entity_name=entity_name,
        entity_path=entity_name,
    )
    _set_input_output_attrs(
        attrs=attrs,
        input_messages=_extract_input_messages(span=span, attrs=attrs),
        output_messages=_extract_output_messages(span=span, attrs=attrs),
    )
    for name, event_attrs in _get_events(span):
        if name in {"memory.query", "memory.content"} and "content" in event_attrs:
            attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(
                _safe_json_loads(event_attrs["content"])
            )
        elif name == "memory.results" and "content" in event_attrs:
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(
                _safe_json_loads(event_attrs["content"])
            )
    if (
        isinstance(operation_name, str)
        and operation_name.startswith("memory.")
        and attrs.get("content") is not None
    ):
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(
            _safe_json_loads(attrs.pop("content"))
        )


def _enrich_chat_span(
    span: ReadableSpan,
    attrs: dict[str, Any],
    inherited_tool_definitions: list[dict[str, Any]] | None = None,
) -> None:
    _set_common_attrs(
        attrs=attrs,
        log_type=LOG_TYPE_CHAT,
        entity_name=STRANDS_OPERATION_CHAT,
        entity_path=STRANDS_OPERATION_CHAT,
    )
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
    model = attrs.get(SpanAttributes.LLM_REQUEST_MODEL)
    if model is not None:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(model)
    tool_definitions = _extract_tool_definitions(attrs) or inherited_tool_definitions
    if tool_definitions:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_dumps(
            tool_definitions, complete=True
        )

    input_messages = _extract_input_messages(span=span, attrs=attrs)
    output_messages = _extract_output_messages(span=span, attrs=attrs)
    _set_input_output_attrs(
        attrs=attrs,
        input_messages=input_messages,
        output_messages=output_messages,
    )
    if input_messages:
        _set_indexed_messages(
            attrs=attrs,
            prefix=_GEN_AI_PROMPT_PREFIX,
            messages=input_messages,
        )
    if output_messages:
        _set_indexed_messages(
            attrs=attrs,
            prefix=_GEN_AI_COMPLETION_PREFIX,
            messages=output_messages,
        )
    provider = attrs.get(SpanAttributes.LLM_SYSTEM)
    if provider in (None, STRANDS_SYSTEM_NAME):
        provider = attrs.get(gen_ai.GEN_AI_PROVIDER_NAME)
    if provider == STRANDS_SYSTEM_NAME:
        attrs.pop(SpanAttributes.LLM_SYSTEM, None)
    elif provider:
        attrs[SpanAttributes.LLM_SYSTEM] = safe_text(provider)
    _set_usage_attrs(attrs=attrs)


def _extract_tool_event_payload(
    span: ReadableSpan,
    event_name: str,
    attr_name: str,
) -> Any:
    for current_event_name, event_attributes in _get_events(span):
        if current_event_name == event_name and attr_name in event_attributes:
            return event_attributes.get(attr_name)
    return None


def _enrich_tool_span(
    span: ReadableSpan,
    attrs: dict[str, Any],
) -> None:
    tool_name = _extract_tool_name(span=span, attrs=attrs)
    _set_common_attrs(
        attrs=attrs,
        log_type=LOG_TYPE_TOOL,
        entity_name=tool_name,
        entity_path=tool_name,
    )

    tool_arguments = attrs.get(gen_ai.GEN_AI_TOOL_CALL_ARGUMENTS)
    if tool_arguments is None:
        tool_arguments = _extract_tool_event_payload(
            span=span,
            event_name=STRANDS_EVENT_TOOL_MESSAGE,
            attr_name="content",
        )
    tool_result = attrs.get(gen_ai.GEN_AI_TOOL_CALL_RESULT)
    if tool_result is None:
        tool_result = _extract_tool_event_payload(
            span=span,
            event_name=STRANDS_EVENT_CHOICE,
            attr_name="message",
        )
    tool_input_payload = {
        "name": tool_name,
        "arguments": to_jsonable(_safe_json_loads(tool_arguments)),
    }
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(tool_input_payload)

    if tool_result is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_string(
            _content_for_message(tool_result)
        )


def _strip_raw_attrs(attrs: dict[str, Any], log_type: str) -> dict[str, Any]:
    attrs = {
        key: value
        for key, value in attrs.items()
        if key not in STRANDS_RAW_ATTRS_TO_STRIP
        and key not in _OFF_CONTRACT_ALIAS_ATTRS
        and not any(
            key.startswith(prefix) for prefix in STRANDS_RAW_ATTR_PREFIXES_TO_STRIP
        )
    }
    if log_type != LOG_TYPE_CHAT:
        attrs = {
            key: value
            for key, value in attrs.items()
            if key not in STRANDS_NON_LLM_ATTRS_TO_STRIP
            and key
            not in {
                SpanAttributes.LLM_SYSTEM,
                SpanAttributes.LLM_REQUEST_TYPE,
                SpanAttributes.LLM_REQUEST_MODEL,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            }
        }
    return attrs


def _preserve_bounded_error(
    span: ReadableSpan, attrs: dict[str, Any], allowed: bool
) -> None:
    status = getattr(span, "status", None)
    if getattr(status, "status_code", None) is StatusCode.ERROR and hasattr(
        span, "_status"
    ):
        description = (
            safe_text(getattr(status, "description", None)) if allowed else None
        )
        span._status = Status(StatusCode.ERROR, description)


def _drop_consumed_native_events(span: ReadableSpan, allowed: bool) -> None:
    events = []
    for name, attributes in _get_events(span):
        if name.startswith(("gen_ai.", "memory.")):
            continue
        events.append(
            Event(name=name, attributes=filter_attributes(attributes, allowed))
        )
    if hasattr(span, "_events"):
        span._events = tuple(events)
    else:
        try:
            span.events = tuple(events)
        except (AttributeError, TypeError):
            pass


def enrich_strands_agents_span(
    span: ReadableSpan,
    *,
    inherited_tool_definitions: list[dict[str, Any]] | None = None,
    capture_content: bool = True,
) -> None:
    original_attrs = getattr(span, "_attributes", None)
    if original_attrs is None:
        return

    attrs = dict(original_attrs)
    allowed = capture_content and content_enabled()
    if not is_strands_agents_span(span=span, attrs=attrs):
        return

    log_type = _extract_log_type(span=span, attrs=attrs)
    if log_type is None:
        span._attributes = filter_attributes(attrs, allowed)
        _preserve_bounded_error(span, attrs, allowed)
        _drop_consumed_native_events(span, allowed)
        return

    if log_type == LOG_TYPE_AGENT:
        _enrich_agent_span(span=span, attrs=attrs)
    elif log_type == LOG_TYPE_TASK:
        _enrich_task_span(span=span, attrs=attrs)
    elif log_type == LOG_TYPE_CHAT:
        _enrich_chat_span(
            span=span,
            attrs=attrs,
            inherited_tool_definitions=inherited_tool_definitions,
        )
    elif log_type == LOG_TYPE_TOOL:
        _enrich_tool_span(span=span, attrs=attrs)

    attrs = _strip_raw_attrs(attrs=attrs, log_type=log_type)
    attrs = filter_attributes(attrs, allowed)
    _preserve_bounded_error(span, attrs, allowed)
    span._attributes = attrs
    _drop_consumed_native_events(span, allowed)


class StrandsAgentsSpanProcessor(SpanProcessor):
    """Translate only native Strands spans, with state scoped to their span IDs."""

    def __init__(self, *, include_tool_definitions: bool = True) -> None:
        self._states: dict[tuple[int, int], dict[str, Any]] = {}
        self._lock = RLock()
        self._include_tool_definitions = include_tool_definitions

    @staticmethod
    def _key(span: Any) -> tuple[int, int]:
        context = span.get_span_context()
        return context.trace_id, context.span_id

    def register_native(
        self, span: Any, allowed: bool, attrs: Mapping[str, Any]
    ) -> None:
        parent = getattr(span, "parent", None)
        with self._lock:
            inherited = (
                self._states.get((parent.trace_id, parent.span_id), {})
                if parent
                else {}
            )
            definitions = (
                _extract_tool_definitions(attrs)
                if allowed and self._include_tool_definitions
                else None
            )
            definitions = (
                definitions or inherited.get("definitions")
                if allowed and self._include_tool_definitions
                else None
            )
            self._states[self._key(span)] = {
                "allowed": allowed,
                "definitions": definitions,
                "provider": inherited.get("provider"),
                "streaming": inherited.get("streaming"),
            }

    def set_model(self, span: Any, model: Any) -> None:
        module = type(model).__module__
        provider = (
            module.removeprefix("strands.models.").split(".")[0]
            if module.startswith("strands.models.")
            else None
        )
        provider = {"openai_responses": "openai", "azure_openai": "openai"}.get(
            provider, provider
        )
        config = getattr(model, "config", {})
        streaming = None
        if module == "strands.models.openai":
            streaming = bool(
                config.get("stream", config.get("params", {}).get("stream", True))
            )
        if module == "strands.models.openai_responses":
            streaming = True
        with self._lock:
            state = self._states.get(self._key(span))
            if state is not None:
                state.update(provider=provider, streaming=streaming)

    def record_provider_usage(
        self, span: Any, usage: Any, input_name: str, output_name: str
    ) -> None:
        with self._lock:
            state = self._states.get(self._key(span))
            if state is not None:
                state["actual_usage"] = {
                    gen_ai.GEN_AI_USAGE_INPUT_TOKENS: getattr(usage, input_name, None),
                    gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS: getattr(
                        usage, output_name, None
                    ),
                    SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS: getattr(
                        usage, "total_tokens", None
                    ),
                }

    def owns(self, span: Any) -> bool:
        with self._lock:
            return self._key(span) in self._states

    def content_allowed(self, span: Any) -> bool:
        with self._lock:
            return bool(self._states.get(self._key(span), {}).get("allowed", False))

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        with self._lock:
            state = self._states.pop(self._key(span), {})
        try:
            if _extract_log_type(span, span.attributes or {}) == LOG_TYPE_CHAT:
                attrs = dict(span.attributes or {})
                if "actual_usage" in state:
                    for key in (
                        gen_ai.GEN_AI_USAGE_INPUT_TOKENS,
                        gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS,
                        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
                        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                    ):
                        attrs.pop(key, None)
                    for key, value in state["actual_usage"].items():
                        if type(value) is int and value >= 0:
                            attrs[key] = value
                if state.get("provider"):
                    attrs[SpanAttributes.LLM_SYSTEM] = state["provider"]
                if state.get("streaming") is not None:
                    attrs[SpanAttributes.GEN_AI_IS_STREAMING] = state["streaming"]
                span._attributes = attrs
            enrich_strands_agents_span(
                span=span,
                inherited_tool_definitions=state.get("definitions"),
                capture_content=state.get("allowed", False),
            )
        except Exception:
            logger.debug("Failed to enrich Strands Agents span", exc_info=True)

    def shutdown(self) -> None:
        with self._lock:
            self._states.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
