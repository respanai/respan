"""Normalize Claude Agent SDK OTEL spans for the Respan OTLP pipeline."""

from __future__ import annotations

import ast
import json
import logging
import threading
from typing import Any, Mapping

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode

from respan_instrumentation_claude_agent_sdk._constants import (
    CLAUDE_AGENT_SDK_AGENT_NAME_ATTR,
    CLAUDE_AGENT_SDK_CONVERSATION_ID_ATTR,
    CLAUDE_AGENT_SDK_INPUT_MESSAGES_ATTR,
    CLAUDE_AGENT_SDK_OPERATION_NAME_ATTR,
    CLAUDE_AGENT_SDK_OUTPUT_MESSAGES_ATTR,
    CLAUDE_AGENT_SDK_STRIP_ATTRS,
    CLAUDE_AGENT_SDK_SYSTEM_INSTRUCTIONS_ATTR,
    CLAUDE_AGENT_SDK_TOOL_CALL_ARGUMENTS_ATTR,
    CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR,
    CLAUDE_AGENT_SDK_TOOL_CALL_RESULT_ATTR,
    CLAUDE_AGENT_SDK_TOOL_DEFINITIONS_ATTR,
    CLAUDE_AGENT_SDK_TOOL_NAME_ATTR,
    CLAUDE_AGENT_SDK_USAGE_INPUT_TOKENS_ATTR,
    CLAUDE_AGENT_SDK_USAGE_OUTPUT_TOKENS_ATTR,
    INPUT_VALUE_ATTR,
    OUTPUT_VALUE_ATTR,
    RESPAN_OVERRIDE_INPUT_ATTR,
    RESPAN_OVERRIDE_TOOL_CALLS_ATTR,
    RESPAN_OVERRIDE_TOOLS_ATTR,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
)
from respan_sdk.utils.serialization import serialize_value

logger = logging.getLogger(__name__)

_CLAUDE_AGENT_OPERATION_NAME = "invoke_agent"
_CLAUDE_TOOL_OPERATION_NAME = "execute_tool"
_GEN_AI_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_GEN_AI_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_COMPLETION_TOOL_CALLS_ATTR = f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"
_LEGACY_RESPAN_SPAN_TOOL_CALLS_ATTR = "respan.span.tool_calls"
_LEGACY_RESPAN_SPAN_TOOLS_ATTR = "respan.span.tools"


def _set_if_unset_span_status(span: ReadableSpan, attrs: Mapping[str, Any]) -> None:
    current_status = getattr(span, "status", None)
    current_status_code = getattr(current_status, "status_code", None)
    if current_status_code not in {None, StatusCode.UNSET}:
        return

    error_message = (
        attrs.get("error.message")
        or attrs.get("exception.message")
        or attrs.get("error.type")
        or attrs.get("exception.type")
    )
    status = (
        Status(StatusCode.ERROR, str(error_message))
        if error_message
        else Status(StatusCode.OK)
    )
    try:
        span._status = status  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Failed to set Claude Agent SDK span status", exc_info=True)


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        stripped = value.strip()
        if stripped[:1] not in {"{", "[", "("}:
            return None
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed_value = _safe_json_loads(value)
        if parsed_value is not None and value.strip()[:1] in {"{", "[", "("}:
            return json.dumps(serialize_value(parsed_value), default=str)
        return value
    return json.dumps(serialize_value(value), default=str)


def _set_if_missing(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if attrs.get(key) in (None, "", (), []):
        attrs[key] = value


def _set_if_present(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    attrs[key] = value


def _pop_attrs(attrs: dict[str, Any], *keys: str) -> None:
    for key in keys:
        attrs.pop(key, None)


def _pop_attr_prefixes(attrs: dict[str, Any], *prefixes: str) -> None:
    keys_to_remove = [
        key
        for key in attrs
        if any(key.startswith(prefix) for prefix in prefixes)
    ]
    for key in keys_to_remove:
        attrs.pop(key, None)


def _extract_first(attrs: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, "", (), []):
            return value
    return None


def _extract_agent_name(span: ReadableSpan, attrs: Mapping[str, Any]) -> str:
    raw_agent_name = attrs.get(CLAUDE_AGENT_SDK_AGENT_NAME_ATTR)
    if isinstance(raw_agent_name, str) and raw_agent_name:
        return raw_agent_name

    span_name = span.name.strip()
    if span_name.startswith(f"{_CLAUDE_AGENT_OPERATION_NAME} "):
        parsed_agent_name = span_name[len(_CLAUDE_AGENT_OPERATION_NAME) + 1 :].strip()
        if parsed_agent_name:
            return parsed_agent_name
    return _CLAUDE_AGENT_OPERATION_NAME


def _extract_model(attrs: Mapping[str, Any]) -> str | None:
    raw_model = _extract_first(
        attrs,
        (
            SpanAttributes.LLM_RESPONSE_MODEL,
            SpanAttributes.LLM_REQUEST_MODEL,
        ),
    )
    if isinstance(raw_model, str) and raw_model:
        return raw_model
    return None


def _extract_usage(
    attrs: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None, int | None]:
    prompt_tokens = attrs.get(CLAUDE_AGENT_SDK_USAGE_INPUT_TOKENS_ATTR)
    completion_tokens = attrs.get(CLAUDE_AGENT_SDK_USAGE_OUTPUT_TOKENS_ATTR)
    cache_hit_tokens = attrs.get(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS)
    cache_creation_tokens = attrs.get(
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS
    )

    normalized_prompt_tokens = (
        prompt_tokens if isinstance(prompt_tokens, int) else None
    )
    normalized_cache_hit_tokens = (
        cache_hit_tokens if isinstance(cache_hit_tokens, int) else None
    )
    normalized_cache_creation_tokens = (
        cache_creation_tokens if isinstance(cache_creation_tokens, int) else None
    )

    if normalized_prompt_tokens is not None and (
        normalized_cache_hit_tokens is not None
        or normalized_cache_creation_tokens is not None
    ):
        uncached_prompt_tokens = normalized_prompt_tokens - (
            (normalized_cache_hit_tokens or 0)
            + (normalized_cache_creation_tokens or 0)
        )
        if uncached_prompt_tokens >= 0:
            normalized_prompt_tokens = uncached_prompt_tokens

    return (
        normalized_prompt_tokens,
        completion_tokens if isinstance(completion_tokens, int) else None,
        normalized_cache_hit_tokens,
        normalized_cache_creation_tokens,
    )


def _extract_messages(attrs: Mapping[str, Any], attr_name: str) -> list[Any] | None:
    messages = _safe_json_loads(attrs.get(attr_name))
    if isinstance(messages, list):
        return messages
    return None


def _extract_normalized_input_messages(attrs: Mapping[str, Any]) -> list[Any] | None:
    input_messages = _extract_messages(attrs, CLAUDE_AGENT_SDK_INPUT_MESSAGES_ATTR)
    if input_messages is None:
        return None

    system_instructions = attrs.get(CLAUDE_AGENT_SDK_SYSTEM_INSTRUCTIONS_ATTR)
    normalized_input_messages = []
    if isinstance(system_instructions, str) and system_instructions:
        normalized_input_messages.append(
            {"role": "system", "content": system_instructions}
        )
    normalized_input_messages.extend(input_messages)
    return normalized_input_messages


def _set_message_attrs(
    attrs: dict[str, Any],
    prefix: str,
    messages: list[Any] | None,
) -> None:
    if messages is None:
        return
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role") or message.get("type")
        content = message.get("content")
        message_prefix = f"{prefix}.{index}"
        if role is not None:
            _set_if_missing(attrs, f"{message_prefix}.role", str(role))
        if content is not None:
            _set_if_missing(
                attrs,
                f"{message_prefix}.content",
                content
                if isinstance(content, str)
                else json.dumps(serialize_value(content), default=str),
            )


def _extract_input_output(attrs: Mapping[str, Any]) -> tuple[str | None, str | None]:
    input_messages = _extract_normalized_input_messages(attrs)
    if input_messages is not None:
        input_value = json.dumps(
            serialize_value(input_messages),
            default=str,
        )
    else:
        input_value = _json_string(
            _extract_first(
                attrs,
                (
                    INPUT_VALUE_ATTR,
                    SpanAttributes.TRACELOOP_ENTITY_INPUT,
                ),
            )
        )

    output_messages = _extract_messages(attrs, CLAUDE_AGENT_SDK_OUTPUT_MESSAGES_ATTR)
    if output_messages is not None:
        output_value = json.dumps(serialize_value(output_messages), default=str)
    else:
        output_value = _json_string(
            _extract_first(
                attrs,
                (
                    OUTPUT_VALUE_ATTR,
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                ),
            )
        )

    return input_value, output_value


def _normalize_tool_definition(tool_definition: Any) -> dict[str, Any] | None:
    if isinstance(tool_definition, str) and tool_definition:
        return {
            "type": "function",
            "function": {"name": tool_definition},
        }

    if not isinstance(tool_definition, Mapping):
        return None

    function_payload = tool_definition.get("function")
    if isinstance(function_payload, Mapping):
        normalized = {
            "type": tool_definition.get("type", "function"),
            "function": {"name": function_payload.get("name")},
        }
        for key in ("description", "parameters", "strict"):
            value = function_payload.get(key)
            if value is not None:
                normalized["function"][key] = value
        if normalized["function"].get("name"):
            return normalized
        return None

    tool_name = tool_definition.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None

    normalized_function: dict[str, Any] = {"name": tool_name}
    description = tool_definition.get("description")
    if description is not None:
        normalized_function["description"] = description
    parameters = tool_definition.get("input_schema") or tool_definition.get("parameters")
    if parameters is not None:
        normalized_function["parameters"] = parameters

    return {
        "type": tool_definition.get("type", "function"),
        "function": normalized_function,
    }


def _extract_tools(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    raw_tool_definitions = _safe_json_loads(
        attrs.get(CLAUDE_AGENT_SDK_TOOL_DEFINITIONS_ATTR)
    )
    if isinstance(raw_tool_definitions, Mapping):
        raw_tool_definitions = [raw_tool_definitions]
    if not isinstance(raw_tool_definitions, list):
        return None

    normalized_tools = []
    for tool_definition in raw_tool_definitions:
        normalized_tool = _normalize_tool_definition(tool_definition)
        if normalized_tool is not None:
            normalized_tools.append(normalized_tool)
    return normalized_tools or None


def _extract_existing_tools(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    parsed_tools = _safe_json_loads(attrs.get(SpanAttributes.LLM_REQUEST_FUNCTIONS))
    if isinstance(parsed_tools, list):
        return [dict(tool) for tool in parsed_tools if isinstance(tool, Mapping)]

    raw_tools = attrs.get(RESPAN_OVERRIDE_TOOLS_ATTR)
    if isinstance(raw_tools, list):
        return [dict(tool) for tool in raw_tools if isinstance(tool, Mapping)]

    parsed_tools = _safe_json_loads(attrs.get(_LEGACY_RESPAN_SPAN_TOOLS_ATTR))
    if isinstance(parsed_tools, list):
        return [dict(tool) for tool in parsed_tools if isinstance(tool, Mapping)]
    return None


def _extract_tool_calls(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    output_messages = _extract_messages(attrs, CLAUDE_AGENT_SDK_OUTPUT_MESSAGES_ATTR)
    if not output_messages:
        return None

    tool_calls = []
    for message in output_messages:
        if not isinstance(message, Mapping):
            continue
        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            continue

        for block in content_blocks:
            if not isinstance(block, Mapping):
                continue

            block_type = block.get("type")
            if block_type not in {None, "tool_use"}:
                continue

            tool_name = block.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            tool_arguments = _json_string(block.get("input", {}))
            tool_call_id = (
                block.get("id")
                or block.get("tool_use_id")
                or attrs.get(CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR)
                or ""
            )

            tool_calls.append(
                {
                    "id": str(tool_call_id),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_arguments or "",
                    },
                }
            )

    return tool_calls or None


def _extract_existing_tool_calls(
    attrs: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    parsed_tool_calls = _safe_json_loads(attrs.get(_COMPLETION_TOOL_CALLS_ATTR))
    if isinstance(parsed_tool_calls, list):
        return [
            tool_call
            for tool_call in parsed_tool_calls
            if isinstance(tool_call, Mapping)
        ]

    raw_tool_calls = attrs.get(RESPAN_OVERRIDE_TOOL_CALLS_ATTR)
    if isinstance(raw_tool_calls, list):
        return [tool_call for tool_call in raw_tool_calls if isinstance(tool_call, Mapping)]

    parsed_tool_calls = _safe_json_loads(attrs.get(_LEGACY_RESPAN_SPAN_TOOL_CALLS_ATTR))
    if isinstance(parsed_tool_calls, list):
        return [
            tool_call
            for tool_call in parsed_tool_calls
            if isinstance(tool_call, Mapping)
        ]
    return None


def _build_tool_call_from_tool_span_attrs(
    attrs: Mapping[str, Any],
) -> dict[str, Any] | None:
    tool_name = _extract_first(
        attrs,
        (
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            CLAUDE_AGENT_SDK_TOOL_NAME_ATTR,
            "tool",
        ),
    )
    if not isinstance(tool_name, str) or not tool_name:
        return None

    tool_arguments = _json_string(
        _extract_first(
            attrs,
            (
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                CLAUDE_AGENT_SDK_TOOL_CALL_ARGUMENTS_ATTR,
                RESPAN_OVERRIDE_INPUT_ATTR,
            ),
        )
    )
    tool_call_id = attrs.get(CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR) or ""

    return {
        "id": str(tool_call_id),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": tool_arguments or "",
        },
    }


def _normalized_tool_arguments(arguments: Any) -> str:
    """Compare structured arguments without depending on JSON formatting."""
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # Match the Python-literal compatibility supported by _json_string.
            parsed_arguments = _safe_json_loads(arguments)
            if parsed_arguments is None:
                return f"raw:{arguments}"
    else:
        parsed_arguments = arguments
    return "json:" + json.dumps(
        serialize_value(parsed_arguments), sort_keys=True, separators=(",", ":"),
        default=str,
    )


def _tool_call_signature(tool_call: Mapping[str, Any]) -> tuple[str, str, str]:
    function_payload = tool_call.get("function")
    if not isinstance(function_payload, Mapping):
        function_payload = {}

    tool_call_id = tool_call.get("id")
    tool_name = function_payload.get("name")
    tool_arguments = function_payload.get("arguments")
    return (
        str(tool_call_id or ""),
        str(tool_name or ""),
        _normalized_tool_arguments(tool_arguments),
    )


def _merge_tool_calls(
    *tool_call_lists: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    merged_tool_calls = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    anonymous_signatures: set[tuple[str, str, str]] = set()

    for tool_call_list in tool_call_lists:
        if not isinstance(tool_call_list, list):
            continue
        for tool_call in tool_call_list:
            if not isinstance(tool_call, Mapping):
                continue
            signature = _tool_call_signature(tool_call)
            tool_call_id = signature[0]
            if tool_call_id:
                existing_call = calls_by_id.get(tool_call_id)
                if existing_call is not None:
                    _merge_tool_call_observation(existing_call, tool_call)
                    continue
            else:
                if signature in anonymous_signatures:
                    continue
                anonymous_signatures.add(signature)
            merged_call = dict(tool_call)
            if tool_call_id:
                calls_by_id[tool_call_id] = merged_call
            merged_tool_calls.append(merged_call)

    return merged_tool_calls or None


def _merge_tool_call_observation(
    existing_call: dict[str, Any], incoming_call: Mapping[str, Any],
) -> None:
    """Fill absent fields without replacing the first populated observation."""
    incoming_function = incoming_call.get("function")
    if not isinstance(incoming_function, Mapping):
        return
    existing_function = existing_call.get("function")
    merged_function = (
        dict(existing_function) if isinstance(existing_function, Mapping) else {}
    )
    conflicting = False
    for field in ("name", "arguments"):
        incoming_value = incoming_function.get(field)
        existing_value = merged_function.get(field)
        if incoming_value is None or incoming_value == "":
            continue
        if existing_value is None or existing_value == "":
            merged_function[field] = incoming_value
        elif field == "arguments":
            conflicting |= (
                _normalized_tool_arguments(existing_value)
                != _normalized_tool_arguments(incoming_value)
            )
        else:
            conflicting |= existing_value != incoming_value
    if conflicting:
        logger.warning(
            "Conflicting Claude Agent SDK tool call for invocation ID %s "
            "(name or arguments differ); keeping the first call",
            existing_call.get("id"),
        )
    else:
        existing_call["function"] = merged_function
        if not existing_call.get("type") and incoming_call.get("type"):
            existing_call["type"] = incoming_call["type"]


def _extract_function_name(payload: Mapping[str, Any]) -> str | None:
    function_payload = payload.get("function")
    if not isinstance(function_payload, Mapping):
        return None
    tool_name = function_payload.get("name")
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    return None


def _rename_tool_definition(
    tool_definition: Mapping[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    normalized_tool_definition = dict(tool_definition)
    function_payload = tool_definition.get("function")
    normalized_function_payload = (
        dict(function_payload) if isinstance(function_payload, Mapping) else {}
    )
    normalized_function_payload["name"] = tool_name
    normalized_tool_definition["function"] = normalized_function_payload
    return normalized_tool_definition


def _reconcile_tools_with_tool_calls(
    tools: list[dict[str, Any]] | None,
    tool_calls: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    normalized_tools = [dict(tool_definition) for tool_definition in tools]
    if not tool_calls:
        return normalized_tools

    tool_call_names = {
        tool_call_name
        for tool_call in tool_calls
        if isinstance(tool_call, Mapping)
        and (tool_call_name := _extract_function_name(tool_call)) is not None
    }
    if not tool_call_names:
        return normalized_tools

    inferred_prefixes = set()
    for tool_definition in normalized_tools:
        tool_name = _extract_function_name(tool_definition)
        if not tool_name:
            continue
        for tool_call_name in tool_call_names:
            if tool_call_name != tool_name and tool_call_name.endswith(tool_name):
                prefix = tool_call_name[: -len(tool_name)]
                if prefix:
                    inferred_prefixes.add(prefix)

    shared_prefix = next(iter(inferred_prefixes)) if len(inferred_prefixes) == 1 else None

    reconciled_tools = []
    for tool_definition in normalized_tools:
        tool_name = _extract_function_name(tool_definition)
        if not tool_name:
            reconciled_tools.append(tool_definition)
            continue

        if tool_name in tool_call_names:
            reconciled_tools.append(tool_definition)
            continue

        suffix_matches = [
            tool_call_name
            for tool_call_name in tool_call_names
            if tool_call_name.endswith(tool_name)
        ]
        if len(set(suffix_matches)) == 1:
            reconciled_tools.append(
                _rename_tool_definition(tool_definition, suffix_matches[0])
            )
            continue

        if shared_prefix and not tool_name.startswith(shared_prefix):
            reconciled_tools.append(
                _rename_tool_definition(tool_definition, f"{shared_prefix}{tool_name}")
            )
            continue

        reconciled_tools.append(tool_definition)

    return reconciled_tools


def _extract_tool_span_name(span: ReadableSpan, attrs: Mapping[str, Any]) -> str:
    span_name = span.name.strip()
    if span_name.startswith(f"{_CLAUDE_TOOL_OPERATION_NAME} "):
        parsed_tool_name = span_name[len(_CLAUDE_TOOL_OPERATION_NAME) + 1 :].strip()
        if parsed_tool_name:
            return parsed_tool_name

    raw_tool_name = _extract_first(
        attrs,
        (
            CLAUDE_AGENT_SDK_TOOL_NAME_ATTR,
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            "tool",
        ),
    )
    if isinstance(raw_tool_name, str) and raw_tool_name:
        return raw_tool_name
    return "tool"


def _get_span_key(span: ReadableSpan) -> tuple[int, int] | None:
    span_context = span.get_span_context()
    if span_context is None:
        return None
    return span_context.trace_id, span_context.span_id


def _get_parent_span_key(span: ReadableSpan) -> tuple[int, int] | None:
    span_context = span.get_span_context()
    if span_context is None:
        return None

    parent_context = getattr(span, "parent", None)
    parent_span_id = getattr(parent_context, "span_id", None)
    if parent_span_id is None:
        return None

    return span_context.trace_id, parent_span_id


def is_claude_agent_sdk_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    operation_name = attrs.get(CLAUDE_AGENT_SDK_OPERATION_NAME_ATTR)
    return (
        operation_name in {_CLAUDE_AGENT_OPERATION_NAME, _CLAUDE_TOOL_OPERATION_NAME}
        or bool(attrs.get(CLAUDE_AGENT_SDK_AGENT_NAME_ATTR))
        or bool(attrs.get(CLAUDE_AGENT_SDK_TOOL_NAME_ATTR))
        or span.name.startswith(_CLAUDE_AGENT_OPERATION_NAME)
        or span.name.startswith(_CLAUDE_TOOL_OPERATION_NAME)
    )


def enrich_claude_agent_sdk_span(span: ReadableSpan) -> None:
    original_attrs = getattr(span, "_attributes", None)
    if original_attrs is None:
        return

    attrs = dict(original_attrs)
    if not is_claude_agent_sdk_span(span, attrs):
        return

    _set_if_missing(
        attrs,
        RESPAN_LOG_METHOD,
        LogMethodChoices.TRACING_INTEGRATION.value,
    )

    operation_name = attrs.get(CLAUDE_AGENT_SDK_OPERATION_NAME_ATTR)
    session_id = attrs.get(CLAUDE_AGENT_SDK_CONVERSATION_ID_ATTR)
    if isinstance(session_id, str) and session_id:
        _set_if_missing(attrs, RESPAN_SESSION_ID, session_id)

    if (
        operation_name == _CLAUDE_TOOL_OPERATION_NAME
        or attrs.get(CLAUDE_AGENT_SDK_TOOL_NAME_ATTR)
        or attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_TOOL
        or span.name.startswith(f"{_CLAUDE_TOOL_OPERATION_NAME} ")
    ):
        tool_name = _extract_tool_span_name(span, attrs)
        # A previous normalizer may already have removed the native attributes.
        tool_input = attrs.get(CLAUDE_AGENT_SDK_TOOL_CALL_ARGUMENTS_ATTR)
        if tool_input is None:
            tool_input = attrs.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
        tool_input = _json_string(tool_input)
        tool_output = attrs.get(CLAUDE_AGENT_SDK_TOOL_CALL_RESULT_ATTR)
        if tool_output is None:
            tool_output = attrs.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
        tool_output = _json_string(tool_output)

        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TOOL
        attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = tool_name
        attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = tool_name
        _set_if_present(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, tool_input)
        _set_if_present(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, tool_output)
        _pop_attrs(
            attrs,
            SpanAttributes.LLM_REQUEST_TYPE,
            SpanAttributes.LLM_REQUEST_MODEL,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            _COMPLETION_TOOL_CALLS_ATTR,
            SpanAttributes.LLM_SYSTEM,
            "cost",
        )
        _pop_attr_prefixes(
            attrs,
            _GEN_AI_PROMPT_PREFIX,
            _GEN_AI_COMPLETION_PREFIX,
        )
        if tool_output is None:
            _pop_attrs(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
        if tool_input is None:
            _pop_attrs(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT)
    else:
        agent_name = _extract_agent_name(span, attrs)
        input_value, output_value = _extract_input_output(attrs)
        input_messages = _extract_normalized_input_messages(attrs)
        output_messages = _extract_messages(attrs, CLAUDE_AGENT_SDK_OUTPUT_MESSAGES_ATTR)
        tool_calls = _merge_tool_calls(
            _extract_existing_tool_calls(attrs), _extract_tool_calls(attrs),
        )
        tools = _reconcile_tools_with_tool_calls(
            tools=_extract_tools(attrs),
            tool_calls=tool_calls,
        )
        model = _extract_model(attrs)
        prompt_tokens, completion_tokens, cache_hit_tokens, cache_creation_tokens = (
            _extract_usage(attrs)
        )

        # Upstream marks invoke_agent spans as chat because they also carry GenAI
        # message attributes. The operation boundary is authoritative here.
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_AGENT
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, input_value)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_value)
        _set_message_attrs(attrs, SpanAttributes.LLM_PROMPTS, input_messages)
        _set_message_attrs(attrs, SpanAttributes.LLM_COMPLETIONS, output_messages)
        if model is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_REQUEST_MODEL, model)
        if prompt_tokens is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        if completion_tokens is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        if prompt_tokens is not None or completion_tokens is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                (prompt_tokens or 0) + (completion_tokens or 0),
            )
        if cache_hit_tokens is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                cache_hit_tokens,
            )
        if cache_creation_tokens is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
                cache_creation_tokens,
            )
        if tools is not None:
            attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json.dumps(
                serialize_value(tools),
                default=str,
            )
        if tool_calls is not None:
            attrs[_COMPLETION_TOOL_CALLS_ATTR] = json.dumps(
                serialize_value(tool_calls),
                default=str,
            )
            _set_if_missing(attrs, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")

    span._attributes = {
        key: value
        for key, value in attrs.items()
        if key not in CLAUDE_AGENT_SDK_STRIP_ATTRS
        and (
            not key.startswith("gen_ai.tool.")
            or (
                key == CLAUDE_AGENT_SDK_TOOL_CALL_ID_ATTR
                and attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_TOOL
            )
        )
    }
    _set_if_unset_span_status(span, span._attributes)


class ClaudeAgentSDKSpanProcessor(SpanProcessor):
    """Normalize native Claude Agent SDK spans into Respan OTLP attributes."""

    def __init__(self) -> None:
        self._pending_tool_calls_by_parent: dict[
            tuple[int, int],
            list[tuple[int, dict[str, Any]]],
        ] = {}
        self._pending_tool_calls_lock = threading.Lock()

    def _store_pending_tool_call(
        self,
        span: ReadableSpan,
        source_attrs: Mapping[str, Any] | None = None,
    ) -> None:
        parent_span_key = _get_parent_span_key(span)
        if parent_span_key is None:
            return

        attrs = source_attrs
        if attrs is None:
            attrs = getattr(span, "_attributes", None)
        if not isinstance(attrs, Mapping):
            return

        tool_call = _build_tool_call_from_tool_span_attrs(attrs)
        if tool_call is None:
            return

        tool_start_time = getattr(span, "start_time", None)
        sort_key = tool_start_time if isinstance(tool_start_time, int) else 0

        with self._pending_tool_calls_lock:
            pending_tool_calls = self._pending_tool_calls_by_parent.setdefault(
                parent_span_key,
                [],
            )
            pending_tool_calls.append((sort_key, tool_call))

    def _consume_pending_tool_calls(
        self,
        span: ReadableSpan,
    ) -> list[dict[str, Any]] | None:
        span_key = _get_span_key(span)
        if span_key is None:
            return None

        with self._pending_tool_calls_lock:
            pending_tool_calls = self._pending_tool_calls_by_parent.pop(span_key, None)

        if not pending_tool_calls:
            return None

        pending_tool_calls.sort(key=lambda entry: entry[0])
        return [tool_call for _, tool_call in pending_tool_calls]

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        try:
            enrich_claude_agent_sdk_span(span)

            attrs = getattr(span, "_attributes", None)
            if not isinstance(attrs, Mapping):
                self._consume_pending_tool_calls(span)
                return

            if attrs.get(RESPAN_LOG_TYPE) == LOG_TYPE_TOOL:
                # The exported tool keeps its invocation ID so the parent call
                # and the tool's result remain correlated after normalization.
                self._store_pending_tool_call(span, attrs)
                # Only agent spans merge queued tool calls into their final attrs.
                # Drop any child calls queued against non-agent parents on span end.
                self._consume_pending_tool_calls(span)
                return

            if attrs.get(RESPAN_LOG_TYPE) != LOG_TYPE_AGENT:
                self._consume_pending_tool_calls(span)
                return

            pending_tool_calls = self._consume_pending_tool_calls(span)
            existing_tool_calls = _extract_existing_tool_calls(attrs)
            merged_tool_calls = _merge_tool_calls(existing_tool_calls, pending_tool_calls)
            if merged_tool_calls is None:
                return

            updated_attrs = dict(attrs)
            updated_attrs[_COMPLETION_TOOL_CALLS_ATTR] = json.dumps(
                serialize_value(merged_tool_calls),
                default=str,
            )
            _set_if_missing(
                updated_attrs,
                f"{SpanAttributes.LLM_COMPLETIONS}.0.role",
                "assistant",
            )
            reconciled_tools = _reconcile_tools_with_tool_calls(
                tools=_extract_existing_tools(updated_attrs),
                tool_calls=merged_tool_calls,
            )
            if reconciled_tools is not None:
                updated_attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json.dumps(
                    serialize_value(reconciled_tools),
                    default=str,
                )
            # The shared exporter synthesizes the final assistant_message child span.
            # Keep the processor focused on parent-span normalization to avoid
            # emitting the same synthetic child twice.
            span._attributes = {
                key: value
                for key, value in updated_attrs.items()
                if key not in CLAUDE_AGENT_SDK_STRIP_ATTRS
            }
            _set_if_unset_span_status(span, span._attributes)
        except Exception:
            logger.exception("Failed to enrich Claude Agent SDK span")

    def shutdown(self) -> None:
        with self._pending_tool_calls_lock:
            self._pending_tool_calls_by_parent.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
