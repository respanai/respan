"""Translate AWS Bedrock Runtime payloads into Respan span fields."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from respan_instrumentation_aws_bedrock._constants import (
    ASSISTANT_ROLE,
    BODY_KEY,
    CONTENT_KEY,
    CONVERSE_OPERATION,
    CONVERSE_STREAM_OPERATION,
    DESCRIPTION_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    INPUT_KEY,
    INPUT_SCHEMA_KEY,
    INVOKE_MODEL_OPERATION,
    INVOKE_MODEL_STREAM_OPERATION,
    MESSAGE_KEY,
    MESSAGES_KEY,
    MODEL_ID_KEY,
    NAME_KEY,
    OUTPUT_KEY,
    ROLE_KEY,
    SYSTEM_KEY,
    SYSTEM_ROLE,
    TEXT_KEY,
    TOOL_CONFIG_KEY,
    TOOL_ROLE,
    TOOLS_KEY,
    TYPE_KEY,
    USAGE_KEY,
    USER_ROLE,
)
from respan_instrumentation_aws_bedrock._privacy import json_text


def serialize_value(*, value):
    from respan_instrumentation_aws_bedrock._privacy import value as builtin_value

    return builtin_value(value)


@dataclass(frozen=True)
class BedrockRequest:
    operation_name: str
    model_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    raw_payload: Any = None


@dataclass(frozen=True)
class BedrockResponse:
    content: str = ""
    role: str = ASSISTANT_ROLE
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw_payload: Any = None


def safe_json(value):
    return json_text(value)


def to_json_attr(value: Any) -> str:
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return default


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _load_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    if isinstance(value, Mapping | list):
        return value
    return serialize_value(value=value)


def _normalize_text_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get(TEXT_KEY)
        if isinstance(text, str):
            return text
        if "json" in content:
            return content["json"]
        if "toolUse" in content:
            return _normalize_tool_call(content["toolUse"])
        if "toolResult" in content:
            return _normalize_tool_result(content["toolResult"])
        if content.get(TYPE_KEY) == "text":
            return content.get(TEXT_KEY, "")
        if content.get(TYPE_KEY) == "tool_use":
            return _normalize_anthropic_tool_use(content)
        if content.get(TYPE_KEY) == "tool_result":
            return _normalize_anthropic_tool_result(content)
        return serialize_value(value=content)
    if isinstance(content, list | tuple):
        normalized = [
            _normalize_text_content(item)
            for item in content
            if _normalize_text_content(item) not in (None, "", [], {})
        ]
        if not normalized:
            return ""
        if all(isinstance(item, str) for item in normalized):
            return "\n".join(normalized)
        return normalized
    return serialize_value(value=content)


def _normalize_message(
    message: Any, *, default_role: str = USER_ROLE
) -> dict[str, Any]:
    role = _field(message, ROLE_KEY, default_role) or default_role
    content = _field(message, CONTENT_KEY)
    if content is None:
        content = _field(message, "contentBlocks")
    if content is None and isinstance(message, str):
        content = message

    normalized_content = _normalize_text_content(content)
    normalized: dict[str, Any] = {
        ROLE_KEY: _normalize_bedrock_role(role),
        CONTENT_KEY: normalized_content,
    }
    tool_calls = _extract_tool_calls_from_content(content)
    if tool_calls and normalized[ROLE_KEY] == ASSISTANT_ROLE:
        normalized["tool_calls"] = tool_calls
    return normalized


def _normalize_bedrock_role(role: Any) -> str:
    if role in {"assistant", "model"}:
        return ASSISTANT_ROLE
    if role == "system":
        return SYSTEM_ROLE
    if role == "tool":
        return TOOL_ROLE
    return USER_ROLE if not isinstance(role, str) else role


def _normalize_system_messages(system: Any) -> list[dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: system}]
    if isinstance(system, list | tuple):
        content = _normalize_text_content(system)
        return [{ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: content}] if content else []
    return [{ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: _normalize_text_content(system)}]


def _normalize_prompt_from_body(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, Mapping):
        return []

    messages: list[dict[str, Any]] = []
    messages.extend(_normalize_system_messages(body.get(SYSTEM_KEY)))

    raw_messages = body.get(MESSAGES_KEY)
    if isinstance(raw_messages, list):
        messages.extend(_normalize_message(message) for message in raw_messages)
        return messages

    for key in ("prompt", "inputText", "input_text", INPUT_KEY):
        value = body.get(key)
        if value is not None:
            messages.append(
                {ROLE_KEY: USER_ROLE, CONTENT_KEY: _normalize_text_content(value)}
            )
            return messages
    return messages


def _normalize_converse_messages(api_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    messages.extend(_normalize_system_messages(api_params.get(SYSTEM_KEY)))
    raw_messages = api_params.get(MESSAGES_KEY, [])
    if isinstance(raw_messages, list):
        messages.extend(_normalize_message(message) for message in raw_messages)
    return messages


def _normalize_anthropic_tool_use(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": block.get("id", ""),
        TYPE_KEY: FUNCTION_TOOL_TYPE,
        FUNCTION_KEY: {
            NAME_KEY: block.get(NAME_KEY, ""),
            "arguments": to_json_attr(block.get(INPUT_KEY, {})),
        },
    }


def _normalize_tool_call(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": block.get("toolUseId", ""),
        TYPE_KEY: FUNCTION_TOOL_TYPE,
        FUNCTION_KEY: {
            NAME_KEY: block.get(NAME_KEY, ""),
            "arguments": to_json_attr(block.get(INPUT_KEY, {})),
        },
    }


def _normalize_anthropic_tool_result(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        ROLE_KEY: TOOL_ROLE,
        "tool_call_id": block.get("tool_use_id", ""),
        CONTENT_KEY: _normalize_text_content(block.get(CONTENT_KEY, "")),
    }


def _normalize_tool_result(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        ROLE_KEY: TOOL_ROLE,
        "tool_call_id": block.get("toolUseId", ""),
        CONTENT_KEY: _normalize_text_content(block.get(CONTENT_KEY, "")),
    }


def _extract_tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    blocks = content if isinstance(content, list | tuple) else [content]
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        if "toolUse" in block:
            tool_call = _normalize_tool_call(block["toolUse"])
        elif block.get(TYPE_KEY) == "tool_use":
            tool_call = _normalize_anthropic_tool_use(block)
        else:
            continue
        function = tool_call.get(FUNCTION_KEY, {})
        if isinstance(function, Mapping) and function.get(NAME_KEY):
            tool_calls.append(tool_call)
    return tool_calls


def _normalize_tool_definition(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, Mapping):
        return None

    if "toolSpec" in tool:
        tool = tool["toolSpec"]

    name = tool.get(NAME_KEY)
    if not isinstance(name, str) or not name:
        return None

    schema = tool.get("inputSchema")
    if schema is None:
        schema = tool.get(INPUT_SCHEMA_KEY)
    if isinstance(schema, Mapping) and "json" in schema:
        schema = schema["json"]
    if schema is None:
        schema = {"type": "object"}

    function: dict[str, Any] = {
        NAME_KEY: name,
        "parameters": schema,
    }
    description = tool.get(DESCRIPTION_KEY)
    if description:
        function[DESCRIPTION_KEY] = description

    return {
        TYPE_KEY: FUNCTION_TOOL_TYPE,
        FUNCTION_KEY: function,
    }


def _extract_tools_from_body(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, Mapping):
        return []
    raw_tools = body.get(TOOLS_KEY)
    if not isinstance(raw_tools, list):
        return []
    return [
        normalized
        for tool in raw_tools
        if (normalized := _normalize_tool_definition(tool)) is not None
    ]


def _extract_tools_from_converse(api_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    tool_config = api_params.get(TOOL_CONFIG_KEY)
    if not isinstance(tool_config, Mapping):
        return []
    raw_tools = tool_config.get(TOOLS_KEY)
    if not isinstance(raw_tools, list):
        return []
    return [
        normalized
        for tool in raw_tools
        if (normalized := _normalize_tool_definition(tool)) is not None
    ]


def parse_bedrock_request(
    *,
    operation_name: str,
    api_params: Mapping[str, Any] | None,
) -> BedrockRequest:
    params = api_params or {}
    model_id = params.get(MODEL_ID_KEY)

    if operation_name in {INVOKE_MODEL_OPERATION, INVOKE_MODEL_STREAM_OPERATION}:
        body = _load_json(params.get(BODY_KEY))
        return BedrockRequest(
            operation_name=operation_name,
            model_id=model_id if isinstance(model_id, str) else None,
            messages=_normalize_prompt_from_body(body),
            tools=_extract_tools_from_body(body),
            raw_payload=body,
        )

    if operation_name in {CONVERSE_OPERATION, CONVERSE_STREAM_OPERATION}:
        return BedrockRequest(
            operation_name=operation_name,
            model_id=model_id if isinstance(model_id, str) else None,
            messages=_normalize_converse_messages(params),
            tools=_extract_tools_from_converse(params),
            raw_payload=serialize_value(value=params),
        )

    return BedrockRequest(
        operation_name=operation_name,
        model_id=model_id if isinstance(model_id, str) else None,
        raw_payload=serialize_value(value=params),
    )


def _usage_from_mapping(value):
    if type(value) is not dict:
        return {}
    names = {
        "input_tokens": (
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
            "promptTokens",
            "inputTextTokenCount",
        ),
        "output_tokens": (
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
        ),
        "total_tokens": ("total_tokens", "totalTokens", "total_token_count"),
        "cache_read_input_tokens": ("cacheReadInputTokens", "cache_read_input_tokens"),
        "cache_creation_input_tokens": (
            "cacheWriteInputTokens",
            "cache_creation_input_tokens",
        ),
    }
    result = {}
    for target, sources in names.items():
        for source in sources:
            count = _coerce_int(value.get(source))
            if count is not None:
                result[target] = count
                break
    return result


def _merge_usage(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        if isinstance(value, int):
            target[key] = value


def _response_from_anthropic_payload(payload: Mapping[str, Any]) -> BedrockResponse:
    content = payload.get(CONTENT_KEY, "")
    return BedrockResponse(
        content=_extract_text_from_response_content(content),
        role=_normalize_bedrock_role(payload.get(ROLE_KEY, ASSISTANT_ROLE)),
        tool_calls=_extract_tool_calls_from_content(content),
        usage=_usage_from_mapping(payload.get(USAGE_KEY)),
        raw_payload=payload,
    )


def _response_from_converse_payload(payload: Mapping[str, Any]) -> BedrockResponse:
    output = payload.get(OUTPUT_KEY)
    message = output.get(MESSAGE_KEY) if isinstance(output, Mapping) else None
    if not isinstance(message, Mapping):
        return BedrockResponse(raw_payload=payload)
    content = message.get(CONTENT_KEY, "")
    return BedrockResponse(
        content=_extract_text_from_response_content(content),
        role=_normalize_bedrock_role(message.get(ROLE_KEY, ASSISTANT_ROLE)),
        tool_calls=_extract_tool_calls_from_content(content),
        usage=_usage_from_mapping(payload.get(USAGE_KEY)),
        raw_payload=payload,
    )


def _response_from_titan_payload(payload: Mapping[str, Any]) -> BedrockResponse:
    text = ""
    usage: dict[str, int] = {}
    input_tokens = _coerce_int(payload.get("inputTextTokenCount"))
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens

    results = payload.get("results")
    if isinstance(results, list) and results:
        first_result = results[0]
        if isinstance(first_result, Mapping):
            text = str(
                first_result.get("outputText") or first_result.get(TEXT_KEY) or ""
            )
            output_tokens = _coerce_int(first_result.get("tokenCount"))
            if output_tokens is not None:
                usage["output_tokens"] = output_tokens
    return BedrockResponse(content=text, usage=usage, raw_payload=payload)


def _extract_text_from_response_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get(TEXT_KEY)
        if isinstance(text, str):
            return text
        if content.get(TYPE_KEY) == "text":
            return str(content.get(TEXT_KEY, ""))
        return ""
    if isinstance(content, list | tuple):
        parts = [
            _extract_text_from_response_content(item)
            for item in content
            if _extract_text_from_response_content(item)
        ]
        return "\n".join(parts)
    return ""


def parse_bedrock_response(
    *,
    operation_name: str,
    response_payload: Any,
) -> BedrockResponse:
    if not isinstance(response_payload, Mapping):
        return BedrockResponse(raw_payload=response_payload)

    if operation_name == CONVERSE_OPERATION:
        return _response_from_converse_payload(response_payload)

    if CONTENT_KEY in response_payload and isinstance(
        response_payload.get(CONTENT_KEY), list
    ):
        return _response_from_anthropic_payload(response_payload)

    if "results" in response_payload or "inputTextTokenCount" in response_payload:
        return _response_from_titan_payload(response_payload)

    for key in ("generation", "outputText", "completion"):
        value = response_payload.get(key)
        if isinstance(value, str):
            return BedrockResponse(
                content=value,
                usage=_usage_from_mapping(
                    response_payload.get(USAGE_KEY) or response_payload
                ),
                raw_payload=response_payload,
            )

    outputs = response_payload.get("outputs")
    if isinstance(outputs, list) and outputs:
        return BedrockResponse(
            content=_extract_text_from_response_content(outputs),
            usage=_usage_from_mapping(
                response_payload.get(USAGE_KEY) or response_payload
            ),
            raw_payload=response_payload,
        )

    return BedrockResponse(
        content=_extract_text_from_response_content(response_payload),
        usage=_usage_from_mapping(response_payload.get(USAGE_KEY) or response_payload),
        raw_payload=response_payload,
    )


def _parse_chunk_payload(event: Mapping[str, Any]) -> Any:
    chunk = event.get("chunk")
    if isinstance(chunk, Mapping):
        bytes_value = chunk.get("bytes")
        if bytes_value is not None:
            return _load_json(bytes_value)
    return None


def parse_bedrock_stream_response(*, operation_name, events):
    text_parts = []
    tools = {}
    usage = {}
    for event in events:
        if type(event) is not dict:
            continue
        if operation_name == CONVERSE_STREAM_OPERATION:
            start = event.get("contentBlockStart", {})
            delta = event.get("contentBlockDelta", {})
            if (
                type(start) is dict
                and type(start.get("start")) is dict
                and "toolUse" in start["start"]
            ):
                tools[start.get("contentBlockIndex", 0)] = _normalize_tool_call(
                    start["start"]["toolUse"]
                )
            block = delta.get("delta", {}) if type(delta) is dict else {}
            if type(block) is dict:
                if type(block.get("text")) is str:
                    text_parts.append(block["text"])
                fragment = (
                    block.get("toolUse", {}).get("input")
                    if type(block.get("toolUse")) is dict
                    else None
                )
                tool = tools.get(delta.get("contentBlockIndex", 0))
                if tool is not None and type(fragment) is str:
                    function = tool["function"]
                    previous = function.get("_fragments", "")
                    function["_fragments"] = previous + fragment
            metadata = event.get("metadata", {})
            if type(metadata) is dict:
                _merge_usage(usage, _usage_from_mapping(metadata.get("usage")))
        else:
            payload = _parse_chunk_payload(event)
            if type(payload) is not dict:
                continue
            kind = payload.get("type")
            index = payload.get("index", 0)
            if (
                kind == "content_block_start"
                and type(payload.get("content_block")) is dict
                and payload["content_block"].get("type") == "tool_use"
            ):
                tools[index] = _normalize_anthropic_tool_use(payload["content_block"])
            if kind == "content_block_delta" and type(payload.get("delta")) is dict:
                delta = payload["delta"]
                if type(delta.get("text")) is str:
                    text_parts.append(delta["text"])
                if type(delta.get("partial_json")) is str and index in tools:
                    function = tools[index]["function"]
                    function["_fragments"] = (
                        function.get("_fragments", "") + delta["partial_json"]
                    )
            for key in ("outputText", "generation", "completion"):
                if type(payload.get(key)) is str:
                    text_parts.append(payload[key])
                    break
            message = payload.get("message", {})
            if type(message) is dict:
                _merge_usage(usage, _usage_from_mapping(message.get("usage")))
            _merge_usage(usage, _usage_from_mapping(payload.get("usage")))
            _merge_usage(
                usage,
                _usage_from_mapping(payload.get("amazon-bedrock-invocationMetrics")),
            )
    for tool in tools.values():
        function = tool["function"]
        fragments = function.pop("_fragments", None)
        if fragments is not None:
            try:
                function["arguments"] = safe_json(json.loads(fragments))
            except ValueError:
                function["arguments"] = fragments
    return BedrockResponse(
        content="".join(text_parts),
        tool_calls=list(tools.values()),
        usage=usage,
        raw_payload=events,
    )
