"""Translate AWS Bedrock Runtime payloads into Respan span fields."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Mapping
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
from respan_sdk.utils.serialization import serialize_value


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


class ReplayableBody:
    """Small file-like wrapper that lets callers read a captured response body."""

    def __init__(self, body: bytes, original_body: Any = None) -> None:
        self._body = body
        self._stream = io.BytesIO(body)
        self._original_body = original_body

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            return self._stream.read()
        return self._stream.read(amt)

    def iter_chunks(self, chunk_size: int = 1024) -> Iterable[bytes]:
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def iter_lines(self, chunk_size: int = 1024) -> Iterable[bytes]:
        pending = b""
        for chunk in self.iter_chunks(chunk_size=chunk_size):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                yield line
        if pending:
            yield pending

    def close(self) -> None:
        close = getattr(self._original_body, "close", None)
        if callable(close):
            close()
        self._stream.close()

    def __iter__(self) -> Iterable[bytes]:
        return self.iter_chunks()

    def __getattr__(self, name: str) -> Any:
        if self._original_body is None:
            raise AttributeError(name)
        return getattr(self._original_body, name)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(serialize_value(value=value), default=str)
    except Exception:
        return str(value)


def to_json_attr(value: Any) -> str:
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
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


def _read_body_bytes(body: Any) -> bytes | None:
    if body is None:
        return None
    if isinstance(body, bytes | bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")

    read = getattr(body, "read", None)
    if not callable(read):
        return None

    data = read()
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, bytes | bytearray):
        return bytes(data)
    return None


def capture_invoke_response_payload(response: Any) -> tuple[Any, Any]:
    """Read and restore `invoke_model` response bodies for translation."""
    if not isinstance(response, dict):
        return response, None

    body = response.get(BODY_KEY)
    body_bytes = _read_body_bytes(body)
    if body_bytes is None:
        return response, None

    response[BODY_KEY] = ReplayableBody(body=body_bytes, original_body=body)
    return response, _load_json(body_bytes)


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
        if value:
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


def _usage_from_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}

    prompt_tokens = (
        _coerce_int(value.get("input_tokens"))
        or _coerce_int(value.get("inputTokens"))
        or _coerce_int(value.get("prompt_tokens"))
        or _coerce_int(value.get("promptTokens"))
        or _coerce_int(value.get("inputTextTokenCount"))
    )
    completion_tokens = (
        _coerce_int(value.get("output_tokens"))
        or _coerce_int(value.get("outputTokens"))
        or _coerce_int(value.get("completion_tokens"))
        or _coerce_int(value.get("completionTokens"))
    )
    total_tokens = (
        _coerce_int(value.get("total_tokens"))
        or _coerce_int(value.get("totalTokens"))
        or _coerce_int(value.get("total_token_count"))
    )

    result: dict[str, int] = {}
    if prompt_tokens is not None:
        result["input_tokens"] = prompt_tokens
    if completion_tokens is not None:
        result["output_tokens"] = completion_tokens
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    if total_tokens is not None:
        result["total_tokens"] = total_tokens
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
    if usage and "total_tokens" not in usage:
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        )
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


def parse_bedrock_stream_response(
    *,
    operation_name: str,
    events: list[Any],
) -> BedrockResponse:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    raw_payloads: list[Any] = []

    for event in events:
        if not isinstance(event, Mapping):
            raw_payloads.append(serialize_value(value=event))
            continue
        raw_payloads.append(serialize_value(value=event))

        if operation_name == CONVERSE_STREAM_OPERATION:
            delta = _field(
                _field(event.get("contentBlockDelta"), "delta", {}), TEXT_KEY
            )
            if isinstance(delta, str):
                text_parts.append(delta)

            start_block = _field(event.get("contentBlockStart"), "start", {})
            if isinstance(start_block, Mapping) and "toolUse" in start_block:
                tool_call = _normalize_tool_call(start_block["toolUse"])
                if tool_call.get(FUNCTION_KEY, {}).get(NAME_KEY):
                    tool_calls.append(tool_call)

            metadata = event.get("metadata")
            if isinstance(metadata, Mapping):
                _merge_usage(usage, _usage_from_mapping(metadata.get(USAGE_KEY)))
            continue

        payload = _parse_chunk_payload(event)
        if not isinstance(payload, Mapping):
            continue
        raw_payloads.append(payload)

        payload_type = payload.get(TYPE_KEY)
        if payload_type == "content_block_delta":
            delta = payload.get("delta")
            text = delta.get(TEXT_KEY) if isinstance(delta, Mapping) else None
            if isinstance(text, str):
                text_parts.append(text)
        elif payload_type == "content_block_start":
            content_block = payload.get("content_block")
            if (
                isinstance(content_block, Mapping)
                and content_block.get(TYPE_KEY) == "tool_use"
            ):
                tool_call = _normalize_anthropic_tool_use(content_block)
                if tool_call.get(FUNCTION_KEY, {}).get(NAME_KEY):
                    tool_calls.append(tool_call)
        elif payload_type == "message_start":
            message = payload.get(MESSAGE_KEY)
            if isinstance(message, Mapping):
                _merge_usage(usage, _usage_from_mapping(message.get(USAGE_KEY)))
        elif payload_type == "message_delta":
            _merge_usage(usage, _usage_from_mapping(payload.get(USAGE_KEY)))

    return BedrockResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        usage=usage,
        raw_payload=raw_payloads,
    )
