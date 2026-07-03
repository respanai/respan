"""Translate AWS SageMaker Runtime payloads into Respan span fields."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.semconv_ai import LLMRequestTypeValues

from respan_instrumentation_sagemaker._constants import (
    ASSISTANT_ROLE,
    BODY_KEY,
    BYTES_KEY,
    CHOICES_KEY,
    CONTENT_KEY,
    CUSTOM_ATTRIBUTES_KEY,
    DESCRIPTION_KEY,
    ENDPOINT_NAME_KEY,
    FUNCTION_KEY,
    FUNCTIONS_KEY,
    FUNCTION_TOOL_TYPE,
    GENERATED_TEXT_KEY,
    INPUT_KEY,
    INPUT_LOCATION_KEY,
    INPUTS_KEY,
    INVOKE_ENDPOINT_ASYNC_OPERATION,
    MESSAGE_KEY,
    MESSAGES_KEY,
    MODEL_KEY,
    NAME_KEY,
    OUTPUT_LOCATION_KEY,
    OUTPUTS_KEY,
    PARAMETERS_KEY,
    PAYLOAD_PART_KEY,
    PROMPT_KEY,
    RESPAN_MODEL_ATTRIBUTE,
    ROLE_KEY,
    SYSTEM_KEY,
    SYSTEM_ROLE,
    TARGET_MODEL_KEY,
    TEXT_KEY,
    TOOL_CALLS_KEY,
    TOOLS_KEY,
    TOOL_ROLE,
    TYPE_KEY,
    USAGE_KEY,
    USER_ROLE,
)
from respan_sdk.utils.serialization import serialize_value


@dataclass(frozen=True)
class SageMakerRequest:
    operation_name: str
    endpoint_name: str | None = None
    model_id: str | None = None
    request_type: str = LLMRequestTypeValues.CHAT.value
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    raw_payload: Any = None


@dataclass(frozen=True)
class SageMakerResponse:
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


def _model_from_custom_attributes(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    for part in value.replace(";", ",").split(","):
        key, separator, raw_model = part.strip().partition("=")
        if separator and key.strip() == RESPAN_MODEL_ATTRIBUTE:
            model = raw_model.strip()
            return model or None
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
    """Read and restore `invoke_endpoint` response bodies for translation."""
    if not isinstance(response, dict):
        return response, None

    body = response.get(BODY_KEY)
    body_bytes = _read_body_bytes(body)
    if body_bytes is None:
        return response, None

    response[BODY_KEY] = ReplayableBody(body=body_bytes, original_body=body)
    return response, _load_json(body_bytes)


def _normalize_role(role: Any) -> str:
    if role in {"assistant", "model"}:
        return ASSISTANT_ROLE
    if role == "system":
        return SYSTEM_ROLE
    if role == "tool":
        return TOOL_ROLE
    return USER_ROLE if not isinstance(role, str) else role


def _normalize_text_content(content: Any) -> Any:
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
        if content.get(TYPE_KEY) in {"tool_call", "function"}:
            return _normalize_tool_call(content)
        if "json" in content:
            return content["json"]
        return serialize_value(value=content)
    if isinstance(content, list | tuple):
        normalized = [
            item
            for item in (_normalize_text_content(item) for item in content)
            if item not in (None, "", [], {})
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
        content = _field(message, TEXT_KEY)
    if content is None and isinstance(message, str):
        content = message

    normalized: dict[str, Any] = {
        ROLE_KEY: _normalize_role(role),
        CONTENT_KEY: _normalize_text_content(content),
    }
    tool_calls = _extract_tool_calls_from_content(content)
    if tool_calls and normalized[ROLE_KEY] == ASSISTANT_ROLE:
        normalized[TOOL_CALLS_KEY] = tool_calls
    return normalized


def _normalize_system_messages(system: Any) -> list[dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: system}]
    if isinstance(system, list | tuple):
        content = _normalize_text_content(system)
        return [{ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: content}] if content else []
    return [{ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: _normalize_text_content(system)}]


def _prompt_message(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _normalize_text_content(value)}]


def _normalize_prompt_from_payload(body: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(body, list):
        if body and all(isinstance(item, Mapping) and ROLE_KEY in item for item in body):
            return (
                LLMRequestTypeValues.CHAT.value,
                [_normalize_message(message) for message in body],
            )
        return (
            LLMRequestTypeValues.CHAT.value,
            _prompt_message(_normalize_text_content(body)),
        )

    if not isinstance(body, Mapping):
        return (LLMRequestTypeValues.CHAT.value, _prompt_message(body))

    messages: list[dict[str, Any]] = []
    messages.extend(_normalize_system_messages(body.get(SYSTEM_KEY)))

    raw_messages = body.get(MESSAGES_KEY)
    if isinstance(raw_messages, list):
        messages.extend(_normalize_message(message) for message in raw_messages)
        return (LLMRequestTypeValues.CHAT.value, messages)

    for key in (PROMPT_KEY, INPUTS_KEY, INPUT_KEY, TEXT_KEY):
        value = body.get(key)
        if value is not None:
            return (LLMRequestTypeValues.CHAT.value, _prompt_message(value))

    return (LLMRequestTypeValues.CHAT.value, messages)


def _normalize_tool_definition(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, Mapping):
        return None

    if FUNCTION_KEY in tool and isinstance(tool[FUNCTION_KEY], Mapping):
        function = dict(tool[FUNCTION_KEY])
    else:
        function = {
            NAME_KEY: tool.get(NAME_KEY),
            "parameters": tool.get(PARAMETERS_KEY)
            or tool.get("input_schema")
            or tool.get("schema")
            or {"type": "object"},
        }
        description = tool.get(DESCRIPTION_KEY)
        if description:
            function[DESCRIPTION_KEY] = description

    name = function.get(NAME_KEY)
    if not isinstance(name, str) or not name:
        return None

    return {
        TYPE_KEY: tool.get(TYPE_KEY, FUNCTION_TOOL_TYPE),
        FUNCTION_KEY: function,
    }


def _extract_tools_from_payload(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, Mapping):
        return []

    raw_tools = body.get(TOOLS_KEY)
    if raw_tools is None:
        raw_tools = body.get(FUNCTIONS_KEY)
    if not isinstance(raw_tools, list):
        return []
    return [
        normalized
        for tool in raw_tools
        if (normalized := _normalize_tool_definition(tool)) is not None
    ]


def _normalize_tool_call(block: Mapping[str, Any]) -> dict[str, Any]:
    function = block.get(FUNCTION_KEY)
    if isinstance(function, Mapping):
        name = function.get(NAME_KEY) or block.get(NAME_KEY) or ""
        arguments = function.get("arguments")
    else:
        name = block.get(NAME_KEY) or ""
        arguments = block.get("arguments") or block.get(INPUT_KEY) or {}

    return {
        "id": block.get("id") or block.get("tool_call_id") or block.get("toolUseId") or "",
        TYPE_KEY: block.get(TYPE_KEY, FUNCTION_TOOL_TYPE),
        FUNCTION_KEY: {
            NAME_KEY: name,
            "arguments": to_json_attr(arguments or {}),
        },
    }


def _extract_tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    blocks = content if isinstance(content, list | tuple) else [content]
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        raw_tool_calls = block.get(TOOL_CALLS_KEY)
        if isinstance(raw_tool_calls, list):
            for raw_tool_call in raw_tool_calls:
                if isinstance(raw_tool_call, Mapping):
                    tool_call = _normalize_tool_call(raw_tool_call)
                    if tool_call.get(FUNCTION_KEY, {}).get(NAME_KEY):
                        tool_calls.append(tool_call)
            continue
        if block.get(TYPE_KEY) in {"tool_call", "function"} or FUNCTION_KEY in block:
            tool_call = _normalize_tool_call(block)
            if tool_call.get(FUNCTION_KEY, {}).get(NAME_KEY):
                tool_calls.append(tool_call)
    return tool_calls


def parse_sagemaker_request(
    *,
    operation_name: str,
    api_params: Mapping[str, Any] | None,
) -> SageMakerRequest:
    params = api_params or {}
    endpoint_name = params.get(ENDPOINT_NAME_KEY)
    target_model = params.get(TARGET_MODEL_KEY)
    custom_model = _model_from_custom_attributes(params.get(CUSTOM_ATTRIBUTES_KEY))
    body = _load_json(params.get(BODY_KEY))

    if operation_name == INVOKE_ENDPOINT_ASYNC_OPERATION:
        raw_payload = {
            INPUT_LOCATION_KEY: params.get(INPUT_LOCATION_KEY),
            "InferenceId": params.get("InferenceId"),
        }
        messages = _prompt_message(params.get(INPUT_LOCATION_KEY))
        return SageMakerRequest(
            operation_name=operation_name,
            endpoint_name=endpoint_name if isinstance(endpoint_name, str) else None,
            model_id=custom_model or (target_model if isinstance(target_model, str) else None),
            request_type=LLMRequestTypeValues.CHAT.value,
            messages=messages,
            raw_payload=raw_payload,
        )

    request_type, messages = _normalize_prompt_from_payload(body)
    tools = _extract_tools_from_payload(body)
    if tools:
        request_type = LLMRequestTypeValues.CHAT.value

    body_model = body.get(MODEL_KEY) if isinstance(body, Mapping) else None
    model_id = custom_model or (target_model if isinstance(target_model, str) else body_model)

    return SageMakerRequest(
        operation_name=operation_name,
        endpoint_name=endpoint_name if isinstance(endpoint_name, str) else None,
        model_id=model_id if isinstance(model_id, str) else None,
        request_type=request_type,
        messages=messages,
        tools=tools,
        raw_payload=body,
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
        or _coerce_int(value.get("generated_tokens"))
        or _coerce_int(value.get("generatedTokens"))
    )
    total_tokens = (
        _coerce_int(value.get("total_tokens"))
        or _coerce_int(value.get("totalTokens"))
        or _coerce_int(value.get("total_token_count"))
    )

    details = value.get("details")
    if isinstance(details, Mapping):
        nested = _usage_from_mapping(details)
        prompt_tokens = prompt_tokens if prompt_tokens is not None else nested.get("input_tokens")
        completion_tokens = (
            completion_tokens
            if completion_tokens is not None
            else nested.get("output_tokens")
        )
        total_tokens = total_tokens if total_tokens is not None else nested.get("total_tokens")

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


def _extract_text_from_response_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in (GENERATED_TEXT_KEY, TEXT_KEY, CONTENT_KEY, "outputText", "generation"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        if content.get(TYPE_KEY) == "text":
            return str(content.get(TEXT_KEY, ""))
        return ""
    if isinstance(content, list | tuple):
        parts = [
            text
            for item in content
            if (text := _extract_text_from_response_content(item))
        ]
        return "\n".join(parts)
    return ""


def _response_from_openai_choice(payload: Mapping[str, Any]) -> SageMakerResponse | None:
    choices = payload.get(CHOICES_KEY)
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        return None

    message = first_choice.get(MESSAGE_KEY)
    if isinstance(message, Mapping):
        content = message.get(CONTENT_KEY, "")
        tool_calls = message.get(TOOL_CALLS_KEY, [])
        return SageMakerResponse(
            content=_extract_text_from_response_content(content),
            role=_normalize_role(message.get(ROLE_KEY, ASSISTANT_ROLE)),
            tool_calls=[
                tool_call
                for raw_tool_call in tool_calls
                if isinstance(raw_tool_call, Mapping)
                if (tool_call := _normalize_tool_call(raw_tool_call)).get(
                    FUNCTION_KEY, {}
                ).get(NAME_KEY)
            ]
            if isinstance(tool_calls, list)
            else [],
            usage=_usage_from_mapping(payload.get(USAGE_KEY)),
            raw_payload=payload,
        )

    text = first_choice.get(TEXT_KEY)
    if isinstance(text, str):
        return SageMakerResponse(
            content=text,
            usage=_usage_from_mapping(payload.get(USAGE_KEY)),
            raw_payload=payload,
        )
    return None


def _response_from_generated_payload(payload: Mapping[str, Any]) -> SageMakerResponse:
    for key in (GENERATED_TEXT_KEY, "generated_texts", "generation", "outputText", "completion"):
        value = payload.get(key)
        if isinstance(value, str):
            return SageMakerResponse(
                content=value,
                usage=_usage_from_mapping(payload.get(USAGE_KEY) or payload),
                raw_payload=payload,
            )
        if isinstance(value, list):
            return SageMakerResponse(
                content=_extract_text_from_response_content(value),
                usage=_usage_from_mapping(payload.get(USAGE_KEY) or payload),
                raw_payload=payload,
            )

    outputs = payload.get(OUTPUTS_KEY)
    if isinstance(outputs, list) and outputs:
        return SageMakerResponse(
            content=_extract_text_from_response_content(outputs),
            usage=_usage_from_mapping(payload.get(USAGE_KEY) or payload),
            raw_payload=payload,
        )

    return SageMakerResponse(
        content=_extract_text_from_response_content(payload),
        usage=_usage_from_mapping(payload.get(USAGE_KEY) or payload),
        raw_payload=payload,
    )


def parse_sagemaker_response(
    *,
    operation_name: str,
    response_payload: Any,
) -> SageMakerResponse:
    if operation_name == INVOKE_ENDPOINT_ASYNC_OPERATION:
        output_location = None
        if isinstance(response_payload, Mapping):
            output_location = response_payload.get(OUTPUT_LOCATION_KEY)
        return SageMakerResponse(
            content=str(output_location or ""),
            raw_payload=response_payload,
        )

    if isinstance(response_payload, list):
        if response_payload and isinstance(response_payload[0], Mapping):
            return _response_from_generated_payload(response_payload[0])
        return SageMakerResponse(
            content=_extract_text_from_response_content(response_payload),
            raw_payload=response_payload,
        )

    if not isinstance(response_payload, Mapping):
        return SageMakerResponse(
            content=str(response_payload or ""),
            raw_payload=response_payload,
        )

    if openai_choice_response := _response_from_openai_choice(response_payload):
        return openai_choice_response

    return _response_from_generated_payload(response_payload)


def _parse_payload_part(event: Mapping[str, Any]) -> Any:
    payload_part = event.get(PAYLOAD_PART_KEY)
    if isinstance(payload_part, Mapping):
        bytes_value = payload_part.get(BYTES_KEY)
        if bytes_value is not None:
            return _load_json(bytes_value)
    return None


def _extract_stream_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        token = payload.get("token")
        if isinstance(token, Mapping):
            text = token.get(TEXT_KEY)
            if isinstance(text, str):
                return text
        choices = payload.get(CHOICES_KEY)
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") if isinstance(choices[0], Mapping) else None
            if isinstance(delta, Mapping):
                content = delta.get(CONTENT_KEY)
                if isinstance(content, str):
                    return content
            text = choices[0].get(TEXT_KEY) if isinstance(choices[0], Mapping) else None
            if isinstance(text, str):
                return text
        return _extract_text_from_response_content(payload)
    return ""


def parse_sagemaker_stream_response(*, events: list[Any]) -> SageMakerResponse:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    raw_payloads: list[Any] = []

    for event in events:
        if not isinstance(event, Mapping):
            raw_payloads.append(serialize_value(value=event))
            continue
        raw_payloads.append(serialize_value(value=event))

        payload = _parse_payload_part(event)
        if payload is None:
            continue
        raw_payloads.append(payload)
        if isinstance(payload, Mapping):
            _merge_usage(usage, _usage_from_mapping(payload.get(USAGE_KEY) or payload))
            extracted_tool_calls = _extract_tool_calls_from_content(payload)
            if extracted_tool_calls:
                tool_calls.extend(extracted_tool_calls)
        text = _extract_stream_text(payload)
        if text:
            text_parts.append(text)

    return SageMakerResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        usage=usage,
        raw_payload=raw_payloads,
    )
