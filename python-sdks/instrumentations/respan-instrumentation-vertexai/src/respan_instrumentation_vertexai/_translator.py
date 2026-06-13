"""Translate Vertex AI SDK payloads into Respan span fields."""

from __future__ import annotations

import json
from typing import Any, Iterable

from respan_instrumentation_vertexai._constants import (
    ARGS_KEY,
    ASSISTANT_ROLE,
    CANDIDATES_KEY,
    CANDIDATES_TOKEN_COUNT_KEY,
    CONTENT_KEY,
    FUNCTION_CALL_KEY,
    FUNCTION_DECLARATIONS_KEY,
    FUNCTION_KEY,
    FUNCTION_RESPONSE_KEY,
    FUNCTION_TOOL_TYPE,
    GENERATION_CONFIG_KEY,
    ID_KEY,
    MODEL_ROLE,
    NAME_KEY,
    PARAMETERS_KEY,
    PARTS_KEY,
    PROMPT_TOKEN_COUNT_KEY,
    RESPONSE_KEY,
    ROLE_KEY,
    STREAM_KEY,
    SYSTEM_INSTRUCTION_KEY,
    SYSTEM_ROLE,
    TEXT_KEY,
    TOOL_CONFIG_KEY,
    TOOL_ROLE,
    TOOLS_KEY,
    TOTAL_TOKEN_COUNT_KEY,
    TYPE_KEY,
    USAGE_METADATA_KEY,
    USER_ROLE,
)
from respan_sdk.utils.serialization import serialize_value


def safe_json(value: Any) -> str:
    """JSON-encode values after applying the SDK's serializer."""
    try:
        return json.dumps(serialize_value(value=value), default=str)
    except Exception:
        return str(value)


def to_json_attr(value: Any) -> str:
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _dump_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _dump_value(nested_value)
            for key, nested_value in value.items()
            if nested_value is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [_dump_value(item) for item in value if item is not None]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True, by_alias=False)
        except TypeError:
            return model_dump()
    return serialize_value(value=value)


def _role(value: Any, default: str = USER_ROLE) -> str:
    role = _field(value, ROLE_KEY, default) or default
    if role == MODEL_ROLE:
        return ASSISTANT_ROLE
    if role == FUNCTION_KEY:
        return TOOL_ROLE
    return str(role)


def _is_content_like(value: Any) -> bool:
    if isinstance(value, dict):
        return PARTS_KEY in value or ROLE_KEY in value
    return hasattr(value, PARTS_KEY) or hasattr(value, ROLE_KEY)


def _is_part_like(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(
            key in value
            for key in (
                TEXT_KEY,
                FUNCTION_CALL_KEY,
                FUNCTION_RESPONSE_KEY,
                "inline_data",
                "file_data",
            )
        )
    return any(
        hasattr(value, attr)
        for attr in (
            TEXT_KEY,
            FUNCTION_CALL_KEY,
            FUNCTION_RESPONSE_KEY,
            "inline_data",
            "file_data",
        )
    )


def _normalize_function_call(value: Any) -> dict[str, Any]:
    function_call: dict[str, Any] = {
        TYPE_KEY: FUNCTION_TOOL_TYPE,
        FUNCTION_KEY: {},
    }
    call_id = _field(value, ID_KEY)
    if call_id:
        function_call[ID_KEY] = call_id
    name = _field(value, NAME_KEY)
    if name:
        function_call[FUNCTION_KEY][NAME_KEY] = name
    args = _field(value, ARGS_KEY)
    if args is not None:
        function_call[FUNCTION_KEY]["arguments"] = to_json_attr(_dump_value(args))
    return function_call


def _normalize_function_response(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        TYPE_KEY: "function_response",
        FUNCTION_KEY: {},
    }
    name = _field(value, NAME_KEY)
    if name:
        result[FUNCTION_KEY][NAME_KEY] = name
    response = _field(value, RESPONSE_KEY)
    if response is not None:
        result[FUNCTION_KEY][RESPONSE_KEY] = _dump_value(response)
    return result


def _normalize_part(part: Any) -> Any:
    if isinstance(part, str):
        return part

    text = _field(part, TEXT_KEY)
    if text is not None:
        return text

    function_call = _field(part, FUNCTION_CALL_KEY)
    if function_call is not None:
        return _normalize_function_call(function_call)

    function_response = _field(part, FUNCTION_RESPONSE_KEY)
    if function_response is not None:
        return _normalize_function_response(function_response)

    dumped = _dump_value(part)
    if isinstance(dumped, dict):
        for key in ("inline_data", "file_data"):
            if dumped.get(key) is not None:
                return {TYPE_KEY: key, key: dumped[key]}
    return dumped


def _normalize_parts(parts: Any) -> Any:
    if parts is None:
        return ""
    if isinstance(parts, (str, bytes)):
        return parts.decode() if isinstance(parts, bytes) else parts
    if not isinstance(parts, (list, tuple)):
        return _normalize_part(parts)

    normalized_parts = [
        normalized
        for part in parts
        if (normalized := _normalize_part(part)) not in (None, "", [], {})
    ]
    if not normalized_parts:
        return ""
    if len(normalized_parts) == 1:
        return normalized_parts[0]
    if all(isinstance(part, str) for part in normalized_parts):
        return "\n".join(normalized_parts)
    return normalized_parts


def _normalize_content(
    content: Any, *, default_role: str = USER_ROLE
) -> dict[str, Any]:
    if isinstance(content, str):
        return {ROLE_KEY: default_role, CONTENT_KEY: content}
    if _is_part_like(content) and not _is_content_like(content):
        return {ROLE_KEY: default_role, CONTENT_KEY: _normalize_part(content)}

    parts = _field(content, PARTS_KEY)
    return {
        ROLE_KEY: _role(content, default=default_role),
        CONTENT_KEY: _normalize_parts(parts),
    }


def normalize_input_messages(
    contents: Any,
    *,
    system_instruction: Any = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if system_instruction is not None:
        messages.append(
            _normalize_content(system_instruction, default_role=SYSTEM_ROLE)
            if _is_content_like(system_instruction)
            else {
                ROLE_KEY: SYSTEM_ROLE,
                CONTENT_KEY: _normalize_parts(system_instruction),
            }
        )

    if contents is None:
        return messages
    if isinstance(contents, str):
        messages.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: contents})
        return messages
    if _is_content_like(contents):
        messages.append(_normalize_content(contents))
        return messages
    if _is_part_like(contents):
        messages.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: _normalize_part(contents)})
        return messages
    if isinstance(contents, (list, tuple)):
        if all(_is_content_like(item) for item in contents):
            messages.extend(_normalize_content(item) for item in contents)
            return messages
        messages.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: _normalize_parts(contents)})
        return messages

    messages.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: serialize_value(value=contents)})
    return messages


def format_input(contents: Any, *, system_instruction: Any = None) -> str:
    return safe_json(
        value=normalize_input_messages(
            contents=contents,
            system_instruction=system_instruction,
        )
    )


def _candidate_contents(response: Any) -> Iterable[Any]:
    for candidate in _field(response, CANDIDATES_KEY, []) or []:
        content = _field(candidate, CONTENT_KEY)
        if content is not None:
            yield content


def _response_text(response: Any) -> str:
    text = _field(response, TEXT_KEY)
    if isinstance(text, str):
        return text

    text_parts: list[str] = []
    for content in _candidate_contents(response):
        content_value = _normalize_content(content, default_role=ASSISTANT_ROLE).get(
            CONTENT_KEY,
            "",
        )
        if isinstance(content_value, str):
            text_parts.append(content_value)
    return "\n".join(part for part in text_parts if part)


def format_output(response_or_chunks: Any) -> str:
    chunks = response_or_chunks if isinstance(response_or_chunks, list) else None
    if chunks is not None:
        return "".join(_response_text(chunk) for chunk in chunks if chunk is not None)
    if response_or_chunks is None:
        return ""
    return _response_text(response_or_chunks)


def extract_usage(response_or_chunks: Any) -> dict[str, int]:
    response = None
    if isinstance(response_or_chunks, list):
        for chunk in reversed(response_or_chunks):
            if _field(chunk, USAGE_METADATA_KEY) is not None:
                response = chunk
                break
        if response is None and response_or_chunks:
            response = response_or_chunks[-1]
    else:
        response = response_or_chunks

    usage = _field(response, USAGE_METADATA_KEY)
    if usage is None:
        return {}

    result: dict[str, int] = {}
    prompt_tokens = _field(usage, PROMPT_TOKEN_COUNT_KEY)
    completion_tokens = _field(usage, CANDIDATES_TOKEN_COUNT_KEY)
    total_tokens = _field(usage, TOTAL_TOKEN_COUNT_KEY)
    if isinstance(prompt_tokens, int):
        result[PROMPT_TOKEN_COUNT_KEY] = prompt_tokens
    if isinstance(completion_tokens, int):
        result[CANDIDATES_TOKEN_COUNT_KEY] = completion_tokens
    if isinstance(total_tokens, int):
        result[TOTAL_TOKEN_COUNT_KEY] = total_tokens
    return result


def extract_tool_calls(response_or_chunks: Any) -> list[dict[str, Any]]:
    chunks = (
        response_or_chunks
        if isinstance(response_or_chunks, list)
        else [response_or_chunks]
    )
    tool_calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response in chunks:
        if response is None:
            continue
        for content in _candidate_contents(response):
            parts = _field(content, PARTS_KEY, []) or []
            for part in parts:
                function_call = _field(part, FUNCTION_CALL_KEY)
                if function_call is None:
                    continue
                normalized = _normalize_function_call(function_call)
                function = normalized.get(FUNCTION_KEY)
                if not isinstance(function, dict) or not function.get(NAME_KEY):
                    continue
                signature = safe_json(value=normalized)
                if signature in seen:
                    continue
                seen.add(signature)
                tool_calls.append(normalized)
    return tool_calls


def _function_declaration_tool(definition: Any) -> dict[str, Any] | None:
    name = _field(definition, NAME_KEY)
    if not name:
        return None
    function: dict[str, Any] = {NAME_KEY: name}
    description = _field(definition, "description")
    if description:
        function["description"] = description
    parameters = _field(definition, PARAMETERS_KEY)
    if parameters is not None:
        function[PARAMETERS_KEY] = _dump_value(parameters)
    return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function}


def extract_tools(tools: Any) -> list[dict[str, Any]]:
    if not tools:
        return []

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        declarations = _field(tool, FUNCTION_DECLARATIONS_KEY) or []
        if isinstance(tool, dict) and FUNCTION_DECLARATIONS_KEY not in tool:
            normalized = _function_declaration_tool(tool)
            if normalized is not None:
                normalized_tools.append(normalized)
        for declaration in declarations:
            normalized = _function_declaration_tool(declaration)
            if normalized is not None:
                normalized_tools.append(normalized)

    return normalized_tools


def _first_arg(args: tuple[Any, ...]) -> Any:
    return args[0] if args else None


def _model_name(instance: Any) -> str | None:
    for attr_name in ("_model_name", "model_name", "_model_id", "model_id"):
        value = getattr(instance, attr_name, None)
        if value:
            return str(value)
    nested_model = getattr(instance, "model", None)
    if nested_model is not None and nested_model is not instance:
        return _model_name(nested_model)
    return None


def _system_instruction(instance: Any) -> Any:
    for attr_name in (SYSTEM_INSTRUCTION_KEY, f"_{SYSTEM_INSTRUCTION_KEY}"):
        value = getattr(instance, attr_name, None)
        if value is not None:
            return value
    nested_model = getattr(instance, "model", None)
    if nested_model is not None and nested_model is not instance:
        return _system_instruction(nested_model)
    return None


def _tools(instance: Any, kwargs: dict[str, Any]) -> Any:
    if kwargs.get(TOOLS_KEY) is not None:
        return kwargs.get(TOOLS_KEY)
    for attr_name in (TOOLS_KEY, f"_{TOOLS_KEY}"):
        value = getattr(instance, attr_name, None)
        if value is not None:
            return value
    nested_model = getattr(instance, "model", None)
    if nested_model is not None and nested_model is not instance:
        return _tools(nested_model, kwargs)
    return None


def request_payload_from_call(
    *,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    contents = kwargs.get("contents", kwargs.get("content", _first_arg(args)))
    return {
        "model": _model_name(instance),
        "contents": contents,
        GENERATION_CONFIG_KEY: kwargs.get(GENERATION_CONFIG_KEY),
        TOOL_CONFIG_KEY: kwargs.get(TOOL_CONFIG_KEY),
        TOOLS_KEY: _tools(instance, kwargs),
        STREAM_KEY: bool(kwargs.get(STREAM_KEY, False)),
        SYSTEM_INSTRUCTION_KEY: _system_instruction(instance),
    }
