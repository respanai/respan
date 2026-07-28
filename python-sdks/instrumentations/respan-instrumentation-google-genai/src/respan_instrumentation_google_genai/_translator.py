"""Translate Google Gen AI SDK payloads into Respan span fields."""

from __future__ import annotations

import inspect
import json
from typing import Any, Iterable

from respan_instrumentation_google_genai._constants import (
    ARGS_KEY,
    AUTOMATIC_FUNCTION_CALLING_HISTORY_KEY,
    BUILTIN_TOOL_FIELDS,
    CANDIDATES_KEY,
    CANDIDATES_TOKEN_COUNT_KEY,
    THOUGHTS_TOKEN_COUNT_KEY,
    CONFIG_KEY,
    CONTENT_KEY,
    CONTENTS_KEY,
    DESCRIPTION_KEY,
    FUNCTION_CALL_KEY,
    FUNCTION_DECLARATIONS_KEY,
    FUNCTION_KEY,
    FUNCTION_RESPONSE_KEY,
    FUNCTION_TOOL_TYPE,
    ID_KEY,
    MODEL_KEY,
    MODEL_ROLE,
    NAME_KEY,
    PARAMETERS_JSON_SCHEMA_KEY,
    PARAMETERS_KEY,
    PARTS_KEY,
    PROMPT_TOKEN_COUNT_KEY,
    RESPONSE_KEY,
    ROLE_KEY,
    SYSTEM_ROLE,
    TEXT_KEY,
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
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True, by_alias=False)
        except TypeError:
            return model_dump()
    return serialize_value(value=value)


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
    response_id = _field(value, ID_KEY)
    if response_id:
        result[ID_KEY] = response_id
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
        for key in ("inline_data", "file_data", "code_execution_result", "tool_call"):
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
        _normalize_part(part)
        for part in parts
        if _normalize_part(part) not in (None, "", [], {})
    ]
    if not normalized_parts:
        return ""
    if len(normalized_parts) == 1:
        return normalized_parts[0]
    if all(isinstance(part, str) for part in normalized_parts):
        return "\n".join(normalized_parts)
    return normalized_parts


def _normalize_content(content: Any, *, default_role: str = USER_ROLE) -> dict[str, Any]:
    if isinstance(content, str):
        return {ROLE_KEY: default_role, CONTENT_KEY: content}
    if _is_part_like(content) and not _is_content_like(content):
        return {ROLE_KEY: default_role, CONTENT_KEY: _normalize_part(content)}

    role = _field(content, ROLE_KEY, default_role) or default_role
    parts = _field(content, PARTS_KEY)
    return {ROLE_KEY: role, CONTENT_KEY: _normalize_parts(parts)}


def normalize_input_messages(contents: Any, config: Any = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    system_instruction = _field(config, "system_instruction")
    if system_instruction is not None:
        messages.append(
            _normalize_content(system_instruction, default_role=SYSTEM_ROLE)
            if _is_content_like(system_instruction)
            else {ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: _normalize_parts(system_instruction)}
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


def format_input(contents: Any, config: Any = None) -> str:
    return safe_json(value=normalize_input_messages(contents=contents, config=config))


def _candidate_contents(response: Any) -> Iterable[Any]:
    for candidate in _field(response, CANDIDATES_KEY, []) or []:
        content = _field(candidate, CONTENT_KEY)
        if content is not None:
            yield content


def _response_text(response: Any) -> str:
    text = _field(response, TEXT_KEY)
    if isinstance(text, str):
        return text
    text_property = getattr(response, TEXT_KEY, None)
    if isinstance(text_property, str):
        return text_property

    text_parts: list[str] = []
    for content in _candidate_contents(response):
        content_value = _normalize_content(content, default_role=MODEL_ROLE).get(
            CONTENT_KEY, ""
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
    thoughts_tokens = _field(usage, THOUGHTS_TOKEN_COUNT_KEY)
    total_tokens = _field(usage, TOTAL_TOKEN_COUNT_KEY)
    if isinstance(prompt_tokens, int):
        result[PROMPT_TOKEN_COUNT_KEY] = prompt_tokens
    if isinstance(completion_tokens, int):
        # Gemini reports thinking tokens separately from candidatesTokenCount and
        # bills them at the output rate, so they belong in the output count. This
        # matches the google-adk instrumentation, which already folds them in.
        result[CANDIDATES_TOKEN_COUNT_KEY] = completion_tokens + (
            thoughts_tokens if isinstance(thoughts_tokens, int) else 0
        )
    if isinstance(total_tokens, int):
        result[TOTAL_TOKEN_COUNT_KEY] = total_tokens
    return result


def _iter_response_contents(response_or_chunks: Any) -> Iterable[Any]:
    chunks = response_or_chunks if isinstance(response_or_chunks, list) else [response_or_chunks]
    for response in chunks:
        if response is None:
            continue
        yield from _candidate_contents(response)
        history = _field(response, AUTOMATIC_FUNCTION_CALLING_HISTORY_KEY, []) or []
        for content in history:
            yield content


def extract_tool_calls(response_or_chunks: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for content in _iter_response_contents(response_or_chunks):
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


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    if annotation in (str, inspect.Signature.empty):
        return {TYPE_KEY: "string"}
    if annotation is int:
        return {TYPE_KEY: "integer"}
    if annotation is float:
        return {TYPE_KEY: "number"}
    if annotation is bool:
        return {TYPE_KEY: "boolean"}
    if annotation in (dict, list):
        return {TYPE_KEY: "object" if annotation is dict else "array"}
    return {TYPE_KEY: "string"}


def _callable_tool_definition(tool: Any) -> dict[str, Any]:
    function: dict[str, Any] = {
        NAME_KEY: getattr(tool, "__name__", tool.__class__.__name__),
    }
    doc = inspect.getdoc(tool)
    if doc:
        function[DESCRIPTION_KEY] = doc

    try:
        signature = inspect.signature(tool)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, parameter in signature.parameters.items():
            if param_name in {"self", "cls"}:
                continue
            properties[param_name] = _annotation_to_json_schema(parameter.annotation)
            if parameter.default is inspect.Signature.empty:
                required.append(param_name)
        if properties:
            parameters: dict[str, Any] = {TYPE_KEY: "object", "properties": properties}
            if required:
                parameters["required"] = required
            function[PARAMETERS_KEY] = parameters

    return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function}


def _function_declaration_tool(definition: Any) -> dict[str, Any] | None:
    name = _field(definition, NAME_KEY)
    if not name:
        return None
    function: dict[str, Any] = {NAME_KEY: name}
    description = _field(definition, DESCRIPTION_KEY)
    if description:
        function[DESCRIPTION_KEY] = description
    parameters = _field(definition, PARAMETERS_JSON_SCHEMA_KEY)
    if parameters is None:
        parameters = _field(definition, PARAMETERS_KEY)
    if parameters is not None:
        function[PARAMETERS_KEY] = _dump_value(parameters)
    return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function}


def extract_tools(config: Any) -> list[dict[str, Any]]:
    tools = _field(config, TOOLS_KEY)
    if not tools:
        return []

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        if callable(tool):
            normalized_tools.append(_callable_tool_definition(tool))
            continue

        declarations = _field(tool, FUNCTION_DECLARATIONS_KEY) or []
        for declaration in declarations:
            normalized = _function_declaration_tool(declaration)
            if normalized is not None:
                normalized_tools.append(normalized)

        for field_name in BUILTIN_TOOL_FIELDS:
            value = _field(tool, field_name)
            if value is not None:
                normalized_tools.append({TYPE_KEY: field_name, field_name: _dump_value(value)})

    return normalized_tools


def request_kwargs_from_call(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        MODEL_KEY: kwargs.get(MODEL_KEY),
        CONTENTS_KEY: kwargs.get(CONTENTS_KEY),
        CONFIG_KEY: kwargs.get(CONFIG_KEY),
    }
