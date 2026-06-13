"""Translate Ollama SDK payloads into Respan span fields."""

from __future__ import annotations

import inspect
import json
from typing import Any, Iterable

from respan_instrumentation_ollama._constants import (
    ARGUMENTS_KEY,
    CONTENT_KEY,
    DESCRIPTION_KEY,
    EMBEDDING_KEY,
    EMBEDDINGS_KEY,
    EVAL_COUNT_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    ID_KEY,
    INPUT_KEY,
    MESSAGE_KEY,
    MESSAGES_KEY,
    MODEL_KEY,
    NAME_KEY,
    PARAMETERS_KEY,
    PROMPT_EVAL_COUNT_KEY,
    RESPONSE_KEY,
    ROLE_KEY,
    SYSTEM_KEY,
    TOOL_CALLS_KEY,
    TOOL_NAME_KEY,
    TOOLS_KEY,
    TYPE_KEY,
    USER_ROLE,
)
from respan_sdk.utils.serialization import serialize_value


def safe_json(value: Any) -> str:
    """JSON-encode values after applying the SDK serializer."""
    try:
        return json.dumps(serialize_value(value=value), default=str)
    except Exception:
        return str(value)


def to_json_attr(value: Any) -> str:
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except Exception:
            pass
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


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def normalize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if not isinstance(messages, (list, tuple)):
        messages = [messages]

    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        role = _field(message, ROLE_KEY, USER_ROLE) or USER_ROLE
        normalized: dict[str, Any] = {ROLE_KEY: str(role)}

        content = _field(message, CONTENT_KEY)
        if content is not None:
            normalized[CONTENT_KEY] = _stringify_content(content)

        tool_name = _field(message, TOOL_NAME_KEY)
        if tool_name is not None:
            normalized[TOOL_NAME_KEY] = str(tool_name)

        tool_calls = normalize_tool_calls(_field(message, TOOL_CALLS_KEY))
        if tool_calls:
            normalized[TOOL_CALLS_KEY] = tool_calls

        normalized_messages.append(normalized)
    return normalized_messages


def normalize_generate_messages(
    *, prompt: Any, system: Any = None
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system not in {None, ""}:
        messages.append({ROLE_KEY: SYSTEM_KEY, CONTENT_KEY: _stringify_content(system)})
    if prompt not in {None, ""}:
        messages.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: _stringify_content(prompt)})
    return messages


def format_chat_input(*, messages: Any, tools: Any = None) -> str:
    payload: dict[str, Any] = {MESSAGES_KEY: normalize_chat_messages(messages)}
    normalized_tools = normalize_tools(tools)
    if normalized_tools:
        payload[TOOLS_KEY] = normalized_tools
    return safe_json(payload)


def format_generate_input(*, prompt: Any, system: Any = None) -> str:
    return safe_json(normalize_generate_messages(prompt=prompt, system=system))


def format_embedding_input(*, prompt: Any = None, input_value: Any = None) -> str:
    value = input_value if input_value is not None else prompt
    return safe_json({INPUT_KEY: _dump_value(value)})


def _iter_responses(response_or_chunks: Any) -> Iterable[Any]:
    if response_or_chunks is None:
        return ()
    if isinstance(response_or_chunks, list):
        return (chunk for chunk in response_or_chunks if chunk is not None)
    return (response_or_chunks,)


def _last_response(response_or_chunks: Any) -> Any:
    if isinstance(response_or_chunks, list):
        for chunk in reversed(response_or_chunks):
            if chunk is not None:
                return chunk
        return None
    return response_or_chunks


def format_chat_output(response_or_chunks: Any) -> str:
    parts: list[str] = []
    for response in _iter_responses(response_or_chunks):
        message = _field(response, MESSAGE_KEY)
        content = _field(message, CONTENT_KEY)
        if content:
            parts.append(str(content))
    return "".join(parts)


def format_generate_output(response_or_chunks: Any) -> str:
    parts: list[str] = []
    for response in _iter_responses(response_or_chunks):
        content = _field(response, RESPONSE_KEY)
        if content:
            parts.append(str(content))
    return "".join(parts)


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    if not isinstance(tool_calls, (list, tuple)):
        tool_calls = [tool_calls]

    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = _field(tool_call, FUNCTION_KEY)
        name = _field(function, NAME_KEY) or _field(tool_call, NAME_KEY)
        if not name:
            continue

        normalized_function: dict[str, Any] = {NAME_KEY: str(name)}
        arguments = _field(function, ARGUMENTS_KEY)
        if arguments is None:
            arguments = _field(tool_call, ARGUMENTS_KEY)
        if arguments is not None:
            normalized_function[ARGUMENTS_KEY] = to_json_attr(_dump_value(arguments))

        normalized: dict[str, Any] = {
            TYPE_KEY: FUNCTION_TOOL_TYPE,
            FUNCTION_KEY: normalized_function,
        }
        call_id = _field(tool_call, ID_KEY)
        if call_id:
            normalized[ID_KEY] = str(call_id)
        normalized_calls.append(normalized)
    return normalized_calls


def extract_chat_tool_calls(response_or_chunks: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response in _iter_responses(response_or_chunks):
        message = _field(response, MESSAGE_KEY)
        for tool_call in normalize_tool_calls(_field(message, TOOL_CALLS_KEY)):
            signature = safe_json(tool_call)
            if signature in seen:
                continue
            seen.add(signature)
            tool_calls.append(tool_call)
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
    if annotation is list:
        return {TYPE_KEY: "array"}
    if annotation is dict:
        return {TYPE_KEY: "object"}
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


def _mapping_tool_definition(tool: Any) -> dict[str, Any] | None:
    dumped = _dump_value(tool)
    if not isinstance(dumped, dict):
        return None

    function = dumped.get(FUNCTION_KEY)
    if isinstance(function, dict) and function.get(NAME_KEY):
        normalized_function: dict[str, Any] = {NAME_KEY: function[NAME_KEY]}
        if function.get(DESCRIPTION_KEY):
            normalized_function[DESCRIPTION_KEY] = function[DESCRIPTION_KEY]
        if function.get(PARAMETERS_KEY) is not None:
            normalized_function[PARAMETERS_KEY] = function[PARAMETERS_KEY]
        return {
            TYPE_KEY: dumped.get(TYPE_KEY, FUNCTION_TOOL_TYPE),
            FUNCTION_KEY: normalized_function,
        }

    name = dumped.get(NAME_KEY)
    if name:
        normalized_function = {NAME_KEY: name}
        if dumped.get(DESCRIPTION_KEY):
            normalized_function[DESCRIPTION_KEY] = dumped[DESCRIPTION_KEY]
        if dumped.get(PARAMETERS_KEY) is not None:
            normalized_function[PARAMETERS_KEY] = dumped[PARAMETERS_KEY]
        return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: normalized_function}

    return None


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if not tools:
        return []
    if not isinstance(tools, (list, tuple)):
        tools = [tools]

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        if callable(tool):
            normalized_tools.append(_callable_tool_definition(tool))
            continue
        normalized = _mapping_tool_definition(tool)
        if normalized is not None:
            normalized_tools.append(normalized)
    return normalized_tools


def extract_usage(response_or_chunks: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    responses = list(_iter_responses(response_or_chunks))
    for response in reversed(responses):
        if PROMPT_EVAL_COUNT_KEY not in result:
            prompt_tokens = _field(response, PROMPT_EVAL_COUNT_KEY)
            if isinstance(prompt_tokens, int):
                result[PROMPT_EVAL_COUNT_KEY] = prompt_tokens
        if EVAL_COUNT_KEY not in result:
            completion_tokens = _field(response, EVAL_COUNT_KEY)
            if isinstance(completion_tokens, int):
                result[EVAL_COUNT_KEY] = completion_tokens
        if PROMPT_EVAL_COUNT_KEY in result and EVAL_COUNT_KEY in result:
            break
    return result


def extract_model(
    *, request_kwargs: dict[str, Any], response_or_chunks: Any = None
) -> str | None:
    model = request_kwargs.get(MODEL_KEY)
    if model:
        return str(model)
    response = _last_response(response_or_chunks)
    response_model = _field(response, MODEL_KEY)
    if response_model:
        return str(response_model)
    return None


def has_embedding_payload(response: Any) -> bool:
    return (
        _field(response, EMBEDDING_KEY) is not None
        or _field(response, EMBEDDINGS_KEY) is not None
    )
