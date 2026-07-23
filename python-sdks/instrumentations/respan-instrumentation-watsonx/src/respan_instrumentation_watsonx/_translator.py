"""Translate IBM watsonx.ai SDK payloads into canonical span fields."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Mapping
from typing import Any

from respan_instrumentation_watsonx._constants import (
    ASSISTANT_ROLE,
    CHOICES_KEY,
    COMPLETION_TOKENS_KEY,
    CONTENT_KEY,
    DELTA_KEY,
    EMBEDDING_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    GENERATED_TEXT_KEY,
    GENERATED_TOKEN_COUNT_KEY,
    INPUT_TOKEN_COUNT_KEY,
    MESSAGE_KEY,
    MODEL_ID_KEY,
    NAME_KEY,
    PROMPT_TOKENS_KEY,
    RESULTS_KEY,
    ROLE_KEY,
    TEXT_KEY,
    TOOL_CALLS_KEY,
    TOTAL_TOKENS_KEY,
    TYPE_KEY,
    USAGE_KEY,
    USER_ROLE,
)
from respan_sdk.utils.serialization import serialize_value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name)
    except Exception:
        return default


def _dump_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _dump_value(nested_value)
            for key, nested_value in value.items()
            if nested_value is not None
        }
    if isinstance(value, list | tuple | set):
        return [_dump_value(item) for item in value if item is not None]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True, by_alias=False)
        except TypeError:
            return model_dump()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return serialize_value(value=value)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(_dump_value(value), default=str)
    except Exception:
        return str(value)


def to_attr_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return safe_json(value)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json(value)


def normalize_text_prompts(prompt: Any) -> list[dict[str, Any]]:
    if prompt is None:
        return []
    if isinstance(prompt, list | tuple):
        return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _coerce_text(item)} for item in prompt]
    return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _coerce_text(prompt)}]


def normalize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, str):
        return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: messages}]
    if isinstance(messages, Mapping):
        return [_normalize_message(messages)]
    if isinstance(messages, list | tuple):
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, Mapping):
                normalized.append(_normalize_message(message))
            else:
                normalized.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: _coerce_text(message)})
        return normalized
    return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _coerce_text(messages)}]


def _normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = message.get(ROLE_KEY) or USER_ROLE
    content = message.get(CONTENT_KEY)
    normalized: dict[str, Any] = {ROLE_KEY: role, CONTENT_KEY: _dump_value(content)}
    tool_calls = message.get(TOOL_CALLS_KEY)
    if tool_calls:
        normalized[TOOL_CALLS_KEY] = normalize_tool_calls(tool_calls)
    return normalized


def format_input_messages(messages: list[dict[str, Any]]) -> str:
    return safe_json(messages)


def _first_choice(response: Any) -> Any:
    choices = _field(response, CHOICES_KEY, []) or []
    if isinstance(choices, list | tuple) and choices:
        return choices[0]
    return None


def _extract_choice_message(response: Any) -> Any:
    choice = _first_choice(response)
    if choice is None:
        return None
    return _field(choice, MESSAGE_KEY)


def _extract_choice_delta(response: Any) -> Any:
    choice = _first_choice(response)
    if choice is None:
        return None
    return _field(choice, DELTA_KEY)


def _iter_results(response: Any) -> Iterable[Any]:
    results = _field(response, RESULTS_KEY, []) or []
    if isinstance(results, list | tuple):
        yield from results


def _text_from_generation_response(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, list | tuple):
        return "\n".join(_text_from_generation_response(item) for item in response)
    for result in _iter_results(response):
        generated_text = _field(result, GENERATED_TEXT_KEY)
        if generated_text is not None:
            return _coerce_text(generated_text)
    generated_text = _field(response, GENERATED_TEXT_KEY)
    if generated_text is not None:
        return _coerce_text(generated_text)
    text = _field(response, TEXT_KEY)
    if text is not None:
        return _coerce_text(text)
    return ""


def format_text_output(response_or_chunks: Any) -> str:
    if isinstance(response_or_chunks, list):
        return "".join(_text_from_generation_response(chunk) for chunk in response_or_chunks)
    return _text_from_generation_response(response_or_chunks)


def _chat_content_from_response(response: Any) -> str:
    message = _extract_choice_message(response)
    if message is not None:
        return _coerce_text(_field(message, CONTENT_KEY))
    delta = _extract_choice_delta(response)
    if delta is not None:
        return _coerce_text(_field(delta, CONTENT_KEY))
    return _text_from_generation_response(response)


def format_chat_output(response_or_chunks: Any) -> str:
    if isinstance(response_or_chunks, list):
        return "".join(_chat_content_from_response(chunk) for chunk in response_or_chunks)
    return _chat_content_from_response(response_or_chunks)


def _usage_sources(response_or_chunks: Any) -> Iterable[Any]:
    responses = response_or_chunks if isinstance(response_or_chunks, list) else [response_or_chunks]
    for response in reversed(responses):
        if response is None:
            continue
        usage = _field(response, USAGE_KEY)
        if usage is not None:
            yield usage
        for result in _iter_results(response):
            yield result
        yield response


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


def extract_usage(response_or_chunks: Any) -> dict[str, int]:
    for source in _usage_sources(response_or_chunks):
        prompt_tokens = _coerce_int(_field(source, PROMPT_TOKENS_KEY))
        if prompt_tokens is None:
            prompt_tokens = _coerce_int(_field(source, INPUT_TOKEN_COUNT_KEY))

        completion_tokens = _coerce_int(_field(source, COMPLETION_TOKENS_KEY))
        if completion_tokens is None:
            completion_tokens = _coerce_int(_field(source, GENERATED_TOKEN_COUNT_KEY))

        total_tokens = _coerce_int(_field(source, TOTAL_TOKENS_KEY))
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        result: dict[str, int] = {}
        if prompt_tokens is not None:
            result[PROMPT_TOKENS_KEY] = prompt_tokens
        if completion_tokens is not None:
            result[COMPLETION_TOKENS_KEY] = completion_tokens
        if total_tokens is not None:
            result[TOTAL_TOKENS_KEY] = total_tokens
        if result:
            return result
    return {}


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
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
    function: dict[str, Any] = {NAME_KEY: getattr(tool, "__name__", tool.__class__.__name__)}
    doc = inspect.getdoc(tool)
    if doc:
        function["description"] = doc
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
            function["parameters"] = parameters
    return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function}


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    if not tools:
        return []
    values = tools if isinstance(tools, list | tuple) else [tools]
    normalized: list[dict[str, Any]] = []
    for tool in values:
        if callable(tool):
            normalized.append(_callable_tool_definition(tool))
            continue
        dumped = _dump_value(tool)
        if not isinstance(dumped, Mapping):
            continue
        if dumped.get(TYPE_KEY) == FUNCTION_TOOL_TYPE and isinstance(
            dumped.get(FUNCTION_KEY), Mapping
        ):
            normalized.append(dict(dumped))
            continue
        function = dumped.get(FUNCTION_KEY)
        if isinstance(function, Mapping):
            normalized.append({TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: dict(function)})
            continue
        name = dumped.get(NAME_KEY)
        if name:
            function_payload = {NAME_KEY: name}
            for key in ("description", "parameters", "schema", "input_schema"):
                if dumped.get(key) is not None:
                    target_key = "parameters" if key in {"schema", "input_schema"} else key
                    function_payload[target_key] = dumped[key]
            normalized.append({TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function_payload})
    return normalized


def _normalize_single_tool_call(tool_call: Any) -> dict[str, Any] | None:
    dumped = _dump_value(tool_call)
    if not isinstance(dumped, Mapping):
        return None
    function = dumped.get(FUNCTION_KEY)
    if not isinstance(function, Mapping):
        name = dumped.get(NAME_KEY)
        if not name:
            return None
        function = {NAME_KEY: name}
        if "arguments" in dumped:
            function["arguments"] = dumped["arguments"]
    if not function.get(NAME_KEY):
        return None
    normalized_function = dict(function)
    if "arguments" in normalized_function:
        normalized_function["arguments"] = to_attr_value(normalized_function["arguments"])
    result = {
        TYPE_KEY: dumped.get(TYPE_KEY) or FUNCTION_TOOL_TYPE,
        FUNCTION_KEY: normalized_function,
    }
    call_id = dumped.get("id")
    if call_id:
        result["id"] = call_id
    return result


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    values = tool_calls if isinstance(tool_calls, list | tuple) else [tool_calls]
    normalized = []
    for tool_call in values:
        value = _normalize_single_tool_call(tool_call)
        if value is not None:
            normalized.append(value)
    return normalized


def extract_chat_tool_calls(response_or_chunks: Any) -> list[dict[str, Any]]:
    responses = response_or_chunks if isinstance(response_or_chunks, list) else [response_or_chunks]
    tool_calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response in responses:
        for source in (_extract_choice_message(response), _extract_choice_delta(response)):
            for tool_call in normalize_tool_calls(_field(source, TOOL_CALLS_KEY)):
                signature = safe_json(tool_call)
                if signature in seen:
                    continue
                seen.add(signature)
                tool_calls.append(tool_call)
    return tool_calls


def normalize_embedding_inputs(inputs: Any) -> list[str]:
    if inputs is None:
        return []
    if isinstance(inputs, str):
        return [inputs]
    if isinstance(inputs, list | tuple):
        return [_coerce_text(item) for item in inputs]
    return [_coerce_text(inputs)]


def embedding_vector_count(response: Any) -> int | None:
    if response is None:
        return None
    if isinstance(response, list | tuple):
        if not response:
            return 0
        if all(isinstance(item, int | float) for item in response):
            return 1
        return len(response)
    results = list(_iter_results(response))
    if results:
        return len(results)
    embedding = _field(response, EMBEDDING_KEY)
    if isinstance(embedding, list | tuple):
        return 1
    return None


def embedding_dimension(response: Any) -> int | None:
    vectors: list[Any] = []
    if isinstance(response, list | tuple):
        if response and all(isinstance(item, int | float) for item in response):
            vectors.append(response)
        else:
            vectors.extend(response)
    else:
        for result in _iter_results(response):
            embedding = _field(result, EMBEDDING_KEY)
            if embedding is not None:
                vectors.append(embedding)
        embedding = _field(response, EMBEDDING_KEY)
        if embedding is not None:
            vectors.append(embedding)
    for vector in vectors:
        if isinstance(vector, list | tuple):
            return len(vector)
    return None


def model_id_from_instance(instance: Any) -> str | None:
    for key in (MODEL_ID_KEY, "_model_id", "deployment_id", "_deployment_id"):
        value = _field(instance, key)
        if value:
            return str(value)
    return None
