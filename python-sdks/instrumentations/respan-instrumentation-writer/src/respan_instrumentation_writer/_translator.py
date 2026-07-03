"""Writer SDK request/response normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_writer._constants import (
    ANSWER_KEY,
    APPLICATION_ID_KEY,
    ARGUMENTS_KEY,
    ASSISTANT_ROLE,
    CHOICES_KEY,
    CONTENT_KEY,
    DATA_KEY,
    DELTA_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    GRAPH_IDS_KEY,
    ID_KEY,
    INDEX_KEY,
    INPUTS_KEY,
    MESSAGE_KEY,
    MESSAGES_KEY,
    MODEL_KEY,
    NAME_KEY,
    PROMPT_KEY,
    QUESTION_KEY,
    ROLE_KEY,
    SUGGESTION_KEY,
    TEXT_KEY,
    TOOL_CALLS_KEY,
    TOOLS_KEY,
    TYPE_KEY,
    USAGE_KEY,
    USER_ROLE,
    VALUE_KEY,
    WRITER_APPLICATION_GENERATE_SPAN_NAME,
    WRITER_APPLICATION_MODEL_NAME,
    WRITER_CHAT_SPAN_NAME,
    WRITER_COMPLETION_SPAN_NAME,
    WRITER_GRAPH_MODEL_NAME,
    WRITER_GRAPH_QUESTION_SPAN_NAME,
    WRITER_SYSTEM_NAME,
    WRITER_TRANSLATION_SPAN_NAME,
    WRITER_VISION_SPAN_NAME,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TEXT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

_WRITER_SENTINEL_CLASS_NAMES = {"Omit", "NotGiven"}
_REQUEST_CONTROL_KEYS = {
    "extra_body",
    "extra_headers",
    "extra_query",
    "timeout",
}


def _is_omitted(value: Any) -> bool:
    return type(value).__name__ in _WRITER_SENTINEL_CLASS_NAMES


def _to_plain(value: Any) -> Any:
    if _is_omitted(value):
        return None

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", by_alias=True, exclude_none=True)
        except TypeError:
            return model_dump()

    if isinstance(value, Mapping):
        return {
            str(key): _to_plain(nested_value)
            for key, nested_value in value.items()
            if not _is_omitted(nested_value)
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value if not _is_omitted(item)]

    if hasattr(value, "__dict__"):
        return {
            str(key): _to_plain(nested_value)
            for key, nested_value in vars(value).items()
            if not key.startswith("_") and not _is_omitted(nested_value)
        }

    return value


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_empty(nested_value)
            for key, nested_value in value.items()
            if nested_value is not None and not _is_omitted(nested_value)
        }
    if isinstance(value, list):
        return [_drop_empty(item) for item in value if item is not None]
    return value


def safe_json(value: Any) -> str:
    return json.dumps(_to_plain(value), default=str)


def _json_content_attr(value: Any) -> str:
    plain_value = _to_plain(value)
    if plain_value is None:
        return ""
    if isinstance(plain_value, str):
        return plain_value
    return safe_json(plain_value)


def _public_request_kwargs(request_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _to_plain(value)
        for key, value in request_kwargs.items()
        if key not in _REQUEST_CONTROL_KEYS and not _is_omitted(value)
    }


def _as_list(value: Any) -> list[Any]:
    if value is None or _is_omitted(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)



def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    for message in _as_list(messages):
        plain_message = _to_plain(message)
        if isinstance(plain_message, Mapping):
            normalized_message = {
                ROLE_KEY: plain_message.get(ROLE_KEY, USER_ROLE),
                CONTENT_KEY: _to_plain(plain_message.get(CONTENT_KEY, "")),
            }
            tool_calls = plain_message.get(TOOL_CALLS_KEY)
            if tool_calls:
                normalized_message[TOOL_CALLS_KEY] = _normalize_tool_calls(tool_calls)
            normalized_messages.append(_drop_empty(normalized_message))
        else:
            normalized_messages.append(
                {ROLE_KEY: USER_ROLE, CONTENT_KEY: _to_plain(plain_message)}
            )
    return normalized_messages


def _normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    plain_tool_call = _to_plain(tool_call)
    if not isinstance(plain_tool_call, Mapping):
        return {}

    function = plain_tool_call.get(FUNCTION_KEY) or {}
    if not isinstance(function, Mapping):
        function = {}

    arguments = function.get(ARGUMENTS_KEY, "")
    if not isinstance(arguments, str):
        arguments = safe_json(arguments)

    return _drop_empty(
        {
            ID_KEY: plain_tool_call.get(ID_KEY),
            TYPE_KEY: plain_tool_call.get(TYPE_KEY, FUNCTION_TOOL_TYPE),
            INDEX_KEY: plain_tool_call.get(INDEX_KEY),
            FUNCTION_KEY: {
                NAME_KEY: function.get(NAME_KEY),
                ARGUMENTS_KEY: arguments,
            },
        }
    )


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    return [
        tool_call
        for tool_call in (_normalize_tool_call(item) for item in _as_list(tool_calls))
        if tool_call
    ]


def _normalize_tools(tools: Any) -> list[Any]:
    return [_drop_empty(_to_plain(tool)) for tool in _as_list(tools)]


def _base_llm_attrs(
    *,
    span_name: str,
    log_type: str,
    request_type: str,
) -> dict[str, Any]:
    attrs = {
        TLSpanAttributes.LLM_SYSTEM: WRITER_SYSTEM_NAME,
        TLSpanAttributes.LLM_REQUEST_TYPE: request_type,
        TLSpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        RESPAN_LOG_TYPE: log_type,
    }
    workflow_name = context_api.get_value(TLSpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_prompt_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    if messages:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(messages)
    for index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        if role is not None:
            attrs[f"{TLSpanAttributes.LLM_PROMPTS}.{index}.role"] = str(role)
        if content is not None:
            attrs[f"{TLSpanAttributes.LLM_PROMPTS}.{index}.content"] = (
                _json_content_attr(content)
            )
        tool_calls = message.get(TOOL_CALLS_KEY)
        if tool_calls:
            attrs[f"{TLSpanAttributes.LLM_PROMPTS}.{index}.tool_calls"] = safe_json(
                tool_calls
            )


def _set_single_prompt_attrs(
    attrs: dict[str, Any],
    *,
    content: Any,
    role: str = USER_ROLE,
) -> None:
    prompt_content = _json_content_attr(content)
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = prompt_content
    attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.role"] = role
    attrs[f"{TLSpanAttributes.LLM_PROMPTS}.0.content"] = prompt_content


def _set_usage_attrs(attrs: dict[str, Any], usage: Any) -> None:
    if usage is None or _is_omitted(usage):
        return

    prompt_tokens = _get_value(usage, "prompt_tokens")
    completion_tokens = _get_value(usage, "completion_tokens")
    total_tokens = _get_value(usage, "total_tokens")

    if prompt_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if total_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens

    prompt_details = _get_value(usage, "prompt_token_details")
    cached_tokens = _get_value(prompt_details, "cached_tokens")
    if cached_tokens:
        attrs[TLSpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cached_tokens


def _choice_message(choice: Any) -> Any:
    message = _get_value(choice, MESSAGE_KEY)
    if message is not None:
        return message
    return _get_value(choice, DELTA_KEY)


def _set_completion_attrs(
    attrs: dict[str, Any],
    completions: list[dict[str, Any]],
) -> None:
    output_values: list[Any] = []
    for index, completion in enumerate(completions):
        role = completion.get(ROLE_KEY, ASSISTANT_ROLE)
        content = completion.get(CONTENT_KEY, "")
        tool_calls = completion.get(TOOL_CALLS_KEY)

        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.{index}.role"] = str(role)
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.{index}.content"] = (
            _json_content_attr(content)
        )
        if tool_calls:
            attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.{index}.tool_calls"] = (
                safe_json(tool_calls)
            )
        output_values.append(content)

    if not output_values:
        return
    if len(output_values) == 1:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_content_attr(
            output_values[0]
        )
    else:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(output_values)


def _chat_completions_from_response(response: Any) -> list[dict[str, Any]]:
    completions: list[dict[str, Any]] = []
    for choice in _as_list(_get_value(response, CHOICES_KEY)):
        message = _choice_message(choice)
        content = _get_value(message, CONTENT_KEY, "")
        tool_calls = _normalize_tool_calls(_get_value(message, TOOL_CALLS_KEY))
        completions.append(
            _drop_empty(
                {
                    ROLE_KEY: _get_value(message, ROLE_KEY, ASSISTANT_ROLE),
                    CONTENT_KEY: content,
                    TOOL_CALLS_KEY: tool_calls,
                }
            )
        )
    return completions


def _merge_streaming_tool_call(
    tool_calls_by_index: dict[int, dict[str, Any]],
    tool_call: Any,
) -> None:
    plain_tool_call = _to_plain(tool_call)
    if not isinstance(plain_tool_call, Mapping):
        return

    index = plain_tool_call.get(INDEX_KEY)
    if not isinstance(index, int):
        index = len(tool_calls_by_index)

    merged = tool_calls_by_index.setdefault(
        index,
        {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: {ARGUMENTS_KEY: ""}},
    )
    if plain_tool_call.get(ID_KEY):
        merged[ID_KEY] = plain_tool_call[ID_KEY]
    if plain_tool_call.get(TYPE_KEY):
        merged[TYPE_KEY] = plain_tool_call[TYPE_KEY]

    function = plain_tool_call.get(FUNCTION_KEY)
    if isinstance(function, Mapping):
        merged_function = merged.setdefault(FUNCTION_KEY, {})
        if function.get(NAME_KEY):
            merged_function[NAME_KEY] = function[NAME_KEY]
        arguments = function.get(ARGUMENTS_KEY)
        if arguments:
            merged_function[ARGUMENTS_KEY] = (
                str(merged_function.get(ARGUMENTS_KEY, "")) + str(arguments)
            )


def _chat_completions_from_chunks(chunks: list[Any]) -> list[dict[str, Any]]:
    content_by_index: dict[int, list[str]] = {}
    role_by_index: dict[int, str] = {}
    tool_calls_by_choice: dict[int, dict[int, dict[str, Any]]] = {}

    for chunk in chunks:
        for choice in _as_list(_get_value(chunk, CHOICES_KEY)):
            index = _get_value(choice, INDEX_KEY, 0)
            if not isinstance(index, int):
                index = 0

            message = _choice_message(choice)
            if message is None:
                continue

            role = _get_value(message, ROLE_KEY)
            if role:
                role_by_index[index] = role

            content = _get_value(message, CONTENT_KEY)
            if content:
                content_by_index.setdefault(index, []).append(str(content))

            tool_calls = _get_value(message, TOOL_CALLS_KEY)
            for tool_call in _as_list(tool_calls):
                _merge_streaming_tool_call(
                    tool_calls_by_choice.setdefault(index, {}),
                    tool_call,
                )

    completions: list[dict[str, Any]] = []
    for index in sorted(set(content_by_index) | set(role_by_index) | set(tool_calls_by_choice)):
        tool_calls = [
            _drop_empty(tool_call)
            for _, tool_call in sorted(tool_calls_by_choice.get(index, {}).items())
        ]
        completions.append(
            _drop_empty(
                {
                    ROLE_KEY: role_by_index.get(index, ASSISTANT_ROLE),
                    CONTENT_KEY: "".join(content_by_index.get(index, [])),
                    TOOL_CALLS_KEY: [tool_call for tool_call in tool_calls if tool_call],
                }
            )
        )
    return completions


def _last_usage_from_chunks(chunks: list[Any]) -> Any:
    for chunk in reversed(chunks):
        usage = _get_value(chunk, USAGE_KEY)
        if usage is not None:
            return usage
    return None


def _last_model_from_chunks(chunks: list[Any]) -> str | None:
    for chunk in reversed(chunks):
        model = _get_value(chunk, MODEL_KEY)
        if model:
            return str(model)
    return None


def build_chat_attrs(
    *,
    request_kwargs: Mapping[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_llm_attrs(
        span_name=WRITER_CHAT_SPAN_NAME,
        log_type=LOG_TYPE_CHAT,
        request_type=LLMRequestTypeValues.CHAT.value,
    )

    model = _get_value(response_or_chunks, MODEL_KEY) or request_kwargs.get(MODEL_KEY)
    chunks = response_or_chunks if isinstance(response_or_chunks, list) else None
    if chunks is not None:
        model = _last_model_from_chunks(chunks) or model
    if model:
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = str(model)

    _set_prompt_attrs(
        attrs=attrs,
        messages=_normalize_messages(request_kwargs.get(MESSAGES_KEY)),
    )

    tools = _normalize_tools(request_kwargs.get(TOOLS_KEY))
    if tools:
        attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(tools)

    if chunks is not None:
        completions = _chat_completions_from_chunks(chunks)
        usage = _last_usage_from_chunks(chunks)
    elif response_or_chunks is not None:
        completions = _chat_completions_from_response(response_or_chunks)
        usage = _get_value(response_or_chunks, USAGE_KEY)
    else:
        completions = []
        usage = None

    _set_completion_attrs(attrs=attrs, completions=completions)
    _set_usage_attrs(attrs=attrs, usage=usage)
    return attrs


def build_completion_attrs(
    *,
    request_kwargs: Mapping[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_llm_attrs(
        span_name=WRITER_COMPLETION_SPAN_NAME,
        log_type=LOG_TYPE_TEXT,
        request_type=LLMRequestTypeValues.COMPLETION.value,
    )
    model = _get_value(response_or_chunks, MODEL_KEY) or request_kwargs.get(MODEL_KEY)
    if model:
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = str(model)

    _set_single_prompt_attrs(attrs=attrs, content=request_kwargs.get(PROMPT_KEY, ""))

    if isinstance(response_or_chunks, list):
        output = "".join(
            str(_get_value(chunk, VALUE_KEY, "")) for chunk in response_or_chunks
        )
        _set_completion_attrs(
            attrs=attrs,
            completions=[{ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: output}],
        )
    elif response_or_chunks is not None:
        completions = [
            {
                ROLE_KEY: ASSISTANT_ROLE,
                CONTENT_KEY: _get_value(choice, TEXT_KEY, ""),
            }
            for choice in _as_list(_get_value(response_or_chunks, CHOICES_KEY))
        ]
        _set_completion_attrs(attrs=attrs, completions=completions)
    return attrs


def _simple_text_operation_attrs(
    *,
    span_name: str,
    request_kwargs: Mapping[str, Any],
    output: Any = None,
    model: str | None = None,
    prompt: Any = None,
) -> dict[str, Any]:
    attrs = _base_llm_attrs(
        span_name=span_name,
        log_type=LOG_TYPE_TEXT,
        request_type=LLMRequestTypeValues.COMPLETION.value,
    )
    if model:
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = model

    if prompt is None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(
            _public_request_kwargs(request_kwargs)
        )
    else:
        _set_single_prompt_attrs(attrs=attrs, content=prompt)

    if output is not None:
        _set_completion_attrs(
            attrs=attrs,
            completions=[{ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: output}],
        )
    return attrs


def build_graph_question_attrs(
    *,
    request_kwargs: Mapping[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    output = None
    if isinstance(response_or_chunks, list):
        for chunk in reversed(response_or_chunks):
            data = _get_value(chunk, DATA_KEY)
            output = _get_value(data, ANSWER_KEY)
            if output:
                break
    elif response_or_chunks is not None:
        output = _get_value(response_or_chunks, ANSWER_KEY)

    return _simple_text_operation_attrs(
        span_name=WRITER_GRAPH_QUESTION_SPAN_NAME,
        request_kwargs=request_kwargs,
        output=output,
        model=WRITER_GRAPH_MODEL_NAME,
        prompt={
            GRAPH_IDS_KEY: request_kwargs.get(GRAPH_IDS_KEY),
            QUESTION_KEY: request_kwargs.get(QUESTION_KEY),
        },
    )


def build_application_generate_attrs(
    *,
    request_kwargs: Mapping[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    output = None
    if isinstance(response_or_chunks, list):
        content_parts: list[str] = []
        for chunk in response_or_chunks:
            delta = _get_value(chunk, DELTA_KEY)
            content = _get_value(delta, CONTENT_KEY)
            if content:
                content_parts.append(str(content))
        output = "".join(content_parts)
    elif response_or_chunks is not None:
        output = _get_value(response_or_chunks, SUGGESTION_KEY)

    return _simple_text_operation_attrs(
        span_name=WRITER_APPLICATION_GENERATE_SPAN_NAME,
        request_kwargs=request_kwargs,
        output=output,
        model=WRITER_APPLICATION_MODEL_NAME,
        prompt={
            APPLICATION_ID_KEY: request_kwargs.get(APPLICATION_ID_KEY),
            INPUTS_KEY: request_kwargs.get(INPUTS_KEY),
        },
    )


def build_vision_attrs(
    *,
    request_kwargs: Mapping[str, Any],
    response: Any = None,
) -> dict[str, Any]:
    return _simple_text_operation_attrs(
        span_name=WRITER_VISION_SPAN_NAME,
        request_kwargs=request_kwargs,
        output=_get_value(response, DATA_KEY) if response is not None else None,
        model=request_kwargs.get(MODEL_KEY),
        prompt=_public_request_kwargs(request_kwargs),
    )


def build_translation_attrs(
    *,
    request_kwargs: Mapping[str, Any],
    response: Any = None,
) -> dict[str, Any]:
    return _simple_text_operation_attrs(
        span_name=WRITER_TRANSLATION_SPAN_NAME,
        request_kwargs=request_kwargs,
        output=_get_value(response, DATA_KEY) if response is not None else None,
        model=request_kwargs.get(MODEL_KEY),
        prompt=_public_request_kwargs(request_kwargs),
    )


def build_tool_attrs(
    *,
    tool_name: str,
    request_kwargs: Mapping[str, Any],
    response: Any = None,
) -> dict[str, Any]:
    attrs = {
        TLSpanAttributes.TRACELOOP_ENTITY_NAME: tool_name,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH: tool_name,
        TLSpanAttributes.TRACELOOP_ENTITY_INPUT: safe_json(
            _public_request_kwargs(request_kwargs)
        ),
        RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
    }
    workflow_name = context_api.get_value(TLSpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name

    if response is not None:
        output = _to_plain(response)
        if isinstance(output, Mapping) and CONTENT_KEY in output:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_content_attr(
                output[CONTENT_KEY]
            )
        elif isinstance(output, Mapping) and ANSWER_KEY in output and output[ANSWER_KEY]:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _json_content_attr(
                output[ANSWER_KEY]
            )
        else:
            attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(output)
    return attrs


def request_kwargs_with_positionals(
    *,
    kwargs: Mapping[str, Any],
    positional_values: tuple[Any, ...],
    positional_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    for index, name in enumerate(positional_names):
        if index < len(positional_values):
            request_kwargs.setdefault(name, positional_values[index])
    return request_kwargs
