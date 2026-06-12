"""Translate LiteLLM callback payloads into canonical Respan span attributes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_litellm._constants import (
    API_BASE_KEY,
    ARGUMENTS_KEY,
    ASSISTANT_ROLE,
    CACHE_HIT_KEY,
    CHOICES_KEY,
    COMPLETION_TOKENS_KEY,
    CONTENT_KEY,
    COST_KEY,
    DELTA_KEY,
    FUNCTION_KEY,
    FUNCTIONS_KEY,
    FUNCTION_TOOL_TYPE,
    ID_KEY,
    LITELLM_CHAT_SPAN_NAME,
    LITELLM_PARAMS_KEY,
    MESSAGE_KEY,
    MESSAGES_KEY,
    METADATA_KEY,
    MODEL_KEY,
    NAME_KEY,
    OPENAI_MODEL_PREFIXES,
    PROMPT_TOKENS_KEY,
    PROVIDER_MODEL_PREFIXES,
    RESPONSE_KEY,
    RESPAN_PARAMS_KEY,
    ROLE_KEY,
    STANDARD_LOGGING_OBJECT_KEY,
    STREAM_KEY,
    TEXT_KEY,
    TOOL_CALLS_KEY,
    TOOLS_KEY,
    TOTAL_TOKENS_KEY,
    TYPE_KEY,
    USAGE_KEY,
    USER_ROLE,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.llm_logging import LogMethodChoices
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
    RESPAN_TRACE_GROUP_ID,
)
from respan_sdk.utils.serialization import serialize_value


def safe_json(value: Any) -> str:
    """Serialize arbitrary LiteLLM values into an OTEL-safe JSON string."""
    try:
        return json.dumps(
            serialize_value(value=value), default=str, separators=(",", ":")
        )
    except Exception:
        return json.dumps(str(value), separators=(",", ":"))


def _to_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method()
            except Exception:
                continue
            if isinstance(converted, Mapping):
                return converted

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return value_dict
    return None


def _get(value: Any, key: str, default: Any = None) -> Any:
    mapping = _to_mapping(value)
    if mapping is not None:
        return mapping.get(key, default)
    return getattr(value, key, default)


def _standard_logging_object(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    standard_logging_object = kwargs.get(STANDARD_LOGGING_OBJECT_KEY)
    mapping = _to_mapping(standard_logging_object)
    return mapping or {}


def _request_messages(kwargs: Mapping[str, Any]) -> list[Any]:
    messages = kwargs.get(MESSAGES_KEY)
    if messages is None:
        messages = _standard_logging_object(kwargs).get(MESSAGES_KEY)
    if messages is None:
        return []
    if isinstance(messages, list):
        return messages
    if isinstance(messages, tuple):
        return list(messages)
    if isinstance(messages, Mapping):
        return [messages]
    return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: messages}]


def _content_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json(value=value)


def _message_role(message: Any) -> str:
    role = _get(message, ROLE_KEY)
    if role is None:
        return USER_ROLE
    return str(role)


def _message_content(message: Any) -> str:
    return _content_to_string(_get(message, CONTENT_KEY))


def _normalized_tool_call(tool_call: Any) -> dict[str, Any] | None:
    function = _get(tool_call, FUNCTION_KEY)
    function_name = _get(function, NAME_KEY)
    if not function_name:
        return None

    arguments = _get(function, ARGUMENTS_KEY, {})
    normalized = {
        TYPE_KEY: str(_get(tool_call, TYPE_KEY, FUNCTION_TOOL_TYPE)),
        FUNCTION_KEY: {
            NAME_KEY: str(function_name),
            ARGUMENTS_KEY: (
                arguments if isinstance(arguments, str) else safe_json(value=arguments)
            ),
        },
    }
    tool_call_id = _get(tool_call, ID_KEY)
    if tool_call_id:
        normalized[ID_KEY] = str(tool_call_id)
    return normalized


def _normalized_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        value = [value]

    normalized = []
    for item in value:
        tool_call = _normalized_tool_call(tool_call=item)
        if tool_call is not None:
            normalized.append(tool_call)
    return normalized


def _normalized_tool_definition(tool: Any) -> dict[str, Any] | None:
    function = _get(tool, FUNCTION_KEY)
    if function is None:
        function = tool
    function_name = _get(function, NAME_KEY)
    if not function_name:
        return None

    tool_type = _get(tool, TYPE_KEY, FUNCTION_TOOL_TYPE)
    function_mapping = _to_mapping(function)
    if function_mapping is None:
        function_mapping = {NAME_KEY: str(function_name)}
    else:
        function_mapping = {
            str(key): serialize_value(value=value)
            for key, value in function_mapping.items()
            if value is not None
        }
        function_mapping[NAME_KEY] = str(function_name)

    return {
        TYPE_KEY: str(tool_type),
        FUNCTION_KEY: function_mapping,
    }


def _tool_definitions(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    tools = kwargs.get(TOOLS_KEY)
    if tools is None and kwargs.get(FUNCTIONS_KEY) is not None:
        tools = [
            {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: function}
            for function in kwargs.get(FUNCTIONS_KEY) or []
        ]

    if not tools:
        return []
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        tools = [tools]

    normalized = []
    for item in tools:
        tool = _normalized_tool_definition(tool=item)
        if tool is not None:
            normalized.append(tool)
    return normalized


def _first_choice(response_obj: Any) -> Any:
    choices = _get(response_obj, CHOICES_KEY, [])
    if choices is None and isinstance(response_obj, Mapping):
        choices = response_obj.get(CHOICES_KEY, [])
    if choices:
        return choices[0]
    return None


def _choice_message(choice: Any) -> Any:
    if choice is None:
        return None
    message = _get(choice, MESSAGE_KEY)
    if message is not None:
        return message
    return _get(choice, DELTA_KEY)


def _response_text(response_obj: Any, kwargs: Mapping[str, Any]) -> str:
    choice = _first_choice(response_obj=response_obj)
    message = _choice_message(choice=choice)
    content = _get(message, CONTENT_KEY)
    if content is not None:
        return _content_to_string(content)

    choice_text = _get(choice, TEXT_KEY)
    if choice_text is not None:
        return _content_to_string(choice_text)

    standard_response = _standard_logging_object(kwargs).get(RESPONSE_KEY)
    if standard_response is not None:
        return _content_to_string(standard_response)

    response_text = _get(response_obj, CONTENT_KEY)
    if response_text is not None:
        return _content_to_string(response_text)

    return ""


def _response_tool_calls(response_obj: Any) -> list[dict[str, Any]]:
    choice = _first_choice(response_obj=response_obj)
    message = _choice_message(choice=choice)
    return _normalized_tool_calls(value=_get(message, TOOL_CALLS_KEY))


def _usage_mapping(response_obj: Any, kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    usage = _get(response_obj, USAGE_KEY)
    mapping = _to_mapping(usage)
    if mapping is not None:
        return mapping

    standard_logging_object = _standard_logging_object(kwargs)
    if standard_logging_object:
        return standard_logging_object
    return {}


def _int_value(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _model_name(response_obj: Any, kwargs: Mapping[str, Any]) -> str | None:
    model = (
        kwargs.get(MODEL_KEY)
        or _get(response_obj, MODEL_KEY)
        or _standard_logging_object(kwargs).get(MODEL_KEY)
    )
    return str(model) if model else None


def _provider_name(model_name: str | None, kwargs: Mapping[str, Any]) -> str:
    litellm_params = kwargs.get(LITELLM_PARAMS_KEY)
    provider = _get(litellm_params, "custom_llm_provider")
    if provider:
        provider_name = str(provider).lower()
        return "google" if provider_name == "gemini" else provider_name

    if not model_name:
        return "litellm"

    model_prefix = model_name.split("/", maxsplit=1)[0].lower()
    if model_prefix in PROVIDER_MODEL_PREFIXES:
        return "google" if model_prefix == "gemini" else model_prefix

    if model_name.lower().startswith(OPENAI_MODEL_PREFIXES):
        return "openai"

    return "litellm"


def _respan_params(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = kwargs.get(METADATA_KEY)
    litellm_params = kwargs.get(LITELLM_PARAMS_KEY)
    litellm_metadata = _get(litellm_params, METADATA_KEY, {})

    for candidate in (litellm_metadata, metadata):
        mapping = _to_mapping(candidate)
        if mapping is None:
            continue
        params = mapping.get(RESPAN_PARAMS_KEY)
        params_mapping = _to_mapping(params)
        if params_mapping is not None:
            return params_mapping
    return {}


def _apply_respan_params(attributes: dict[str, Any], kwargs: Mapping[str, Any]) -> str:
    params = _respan_params(kwargs=kwargs)
    span_name = str(params.get("span_name") or LITELLM_CHAT_SPAN_NAME)

    workflow_name = params.get("workflow_name")
    if workflow_name and "trace_group_identifier" not in params:
        attributes.setdefault(RESPAN_TRACE_GROUP_ID, str(workflow_name))

    for key, value in params.items():
        if key in {
            "parent_span_id",
            "span_id",
            "span_name",
            "trace_id",
            "trace_name",
            "workflow_name",
        }:
            continue
        attr_key = RESPAN_SPAN_ATTRIBUTES_MAP.get(str(key))
        if attr_key is None:
            continue
        if attr_key == RESPAN_METADATA and isinstance(value, Mapping):
            for metadata_key, metadata_value in value.items():
                attributes[f"{RESPAN_METADATA}.{metadata_key}"] = (
                    metadata_value
                    if isinstance(metadata_value, str)
                    else str(metadata_value)
                )
        else:
            attributes[attr_key] = value

    return span_name


def build_litellm_span_data(
    *,
    kwargs: Mapping[str, Any],
    response_obj: Any,
    error: Exception | None = None,
    include_content: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Build canonical span name and attributes from a LiteLLM callback event."""
    model_name = _model_name(response_obj=response_obj, kwargs=kwargs)
    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        SpanAttributes.LLM_SYSTEM: _provider_name(model_name=model_name, kwargs=kwargs),
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: LITELLM_CHAT_SPAN_NAME,
        SpanAttributes.TRACELOOP_ENTITY_PATH: LITELLM_CHAT_SPAN_NAME,
    }
    span_name = _apply_respan_params(attributes=attributes, kwargs=kwargs)
    attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] = span_name
    attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] = span_name

    if model_name:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name
    if kwargs.get(STREAM_KEY):
        attributes[SpanAttributes.LLM_IS_STREAMING] = True

    if include_content:
        messages = _request_messages(kwargs=kwargs)
        attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(value=messages)
        attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = (
            str(error) if error is not None else _response_text(response_obj, kwargs)
        )

        for message_index, message in enumerate(messages):
            prompt_prefix = f"{SpanAttributes.LLM_PROMPTS}.{message_index}"
            attributes[f"{prompt_prefix}.role"] = _message_role(message=message)
            attributes[f"{prompt_prefix}.content"] = _message_content(message=message)
            tool_calls = _normalized_tool_calls(value=_get(message, TOOL_CALLS_KEY))
            if tool_calls:
                attributes[f"{prompt_prefix}.tool_calls"] = safe_json(value=tool_calls)

        completion_text = (
            str(error)
            if error is not None
            else _response_text(
                response_obj=response_obj,
                kwargs=kwargs,
            )
        )
        if completion_text:
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = ASSISTANT_ROLE
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = completion_text

        response_tool_calls = _response_tool_calls(response_obj=response_obj)
        if response_tool_calls:
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = ASSISTANT_ROLE
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"] = safe_json(
                value=response_tool_calls
            )

        tools = _tool_definitions(kwargs=kwargs)
        if tools:
            attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(value=tools)

    usage = _usage_mapping(response_obj=response_obj, kwargs=kwargs)
    prompt_tokens = _int_value(mapping=usage, key=PROMPT_TOKENS_KEY)
    completion_tokens = _int_value(mapping=usage, key=COMPLETION_TOKENS_KEY)
    total_tokens = _int_value(mapping=usage, key=TOTAL_TOKENS_KEY)
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    if prompt_tokens is not None:
        attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if total_tokens is not None:
        attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens

    standard_logging_object = _standard_logging_object(kwargs)
    cost = standard_logging_object.get(COST_KEY, kwargs.get(COST_KEY))
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        attributes[f"{RESPAN_METADATA}.response_cost"] = str(cost)
    cache_hit = standard_logging_object.get(CACHE_HIT_KEY, kwargs.get(CACHE_HIT_KEY))
    if isinstance(cache_hit, bool):
        attributes[f"{RESPAN_METADATA}.cache_hit"] = str(cache_hit).lower()
    api_base = kwargs.get(API_BASE_KEY) or standard_logging_object.get(API_BASE_KEY)
    if api_base:
        attributes[f"{RESPAN_METADATA}.api_base"] = str(api_base)

    return span_name, attributes
