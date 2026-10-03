"""Direct PydanticAI span normalization for the Respan OTLP pipeline."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_SPEECH,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_TRANSCRIPTION,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
)

from respan_instrumentation_pydantic_ai._constants import (
    FINAL_RESULT_ATTR,
    MODEL_NAME_ATTR,
    PYDANTIC_AI_AGGREGATED_USAGE_INPUT_TOKENS_ATTR,
    PYDANTIC_AI_AGGREGATED_USAGE_OUTPUT_TOKENS_ATTR,
    PYDANTIC_AI_AGGREGATED_USAGE_TOTAL_TOKENS_ATTR,
    PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
    PYDANTIC_AI_REASONING_TOKENS_ATTR,
    PYDANTIC_AI_REQUEST_PARAMETERS_ATTR,
    PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME,
    PYDANTIC_AI_STRIP_ATTRS,
    PYDANTIC_AI_TOOLS_ATTR,
    RESPAN_OVERRIDE_MODEL_ATTR,
    RESPAN_RESPONSE_FORMAT_ATTR,
)
from respan_instrumentation_pydantic_ai._serialization import (
    json_string,
    json_value,
    parse_json,
    safe_text,
)

logger = logging.getLogger(__name__)

_PYDANTIC_AI_OPERATION_TO_LOG_TYPE = {
    "chat": LOG_TYPE_CHAT,
    "embedding": LOG_TYPE_EMBEDDING,
    "embeddings": LOG_TYPE_EMBEDDING,
    "response": LOG_TYPE_CHAT,
    "speech": LOG_TYPE_SPEECH,
    "transcription": LOG_TYPE_TRANSCRIPTION,
}
_USAGE_LOG_TYPES = frozenset(
    {
        LOG_TYPE_CHAT,
        LOG_TYPE_EMBEDDING,
        LOG_TYPE_SPEECH,
        LOG_TYPE_TRANSCRIPTION,
    }
)
_NESTED_PROVIDER_USAGE_SUPPRESSIBLE_LOG_TYPES = frozenset(_USAGE_LOG_TYPES)
_RAW_USAGE_ATTRIBUTE_NAMES = frozenset(
    {
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        PYDANTIC_AI_AGGREGATED_USAGE_INPUT_TOKENS_ATTR,
        PYDANTIC_AI_AGGREGATED_USAGE_OUTPUT_TOKENS_ATTR,
        PYDANTIC_AI_AGGREGATED_USAGE_TOTAL_TOKENS_ATTR,
    }
)
_CHAT_ROLES = frozenset({"assistant", "system", "tool", "user"})
_KIND_TO_CHAT_ROLE = {
    "request": "user",
    "response": "assistant",
}
_PART_KIND_TO_CHAT_ROLE = {
    "system-prompt": "system",
    "user-prompt": "user",
    "retry-prompt": "user",
    "tool-return": "tool",
    "text": "assistant",
}
_PRIMITIVE_ATTR_TYPES = (str, bool, int, float, bytes)


def _safe_json_loads(value: Any) -> Any:
    parsed = parse_json(value)
    return None if parsed is value and isinstance(value, str) else parsed


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    parsed = parse_json(value)
    return json_string(parsed)


def _extract_request_parameters(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    request_parameters = _safe_json_loads(
        attrs.get(PYDANTIC_AI_REQUEST_PARAMETERS_ATTR)
    )
    if isinstance(request_parameters, dict):
        return request_parameters
    return None


def _extract_messages(attrs: Mapping[str, Any], attr_name: str) -> list[Any] | None:
    value = _safe_json_loads(attrs.get(attr_name))
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        return [value]
    return None


def _is_chat_role(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _CHAT_ROLES


def _normalize_chat_role(value: Any, default_role: str) -> str:
    if not isinstance(value, str):
        return default_role
    normalized = value.strip().lower()
    if normalized in _CHAT_ROLES:
        return normalized
    return _KIND_TO_CHAT_ROLE.get(normalized, default_role)


def _normalize_part_role(value: Any, default_role: str) -> str:
    if not isinstance(value, str):
        return default_role
    normalized = value.strip().lower()
    if normalized in _CHAT_ROLES:
        return normalized
    return _PART_KIND_TO_CHAT_ROLE.get(normalized, default_role)


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, list):
        parts = [_content_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        for key in ("content", "text", "args", "result"):
            nested = value.get(key)
            if nested not in (None, "", (), []):
                return _content_to_text(nested)
        if value.get("type") in {"text", "thinking", "reasoning"}:
            # Native include_content=False retains only the part type.
            return ""
        return json_string(value)
    return safe_text(value)


def _normalize_tool_call(part: Mapping[str, Any]) -> dict[str, Any] | None:
    if "arguments" not in part and "args" not in part:
        return None
    name = part.get("name") or part.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    call_id = part.get("id") or part.get("tool_call_id")
    arguments = part.get("arguments", part.get("args", {}))
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": safe_text(name),
            "arguments": json_string(parse_json(arguments)),
        },
    }
    if isinstance(call_id, str) and call_id:
        normalized["id"] = safe_text(call_id)
    return normalized


def _normalize_tool_result(part: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"result": json_value(part.get("result"))}
    name = part.get("name") or part.get("tool_name")
    call_id = part.get("id") or part.get("tool_call_id")
    if isinstance(name, str) and name:
        payload["name"] = safe_text(name)
    if isinstance(call_id, str) and call_id:
        payload["tool_call_id"] = safe_text(call_id)
    return payload


def _messages_from_parts(
    parts: Any,
    default_role: str,
    *,
    allow_part_roles: bool,
) -> list[dict[str, Any]]:
    if not isinstance(parts, list):
        content = _content_to_text(parts)
        return [{"role": default_role, "content": content}] if content else []

    messages: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for part in parts:
        role = default_role
        if allow_part_roles and isinstance(part, Mapping):
            role = _normalize_part_role(
                part.get("role")
                or part.get("part_kind")
                or part.get("kind")
                or part.get("type"),
                default_role,
            )
        part_type = (
            safe_text(
                part.get("type") or part.get("part_kind") or part.get("kind") or ""
            ).lower()
            if isinstance(part, Mapping)
            else ""
        )
        if isinstance(part, Mapping) and part_type in {
            "tool-call",
            "tool_call",
            "toolcall",
        }:
            tool_call = _normalize_tool_call(part)
            if tool_call is not None:
                if current is None or current.get("role") != "assistant":
                    current = {"role": "assistant"}
                    messages.append(current)
                current.setdefault("tool_calls", []).append(tool_call)
            continue
        if isinstance(part, Mapping) and part_type in {
            "tool-call-response",
            "tool_call_response",
            "tool-return",
            "tool_return",
        }:
            if "result" not in part:
                continue
            messages.append(
                {"role": "tool", "content": json_string(_normalize_tool_result(part))}
            )
            current = None
            continue
        content = _content_to_text(part)
        if content:
            if (
                current is not None
                and current.get("role") == role
                and "content" not in current
            ):
                current["content"] = content
            else:
                current = {"role": role, "content": content}
                messages.append(current)
    return messages


def _message_to_chat_messages(
    message: Any,
    default_role: str,
) -> list[dict[str, Any]]:
    if not isinstance(message, Mapping):
        content = _content_to_text(message)
        return [{"role": default_role, "content": content}] if content else []

    explicit_role = _is_chat_role(message.get("role"))
    role = _normalize_chat_role(
        message.get("role") or message.get("kind"), default_role
    )
    content = message.get("content")
    if content in (None, "", (), []) and "parts" in message:
        return _messages_from_parts(
            message.get("parts"),
            role,
            allow_part_roles=not explicit_role,
        )

    content_text = _content_to_text(content)
    return [{"role": role, "content": content_text}] if content_text else []


def _normalize_chat_messages(
    messages: list[Any] | None,
    default_role: str,
) -> list[dict[str, Any]] | None:
    if messages is None:
        return None

    normalized_messages = []
    for message in messages:
        normalized_messages.extend(_message_to_chat_messages(message, default_role))
    return normalized_messages or None


def _chat_output_value(messages: list[dict[str, Any]]) -> Any:
    if len(messages) == 1:
        return messages[0]
    return messages


def _is_homogeneous_primitive_array(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return True
    if not all(isinstance(item, _PRIMITIVE_ATTR_TYPES) for item in value):
        return False
    first_type = type(value[0])
    return all(type(item) is first_type for item in value)


def _coerce_otel_attribute_value(value: Any) -> Any:
    if value is None or isinstance(value, _PRIMITIVE_ATTR_TYPES):
        return value
    if _is_homogeneous_primitive_array(value):
        return value
    return json_string(value)


def _extract_tool_names(attrs: Mapping[str, Any]) -> list[str] | None:
    raw_tools = _safe_json_loads(attrs.get(PYDANTIC_AI_TOOLS_ATTR))
    if not isinstance(raw_tools, list):
        return None
    tool_names = [tool_name for tool_name in raw_tools if isinstance(tool_name, str)]
    return tool_names or None


def _normalize_tool_definition(
    tool_definition: Mapping[str, Any],
) -> dict[str, Any] | None:
    function_payload = tool_definition.get("function")
    if isinstance(function_payload, Mapping):
        normalized = {
            "type": tool_definition.get("type", "function"),
            "function": {"name": safe_text(function_payload.get("name"))},
        }
        for key in ("description", "parameters", "strict"):
            value = function_payload.get(key)
            if value is not None:
                normalized["function"][key] = json_value(value)
        if normalized["function"].get("name"):
            return normalized
        return None

    tool_name = tool_definition.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None

    normalized_function: dict[str, Any] = {"name": safe_text(tool_name)}
    description = tool_definition.get("description")
    if description is not None:
        normalized_function["description"] = safe_text(description)
    parameters = tool_definition.get("parameters") or tool_definition.get(
        "parameters_json_schema"
    )
    if parameters is not None:
        normalized_function["parameters"] = json_value(parameters)
    strict = tool_definition.get("strict")
    if strict is not None:
        normalized_function["strict"] = strict

    return {
        "type": tool_definition.get("type", "function"),
        "function": normalized_function,
    }


def _extract_tools(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    tool_definitions = _safe_json_loads(
        attrs.get(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
    )
    if not isinstance(tool_definitions, list):
        request_parameters = _extract_request_parameters(attrs)
        if request_parameters is None:
            return None
        tool_definitions = [
            *(request_parameters.get("function_tools") or []),
            *(request_parameters.get("output_tools") or []),
        ]

    normalized_tools = []
    for tool_definition in tool_definitions:
        if not isinstance(tool_definition, Mapping):
            continue
        normalized_tool = _normalize_tool_definition(tool_definition)
        if normalized_tool is not None:
            normalized_tools.append(normalized_tool)
    return normalized_tools or None


def _extract_response_format(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    existing = attrs.get(RESPAN_RESPONSE_FORMAT_ATTR)
    if isinstance(existing, dict):
        return existing
    parsed_existing = _safe_json_loads(existing)
    if isinstance(parsed_existing, dict):
        return parsed_existing

    request_parameters = _extract_request_parameters(attrs)
    if request_parameters is None:
        return None

    output_mode = request_parameters.get("output_mode")
    if output_mode == "text":
        return {"type": "text"}
    if output_mode == "image":
        return {"type": "image"}
    if output_mode not in {"native", "prompted"}:
        return None

    output_object = request_parameters.get("output_object")
    if not isinstance(output_object, dict):
        return {"type": "json_schema"}

    json_schema_payload: dict[str, Any] = {
        "schema": output_object.get("json_schema") or {}
    }
    for key in ("name", "description", "strict"):
        value = output_object.get(key)
        if value is not None:
            json_schema_payload[key] = value

    return {"type": "json_schema", "json_schema": json_schema_payload}


def _extract_model(attrs: Mapping[str, Any]) -> str | None:
    for key in (
        SpanAttributes.LLM_REQUEST_MODEL,
        MODEL_NAME_ATTR,
        RESPAN_OVERRIDE_MODEL_ATTR,
    ):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _get_int_attr(attrs: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _extract_usage(
    attrs: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None]:
    prompt_tokens = _get_int_attr(
        attrs,
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
        PYDANTIC_AI_AGGREGATED_USAGE_INPUT_TOKENS_ATTR,
    )
    completion_tokens = _get_int_attr(
        attrs,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        PYDANTIC_AI_AGGREGATED_USAGE_OUTPUT_TOKENS_ATTR,
    )
    total_tokens = _get_int_attr(
        attrs,
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
        PYDANTIC_AI_AGGREGATED_USAGE_TOTAL_TOKENS_ATTR,
    )
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return prompt_tokens, completion_tokens, total_tokens


def _set_message_attrs(
    attrs: dict[str, Any],
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for index, message in enumerate(messages):
        message_prefix = f"{prefix}.{index}"
        _set_if_missing(attrs, f"{message_prefix}.role", message["role"])
        content = message.get("content")
        if content not in (None, ""):
            _set_if_missing(attrs, f"{message_prefix}.content", content)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            _set_if_missing(
                attrs,
                f"{message_prefix}.tool_calls",
                json_string(tool_calls),
            )


def _entity_path(span: ReadableSpan, entity_name: str) -> str:
    return "" if getattr(span, "parent", None) is None else entity_name


def _agent_content(attrs: Mapping[str, Any]) -> tuple[Any | None, Any | None]:
    all_messages = _safe_json_loads(attrs.get("pydantic_ai.all_messages"))
    normalized = _normalize_chat_messages(
        all_messages if isinstance(all_messages, list) else None,
        "user",
    )
    output = parse_json(attrs.get(FINAL_RESULT_ATTR))
    if output is None and normalized and normalized[-1].get("role") == "assistant":
        output = normalized[-1]
    if normalized and normalized[-1].get("role") == "assistant":
        normalized = normalized[:-1]
    return normalized or None, output


def _get_span_key(span: Any) -> tuple[int, int] | None:
    try:
        span_context = span.get_span_context()
    except Exception:  # noqa: BLE001 - vendor span objects may expose hostile hooks.
        return None

    trace_id = getattr(span_context, "trace_id", None)
    span_id = getattr(span_context, "span_id", None)
    if isinstance(trace_id, int) and isinstance(span_id, int):
        return trace_id, span_id
    return None


def _get_parent_span_key(span: Any) -> tuple[int, int] | None:
    span_key = _get_span_key(span)
    parent_span_id = getattr(getattr(span, "parent", None), "span_id", None)
    if span_key is None or not isinstance(parent_span_id, int):
        return None
    return span_key[0], parent_span_id


def _span_has_raw_usage_attributes(attrs: Mapping[str, Any]) -> bool:
    return any(
        isinstance(attrs.get(attribute_name), int)
        for attribute_name in _RAW_USAGE_ATTRIBUTE_NAMES
    )


def _should_map_usage_fields(
    log_type: str | None,
    suppress_nested_provider_usage: bool = False,
) -> bool:
    if log_type not in _USAGE_LOG_TYPES:
        return False
    return not (
        suppress_nested_provider_usage
        and log_type in _NESTED_PROVIDER_USAGE_SUPPRESSIBLE_LOG_TYPES
    )


def _enrich_nested_provider_span(
    span: ReadableSpan,
    attrs: dict[str, Any],
) -> None:
    if is_pydantic_ai_span(span, attrs):
        return
    if not _span_has_raw_usage_attributes(attrs):
        return

    log_type = _extract_log_type(span, attrs)
    if log_type not in _NESTED_PROVIDER_USAGE_SUPPRESSIBLE_LOG_TYPES:
        return

    _set_if_missing(
        attrs, RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value
    )
    _set_if_missing(attrs, RESPAN_LOG_TYPE, log_type)

    prompt_tokens, completion_tokens, total_tokens = _extract_usage(attrs)
    if prompt_tokens is not None:
        _set_if_missing(attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    if completion_tokens is not None:
        _set_if_missing(
            attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
        )
    if total_tokens is not None:
        _set_if_missing(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    output_messages = attrs.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    if output_messages is not None:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_messages)


def _extract_log_type(span: ReadableSpan, attrs: Mapping[str, Any]) -> str | None:
    if isinstance(attrs.get(GenAIAttributes.GEN_AI_TOOL_NAME), str):
        return LOG_TYPE_TOOL

    operation_name = attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
    if isinstance(operation_name, str):
        operation_log_type = _PYDANTIC_AI_OPERATION_TO_LOG_TYPE.get(operation_name)
        if operation_log_type is not None:
            return operation_log_type

    if isinstance(attrs.get(GenAIAttributes.GEN_AI_AGENT_NAME), str) or isinstance(
        attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR), str
    ):
        return LOG_TYPE_AGENT

    if span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME and _extract_tool_names(attrs):
        return LOG_TYPE_TASK
    return None


def is_pydantic_ai_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    scope = getattr(getattr(span, "instrumentation_scope", None), "name", None)
    if scope:
        return scope == "pydantic-ai" or scope.startswith("pydantic-ai.")
    return (
        bool(attrs.get(SpanAttributes.LLM_SYSTEM))
        or PYDANTIC_AI_REQUEST_PARAMETERS_ATTR in attrs
        or GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in attrs
        or bool(attrs.get(GenAIAttributes.GEN_AI_TOOL_NAME))
        or bool(attrs.get(GenAIAttributes.GEN_AI_AGENT_NAME))
        or bool(attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR))
        or bool(attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS))
        or bool(attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_RESULT))
        or bool(attrs.get(PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR))
        or bool(attrs.get(PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR))
        or span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME
        or FINAL_RESULT_ATTR in attrs
    )


def _set_if_missing(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    existing = attrs.get(key)
    if existing in (None, "", (), []):
        attrs[key] = value


def enrich_pydantic_ai_span(
    span: ReadableSpan,
    suppress_nested_provider_usage: bool = False,
) -> None:
    original_attrs = getattr(span, "_attributes", None)
    if original_attrs is None:
        return

    attrs = dict(original_attrs)
    if not is_pydantic_ai_span(span, attrs):
        return

    log_type = _extract_log_type(span, attrs)
    if log_type is None:
        return

    _set_if_missing(
        attrs, RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value
    )
    _set_if_missing(attrs, RESPAN_LOG_TYPE, log_type)

    if log_type in _USAGE_LOG_TYPES:
        provider = attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        if provider:
            _set_if_missing(attrs, SpanAttributes.LLM_SYSTEM, provider)
        model = _extract_model(attrs)
        if model is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_REQUEST_MODEL, model)

    if _should_map_usage_fields(
        log_type,
        suppress_nested_provider_usage=suppress_nested_provider_usage,
    ):
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(attrs)
        if prompt_tokens is not None:
            _set_if_missing(
                attrs, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
            )
            _set_if_missing(
                attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens
            )
        if completion_tokens is not None:
            _set_if_missing(
                attrs,
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                completion_tokens,
            )
            _set_if_missing(
                attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
            )
        if total_tokens is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
        for source, destination in (
            (
                GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            ),
            (
                GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            ),
            (
                GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
                SpanAttributes.LLM_USAGE_REASONING_TOKENS,
            ),
        ):
            value = _get_int_attr(attrs, source)
            if (
                source == GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS
                and value is None
            ):
                value = _get_int_attr(attrs, PYDANTIC_AI_REASONING_TOKENS_ATTR)
            if value is not None:
                _set_if_missing(attrs, destination, value)

    if log_type == LOG_TYPE_EMBEDDING:
        _set_if_missing(
            attrs, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.EMBEDDING.value
        )
        _set_if_missing(
            attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, "pydantic_ai.embeddings"
        )
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            _entity_path(span, "pydantic_ai.embeddings"),
        )
        if attrs.get("inputs") is not None:
            _set_if_missing(
                attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, attrs["inputs"]
            )
        if attrs.get("embeddings") is not None:
            _set_if_missing(
                attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, attrs["embeddings"]
            )
        for key in (
            "inputs",
            "embeddings",
            "embedding_settings",
            "inputs_count",
            "input_type",
        ):
            attrs.pop(key, None)

    if log_type == LOG_TYPE_CHAT:
        response_format = _extract_response_format(attrs)
        if response_format is not None:
            attrs[RESPAN_RESPONSE_FORMAT_ATTR] = json_string(response_format)

        tools = _extract_tools(attrs)
        if tools is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.LLM_REQUEST_FUNCTIONS,
                json_string(tools),
            )

    tool_name = attrs.get(GenAIAttributes.GEN_AI_TOOL_NAME)
    tool_name = tool_name if isinstance(tool_name, str) else None
    agent_name = attrs.get(GenAIAttributes.GEN_AI_AGENT_NAME)
    if not isinstance(agent_name, str):
        legacy_agent_name = attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR)
        agent_name = legacy_agent_name if isinstance(legacy_agent_name, str) else None

    tool_input = _json_string(
        attrs.get(
            GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS,
            attrs.get(PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR),
        )
    )
    tool_output = _json_string(
        attrs.get(
            GenAIAttributes.GEN_AI_TOOL_CALL_RESULT,
            attrs.get(PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR),
        )
    )

    if log_type in {LOG_TYPE_AGENT, LOG_TYPE_TASK, LOG_TYPE_TOOL}:
        for key in (
            SpanAttributes.LLM_SYSTEM,
            SpanAttributes.LLM_REQUEST_MODEL,
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            RESPAN_RESPONSE_FORMAT_ATTR,
            GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
        ):
            attrs.pop(key, None)
        for key in list(attrs):
            if key.startswith(
                ("gen_ai.usage.", "gen_ai.aggregated_usage.", "llm.usage.")
            ):
                attrs.pop(key, None)

    if suppress_nested_provider_usage and log_type in _USAGE_LOG_TYPES:
        for key in list(attrs):
            if key in _RAW_USAGE_ATTRIBUTE_NAMES or key.startswith(
                ("gen_ai.usage.", "llm.usage.")
            ):
                attrs.pop(key, None)

    if log_type == LOG_TYPE_TOOL and tool_name is not None:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, tool_name)
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            _entity_path(span, tool_name),
        )
        if tool_input is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json_string({"name": tool_name, "arguments": parse_json(tool_input)}),
            )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, tool_output)

    if log_type == LOG_TYPE_CHAT:
        entity_name = "pydantic_ai.chat"
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            _entity_path(span, entity_name),
        )
        _set_if_missing(
            attrs,
            SpanAttributes.LLM_REQUEST_TYPE,
            LLMRequestTypeValues.CHAT.value,
        )
        input_messages = _normalize_chat_messages(
            _extract_messages(attrs, GenAIAttributes.GEN_AI_INPUT_MESSAGES),
            "user",
        )
        output_messages = _normalize_chat_messages(
            _extract_messages(attrs, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES),
            "assistant",
        )
        if input_messages is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json_string(input_messages),
            )
            _set_message_attrs(attrs, SpanAttributes.LLM_PROMPTS, input_messages)
        if output_messages is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json_string(_chat_output_value(output_messages)),
            )
            _set_message_attrs(attrs, SpanAttributes.LLM_COMPLETIONS, output_messages)

    if log_type == LOG_TYPE_AGENT and agent_name is not None:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, agent_name)
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            _entity_path(span, agent_name),
        )
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, agent_name)
        agent_input, agent_output = _agent_content(attrs)
        if agent_input is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json_string(agent_input),
            )
        if agent_output is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json_string(agent_output),
            )

    if log_type == LOG_TYPE_TASK and span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, "running_tools")
        _set_if_missing(
            attrs,
            SpanAttributes.TRACELOOP_ENTITY_PATH,
            _entity_path(span, "running_tools"),
        )

    span._attributes = {
        key: _coerce_otel_attribute_value(value)
        for key, value in attrs.items()
        if key not in PYDANTIC_AI_STRIP_ATTRS
    }


class PydanticAISpanProcessor(SpanProcessor):
    """Normalize raw PydanticAI spans into Respan's OTLP conventions."""

    def __init__(self) -> None:
        self._nested_provider_usage_parent_keys: set[tuple[int, int]] = set()
        self._active_pydantic_span_keys: set[tuple[int, int]] = set()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if is_pydantic_ai_span(span, dict(getattr(span, "attributes", None) or {})):
            key = _get_span_key(span)
            if key is not None:
                self._active_pydantic_span_keys.add(key)

    def on_end(self, span: ReadableSpan) -> None:
        attrs = dict(getattr(span, "_attributes", None) or {})
        if not is_pydantic_ai_span(span, attrs):
            parent_span_key = _get_parent_span_key(span)
            if (
                parent_span_key in self._active_pydantic_span_keys
                and _span_has_raw_usage_attributes(attrs)
            ):
                _enrich_nested_provider_span(span, attrs)
                span._attributes = attrs
                self._nested_provider_usage_parent_keys.add(parent_span_key)
            return

        span_key = _get_span_key(span)
        try:
            enrich_pydantic_ai_span(
                span,
                suppress_nested_provider_usage=(
                    span_key in self._nested_provider_usage_parent_keys
                ),
            )
        except Exception:
            logger.exception("Failed to enrich PydanticAI span")
        finally:
            if span_key is not None:
                self._nested_provider_usage_parent_keys.discard(span_key)
                self._active_pydantic_span_keys.discard(span_key)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
