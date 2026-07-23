"""Direct PydanticAI span normalization for the Respan OTLP pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_pydantic_ai._constants import (
    FINAL_RESULT_ATTR,
    MODEL_NAME_ATTR,
    PYDANTIC_AI_AGENT_NAME_ATTR,
    PYDANTIC_AI_AGGREGATED_USAGE_INPUT_TOKENS_ATTR,
    PYDANTIC_AI_AGGREGATED_USAGE_OUTPUT_TOKENS_ATTR,
    PYDANTIC_AI_AGGREGATED_USAGE_TOTAL_TOKENS_ATTR,
    PYDANTIC_AI_INPUT_MESSAGES_ATTR,
    PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
    PYDANTIC_AI_OPERATION_NAME_ATTR,
    PYDANTIC_AI_OUTPUT_MESSAGES_ATTR,
    PYDANTIC_AI_REQUEST_PARAMETERS_ATTR,
    PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME,
    PYDANTIC_AI_STRIP_ATTRS,
    PYDANTIC_AI_TOOL_CALL_ARGUMENTS_ATTR,
    PYDANTIC_AI_TOOL_CALL_RESULT_ATTR,
    PYDANTIC_AI_TOOL_DEFINITIONS_ATTR,
    PYDANTIC_AI_TOOL_NAME_ATTR,
    PYDANTIC_AI_TOOLS_ATTR,
    PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR,
    PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR,
    RESPAN_OVERRIDE_MODEL_ATTR,
    RESPAN_RESPONSE_FORMAT_ATTR,
)
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

logger = logging.getLogger(__name__)

_PYDANTIC_AI_OPERATION_TO_LOG_TYPE = {
    "chat": LOG_TYPE_CHAT,
    "embedding": LOG_TYPE_EMBEDDING,
    "response": LOG_TYPE_CHAT,
    "speech": LOG_TYPE_SPEECH,
    "transcription": LOG_TYPE_TRANSCRIPTION,
}
_USAGE_LOG_TYPES = frozenset(
    {
        LOG_TYPE_CHAT,
        LOG_TYPE_EMBEDDING,
        LOG_TYPE_SPEECH,
        LOG_TYPE_TOOL,
        LOG_TYPE_TRANSCRIPTION,
    }
)
_NESTED_PROVIDER_USAGE_SUPPRESSIBLE_LOG_TYPES = frozenset(
    _USAGE_LOG_TYPES - {LOG_TYPE_TOOL}
)
_RAW_USAGE_ATTRIBUTE_NAMES = frozenset(
    {
        PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR,
        PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR,
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
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _extract_request_parameters(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    request_parameters = _safe_json_loads(attrs.get(PYDANTIC_AI_REQUEST_PARAMETERS_ATTR))
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
        return value
    if isinstance(value, list):
        parts = [_content_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        for key in ("content", "text", "args", "result"):
            nested = value.get(key)
            if nested not in (None, "", (), []):
                return _content_to_text(nested)
        return json.dumps(value, default=str)
    return str(value)


def _messages_from_parts(
    parts: Any,
    default_role: str,
    *,
    allow_part_roles: bool,
) -> list[dict[str, str]]:
    if not isinstance(parts, list):
        content = _content_to_text(parts)
        return [{"role": default_role, "content": content}] if content else []

    messages = []
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
        content = _content_to_text(part)
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _message_to_chat_messages(
    message: Any,
    default_role: str,
) -> list[dict[str, str]]:
    if not isinstance(message, Mapping):
        content = _content_to_text(message)
        return [{"role": default_role, "content": content}] if content else []

    explicit_role = _is_chat_role(message.get("role"))
    role = _normalize_chat_role(message.get("role") or message.get("kind"), default_role)
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
) -> list[dict[str, str]] | None:
    if messages is None:
        return None

    normalized_messages = []
    for message in messages:
        normalized_messages.extend(_message_to_chat_messages(message, default_role))
    return normalized_messages or None


def _chat_output_value(messages: list[dict[str, str]]) -> Any:
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
    return json.dumps(value, default=str)


def _extract_tool_names(attrs: Mapping[str, Any]) -> list[str] | None:
    raw_tools = _safe_json_loads(attrs.get(PYDANTIC_AI_TOOLS_ATTR))
    if not isinstance(raw_tools, list):
        return None
    tool_names = [tool_name for tool_name in raw_tools if isinstance(tool_name, str)]
    return tool_names or None


def _normalize_tool_definition(tool_definition: Mapping[str, Any]) -> dict[str, Any] | None:
    function_payload = tool_definition.get("function")
    if isinstance(function_payload, Mapping):
        normalized = {
            "type": tool_definition.get("type", "function"),
            "function": {"name": function_payload.get("name")},
        }
        for key in ("description", "parameters", "strict"):
            value = function_payload.get(key)
            if value is not None:
                normalized["function"][key] = value
        if normalized["function"].get("name"):
            return normalized
        return None

    tool_name = tool_definition.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None

    normalized_function: dict[str, Any] = {"name": tool_name}
    description = tool_definition.get("description")
    if description is not None:
        normalized_function["description"] = description
    parameters = tool_definition.get("parameters") or tool_definition.get(
        "parameters_json_schema"
    )
    if parameters is not None:
        normalized_function["parameters"] = parameters
    strict = tool_definition.get("strict")
    if strict is not None:
        normalized_function["strict"] = strict

    return {
        "type": tool_definition.get("type", "function"),
        "function": normalized_function,
    }


def _extract_tools(attrs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    tool_definitions = _safe_json_loads(attrs.get(PYDANTIC_AI_TOOL_DEFINITIONS_ATTR))
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
    for key in (SpanAttributes.LLM_REQUEST_MODEL, MODEL_NAME_ATTR, RESPAN_OVERRIDE_MODEL_ATTR):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _get_int_attr(attrs: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, int):
            return value
    return None


def _extract_usage(attrs: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
    prompt_tokens = _get_int_attr(
        attrs,
        PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR,
        PYDANTIC_AI_AGGREGATED_USAGE_INPUT_TOKENS_ATTR,
    )
    completion_tokens = _get_int_attr(
        attrs,
        PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR,
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
    messages: list[dict[str, str]],
) -> None:
    for index, message in enumerate(messages):
        message_prefix = f"{prefix}.{index}"
        _set_if_missing(attrs, f"{message_prefix}.role", message["role"])
        _set_if_missing(attrs, f"{message_prefix}.content", message["content"])


def _get_span_key(span: Any) -> tuple[int, int] | None:
    try:
        span_context = span.get_span_context()
    except Exception:
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
    if (
        suppress_nested_provider_usage
        and log_type in _NESTED_PROVIDER_USAGE_SUPPRESSIBLE_LOG_TYPES
    ):
        return False
    return True


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

    _set_if_missing(attrs, RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    _set_if_missing(attrs, RESPAN_LOG_TYPE, log_type)

    prompt_tokens, completion_tokens, total_tokens = _extract_usage(attrs)
    if prompt_tokens is not None:
        _set_if_missing(attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    if completion_tokens is not None:
        _set_if_missing(attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
    if total_tokens is not None:
        _set_if_missing(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    output_messages = attrs.get(PYDANTIC_AI_OUTPUT_MESSAGES_ATTR)
    if output_messages is not None:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, output_messages)


def _extract_log_type(span: ReadableSpan, attrs: Mapping[str, Any]) -> str | None:
    if isinstance(attrs.get(PYDANTIC_AI_TOOL_NAME_ATTR), str):
        return LOG_TYPE_TOOL

    operation_name = attrs.get(PYDANTIC_AI_OPERATION_NAME_ATTR)
    if isinstance(operation_name, str):
        operation_log_type = _PYDANTIC_AI_OPERATION_TO_LOG_TYPE.get(operation_name)
        if operation_log_type is not None:
            return operation_log_type

    if isinstance(attrs.get(PYDANTIC_AI_AGENT_NAME_ATTR), str) or isinstance(
        attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR), str
    ):
        return LOG_TYPE_AGENT

    if span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME and _extract_tool_names(attrs):
        return LOG_TYPE_TASK
    return None


def is_pydantic_ai_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    return (
        bool(attrs.get(SpanAttributes.LLM_SYSTEM))
        or PYDANTIC_AI_REQUEST_PARAMETERS_ATTR in attrs
        or PYDANTIC_AI_TOOL_DEFINITIONS_ATTR in attrs
        or bool(attrs.get(PYDANTIC_AI_TOOL_NAME_ATTR))
        or bool(attrs.get(PYDANTIC_AI_AGENT_NAME_ATTR))
        or bool(attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR))
        or bool(attrs.get(PYDANTIC_AI_TOOL_CALL_ARGUMENTS_ATTR))
        or bool(attrs.get(PYDANTIC_AI_TOOL_CALL_RESULT_ATTR))
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

    _set_if_missing(attrs, RESPAN_LOG_METHOD, LogMethodChoices.TRACING_INTEGRATION.value)
    _set_if_missing(attrs, RESPAN_LOG_TYPE, log_type)

    model = _extract_model(attrs)
    if model is not None:
        _set_if_missing(attrs, SpanAttributes.LLM_REQUEST_MODEL, model)

    if _should_map_usage_fields(
        log_type,
        suppress_nested_provider_usage=suppress_nested_provider_usage,
    ):
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(attrs)
        if prompt_tokens is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        if completion_tokens is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        if total_tokens is not None:
            _set_if_missing(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    response_format = _extract_response_format(attrs)
    if response_format is not None:
        attrs[RESPAN_RESPONSE_FORMAT_ATTR] = json.dumps(response_format, default=str)

    tools = _extract_tools(attrs)
    if tools is not None:
        _set_if_missing(
            attrs,
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            json.dumps(tools, default=str),
        )

    tool_name = attrs.get(PYDANTIC_AI_TOOL_NAME_ATTR)
    tool_name = tool_name if isinstance(tool_name, str) else None
    agent_name = attrs.get(PYDANTIC_AI_AGENT_NAME_ATTR)
    if not isinstance(agent_name, str):
        legacy_agent_name = attrs.get(PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR)
        agent_name = legacy_agent_name if isinstance(legacy_agent_name, str) else None

    tool_input = _json_string(
        attrs.get(
            PYDANTIC_AI_TOOL_CALL_ARGUMENTS_ATTR, attrs.get(PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR)
        )
    )
    tool_output = _json_string(
        attrs.get(
            PYDANTIC_AI_TOOL_CALL_RESULT_ATTR, attrs.get(PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR)
        )
    )

    if log_type == LOG_TYPE_TOOL and tool_name is not None:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, tool_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, tool_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, tool_input)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, tool_output)

    if log_type == LOG_TYPE_CHAT:
        _set_if_missing(
            attrs,
            SpanAttributes.LLM_REQUEST_TYPE,
            LLMRequestTypeValues.CHAT.value,
        )
        input_messages = _normalize_chat_messages(
            _extract_messages(attrs, PYDANTIC_AI_INPUT_MESSAGES_ATTR),
            "user",
        )
        output_messages = _normalize_chat_messages(
            _extract_messages(attrs, PYDANTIC_AI_OUTPUT_MESSAGES_ATTR),
            "assistant",
        )
        if input_messages is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                json.dumps(input_messages, default=str),
            )
            _set_message_attrs(attrs, SpanAttributes.LLM_PROMPTS, input_messages)
        if output_messages is not None:
            _set_if_missing(
                attrs,
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                json.dumps(_chat_output_value(output_messages), default=str),
            )
            _set_message_attrs(attrs, SpanAttributes.LLM_COMPLETIONS, output_messages)

    if log_type == LOG_TYPE_AGENT and agent_name is not None:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, agent_name)
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, agent_name)

    if log_type == LOG_TYPE_TASK and span.name == PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME:
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, "running_tools")
        _set_if_missing(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, "running_tools")

    span._attributes = {
        key: _coerce_otel_attribute_value(value)
        for key, value in attrs.items()
        if key not in PYDANTIC_AI_STRIP_ATTRS
    }


class PydanticAISpanProcessor(SpanProcessor):
    """Normalize raw PydanticAI spans into Respan's OTLP conventions."""

    def __init__(self) -> None:
        self._nested_provider_usage_parent_keys: set[tuple[int, int]] = set()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        attrs = dict(getattr(span, "_attributes", None) or {})
        _enrich_nested_provider_span(span, attrs)
        span._attributes = attrs
        if (
            not is_pydantic_ai_span(span, attrs)
            and _span_has_raw_usage_attributes(attrs)
        ):
            parent_span_key = _get_parent_span_key(span)
            if parent_span_key is not None:
                self._nested_provider_usage_parent_keys.add(parent_span_key)

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

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
