"""Normalize Microsoft Agent Framework native OTEL spans for Respan export."""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_microsoft_agent_framework._constants import (
    AGENT_FRAMEWORK_SCOPE_PREFIX,
    AGENT_FRAMEWORK_SYSTEM,
    ATTR_GEN_AI_AGENT_NAME,
    ATTR_GEN_AI_CONVERSATION_ID,
    ATTR_GEN_AI_INPUT_MESSAGES,
    ATTR_GEN_AI_OPERATION_NAME,
    ATTR_GEN_AI_OUTPUT_MESSAGES,
    ATTR_GEN_AI_PROVIDER_NAME,
    ATTR_GEN_AI_RESPONSE_MODEL,
    ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
    ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
    ATTR_GEN_AI_TOOL_CALL_ID,
    ATTR_GEN_AI_TOOL_CALL_RESULT,
    ATTR_GEN_AI_TOOL_DEFINITIONS,
    ATTR_GEN_AI_TOOL_NAME,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    ATTR_WORKFLOW_EDGE_GROUP_ID,
    ATTR_WORKFLOW_EXECUTOR_ID,
    ATTR_WORKFLOW_ID,
    ATTR_WORKFLOW_NAME,
    OPERATION_CHAT,
    OPERATION_CREATE_AGENT,
    OPERATION_EXECUTE_TOOL,
    OPERATION_INVOKE_AGENT,
    TASK_SPAN_PREFIXES,
    TOP_LEVEL_ALIAS_ATTRS,
    WORKFLOW_SPAN_PREFIXES,
)
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SESSION_ID,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)

logger = logging.getLogger(__name__)


_OFF_CONTRACT_ALIASES = TOP_LEVEL_ALIAS_ATTRS | frozenset(
    {
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
    }
)

_RAW_ATTRS_TO_STRIP = frozenset(
    {
        ATTR_GEN_AI_INPUT_MESSAGES,
        ATTR_GEN_AI_OUTPUT_MESSAGES,
        ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
        ATTR_GEN_AI_TOOL_DEFINITIONS,
    }
)

_ERROR_TYPE_ATTR = "error.type"
_EXCEPTION_TYPE_ATTR = "exception.type"
_EXCEPTION_MESSAGE_ATTR = "exception.message"
_STATUS_CODE_ATTR = "status_code"


def _span_attr(name: str, fallback: str) -> str:
    return str(getattr(SpanAttributes, name, fallback))


_GEN_AI_SYSTEM = SpanAttributes.LLM_SYSTEM
_GEN_AI_OPERATION_NAME = _span_attr("GEN_AI_OPERATION_NAME", ATTR_GEN_AI_OPERATION_NAME)
_GEN_AI_PROVIDER_NAME = _span_attr("GEN_AI_PROVIDER_NAME", ATTR_GEN_AI_PROVIDER_NAME)
_GEN_AI_RESPONSE_MODEL = ATTR_GEN_AI_RESPONSE_MODEL
_GEN_AI_AGENT_NAME = _span_attr("GEN_AI_AGENT_NAME", ATTR_GEN_AI_AGENT_NAME)
_GEN_AI_TOOL_NAME = _span_attr("GEN_AI_TOOL_NAME", ATTR_GEN_AI_TOOL_NAME)
_GEN_AI_TOOL_PREFIX = f"{ATTR_GEN_AI_TOOL_NAME.rsplit('.', 1)[0]}."
_GEN_AI_TOOL_CALL_ARGUMENTS = _span_attr(
    "GEN_AI_TOOL_CALL_ARGUMENTS",
    ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
)
_GEN_AI_TOOL_CALL_RESULT = _span_attr(
    "GEN_AI_TOOL_CALL_RESULT",
    ATTR_GEN_AI_TOOL_CALL_RESULT,
)
_GEN_AI_USAGE_INPUT_TOKENS = _span_attr(
    "GEN_AI_USAGE_INPUT_TOKENS",
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
)
_GEN_AI_USAGE_OUTPUT_TOKENS = _span_attr(
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
)


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        stripped = value.strip()
        if not stripped or stripped[:1] not in {"{", "[", "("}:
            return value
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value


def _json_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(_to_jsonable(value), default=str)


def _to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {
            str(key): _to_jsonable(item, depth=depth + 1)
            for key, item in value.items()
            if not callable(item)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item, depth=depth + 1) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(
                model_dump(mode="json", exclude_none=True),
                depth=depth + 1,
            )
        except TypeError:
            return _to_jsonable(model_dump(), depth=depth + 1)

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return _to_jsonable(to_dict(), depth=depth + 1)

    if hasattr(value, "__dict__"):
        return {
            key: _to_jsonable(item, depth=depth + 1)
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    return repr(value)


def _set_if_present(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, "", (), []):
        attrs[key] = value


def _set_if_missing(attrs: dict[str, Any], key: str, value: Any) -> None:
    if value in (None, "", (), []):
        return
    if attrs.get(key) in (None, "", (), []):
        attrs[key] = value


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _scope_name(span: ReadableSpan) -> str:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None:
        name = getattr(scope, "name", "")
        if isinstance(name, str):
            return name
    instrumentation_info = getattr(span, "instrumentation_info", None)
    name = getattr(instrumentation_info, "name", "")
    return name if isinstance(name, str) else ""


def _span_name(span: ReadableSpan) -> str:
    return str(getattr(span, "name", "") or "")


def is_agent_framework_span(span: ReadableSpan, attrs: Mapping[str, Any]) -> bool:
    scope_name = _scope_name(span)
    if scope_name.startswith(AGENT_FRAMEWORK_SCOPE_PREFIX):
        return True
    if attrs.get(_GEN_AI_SYSTEM) == AGENT_FRAMEWORK_SYSTEM:
        return True
    if any(key.startswith("agent_framework.") for key in attrs):
        return True
    if ATTR_WORKFLOW_NAME in attrs or ATTR_WORKFLOW_ID in attrs:
        return True
    return False


def _operation_name(attrs: Mapping[str, Any]) -> str | None:
    operation = attrs.get(_GEN_AI_OPERATION_NAME) or attrs.get(ATTR_GEN_AI_OPERATION_NAME)
    return str(operation) if operation else None


def _has_any_prefix(span_name: str, prefixes: Sequence[str]) -> bool:
    return any(span_name == prefix or span_name.startswith(f"{prefix} ") for prefix in prefixes)


def _log_type(span: ReadableSpan, attrs: Mapping[str, Any]) -> str | None:
    operation = _operation_name(attrs)
    span_name = _span_name(span)

    if operation == OPERATION_CHAT:
        return LOG_TYPE_CHAT
    if operation == OPERATION_EXECUTE_TOOL or attrs.get(_GEN_AI_TOOL_NAME):
        return LOG_TYPE_TOOL
    if operation in {OPERATION_INVOKE_AGENT, OPERATION_CREATE_AGENT}:
        return LOG_TYPE_AGENT
    if _has_any_prefix(span_name, WORKFLOW_SPAN_PREFIXES) or ATTR_WORKFLOW_NAME in attrs:
        return LOG_TYPE_WORKFLOW
    if _has_any_prefix(span_name, TASK_SPAN_PREFIXES):
        return LOG_TYPE_TASK
    if attrs.get(_GEN_AI_AGENT_NAME):
        return LOG_TYPE_AGENT
    if attrs.get(SpanAttributes.LLM_REQUEST_MODEL) or attrs.get(_GEN_AI_PROVIDER_NAME):
        return LOG_TYPE_CHAT
    return None


def _suffix_name(span_name: str, prefix: str, fallback: str) -> str:
    if span_name.startswith(f"{prefix} "):
        suffix = span_name[len(prefix) + 1 :].strip()
        if suffix:
            return suffix
    return fallback


def _entity_name_for_log_type(
    span: ReadableSpan,
    attrs: Mapping[str, Any],
    log_type: str,
) -> str:
    span_name = _span_name(span)
    if log_type == LOG_TYPE_AGENT:
        agent_name = attrs.get(_GEN_AI_AGENT_NAME) or attrs.get(ATTR_GEN_AI_AGENT_NAME)
        if agent_name:
            return str(agent_name)
        return _suffix_name(span_name, OPERATION_INVOKE_AGENT, span_name or "agent")
    if log_type == LOG_TYPE_TOOL:
        tool_name = attrs.get(_GEN_AI_TOOL_NAME) or attrs.get(ATTR_GEN_AI_TOOL_NAME)
        if tool_name:
            return str(tool_name)
        return _suffix_name(span_name, OPERATION_EXECUTE_TOOL, span_name or "tool")
    if log_type == LOG_TYPE_WORKFLOW:
        return str(
            attrs.get(ATTR_WORKFLOW_NAME)
            or attrs.get(ATTR_WORKFLOW_ID)
            or span_name
            or "workflow"
        )
    if log_type == LOG_TYPE_TASK:
        return str(
            attrs.get(ATTR_WORKFLOW_EXECUTOR_ID)
            or attrs.get(ATTR_WORKFLOW_EDGE_GROUP_ID)
            or span_name
            or "task"
        )
    if log_type == LOG_TYPE_CHAT:
        return span_name or "agent_framework.chat"
    return span_name or "agent_framework"


def _message_role(message: Mapping[str, Any], *, default: str) -> str:
    role = message.get("role") or message.get("author") or default
    if role == "model":
        return "assistant"
    return str(role)


def _part_text(part: Mapping[str, Any]) -> str | None:
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return None


def _normalize_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(_to_jsonable(value), default=str)


def _tool_call_from_part(part: Mapping[str, Any]) -> dict[str, Any] | None:
    part_type = part.get("type")
    function_payload = part.get("function")
    if isinstance(function_payload, Mapping):
        name = function_payload.get("name")
        arguments = function_payload.get("arguments")
    else:
        name = part.get("name") or part.get("tool_name")
        arguments = part.get("arguments") or part.get("args") or part.get("input")

    if not name and part_type not in {"tool_call", "function_call"}:
        return None
    if not name:
        return None

    tool_call: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _normalize_arguments(arguments),
        },
    }
    call_id = part.get("id") or part.get("call_id") or part.get("tool_call_id")
    if call_id:
        tool_call["id"] = str(call_id)
    return tool_call


def _message_content_and_tool_calls(
    message: Mapping[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    content = message.get("content")
    if isinstance(content, str):
        text_parts = [content]
    else:
        text_parts = []

    tool_calls: list[dict[str, Any]] = []
    parts = message.get("parts")
    if isinstance(parts, Sequence) and not isinstance(parts, (str, bytes, bytearray)):
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            text = _part_text(part)
            if text:
                text_parts.append(text)
            tool_call = _tool_call_from_part(part)
            if tool_call is not None:
                tool_calls.append(tool_call)
            tool_response = part.get("response") or part.get("result")
            if tool_response is not None and not text:
                text_parts.append(str(tool_response))

    direct_tool_calls = message.get("tool_calls")
    parsed_tool_calls = _safe_json_loads(direct_tool_calls)
    if isinstance(parsed_tool_calls, Sequence) and not isinstance(
        parsed_tool_calls,
        (str, bytes, bytearray),
    ):
        for tool_call in parsed_tool_calls:
            if isinstance(tool_call, Mapping):
                normalized = _tool_call_from_part(tool_call)
                if normalized is not None:
                    tool_calls.append(normalized)

    content_text = "\n".join(text_parts) if text_parts else None
    return content_text, tool_calls


def _message_list(value: Any) -> list[Mapping[str, Any]]:
    parsed = _safe_json_loads(value)
    if isinstance(parsed, Mapping):
        return [parsed]
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes, bytearray)):
        return []
    return [message for message in parsed if isinstance(message, Mapping)]


def _apply_messages(
    attrs: dict[str, Any],
    *,
    source_messages: list[Mapping[str, Any]],
    target_prefix: str,
    default_role: str,
    start_index: int = 0,
) -> None:
    for offset, message in enumerate(source_messages):
        index = start_index + offset
        prefix = f"{target_prefix}.{index}"
        attrs.setdefault(f"{prefix}.role", _message_role(message, default=default_role))
        content, tool_calls = _message_content_and_tool_calls(message)
        if content is not None:
            attrs.setdefault(f"{prefix}.content", content)
        if tool_calls:
            attrs.setdefault(f"{prefix}.tool_calls", json.dumps(tool_calls, default=str))


def _system_instruction_messages(value: Any) -> list[dict[str, Any]]:
    parsed = _safe_json_loads(value)
    if parsed in (None, "", [], ()):
        return []
    if isinstance(parsed, str):
        return [{"role": "system", "content": parsed}]
    if isinstance(parsed, Mapping):
        content, _tool_calls = _message_content_and_tool_calls(parsed)
        return [{"role": "system", "content": content or _json_string(parsed) or ""}]
    if isinstance(parsed, Sequence):
        text_parts: list[str] = []
        for item in parsed:
            if isinstance(item, Mapping):
                text = _part_text(item)
                if text:
                    text_parts.append(text)
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return [{"role": "system", "content": "\n".join(text_parts)}]
    return [{"role": "system", "content": _json_string(parsed) or ""}]


def _normalize_tool_definition(tool: Mapping[str, Any]) -> dict[str, Any] | None:
    if isinstance(tool.get("function"), Mapping):
        function_payload = dict(tool["function"])
        name = function_payload.get("name")
        if not name:
            return None
        return {
            "type": str(tool.get("type") or "function"),
            "function": function_payload,
        }

    name = tool.get("name") or tool.get("tool_name")
    if not name:
        return None

    function: dict[str, Any] = {"name": str(name)}
    for key in ("description", "parameters"):
        if key in tool:
            function[key] = _to_jsonable(tool[key])
    return {"type": "function", "function": function}


def _tool_definitions(value: Any) -> list[dict[str, Any]]:
    parsed = _safe_json_loads(value)
    if isinstance(parsed, Mapping):
        candidates = parsed.get("tools") or parsed.get("functions") or [parsed]
    else:
        candidates = parsed
    if not isinstance(candidates, Sequence) or isinstance(
        candidates,
        (str, bytes, bytearray),
    ):
        return []
    definitions: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        normalized = _normalize_tool_definition(item)
        if normalized is not None:
            definitions.append(normalized)
    return definitions


def _apply_chat_attrs(
    span: ReadableSpan,
    attrs: dict[str, Any],
    raw_attrs: Mapping[str, Any],
) -> None:
    attrs[RESPAN_LOG_TYPE] = LOG_TYPE_CHAT
    attrs.setdefault(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)

    provider = raw_attrs.get(_GEN_AI_PROVIDER_NAME) or raw_attrs.get(
        ATTR_GEN_AI_PROVIDER_NAME
    )
    if attrs.get(_GEN_AI_SYSTEM) in (None, "", AGENT_FRAMEWORK_SYSTEM):
        attrs[_GEN_AI_SYSTEM] = str(provider).lower() if provider else AGENT_FRAMEWORK_SYSTEM

    model = raw_attrs.get(SpanAttributes.LLM_REQUEST_MODEL) or raw_attrs.get(
        _GEN_AI_RESPONSE_MODEL
    )
    _set_if_missing(attrs, SpanAttributes.LLM_REQUEST_MODEL, model)

    input_tokens = _int_value(raw_attrs.get(_GEN_AI_USAGE_INPUT_TOKENS))
    output_tokens = _int_value(raw_attrs.get(_GEN_AI_USAGE_OUTPUT_TOKENS))
    if input_tokens is not None:
        attrs.setdefault(_GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        attrs.setdefault(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
    if output_tokens is not None:
        attrs.setdefault(_GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        attrs.setdefault(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)
    if input_tokens is not None or output_tokens is not None:
        attrs.setdefault(
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            (input_tokens or 0) + (output_tokens or 0),
        )

    system_messages = _system_instruction_messages(
        raw_attrs.get(ATTR_GEN_AI_SYSTEM_INSTRUCTIONS)
    )
    prompt_messages = [
        *system_messages,
        *_message_list(raw_attrs.get(ATTR_GEN_AI_INPUT_MESSAGES)),
    ]
    completion_messages = _message_list(raw_attrs.get(ATTR_GEN_AI_OUTPUT_MESSAGES))
    _apply_messages(
        attrs,
        source_messages=prompt_messages,
        target_prefix=SpanAttributes.LLM_PROMPTS,
        default_role="user",
    )
    _apply_messages(
        attrs,
        source_messages=completion_messages,
        target_prefix=SpanAttributes.LLM_COMPLETIONS,
        default_role="assistant",
    )

    if prompt_messages:
        attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_INPUT, _json_string(prompt_messages))
    if completion_messages:
        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_string(completion_messages),
        )

    tool_definitions = _tool_definitions(raw_attrs.get(ATTR_GEN_AI_TOOL_DEFINITIONS))
    if tool_definitions:
        attrs.setdefault(
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            json.dumps(tool_definitions, default=str),
        )

    attrs.setdefault(
        SpanAttributes.TRACELOOP_ENTITY_NAME,
        _entity_name_for_log_type(span, attrs, LOG_TYPE_CHAT),
    )
    attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, "")


def _apply_tool_attrs(
    span: ReadableSpan,
    attrs: dict[str, Any],
    raw_attrs: Mapping[str, Any],
) -> None:
    attrs[RESPAN_LOG_TYPE] = LOG_TYPE_TOOL
    tool_name = _entity_name_for_log_type(span, raw_attrs, LOG_TYPE_TOOL)
    attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_NAME, tool_name)
    attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_PATH, tool_name)

    arguments = raw_attrs.get(_GEN_AI_TOOL_CALL_ARGUMENTS) or raw_attrs.get(
        ATTR_GEN_AI_TOOL_CALL_ARGUMENTS
    )
    input_payload = {
        "name": tool_name,
        "arguments": _safe_json_loads(arguments),
    }
    call_id = raw_attrs.get(ATTR_GEN_AI_TOOL_CALL_ID)
    if call_id:
        input_payload["id"] = str(call_id)
    attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_INPUT, _json_string(input_payload))

    result = raw_attrs.get(_GEN_AI_TOOL_CALL_RESULT) or raw_attrs.get(
        ATTR_GEN_AI_TOOL_CALL_RESULT
    )
    if result is not None:
        attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, _json_string(result))

    for key in list(attrs):
        if key.startswith(_GEN_AI_TOOL_PREFIX):
            attrs.pop(key, None)
    for key in (
        SpanAttributes.LLM_REQUEST_TYPE,
        SpanAttributes.LLM_REQUEST_MODEL,
        SpanAttributes.LLM_REQUEST_FUNCTIONS,
        _GEN_AI_SYSTEM,
    ):
        attrs.pop(key, None)


def _apply_common_span_attrs(
    span: ReadableSpan,
    attrs: dict[str, Any],
    raw_attrs: Mapping[str, Any],
    log_type: str,
) -> None:
    attrs[RESPAN_LOG_TYPE] = log_type
    entity_name = _entity_name_for_log_type(span, raw_attrs, log_type)
    attrs.setdefault(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
    attrs.setdefault(
        SpanAttributes.TRACELOOP_ENTITY_PATH,
        "" if log_type == LOG_TYPE_WORKFLOW else entity_name,
    )

    input_value = raw_attrs.get(ATTR_GEN_AI_INPUT_MESSAGES)
    output_value = raw_attrs.get(ATTR_GEN_AI_OUTPUT_MESSAGES)
    if input_value is not None:
        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            _json_string(_safe_json_loads(input_value)),
        )
    if output_value is not None:
        attrs.setdefault(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            _json_string(_safe_json_loads(output_value)),
        )

    conversation_id = raw_attrs.get(ATTR_GEN_AI_CONVERSATION_ID)
    if conversation_id and attrs.get(RESPAN_SESSION_ID) in (None, ""):
        attrs[RESPAN_SESSION_ID] = str(conversation_id)


def _span_status_is_error(span: ReadableSpan) -> bool:
    status = getattr(span, "status", None)
    status_code = getattr(status, "status_code", None)
    status_name = getattr(status_code, "name", status_code)
    return str(status_name).upper() == "ERROR"


def _error_message(span: ReadableSpan, raw_attrs: Mapping[str, Any]) -> str | None:
    for key in (ERROR_MESSAGE_ATTR, _EXCEPTION_MESSAGE_ATTR):
        value = raw_attrs.get(key)
        if value not in (None, ""):
            return str(value)

    status_description = getattr(getattr(span, "status", None), "description", None)
    if status_description not in (None, ""):
        return str(status_description)

    for key in (_ERROR_TYPE_ATTR, _EXCEPTION_TYPE_ATTR):
        value = raw_attrs.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _apply_error_attrs(
    span: ReadableSpan,
    attrs: dict[str, Any],
    raw_attrs: Mapping[str, Any],
) -> None:
    is_error = any(
        raw_attrs.get(key) not in (None, "")
        for key in (
            ERROR_MESSAGE_ATTR,
            _EXCEPTION_MESSAGE_ATTR,
            _ERROR_TYPE_ATTR,
            _EXCEPTION_TYPE_ATTR,
        )
    ) or _span_status_is_error(span)
    if not is_error:
        return

    status_code = _int_value(attrs.get(_STATUS_CODE_ATTR))
    if status_code is None or status_code < 400:
        attrs[_STATUS_CODE_ATTR] = 500

    message = _error_message(span=span, raw_attrs=raw_attrs)
    if message:
        attrs.setdefault(ERROR_MESSAGE_ATTR, message)


def _cleanup_attrs(attrs: dict[str, Any]) -> None:
    attrs.pop(SpanAttributes.TRACELOOP_SPAN_KIND, None)
    for key in _RAW_ATTRS_TO_STRIP | _OFF_CONTRACT_ALIASES:
        attrs.pop(key, None)


class AgentFrameworkSpanProcessor(SpanProcessor):
    """Translate Microsoft Agent Framework spans before Respan export."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        raw_attrs = dict(getattr(span, "attributes", None) or {})
        if not is_agent_framework_span(span, raw_attrs):
            return

        attrs = dict(getattr(span, "_attributes", None) or raw_attrs)
        log_type = _log_type(span, raw_attrs)
        if log_type is None:
            return

        if log_type == LOG_TYPE_CHAT:
            _apply_chat_attrs(span=span, attrs=attrs, raw_attrs=raw_attrs)
        elif log_type == LOG_TYPE_TOOL:
            _apply_tool_attrs(span=span, attrs=attrs, raw_attrs=raw_attrs)
        else:
            _apply_common_span_attrs(
                span=span,
                attrs=attrs,
                raw_attrs=raw_attrs,
                log_type=log_type,
            )

        _apply_error_attrs(span=span, attrs=attrs, raw_attrs=raw_attrs)
        _cleanup_attrs(attrs)
        span._attributes = attrs

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
