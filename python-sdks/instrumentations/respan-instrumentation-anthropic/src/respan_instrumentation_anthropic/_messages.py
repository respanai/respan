"""Anthropic message normalization and synthetic span helpers."""

from __future__ import annotations

import json
import logging
import time
from threading import Lock
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    GEN_AI_SYSTEM,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TYPE,
    LLM_USAGE_COMPLETION_TOKENS,
    LLM_USAGE_PROMPT_TOKENS,
    RESPAN_LOG_TYPE,
)
from respan_instrumentation_anthropic._gateway_link import link_to_gateway_log
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
    generate_unique_id,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_anthropic._constants import (
    ANTHROPIC_CHAT_SPAN_NAME,
    ANTHROPIC_SYSTEM_NAME,
    ARGUMENTS_KEY,
    ASSISTANT_ROLE,
    CACHE_CREATION_INPUT_TOKENS_KEY,
    CACHE_READ_INPUT_TOKENS_KEY,
    CONTENT_KEY,
    DESCRIPTION_KEY,
    FUNCTION_KEY,
    FUNCTION_TOOL_TYPE,
    GEN_AI_COMPLETION_ROLE_ATTR,
    GEN_AI_COMPLETION_TOOL_CALLS_ATTR,
    ID_KEY,
    INPUT_KEY,
    INPUT_SCHEMA_KEY,
    INPUT_TOKENS_KEY,
    IS_ERROR_KEY,
    MESSAGES_KEY,
    MODEL_KEY,
    NAME_KEY,
    OUTPUT_TOKENS_KEY,
    PARAMETERS_KEY,
    PENDING_EXPIRES_AT_NS_KEY,
    PENDING_PARENT_ID_KEY,
    PENDING_START_NS_KEY,
    PENDING_TOOL_CALL_KEY,
    PENDING_TOOL_DEFINITION_KEY,
    ROLE_KEY,
    SERIALIZED_CONTENT_FIELD_NAMES,
    SYSTEM_KEY,
    SYSTEM_ROLE,
    TEXT_BLOCK_TYPE,
    TEXT_KEY,
    TOOL_CALL_ID_KEY,
    TOOL_CALLS_OVERRIDE,
    TOOL_RESULT_BLOCK_TYPE,
    TOOL_ROLE,
    TOOL_USE_BLOCK_TYPE,
    TOOL_USE_ID_KEY,
    TOOLS_KEY,
    TYPE_KEY,
    USAGE_KEY,
)

logger = logging.getLogger(__name__)

_PENDING_TOOL_CALL_TTL_NS = 15 * 60 * 1_000_000_000
_PENDING_TOOL_CALLS: dict[tuple[str, str], dict[str, Any]] = {}
_PENDING_TOOL_CALLS_LOCK = Lock()


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(serialize_value(value=value), default=str)
    except Exception:
        return str(value)


def _block_attr(block: Any, attr: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(attr, default)
    return getattr(block, attr, default)


def _block_type(block: Any) -> Any:
    return _block_attr(block=block, attr=TYPE_KEY)


def _iter_content_blocks(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, list):
        return content
    return [content]


def _maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped_value = value.strip()
    if not stripped_value or stripped_value[0] not in ("{", "["):
        return value

    try:
        return json.loads(stripped_value)
    except Exception:
        return value


def _serialize_content_value(value: Any) -> Any:
    """Serialize Anthropic content while preserving structured blocks."""
    if isinstance(value, list):
        serialized_parts = [_serialize_content_block(block=block) for block in value]
        non_empty_parts = [
            part for part in serialized_parts if part not in ("", None, [], {})
        ]
        if not non_empty_parts:
            return ""
        if all(isinstance(part, str) for part in non_empty_parts):
            return "\n".join(non_empty_parts)
        return non_empty_parts

    if isinstance(value, dict):
        if _block_type(block=value):
            return _serialize_content_block(block=value)
        return {
            str(key): _serialize_content_value(value=nested_value)
            for key, nested_value in value.items()
        }

    if _block_type(block=value):
        return _serialize_content_block(block=value)

    return serialize_value(value=value)


def _serialize_content_block(block: Any) -> Any:
    """Serialize a single Anthropic content block."""
    if isinstance(block, str):
        return block

    block_type = _block_type(block=block)
    if block_type == TEXT_BLOCK_TYPE:
        return _block_attr(block=block, attr=TEXT_KEY, default="")

    if block_type:
        serialized_block = {TYPE_KEY: block_type}
        for field_name in SERIALIZED_CONTENT_FIELD_NAMES:
            if field_name == TYPE_KEY:
                continue

            field_value = _block_attr(block=block, attr=field_name, default=None)
            if field_value is not None:
                serialized_block[field_name] = _serialize_content_value(
                    value=field_value
                )
        return serialized_block

    text_value = _block_attr(block=block, attr=TEXT_KEY, default=None)
    if text_value is not None:
        return text_value

    return serialize_value(value=block)


def _normalize_content_block(block: Any) -> str:
    """Normalize a content block into human-readable text for turn tracking."""
    serialized_value = _serialize_content_block(block=block)
    if serialized_value in ("", None, [], {}):
        return ""
    if isinstance(serialized_value, str):
        return serialized_value
    return _safe_json(value=serialized_value)


def _extract_text_content(content: Any) -> str:
    """Extract joined text blocks without JSON-encoding plain strings."""
    serialized_content = _serialize_content_value(value=content)
    if serialized_content in ("", None, [], {}):
        return ""
    if isinstance(serialized_content, str):
        return serialized_content
    if not isinstance(serialized_content, list):
        return ""

    text_parts: list[str] = []
    for item in serialized_content:
        if isinstance(item, str) and item:
            text_parts.append(item)
        elif isinstance(item, dict) and item.get(TYPE_KEY) == TEXT_BLOCK_TYPE:
            text_value = item.get(TEXT_KEY)
            if isinstance(text_value, str) and text_value:
                text_parts.append(text_value)
    return "\n".join(text_parts)


def _normalize_tool_result_content(content: Any) -> Any:
    """Preserve structured tool results instead of double-stringifying JSON."""
    normalized_content = _serialize_content_value(value=content)
    if isinstance(normalized_content, str):
        return _maybe_parse_json_string(value=normalized_content)
    return normalized_content


def _format_tool_result_output(content: Any) -> str:
    normalized_content = _normalize_tool_result_content(content=content)
    if isinstance(normalized_content, str):
        return normalized_content
    return _safe_json(value=normalized_content)


def _extract_tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
    """Extract tool_use blocks from arbitrary Anthropic content payloads."""
    tool_calls: list[dict[str, Any]] = []
    for block in _iter_content_blocks(content=content):
        if _block_type(block=block) != TOOL_USE_BLOCK_TYPE:
            continue

        tool_name = _block_attr(block=block, attr=NAME_KEY, default="")
        if not isinstance(tool_name, str) or not tool_name:
            continue

        tool_calls.append(
            {
                ID_KEY: _block_attr(block=block, attr=ID_KEY, default=""),
                TYPE_KEY: FUNCTION_TOOL_TYPE,
                FUNCTION_KEY: {
                    NAME_KEY: tool_name,
                    ARGUMENTS_KEY: _safe_json(
                        value=_block_attr(block=block, attr=INPUT_KEY, default={})
                    ),
                },
            }
        )

    return tool_calls


def _extract_tool_results(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract tool_result blocks from request messages."""
    tool_results: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get(CONTENT_KEY, "")
        else:
            content = getattr(msg, CONTENT_KEY, "")

        for block in _iter_content_blocks(content=content):
            if _block_type(block=block) != TOOL_RESULT_BLOCK_TYPE:
                continue

            tool_call_id = _block_attr(block=block, attr=TOOL_USE_ID_KEY, default="")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue

            tool_results.append(
                {
                    TOOL_CALL_ID_KEY: tool_call_id,
                    CONTENT_KEY: _block_attr(block=block, attr=CONTENT_KEY, default=""),
                    IS_ERROR_KEY: bool(
                        _block_attr(block=block, attr=IS_ERROR_KEY, default=False)
                    ),
                }
            )

    return tool_results


def _extract_input_tool_calls(messages: list[Any]) -> dict[str, dict[str, Any]]:
    """Map tool call IDs from assistant input messages to normalized tool_calls."""
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get(ROLE_KEY, "")
            content = msg.get(CONTENT_KEY, "")
        else:
            role = getattr(msg, ROLE_KEY, "")
            content = getattr(msg, CONTENT_KEY, "")

        if role != ASSISTANT_ROLE:
            continue

        for tool_call in _extract_tool_calls_from_content(content=content):
            tool_call_id = tool_call.get(ID_KEY)
            if isinstance(tool_call_id, str) and tool_call_id:
                tool_calls_by_id[tool_call_id] = tool_call

    return tool_calls_by_id


def _format_input_messages(messages: list[Any], system: Any = None) -> str:
    """Normalize Anthropic messages to chat-style JSON."""
    normalized_messages: list[dict[str, Any]] = []

    if system is not None:
        if isinstance(system, str):
            normalized_messages.append({ROLE_KEY: SYSTEM_ROLE, CONTENT_KEY: system})
        else:
            normalized_messages.append(
                {
                    ROLE_KEY: SYSTEM_ROLE,
                    CONTENT_KEY: _serialize_content_value(value=system),
                }
            )

    for message in messages:
        if isinstance(message, dict):
            role = message.get(ROLE_KEY, "")
            content = message.get(CONTENT_KEY, "")
        elif hasattr(message, ROLE_KEY) and hasattr(message, CONTENT_KEY):
            role = getattr(message, ROLE_KEY)
            content = getattr(message, CONTENT_KEY)
        else:
            role = ""
            content = str(message)

        tool_calls = _extract_tool_calls_from_content(content=content)
        if role == ASSISTANT_ROLE and tool_calls:
            normalized_messages.append(
                {
                    ROLE_KEY: ASSISTANT_ROLE,
                    CONTENT_KEY: _extract_text_content(content=content),
                    TOOL_CALLS_OVERRIDE: tool_calls,
                }
            )
            continue

        tool_messages: list[dict[str, Any]] = []
        residual_blocks: list[Any] = []
        for block in _iter_content_blocks(content=content):
            if _block_type(block=block) != TOOL_RESULT_BLOCK_TYPE:
                residual_blocks.append(block)
                continue

            tool_messages.append(
                {
                    ROLE_KEY: TOOL_ROLE,
                    TOOL_CALL_ID_KEY: _block_attr(
                        block=block, attr=TOOL_USE_ID_KEY, default=""
                    ),
                    CONTENT_KEY: _normalize_tool_result_content(
                        content=_block_attr(block=block, attr=CONTENT_KEY, default="")
                    ),
                }
            )

        if tool_messages:
            if residual_blocks:
                residual_content = _serialize_content_value(value=residual_blocks)
                if residual_content not in ("", None, [], {}):
                    normalized_messages.append(
                        {ROLE_KEY: role, CONTENT_KEY: residual_content}
                    )
            normalized_messages.extend(tool_messages)
            continue

        normalized_messages.append(
            {ROLE_KEY: role, CONTENT_KEY: _serialize_content_value(value=content)}
        )

    return _safe_json(value=normalized_messages)


def _format_output(message: Any) -> str:
    """Extract assistant response text from an Anthropic Message."""
    content = getattr(message, CONTENT_KEY, None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    text_content = _extract_text_content(content=content)
    if text_content:
        return text_content

    if _extract_tool_calls_from_content(content=content):
        return ""

    return _safe_json(value=_serialize_content_value(value=content))


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    return _extract_tool_calls_from_content(content=getattr(message, CONTENT_KEY, None))


def _format_tools(tools: Any) -> list[dict[str, Any]]:
    """Normalize Anthropic tool definitions into chat-completions shape."""
    if not tools:
        return []

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get(NAME_KEY, "")
            description = tool.get(DESCRIPTION_KEY, "")
            parameters = tool.get(INPUT_SCHEMA_KEY, {})
        elif hasattr(tool, NAME_KEY):
            name = getattr(tool, NAME_KEY)
            description = getattr(tool, DESCRIPTION_KEY, "")
            parameters = getattr(tool, INPUT_SCHEMA_KEY, {})
        else:
            continue

        tool_definition: dict[str, Any] = {
            TYPE_KEY: FUNCTION_TOOL_TYPE,
            FUNCTION_KEY: {NAME_KEY: name},
        }
        if description:
            tool_definition[FUNCTION_KEY][DESCRIPTION_KEY] = description
        if parameters:
            tool_definition[FUNCTION_KEY][PARAMETERS_KEY] = parameters
        normalized_tools.append(tool_definition)

    return normalized_tools


def _active_workflow_name() -> str | None:
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_WORKFLOW_NAME)
    if not workflow_name:
        workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    return str(workflow_name) if workflow_name else None


def _build_base_chat_attrs(span_name: str) -> dict[str, Any]:
    attrs = {
        GEN_AI_SYSTEM: ANTHROPIC_SYSTEM_NAME,
        LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
    }
    workflow_name = _active_workflow_name()
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _apply_tool_call_attrs(
    attrs: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> None:
    if not tool_calls:
        return
    attrs[GEN_AI_COMPLETION_TOOL_CALLS_ATTR] = _safe_json(value=tool_calls)
    attrs[GEN_AI_COMPLETION_ROLE_ATTR] = ASSISTANT_ROLE


def _apply_tool_definition_attrs(
    attrs: dict[str, Any], tools: list[dict[str, Any]]
) -> None:
    if not tools:
        return
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = _safe_json(value=tools)


def _build_span_attrs(kwargs: dict[str, Any], message: Any) -> dict[str, Any]:
    """Build span attributes from request kwargs and response Message."""
    attrs = _build_base_chat_attrs(span_name=ANTHROPIC_CHAT_SPAN_NAME)

    model = getattr(message, MODEL_KEY, None) or kwargs.get(MODEL_KEY)
    if model:
        attrs[LLM_REQUEST_MODEL] = model

    messages = kwargs.get(MESSAGES_KEY, [])
    system = kwargs.get(SYSTEM_KEY)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _format_input_messages(
        messages=messages,
        system=system,
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = _format_output(message=message)

    usage = getattr(message, USAGE_KEY, None)
    if usage is not None:
        attrs[LLM_USAGE_PROMPT_TOKENS] = getattr(usage, INPUT_TOKENS_KEY, 0)
        attrs[LLM_USAGE_COMPLETION_TOKENS] = getattr(usage, OUTPUT_TOKENS_KEY, 0)
        cache_read_input_tokens = getattr(usage, CACHE_READ_INPUT_TOKENS_KEY, 0)
        cache_creation_input_tokens = getattr(usage, CACHE_CREATION_INPUT_TOKENS_KEY, 0)
        if cache_read_input_tokens:
            attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = (
                cache_read_input_tokens
            )
        if cache_creation_input_tokens:
            attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = (
                cache_creation_input_tokens
            )

    _apply_tool_call_attrs(attrs=attrs, tool_calls=_extract_tool_calls(message=message))
    _apply_tool_definition_attrs(
        attrs=attrs,
        tools=_format_tools(tools=kwargs.get(TOOLS_KEY)),
    )
    return attrs


def _build_error_attrs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build minimal span attributes for a failed Anthropic request."""
    attrs = _build_base_chat_attrs(span_name=ANTHROPIC_CHAT_SPAN_NAME)
    model = kwargs.get(MODEL_KEY)
    if model:
        attrs[LLM_REQUEST_MODEL] = model

    messages = kwargs.get(MESSAGES_KEY, [])
    system = kwargs.get(SYSTEM_KEY)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = _format_input_messages(
        messages=messages,
        system=system,
    )
    _apply_tool_definition_attrs(
        attrs=attrs,
        tools=_format_tools(tools=kwargs.get(TOOLS_KEY)),
    )
    return attrs


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    """Return active OTEL trace/span IDs for parenting injected spans."""
    try:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
    except Exception:
        return None, None

    trace_id = getattr(span_context, "trace_id", 0) or 0
    span_id = getattr(span_context, "span_id", 0) or 0
    if trace_id == 0 or span_id == 0:
        return None, None

    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _generate_trace_id() -> str:
    return generate_unique_id()


def _generate_span_id() -> str:
    return generate_unique_id()[:16]


def _resolve_injected_trace_parent_ids() -> tuple[str, str | None]:
    trace_id, parent_id = _current_trace_parent_ids()
    if trace_id is None:
        trace_id = _generate_trace_id()
    return trace_id, parent_id


def _prune_expired_pending_tool_calls(*, now_ns: int | None = None) -> None:
    expiration_cutoff_ns = now_ns if now_ns is not None else time.time_ns()
    with _PENDING_TOOL_CALLS_LOCK:
        expired_keys = [
            key
            for key, entry in _PENDING_TOOL_CALLS.items()
            if entry.get(PENDING_EXPIRES_AT_NS_KEY, 0) <= expiration_cutoff_ns
        ]
        for key in expired_keys:
            _PENDING_TOOL_CALLS.pop(key, None)


def _emit_span(
    attrs: dict[str, Any],
    start_ns: int,
    error_message: str | None = None,
    *,
    name: str = ANTHROPIC_CHAT_SPAN_NAME,
    trace_id: str | None = None,
    parent_id: str | None = None,
    span_id: str | None = None,
    end_ns: int | None = None,
    status_code: int = 200,
    gateway_log_id: str | None = None,
) -> None:
    """Build a ReadableSpan and inject it into the OTEL pipeline."""
    try:
        if error_message:
            attrs = dict(attrs)
            attrs.setdefault("error.message", error_message)
            attrs.setdefault("status_code", status_code if status_code >= 400 else 500)
        if gateway_log_id:
            attrs = link_to_gateway_log(attrs, gateway_log_id)

        resolved_trace_id = trace_id
        resolved_parent_id = parent_id
        if resolved_trace_id is None and resolved_parent_id is None:
            resolved_trace_id, resolved_parent_id = _current_trace_parent_ids()

        span = build_readable_span(
            name=name,
            trace_id=resolved_trace_id,
            span_id=span_id,
            parent_id=resolved_parent_id,
            start_time_ns=start_ns,
            end_time_ns=end_ns if end_ns is not None else time.time_ns(),
            attributes=attrs,
            error_message=error_message,
            status_code=status_code,
        )
        inject_span(span=span)
    except Exception:
        logger.debug("Failed to emit Anthropic span", exc_info=True)


def _find_tool_definition(
    tools: list[dict[str, Any]], tool_name: str
) -> dict[str, Any]:
    for tool in tools:
        function = tool.get(FUNCTION_KEY)
        if not isinstance(function, dict):
            continue
        if function.get(NAME_KEY) == tool_name:
            return tool
    return {TYPE_KEY: FUNCTION_TOOL_TYPE, FUNCTION_KEY: {NAME_KEY: tool_name}}


def _register_pending_tool_calls(
    *,
    trace_id: str,
    parent_id: str,
    tool_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """Store tool calls until a matching tool_result arrives later."""
    registered_at_ns = time.time_ns()
    expires_at_ns = registered_at_ns + _PENDING_TOOL_CALL_TTL_NS
    _prune_expired_pending_tool_calls(now_ns=registered_at_ns)
    with _PENDING_TOOL_CALLS_LOCK:
        for tool_call in tool_calls:
            function = tool_call.get(FUNCTION_KEY)
            if not isinstance(function, dict):
                continue

            tool_name = function.get(NAME_KEY)
            tool_call_id = tool_call.get(ID_KEY)
            if (
                not isinstance(tool_name, str)
                or not tool_name
                or not isinstance(tool_call_id, str)
                or not tool_call_id
            ):
                continue

            _PENDING_TOOL_CALLS[(trace_id, tool_call_id)] = {
                PENDING_TOOL_CALL_KEY: tool_call,
                PENDING_TOOL_DEFINITION_KEY: _find_tool_definition(
                    tools=tools, tool_name=tool_name
                ),
                PENDING_PARENT_ID_KEY: parent_id,
                PENDING_START_NS_KEY: registered_at_ns,
                PENDING_EXPIRES_AT_NS_KEY: expires_at_ns,
            }


def _emit_tool_span(
    *,
    trace_id: str,
    parent_id: str,
    start_ns: int,
    end_ns: int,
    tool_call: dict[str, Any],
    tool_definition: dict[str, Any],
    tool_output: str,
    is_error: bool = False,
) -> None:
    function = tool_call.get(FUNCTION_KEY)
    if not isinstance(function, dict):
        return

    tool_name = function.get(NAME_KEY)
    if not isinstance(tool_name, str) or not tool_name:
        return

    tool_input = function.get(ARGUMENTS_KEY, "")
    canonical_input = _safe_json(
        value={
            NAME_KEY: tool_name,
            ARGUMENTS_KEY: _maybe_parse_json_string(value=tool_input),
        }
    )
    canonical_output = _safe_json(value=_maybe_parse_json_string(value=tool_output))
    attrs = {
        SpanAttributes.TRACELOOP_ENTITY_NAME: tool_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: tool_name,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: canonical_input,
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: canonical_output,
        RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
    }
    workflow_name = _active_workflow_name()
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name

    _emit_span(
        attrs=attrs,
        start_ns=start_ns,
        name=tool_name,
        trace_id=trace_id,
        parent_id=parent_id,
        span_id=_generate_span_id(),
        end_ns=end_ns,
        error_message=tool_output if is_error else None,
        status_code=500 if is_error else 200,
    )


def _emit_resolved_tool_spans(
    *,
    kwargs: dict[str, Any],
    trace_id: str,
    fallback_parent_id: str,
    end_ns: int,
    tools: list[dict[str, Any]],
) -> None:
    """Emit tool spans once a later request provides matching tool_result blocks."""
    messages = kwargs.get(MESSAGES_KEY, [])
    _prune_expired_pending_tool_calls(now_ns=end_ns)
    tool_results = _extract_tool_results(messages=messages)
    if not tool_results:
        return

    input_tool_calls = _extract_input_tool_calls(messages=messages)
    for tool_result in tool_results:
        tool_call_id = tool_result[TOOL_CALL_ID_KEY]
        with _PENDING_TOOL_CALLS_LOCK:
            pending_tool_call = _PENDING_TOOL_CALLS.pop((trace_id, tool_call_id), None)

        if pending_tool_call is not None:
            tool_call = pending_tool_call[PENDING_TOOL_CALL_KEY]
            tool_definition = pending_tool_call[PENDING_TOOL_DEFINITION_KEY]
            parent_id = pending_tool_call[PENDING_PARENT_ID_KEY]
            start_ns = pending_tool_call[PENDING_START_NS_KEY]
        else:
            tool_call = input_tool_calls.get(tool_call_id)
            if tool_call is None:
                continue

            function = tool_call.get(FUNCTION_KEY)
            if not isinstance(function, dict):
                continue

            tool_name = function.get(NAME_KEY)
            if not isinstance(tool_name, str) or not tool_name:
                continue

            tool_definition = _find_tool_definition(tools=tools, tool_name=tool_name)
            parent_id = fallback_parent_id
            start_ns = end_ns

        _emit_tool_span(
            trace_id=trace_id,
            parent_id=parent_id,
            start_ns=start_ns,
            end_ns=end_ns,
            tool_call=tool_call,
            tool_definition=tool_definition,
            tool_output=_format_tool_result_output(content=tool_result[CONTENT_KEY]),
            is_error=tool_result[IS_ERROR_KEY],
        )


def _emit_message_spans(
    kwargs: dict[str, Any],
    message: Any,
    start_ns: int,
    gateway_log_id: str | None = None,
) -> None:
    """Emit chat spans and resolve tool spans once tool_result content is available."""
    normalized_tools = _format_tools(tools=kwargs.get(TOOLS_KEY))
    tool_calls = _extract_tool_calls(message=message)
    attrs = _build_span_attrs(kwargs=kwargs, message=message)
    trace_id, parent_id = _resolve_injected_trace_parent_ids()
    chat_span_id = _generate_span_id()

    _emit_span(
        attrs=attrs,
        start_ns=start_ns,
        name=ANTHROPIC_CHAT_SPAN_NAME,
        trace_id=trace_id,
        parent_id=parent_id,
        span_id=chat_span_id,
        gateway_log_id=gateway_log_id,
    )

    _emit_resolved_tool_spans(
        kwargs=kwargs,
        trace_id=trace_id,
        fallback_parent_id=chat_span_id,
        end_ns=start_ns,
        tools=normalized_tools,
    )

    if tool_calls:
        _register_pending_tool_calls(
            trace_id=trace_id,
            parent_id=chat_span_id,
            tool_calls=tool_calls,
            tools=normalized_tools,
        )
