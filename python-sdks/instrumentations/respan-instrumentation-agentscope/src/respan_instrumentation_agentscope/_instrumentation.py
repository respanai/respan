"""Native AgentScope instrumentation plugin for Respan."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes

from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_METHOD,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)
from respan_sdk.utils.data_processing.id_processing import (
    ensure_span_id,
    ensure_trace_id,
    format_span_id,
    format_trace_id,
)
from respan_sdk.utils.serialization import serialize_value
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)

AGENTSCOPE_INSTRUMENTATION_NAME = "agentscope"
AGENTSCOPE_AGENT_MODULE = "agentscope.agent"
AGENTSCOPE_MODEL_MODULE = "agentscope.model"
AGENTSCOPE_TOOL_MODULE = "agentscope.tool"
AGENT_CLASS_NAME = "Agent"
TOOLKIT_CLASS_NAME = "Toolkit"

REPLY_METHOD_NAME = "reply"
REPLY_STREAM_METHOD_NAME = "reply_stream"
CALL_TOOL_METHOD_NAME = "call_tool"
MODEL_CALL_METHOD_NAME = "__call__"

RESPAN_AGENTSCOPE_WRAPPED_ATTR = "_respan_agentscope_wrapped"
RESPAN_AGENTSCOPE_ORIGINALS_ATTR = "_respan_agentscope_originals"

AGENT_SPAN_NAME = "agentscope.agent"
MODEL_SPAN_NAME = "agentscope.model_call"
TOOL_SPAN_NAME = "agentscope.tool"

AGENTSCOPE_AGENT_NAME_ATTR = "agentscope.agent.name"
AGENTSCOPE_REPLY_ID_ATTR = "agentscope.reply.id"
AGENTSCOPE_SESSION_ID_ATTR = "agentscope.session.id"
AGENTSCOPE_MODEL_NAME_ATTR = "agentscope.model.name"
AGENTSCOPE_TOOL_NAME_ATTR = "agentscope.tool.name"
AGENTSCOPE_TOOL_CALL_ID_ATTR = "agentscope.tool.call.id"
AGENTSCOPE_TOOL_STATUS_ATTR = "agentscope.tool.status"

NAME_KEY = "name"
ROLE_KEY = "role"
CONTENT_KEY = "content"
ID_KEY = "id"
INPUT_KEY = "input"
STATE_KEY = "state"
TYPE_KEY = "type"
FUNCTION_KEY = "function"
ARGUMENTS_KEY = "arguments"
TOOL_CALLS_KEY = "tool_calls"
ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"
FUNCTION_TYPE = "function"
STATUS_CODE_ATTR = "status_code"

AGENTSCOPE_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
AGENTSCOPE_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
GEN_AI_USAGE_INPUT_TOKENS = GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
GEN_AI_USAGE_OUTPUT_TOKENS = GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
LLM_USAGE_CACHE_READ_INPUT_TOKENS = SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS

_MODEL_CLASS_NAMES = frozenset(
    {
        "ChatModelBase",
        "OpenAIChatModel",
        "OpenAIResponseModel",
        "AnthropicChatModel",
        "DashScopeChatModel",
        "DeepSeekChatModel",
        "GeminiChatModel",
        "MoonshotChatModel",
        "XAIChatModel",
        "OllamaChatModel",
    }
)


@dataclass(frozen=True)
class _RunContext:
    trace_id: str
    root_span_id: str
    parent_span_id: str | None


_CURRENT_RUN_CONTEXT: ContextVar[_RunContext | None] = ContextVar(
    "respan_agentscope_current_run_context",
    default=None,
)


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _current_parent_ids() -> tuple[str | None, str | None]:
    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()
    if not getattr(span_context, "is_valid", False):
        return None, None
    return (
        format_trace_id(span_context.trace_id),
        format_span_id(span_context.span_id),
    )


def _create_run_context(*, span_name: str, started_at_ns: int) -> _RunContext:
    current_context = _CURRENT_RUN_CONTEXT.get()
    root_span_id = format_span_id(
        ensure_span_id(val=f"{span_name}:{started_at_ns}:root")
    )
    if current_context is not None:
        return _RunContext(
            trace_id=current_context.trace_id,
            root_span_id=root_span_id,
            parent_span_id=current_context.root_span_id,
        )

    parent_trace_id, parent_span_id = _current_parent_ids()
    return _RunContext(
        trace_id=parent_trace_id
        or format_trace_id(ensure_trace_id(val=f"{span_name}:{started_at_ns}")),
        root_span_id=root_span_id,
        parent_span_id=parent_span_id,
    )


@contextmanager
def _use_run_context(run_context: _RunContext):
    token = _CURRENT_RUN_CONTEXT.set(run_context)
    try:
        yield
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return getattr(value, key, default)
    except Exception:
        return default


def _object_method(value: Any, key: str) -> Any | None:
    try:
        method = getattr(value, key, None)
    except Exception:
        return None
    if callable(method):
        return method
    return None


def _object_has_attr(value: Any, key: str) -> bool:
    try:
        getattr(value, key)
    except Exception:
        return False
    return True


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)

    model_dump = _object_method(value=value, key="model_dump")
    if model_dump is not None:
        try:
            converted = model_dump(mode="json")
            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            pass

    to_dict = _object_method(value=value, key="to_dict")
    if to_dict is not None:
        try:
            converted = to_dict()
            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            pass

    value_dict = _object_value(value=value, key="__dict__")
    if isinstance(value_dict, Mapping):
        return {
            key: item
            for key, item in value_dict.items()
            if not str(key).startswith("_")
        }

    return {"value": value}


def _json_string(value: Any) -> str:
    try:
        return json.dumps(
            serialize_value(value=value),
            default=str,
            separators=(",", ":"),
        )
    except Exception:
        return json.dumps(str(value), separators=(",", ":"))


def _attribute_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_string(value=value)


def _set_if_present(attributes: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        attributes[key] = value


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_provider(value: Any) -> str | None:
    if value is None:
        return None
    provider = str(value).strip().lower()
    if not provider:
        return None
    return provider.replace(" ", "_")


def _status_code_from_state(value: Any) -> int:
    normalized = str(_object_value(value=value, key="value", default=value or "")).lower()
    if any(marker in normalized for marker in ("error", "denied", "interrupted")):
        return 500
    return 200


def _call_original(
    original_method: Any,
    is_bound_method: bool,
    instance: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if is_bound_method:
        return original_method(*args, **kwargs)
    return original_method(instance, *args, **kwargs)


def _is_async_iterator(value: Any) -> bool:
    if inspect.isawaitable(value):
        return False
    return _object_has_attr(value=value, key="__aiter__")


def _block_payload(block: Any) -> Any:
    if isinstance(block, str):
        return block

    block_type = _object_value(value=block, key=TYPE_KEY)
    if block_type == "text" or _object_value(value=block, key="text") is not None:
        return _object_value(value=block, key="text")
    if block_type == "thinking" or _object_value(value=block, key="thinking") is not None:
        return _object_value(value=block, key="thinking")
    if block_type == "tool_call" or _object_value(value=block, key=INPUT_KEY) is not None:
        return {
            ID_KEY: _object_value(value=block, key=ID_KEY),
            NAME_KEY: _object_value(value=block, key=NAME_KEY),
            INPUT_KEY: _object_value(value=block, key=INPUT_KEY),
            STATE_KEY: str(_object_value(value=block, key=STATE_KEY, default="")),
        }
    if block_type == "tool_result" or _object_value(value=block, key="output") is not None:
        return {
            ID_KEY: _object_value(value=block, key=ID_KEY),
            NAME_KEY: _object_value(value=block, key=NAME_KEY),
            "output": _object_value(value=block, key="output"),
            STATE_KEY: str(_object_value(value=block, key=STATE_KEY, default="")),
        }
    return _object_to_dict(value=block)


def _content_blocks(value: Any) -> list[Any]:
    if value is None:
        return []

    get_content_blocks = _object_method(value=value, key="get_content_blocks")
    if get_content_blocks is not None:
        try:
            blocks = get_content_blocks()
            if isinstance(blocks, list):
                return blocks
        except Exception:
            pass

    content = _object_value(value=value, key=CONTENT_KEY)
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [content]


def _content_value(value: Any) -> Any:
    if value is None:
        return None
    get_text_content = _object_method(value=value, key="get_text_content")
    if get_text_content is not None:
        try:
            text = get_text_content()
        except Exception:
            text = None
        if text:
            return text

    blocks = _content_blocks(value=value)
    if not blocks:
        return _object_value(value=value, key=CONTENT_KEY, default=value)

    payloads = [_block_payload(block=block) for block in blocks]
    text_parts = [item for item in payloads if isinstance(item, str)]
    if text_parts and len(text_parts) == len(payloads):
        return "".join(text_parts)
    return payloads


def _normalize_message(message: Any) -> dict[str, Any]:
    role = _object_value(value=message, key=ROLE_KEY)
    name = _object_value(value=message, key=NAME_KEY)
    content = _content_value(value=message)
    normalized: dict[str, Any] = {
        ROLE_KEY: str(role or USER_ROLE),
        CONTENT_KEY: content,
    }
    if name:
        normalized[NAME_KEY] = str(name)

    tool_calls = _tool_calls_from_blocks(_content_blocks(value=message))
    if tool_calls:
        normalized[TOOL_CALLS_KEY] = tool_calls
    return normalized


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [_normalize_message(message=item) for item in value]


def _set_message_attributes(
    attributes: dict[str, Any],
    prefix: str,
    messages: list[dict[str, Any]],
) -> None:
    for message_index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        tool_calls = message.get(TOOL_CALLS_KEY)

        if role is not None:
            attributes[f"{prefix}{message_index}.role"] = str(role)
        if content is not None:
            attributes[f"{prefix}{message_index}.content"] = _attribute_string(
                value=content
            )
        if tool_calls:
            attributes[f"{prefix}{message_index}.{TOOL_CALLS_KEY}"] = _json_string(
                value=tool_calls
            )


def _tool_calls_from_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        block_type = _object_value(value=block, key=TYPE_KEY)
        tool_name = _object_value(value=block, key=NAME_KEY)
        tool_input = _object_value(value=block, key=INPUT_KEY)
        if block_type != "tool_call" and tool_input is None:
            continue
        if not tool_name:
            continue
        tool_calls.append(
            {
                ID_KEY: str(_object_value(value=block, key=ID_KEY) or tool_name),
                TYPE_KEY: FUNCTION_TYPE,
                FUNCTION_KEY: {
                    NAME_KEY: str(tool_name),
                    ARGUMENTS_KEY: str(tool_input or "{}"),
                },
            }
        )
    return tool_calls


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None, int | None]:
    usage = _object_value(value=response, key="usage")
    input_tokens = _coerce_int(
        _object_value(value=usage, key="input_tokens")
        or _object_value(value=usage, key="prompt_tokens")
    )
    output_tokens = _coerce_int(
        _object_value(value=usage, key="output_tokens")
        or _object_value(value=usage, key="completion_tokens")
    )
    total_tokens = _coerce_int(_object_value(value=usage, key="total_tokens"))
    cache_read_tokens = _coerce_int(
        _object_value(value=usage, key="cache_input_tokens")
        or _object_value(value=usage, key="cache_read_input_tokens")
    )

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return input_tokens, output_tokens, total_tokens, cache_read_tokens


def _model_name(model: Any) -> str:
    return str(_object_value(value=model, key="model") or type(model).__name__)


def _agent_name(agent: Any) -> str:
    return str(_object_value(value=agent, key=NAME_KEY) or type(agent).__name__)


def _agent_metadata(agent: Any) -> dict[str, Any]:
    state = _object_value(value=agent, key="state")
    metadata: dict[str, Any] = {}
    for public_key, source_key in (
        (AGENTSCOPE_SESSION_ID_ATTR, "session_id"),
        (AGENTSCOPE_REPLY_ID_ATTR, "reply_id"),
    ):
        value = _object_value(value=state, key=source_key)
        if value is not None:
            metadata[public_key] = value
    return metadata


def _emit_span(
    *,
    name: str,
    attributes: dict[str, Any],
    start_time_ns: int,
    end_time_ns: int,
    status_code: int = 200,
    error_message: str | None = None,
    run_context: _RunContext | None = None,
) -> None:
    try:
        if run_context is not None:
            trace_id = run_context.trace_id
            span_id = run_context.root_span_id
            parent_id = run_context.parent_span_id
        else:
            current_context = _CURRENT_RUN_CONTEXT.get()
            if current_context is None:
                trace_id, parent_id = _current_parent_ids()
            else:
                trace_id, parent_id = current_context.trace_id, current_context.root_span_id
            span_id = None

        if status_code >= 400:
            attributes.setdefault(STATUS_CODE_ATTR, status_code)
            if error_message:
                attributes.setdefault(ERROR_MESSAGE_ATTR, error_message)

        span = build_readable_span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            attributes=attributes,
            status_code=status_code,
            error_message=error_message,
        )
        inject_span(span=span)
    except Exception:
        logger.debug("Failed to emit AgentScope span %r", name, exc_info=True)


def _agent_input_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, Mapping) and CONTENT_KEY in value:
        return value.get(CONTENT_KEY)
    content = _content_value(value=value)
    if content is not None and content is not value:
        return content
    return value


def _agent_attributes(
    *,
    agent: Any,
    input_value: Any,
    output_value: Any,
) -> dict[str, Any]:
    entity_name = _agent_name(agent=agent)
    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_AGENT,
        SpanAttributes.TRACELOOP_ENTITY_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        SpanAttributes.TRACELOOP_WORKFLOW_NAME: entity_name,
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _attribute_string(
            value=_agent_input_value(value=input_value)
        ),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: _attribute_string(
            value=_content_value(value=output_value)
        ),
        AGENTSCOPE_AGENT_NAME_ATTR: entity_name,
    }
    for key, value in _agent_metadata(agent=agent).items():
        attributes[key] = str(value)
    metadata = _object_value(value=output_value, key="metadata")
    if metadata:
        attributes[RESPAN_METADATA] = _json_string(value=metadata)
    return attributes


def _model_attributes(
    *,
    model: Any,
    messages: Any,
    tools: Any,
    response: Any,
) -> dict[str, Any]:
    normalized_messages = _normalize_messages(value=messages)
    completion_message = {
        ROLE_KEY: ASSISTANT_ROLE,
        CONTENT_KEY: _content_value(value=response),
    }
    completion_tool_calls = _tool_calls_from_blocks(_content_blocks(value=response))
    if completion_tool_calls:
        completion_message[TOOL_CALLS_KEY] = completion_tool_calls

    model_name = _model_name(model=model)
    provider = _normalize_provider(
        _object_value(value=model, key="provider")
        or _object_value(value=model, key="model_type")
        or type(model).__name__.replace("ChatModel", "")
    )
    input_tokens, output_tokens, total_tokens, cache_read_tokens = _extract_usage(
        response=response
    )

    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
        SpanAttributes.TRACELOOP_ENTITY_NAME: "model_call",
        SpanAttributes.TRACELOOP_ENTITY_PATH: "model_call",
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _json_string(value=normalized_messages),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: _attribute_string(
            value=completion_message.get(CONTENT_KEY)
        ),
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.LLM_REQUEST_MODEL: model_name,
        AGENTSCOPE_MODEL_NAME_ATTR: model_name,
    }
    _set_if_present(attributes=attributes, key=SpanAttributes.LLM_SYSTEM, value=provider)

    if tools:
        attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = _json_string(value=tools)

    _set_if_present(
        attributes=attributes,
        key=GEN_AI_USAGE_INPUT_TOKENS,
        value=input_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=GEN_AI_USAGE_OUTPUT_TOKENS,
        value=output_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        value=input_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        value=output_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        value=total_tokens,
    )
    _set_if_present(
        attributes=attributes,
        key=LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        value=cache_read_tokens,
    )

    _set_message_attributes(
        attributes=attributes,
        prefix=AGENTSCOPE_PROMPT_PREFIX,
        messages=normalized_messages,
    )
    _set_message_attributes(
        attributes=attributes,
        prefix=AGENTSCOPE_COMPLETION_PREFIX,
        messages=[completion_message],
    )
    return attributes


def _tool_call_input(tool_call: Any) -> dict[str, Any]:
    return {
        NAME_KEY: _object_value(value=tool_call, key=NAME_KEY),
        ARGUMENTS_KEY: _object_value(value=tool_call, key=INPUT_KEY),
    }


def _tool_output(chunks: list[Any]) -> Any:
    if not chunks:
        return ""
    final = chunks[-1]
    content = _content_value(value=final)
    if content is not None:
        return content
    return _object_to_dict(value=final)


def _tool_attributes(*, tool_call: Any, chunks: list[Any]) -> dict[str, Any]:
    tool_name = str(_object_value(value=tool_call, key=NAME_KEY) or "tool")
    tool_call_id = _object_value(value=tool_call, key=ID_KEY)
    final_state = _object_value(value=chunks[-1], key=STATE_KEY) if chunks else None
    attributes: dict[str, Any] = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
        SpanAttributes.TRACELOOP_ENTITY_NAME: tool_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: f"tool.{tool_name}",
        SpanAttributes.TRACELOOP_ENTITY_INPUT: _json_string(
            value=_tool_call_input(tool_call=tool_call)
        ),
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT: _attribute_string(
            value=_tool_output(chunks=chunks)
        ),
        AGENTSCOPE_TOOL_NAME_ATTR: tool_name,
    }
    _set_if_present(
        attributes=attributes,
        key=AGENTSCOPE_TOOL_CALL_ID_ATTR,
        value=tool_call_id,
    )
    _set_if_present(
        attributes=attributes,
        key=AGENTSCOPE_TOOL_STATUS_ATTR,
        value=str(_object_value(value=final_state, key="value", default=final_state)),
    )
    return attributes


def _emit_agent_call(
    *,
    agent: Any,
    input_value: Any,
    output_value: Any,
    started_at_ns: int,
    ended_at_ns: int,
    run_context: _RunContext,
) -> None:
    _emit_span(
        name=AGENT_SPAN_NAME,
        attributes=_agent_attributes(
            agent=agent,
            input_value=input_value,
            output_value=output_value,
        ),
        start_time_ns=started_at_ns,
        end_time_ns=ended_at_ns,
        run_context=run_context,
    )


def _emit_agent_error(
    *,
    agent: Any,
    input_value: Any,
    exception: Exception,
    started_at_ns: int,
    ended_at_ns: int,
    run_context: _RunContext,
) -> None:
    _emit_span(
        name=AGENT_SPAN_NAME,
        attributes=_agent_attributes(
            agent=agent,
            input_value=input_value,
            output_value={"error": str(exception)},
        ),
        start_time_ns=started_at_ns,
        end_time_ns=ended_at_ns,
        status_code=500,
        error_message=str(exception),
        run_context=run_context,
    )


def _emit_model_call(
    *,
    model: Any,
    messages: Any,
    tools: Any,
    response: Any,
    started_at_ns: int,
    ended_at_ns: int,
    status_code: int = 200,
    error_message: str | None = None,
) -> None:
    _emit_span(
        name=MODEL_SPAN_NAME,
        attributes=_model_attributes(
            model=model,
            messages=messages,
            tools=tools,
            response=response,
        ),
        start_time_ns=started_at_ns,
        end_time_ns=ended_at_ns,
        status_code=status_code,
        error_message=error_message,
    )


def _emit_tool_call(
    *,
    tool_call: Any,
    chunks: list[Any],
    started_at_ns: int,
    ended_at_ns: int,
    status_code: int | None = None,
    error_message: str | None = None,
) -> None:
    final_state = _object_value(value=chunks[-1], key=STATE_KEY) if chunks else None
    _emit_span(
        name=TOOL_SPAN_NAME,
        attributes=_tool_attributes(tool_call=tool_call, chunks=chunks),
        start_time_ns=started_at_ns,
        end_time_ns=ended_at_ns,
        status_code=status_code or _status_code_from_state(value=final_state),
        error_message=error_message,
    )


def _agent_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("inputs")


def _model_messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("messages")


def _model_tools(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if len(args) > 1:
        return args[1]
    return kwargs.get("tools")


def _tool_call_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("tool_call")


def _wrap_agent_reply(original_method: Any, *, is_bound_method: bool) -> Any:
    @functools.wraps(original_method)
    async def wrapped_agent_reply(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at_ns = time.time_ns()
        input_value = _agent_input(args=args, kwargs=kwargs)
        run_context = _create_run_context(
            span_name=AGENT_SPAN_NAME,
            started_at_ns=started_at_ns,
        )
        with _use_run_context(run_context=run_context):
            try:
                result = await _call_original(
                    original_method,
                    is_bound_method,
                    self,
                    *args,
                    **kwargs,
                )
            except Exception as exception:
                _emit_agent_error(
                    agent=self,
                    input_value=input_value,
                    exception=exception,
                    started_at_ns=started_at_ns,
                    ended_at_ns=time.time_ns(),
                    run_context=run_context,
                )
                raise

        _emit_agent_call(
            agent=self,
            input_value=input_value,
            output_value=result,
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
            run_context=run_context,
        )
        return result

    setattr(wrapped_agent_reply, RESPAN_AGENTSCOPE_WRAPPED_ATTR, True)
    return wrapped_agent_reply


def _wrap_agent_reply_stream(original_method: Any, *, is_bound_method: bool) -> Any:
    @functools.wraps(original_method)
    def wrapped_agent_reply_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        async def stream() -> AsyncIterator[Any]:
            started_at_ns = time.time_ns()
            input_value = _agent_input(args=args, kwargs=kwargs)
            run_context = _create_run_context(
                span_name=AGENT_SPAN_NAME,
                started_at_ns=started_at_ns,
            )
            items: list[Any] = []
            with _use_run_context(run_context=run_context):
                try:
                    result = _call_original(
                        original_method,
                        is_bound_method,
                        self,
                        *args,
                        **kwargs,
                    )
                    async for item in result:
                        items.append(item)
                        yield item
                except Exception as exception:
                    _emit_agent_error(
                        agent=self,
                        input_value=input_value,
                        exception=exception,
                        started_at_ns=started_at_ns,
                        ended_at_ns=time.time_ns(),
                        run_context=run_context,
                    )
                    raise

            output_value = items[-1] if items else None
            _emit_agent_call(
                agent=self,
                input_value=input_value,
                output_value=output_value,
                started_at_ns=started_at_ns,
                ended_at_ns=time.time_ns(),
                run_context=run_context,
            )

        return stream()

    setattr(wrapped_agent_reply_stream, RESPAN_AGENTSCOPE_WRAPPED_ATTR, True)
    return wrapped_agent_reply_stream


def _wrap_model_stream(
    *,
    async_iterator: Any,
    model: Any,
    messages: Any,
    tools: Any,
    started_at_ns: int,
) -> AsyncIterator[Any]:
    async def stream() -> AsyncIterator[Any]:
        chunks: list[Any] = []
        try:
            async for chunk in async_iterator:
                chunks.append(chunk)
                yield chunk
        except Exception as exception:
            response = chunks[-1] if chunks else {"error": str(exception)}
            _emit_model_call(
                model=model,
                messages=messages,
                tools=tools,
                response=response,
                started_at_ns=started_at_ns,
                ended_at_ns=time.time_ns(),
                status_code=500,
                error_message=str(exception),
            )
            raise
        else:
            completed_response = next(
                (chunk for chunk in reversed(chunks) if _object_value(chunk, "is_last")),
                chunks[-1] if chunks else None,
            )
            _emit_model_call(
                model=model,
                messages=messages,
                tools=tools,
                response=completed_response,
                started_at_ns=started_at_ns,
                ended_at_ns=time.time_ns(),
            )

    return stream()


def _wrap_model_call(original_method: Any, *, is_bound_method: bool) -> Any:
    @functools.wraps(original_method)
    async def wrapped_model_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        started_at_ns = time.time_ns()
        messages = _model_messages(args=args, kwargs=kwargs)
        tools = _model_tools(args=args, kwargs=kwargs)
        try:
            result = await _call_original(
                original_method,
                is_bound_method,
                self,
                *args,
                **kwargs,
            )
        except Exception as exception:
            _emit_model_call(
                model=self,
                messages=messages,
                tools=tools,
                response={"error": str(exception)},
                started_at_ns=started_at_ns,
                ended_at_ns=time.time_ns(),
                status_code=500,
                error_message=str(exception),
            )
            raise

        if _is_async_iterator(value=result):
            return _wrap_model_stream(
                async_iterator=result,
                model=self,
                messages=messages,
                tools=tools,
                started_at_ns=started_at_ns,
            )

        _emit_model_call(
            model=self,
            messages=messages,
            tools=tools,
            response=result,
            started_at_ns=started_at_ns,
            ended_at_ns=time.time_ns(),
        )
        return result

    setattr(wrapped_model_call, RESPAN_AGENTSCOPE_WRAPPED_ATTR, True)
    return wrapped_model_call


def _wrap_toolkit_call_tool(original_method: Any, *, is_bound_method: bool) -> Any:
    @functools.wraps(original_method)
    def wrapped_call_tool(self: Any, *args: Any, **kwargs: Any) -> Any:
        async def stream() -> AsyncIterator[Any]:
            started_at_ns = time.time_ns()
            tool_call = _tool_call_arg(args=args, kwargs=kwargs)
            chunks: list[Any] = []
            try:
                result = _call_original(
                    original_method,
                    is_bound_method,
                    self,
                    *args,
                    **kwargs,
                )
                async for chunk in result:
                    chunks.append(chunk)
                    yield chunk
            except Exception as exception:
                _emit_tool_call(
                    tool_call=tool_call,
                    chunks=chunks,
                    started_at_ns=started_at_ns,
                    ended_at_ns=time.time_ns(),
                    status_code=500,
                    error_message=str(exception),
                )
                raise

            _emit_tool_call(
                tool_call=tool_call,
                chunks=chunks,
                started_at_ns=started_at_ns,
                ended_at_ns=time.time_ns(),
            )

        return stream()

    setattr(wrapped_call_tool, RESPAN_AGENTSCOPE_WRAPPED_ATTR, True)
    return wrapped_call_tool


def _model_classes_from_module(module: Any) -> list[type[Any]]:
    classes: list[type[Any]] = []
    for _, value in vars(module).items():
        if not inspect.isclass(value):
            continue
        class_name = getattr(value, "__name__", "")
        is_model_class = (
            class_name in _MODEL_CLASS_NAMES
            or class_name.endswith("ChatModel")
            or class_name.endswith("ChatModelBase")
        )
        if is_model_class and _object_has_attr(value=value, key=MODEL_CALL_METHOD_NAME):
            classes.append(value)
    return list(dict.fromkeys(classes))


class AgentScopeInstrumentor:
    """Respan instrumentor for AgentScope.

    The default activation patches AgentScope's public Agent, ChatModelBase, and
    Toolkit classes. Specific instances can also be supplied for applications
    that define custom model or toolkit subclasses outside AgentScope modules.
    """

    name = AGENTSCOPE_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        agent: Any | None = None,
        model: Any | None = None,
        toolkit: Any | None = None,
        instrument_models: bool = True,
        instrument_tools: bool = True,
    ) -> None:
        self._agent = agent
        self._model = model
        self._toolkit = toolkit
        self._instrument_models = instrument_models
        self._instrument_tools = instrument_tools
        self._patches: list[tuple[Any, str, Any]] = []
        self._is_instrumented = False

    def _patch_method(
        self,
        *,
        target: Any,
        method_name: str,
        wrapper_factory: Any,
        is_bound_method: bool,
    ) -> bool:
        original_method = _object_value(value=target, key=method_name)
        if original_method is None:
            return False
        if _object_value(
            value=original_method,
            key=RESPAN_AGENTSCOPE_WRAPPED_ATTR,
            default=False,
        ):
            return True

        wrapped_method = wrapper_factory(
            original_method,
            is_bound_method=is_bound_method,
        )
        if is_bound_method:
            wrapped_method = MethodType(wrapped_method, target)

        setattr(target, method_name, wrapped_method)
        originals = _object_value(
            value=target,
            key=RESPAN_AGENTSCOPE_ORIGINALS_ATTR,
            default={},
        )
        originals[method_name] = original_method
        setattr(target, RESPAN_AGENTSCOPE_ORIGINALS_ATTR, originals)
        self._patches.append((target, method_name, original_method))
        return True

    def _patch_agent_target(self, target: Any, *, is_bound_method: bool) -> bool:
        patched = False
        patched |= self._patch_method(
            target=target,
            method_name=REPLY_METHOD_NAME,
            wrapper_factory=_wrap_agent_reply,
            is_bound_method=is_bound_method,
        )
        patched |= self._patch_method(
            target=target,
            method_name=REPLY_STREAM_METHOD_NAME,
            wrapper_factory=_wrap_agent_reply_stream,
            is_bound_method=is_bound_method,
        )
        return patched

    def _patch_model_target(self, target: Any, *, is_bound_method: bool) -> bool:
        if is_bound_method:
            target = type(target)
            is_bound_method = False
        return self._patch_method(
            target=target,
            method_name=MODEL_CALL_METHOD_NAME,
            wrapper_factory=_wrap_model_call,
            is_bound_method=is_bound_method,
        )

    def _patch_toolkit_target(self, target: Any, *, is_bound_method: bool) -> bool:
        return self._patch_method(
            target=target,
            method_name=CALL_TOOL_METHOD_NAME,
            wrapper_factory=_wrap_toolkit_call_tool,
            is_bound_method=is_bound_method,
        )

    @staticmethod
    def _load_agent_class() -> type[Any]:
        module = importlib.import_module(AGENTSCOPE_AGENT_MODULE)
        return getattr(module, AGENT_CLASS_NAME)

    @staticmethod
    def _load_model_module() -> Any:
        return importlib.import_module(AGENTSCOPE_MODEL_MODULE)

    @staticmethod
    def _load_toolkit_class() -> type[Any]:
        module = importlib.import_module(AGENTSCOPE_TOOL_MODULE)
        return getattr(module, TOOLKIT_CLASS_NAME)

    def activate(self) -> None:
        """Activate AgentScope instrumentation."""
        if self._is_instrumented:
            return

        if not _is_respan_tracing_enabled():
            logger.info(
                "AgentScope instrumentation skipped because Respan tracing is disabled"
            )
            return

        patched_any = False

        if self._agent is not None:
            patched_any |= self._patch_agent_target(
                self._agent,
                is_bound_method=True,
            )
        else:
            try:
                patched_any |= self._patch_agent_target(
                    self._load_agent_class(),
                    is_bound_method=False,
                )
            except ImportError as exception:
                logger.warning(
                    "Failed to activate AgentScope agent instrumentation - missing dependency: %s",
                    exception,
                )

        if self._instrument_models:
            if self._model is not None:
                patched_any |= self._patch_model_target(
                    self._model,
                    is_bound_method=True,
                )
            else:
                try:
                    model_module = self._load_model_module()
                except ImportError as exception:
                    logger.info(
                        "AgentScope model instrumentation skipped - missing dependency: %s",
                        exception,
                    )
                else:
                    for model_class in _model_classes_from_module(model_module):
                        patched_any |= self._patch_model_target(
                            model_class,
                            is_bound_method=False,
                        )

        if self._instrument_tools:
            if self._toolkit is not None:
                patched_any |= self._patch_toolkit_target(
                    self._toolkit,
                    is_bound_method=True,
                )
            else:
                try:
                    patched_any |= self._patch_toolkit_target(
                        self._load_toolkit_class(),
                        is_bound_method=False,
                    )
                except ImportError as exception:
                    logger.info(
                        "AgentScope toolkit instrumentation skipped - missing dependency: %s",
                        exception,
                    )

        self._is_instrumented = bool(patched_any)
        if self._is_instrumented:
            logger.info("AgentScope instrumentation activated")

    def deactivate(self) -> None:
        """Restore patched AgentScope methods."""
        for target, method_name, original_method in reversed(self._patches):
            try:
                setattr(target, method_name, original_method)
                originals = _object_value(
                    value=target,
                    key=RESPAN_AGENTSCOPE_ORIGINALS_ATTR,
                    default={},
                )
                if isinstance(originals, dict):
                    originals.pop(method_name, None)
                    if originals:
                        setattr(target, RESPAN_AGENTSCOPE_ORIGINALS_ATTR, originals)
                    elif _object_has_attr(
                        value=target,
                        key=RESPAN_AGENTSCOPE_ORIGINALS_ATTR,
                    ):
                        delattr(target, RESPAN_AGENTSCOPE_ORIGINALS_ATTR)
            except Exception:
                logger.debug(
                    "Failed to restore AgentScope method %s on %r",
                    method_name,
                    target,
                    exc_info=True,
                )
        self._patches.clear()
        self._is_instrumented = False
        logger.info("AgentScope instrumentation deactivated")
