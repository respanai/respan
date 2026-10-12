"""Emit contract-compliant OpenAI SDK calls into the active OTel pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TEXT,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_GATEWAY_LOG_ID,
    RESPAN_LOG_TYPE,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_openai import _translator as tr
from respan_instrumentation_openai._constants import (
    ASSISTANT_ROLE,
    CHAT_SPAN_NAME,
    COMPLETION_SPAN_NAME,
    EMBEDDING_SPAN_NAME,
    GEN_AI_RESPONSE_ID,
    OPENAI_SYSTEM,
    REQUEST_TYPE_CHAT,
    REQUEST_TYPE_COMPLETION,
    REQUEST_TYPE_EMBEDDING,
    RESPONSE_SPAN_NAME,
)

logger = logging.getLogger(__name__)

_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."
_USAGE_INPUT = getattr(
    SpanAttributes, "LLM_USAGE_INPUT_TOKENS", "gen_ai.usage.input_tokens"
)
_USAGE_OUTPUT = getattr(
    SpanAttributes, "LLM_USAGE_OUTPUT_TOKENS", "gen_ai.usage.output_tokens"
)


# A span linked to a Respan gateway log row is a link only. The gateway row
# already carries the call's usage, cost and content; the span reporting them
# again would count the call twice (the server estimates tokens from content
# when usage is absent), so both are dropped and the trace shows the row.
_LINKED_DROPPED_PREFIXES = (
    "gen_ai.usage.",
    "llm.usage.",
    f"{SpanAttributes.LLM_PROMPTS}.",
    f"{SpanAttributes.LLM_COMPLETIONS}.",
)
_LINKED_DROPPED_KEYS = frozenset(
    {SpanAttributes.TRACELOOP_ENTITY_INPUT, SpanAttributes.TRACELOOP_ENTITY_OUTPUT}
)


def _link_to_gateway_log(attrs: dict[str, Any], gateway_log_id: str) -> None:
    attrs[RESPAN_GATEWAY_LOG_ID] = gateway_log_id
    for key in [
        key
        for key in attrs
        if key in _LINKED_DROPPED_KEYS or key.startswith(_LINKED_DROPPED_PREFIXES)
    ]:
        del attrs[key]


def current_trace_parent_ids() -> tuple[str | None, str | None]:
    context = trace.get_current_span().get_span_context()
    trace_id = getattr(context, "trace_id", 0) or 0
    span_id = getattr(context, "span_id", 0) or 0
    if not trace_id or not span_id:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _base_attrs(
    *, span_name: str, log_type: str, request_type: str, is_streaming: bool
) -> dict[str, Any]:
    # Auto-emitted spans intentionally do not set traceloop.span.kind.
    return {
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        SpanAttributes.LLM_SYSTEM: OPENAI_SYSTEM,
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.GEN_AI_IS_STREAMING: is_streaming,
        RESPAN_LOG_TYPE: log_type,
    }


def _set_model(
    attrs: dict[str, Any], request_kwargs: dict[str, Any], response: Any
) -> None:
    model = tr.request_model(request_kwargs)
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = model
    response_model = tr.response_model(response)
    if response_model:
        attrs[SpanAttributes.LLM_RESPONSE_MODEL] = response_model
    response_id = tr.response_id(response)
    if response_id:
        attrs[GEN_AI_RESPONSE_ID] = response_id


def _set_prompt_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role is not None:
            attrs[f"{_PROMPT_PREFIX}{index}.role"] = str(role)
        if content is not None:
            attrs[f"{_PROMPT_PREFIX}{index}.content"] = tr.to_attr_value(content)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            attrs[f"{_PROMPT_PREFIX}{index}.tool_calls"] = tr.safe_json(tool_calls)


def _set_completion_attrs(
    attrs: dict[str, Any], content: str, tool_calls: list[dict[str, Any]] | None = None
) -> None:
    attrs[f"{_COMPLETION_PREFIX}0.role"] = ASSISTANT_ROLE
    attrs[f"{_COMPLETION_PREFIX}0.content"] = content
    if tool_calls:
        attrs[f"{_COMPLETION_PREFIX}0.tool_calls"] = tr.safe_json(tool_calls)


def _set_usage(attrs: dict[str, Any], response: Any) -> None:
    usage = tr.extract_usage(response)
    if "prompt" in usage:
        attrs[_USAGE_INPUT] = usage["prompt"]
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage["prompt"]
    if "completion" in usage:
        attrs[_USAGE_OUTPUT] = usage["completion"]
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage["completion"]
    if "total" in usage:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage["total"]


def build_chat_attrs(
    *, request_kwargs: dict[str, Any], response: Any = None
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=CHAT_SPAN_NAME,
        log_type=LOG_TYPE_CHAT,
        request_type=REQUEST_TYPE_CHAT,
        is_streaming=bool(request_kwargs.get("stream")),
    )
    messages = tr.normalize_chat_messages(request_kwargs.get("messages"))
    managed_prompt = tr.managed_prompt_input(request_kwargs.get("extra_body"))
    if not messages and managed_prompt is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = tr.safe_json(managed_prompt)
    else:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = tr.format_input_messages(
            messages
        )
    _set_prompt_attrs(attrs, messages)
    tools = tr.normalize_tools(request_kwargs.get("tools"))
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = tr.safe_json(tools)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        output_message = tr.chat_output_message(response)
        tool_calls = tr.extract_chat_tool_calls(response)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = tr.safe_json(output_message)
        _set_completion_attrs(
            attrs, tr.format_chat_output(response), tool_calls=tool_calls
        )
        _set_usage(attrs, response)
    return attrs


def build_completion_attrs(
    *, request_kwargs: dict[str, Any], response: Any = None
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=COMPLETION_SPAN_NAME,
        log_type=LOG_TYPE_TEXT,
        request_type=REQUEST_TYPE_COMPLETION,
        is_streaming=bool(request_kwargs.get("stream")),
    )
    messages = tr.normalize_text_prompts(request_kwargs.get("prompt"))
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = tr.format_input_messages(messages)
    _set_prompt_attrs(attrs, messages)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        output = tr.format_completion_output(response)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = tr.safe_json({"text": output})
        _set_completion_attrs(attrs, output)
        _set_usage(attrs, response)
    return attrs


def build_response_attrs(
    *, request_kwargs: dict[str, Any], response: Any = None
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=RESPONSE_SPAN_NAME,
        log_type=LOG_TYPE_CHAT,
        request_type=REQUEST_TYPE_CHAT,
        is_streaming=bool(request_kwargs.get("stream")),
    )
    messages = tr.normalize_responses_input(
        request_kwargs.get("input"), request_kwargs.get("instructions")
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = tr.format_input_messages(messages)
    _set_prompt_attrs(attrs, messages)
    tools = tr.normalize_tools(request_kwargs.get("tools"))
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = tr.safe_json(tools)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        tool_calls = tr.extract_responses_tool_calls(response)
        output = tr.format_responses_output(response)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = tr.safe_json(
            tr.responses_output_payload(response)
        )
        _set_completion_attrs(attrs, output, tool_calls=tool_calls)
        _set_usage(attrs, response)
    return attrs


def build_embedding_attrs(
    *, request_kwargs: dict[str, Any], response: Any = None
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=EMBEDDING_SPAN_NAME,
        log_type=LOG_TYPE_EMBEDDING,
        request_type=REQUEST_TYPE_EMBEDDING,
        is_streaming=False,
    )
    inputs = tr.normalize_embedding_inputs(request_kwargs.get("input"))
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = tr.safe_json(inputs)
    if inputs:
        attrs[f"{_PROMPT_PREFIX}0.content"] = tr.safe_json(inputs)
    _set_model(attrs, request_kwargs, response)
    if response is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = tr.safe_json(
            tr.embedding_payload(response)
        )
        _set_usage(attrs, response)
    return attrs


def emit_span(
    *,
    span_name: str,
    attrs: dict[str, Any],
    start_ns: int,
    error_message: str | None = None,
    error_type: str | None = None,
    status_code: int = 200,
    trace_id: str | None = None,
    parent_id: str | None = None,
    gateway_log_id: str | None = None,
) -> None:
    try:
        if trace_id is None and parent_id is None:
            trace_id, parent_id = current_trace_parent_ids()
        attrs["status_code"] = status_code
        if error_message:
            attrs["error.message"] = error_message
            attrs["error.type"] = error_type or "OpenAIError"
            attrs.setdefault(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                tr.safe_json(
                    {
                        "error": error_type or "OpenAIError",
                        "message": error_message,
                        "status": "error",
                        "status_code": status_code,
                    }
                ),
            )
        if gateway_log_id:
            _link_to_gateway_log(attrs, gateway_log_id)
        span = build_readable_span(
            name=span_name,
            trace_id=trace_id,
            parent_id=parent_id,
            start_time_ns=start_ns,
            end_time_ns=time.time_ns(),
            attributes=attrs,
            error_message=error_message,
            status_code=status_code,
        )
        inject_span(span=span)
    except BaseException:
        logger.debug("Failed to emit OpenAI span %r", span_name, exc_info=True)


def _emit(
    builder: Callable[..., dict[str, Any]],
    span_name: str,
    *,
    request_kwargs: dict[str, Any],
    response: Any,
    start_ns: int,
    error_message: str | None,
    error_type: str | None,
    status_code: int,
    trace_id: str | None,
    parent_id: str | None,
    gateway_log_id: str | None = None,
) -> None:
    try:
        attrs = builder(request_kwargs=request_kwargs, response=response)
    except BaseException:
        logger.debug("Failed to build attrs for %r", span_name, exc_info=True)
        attrs = {}
    emit_span(
        span_name=span_name,
        attrs=attrs,
        start_ns=start_ns,
        error_message=error_message,
        error_type=error_type,
        status_code=status_code,
        trace_id=trace_id,
        parent_id=parent_id,
        gateway_log_id=gateway_log_id,
    )


def emit_chat_span(
    *,
    request_kwargs,
    start_ns,
    response=None,
    error_message=None,
    error_type=None,
    status_code=200,
    trace_id=None,
    parent_id=None,
    gateway_log_id=None,
) -> None:
    _emit(
        build_chat_attrs,
        CHAT_SPAN_NAME,
        request_kwargs=request_kwargs,
        response=response,
        start_ns=start_ns,
        error_message=error_message,
        error_type=error_type,
        status_code=status_code,
        trace_id=trace_id,
        parent_id=parent_id,
        gateway_log_id=gateway_log_id,
    )


def emit_completion_span(
    *,
    request_kwargs,
    start_ns,
    response=None,
    error_message=None,
    error_type=None,
    status_code=200,
    trace_id=None,
    parent_id=None,
    gateway_log_id=None,
) -> None:
    _emit(
        build_completion_attrs,
        COMPLETION_SPAN_NAME,
        request_kwargs=request_kwargs,
        response=response,
        start_ns=start_ns,
        error_message=error_message,
        error_type=error_type,
        status_code=status_code,
        trace_id=trace_id,
        parent_id=parent_id,
        gateway_log_id=gateway_log_id,
    )


def emit_response_span(
    *,
    request_kwargs,
    start_ns,
    response=None,
    error_message=None,
    error_type=None,
    status_code=200,
    trace_id=None,
    parent_id=None,
    gateway_log_id=None,
) -> None:
    _emit(
        build_response_attrs,
        RESPONSE_SPAN_NAME,
        request_kwargs=request_kwargs,
        response=response,
        start_ns=start_ns,
        error_message=error_message,
        error_type=error_type,
        status_code=status_code,
        trace_id=trace_id,
        parent_id=parent_id,
        gateway_log_id=gateway_log_id,
    )


def emit_embedding_span(
    *,
    request_kwargs,
    start_ns,
    response=None,
    error_message=None,
    error_type=None,
    status_code=200,
    trace_id=None,
    parent_id=None,
    gateway_log_id=None,
) -> None:
    _emit(
        build_embedding_attrs,
        EMBEDDING_SPAN_NAME,
        request_kwargs=request_kwargs,
        response=response,
        start_ns=start_ns,
        error_message=error_message,
        error_type=error_type,
        status_code=status_code,
        trace_id=trace_id,
        parent_id=parent_id,
        gateway_log_id=gateway_log_id,
    )
