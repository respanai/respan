"""Emit IBM watsonx.ai SDK calls as OTEL ReadableSpan objects."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_watsonx._constants import (
    ASSISTANT_ROLE,
    COMPLETION_TOKENS_KEY,
    CONTENT_KEY,
    MODEL_ID_KEY,
    PROMPT_TOKENS_KEY,
    ROLE_KEY,
    TOTAL_TOKENS_KEY,
    WATSONX_CHAT_SPAN_NAME,
    WATSONX_EMBEDDING_SPAN_NAME,
    WATSONX_SYSTEM_NAME,
    WATSONX_TEXT_SPAN_NAME,
)
from respan_instrumentation_watsonx._translator import (
    embedding_dimension,
    embedding_vector_count,
    extract_chat_tool_calls,
    extract_usage,
    format_chat_output,
    format_input_messages,
    format_text_output,
    model_id_from_instance,
    normalize_chat_messages,
    normalize_embedding_inputs,
    normalize_text_prompts,
    normalize_tools,
    safe_json,
    to_attr_value,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TEXT,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_sdk.utils.data_processing.id_processing import format_span_id, format_trace_id
from respan_tracing.utils.span_factory import build_readable_span, inject_span

logger = logging.getLogger(__name__)

_GEN_AI_PROMPT_PREFIX = f"{TLSpanAttributes.LLM_PROMPTS}."
_GEN_AI_COMPLETION_PREFIX = f"{TLSpanAttributes.LLM_COMPLETIONS}."


def _request_type_value(name: str, fallback: str) -> str:
    value = getattr(LLMRequestTypeValues, name, None)
    return getattr(value, "value", fallback)


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
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


def _base_attrs(*, span_name: str, log_type: str, request_type: str) -> dict[str, Any]:
    attrs = {
        TLSpanAttributes.LLM_SYSTEM: WATSONX_SYSTEM_NAME,
        TLSpanAttributes.LLM_REQUEST_TYPE: request_type,
        TLSpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        RESPAN_LOG_TYPE: log_type,
    }
    workflow_name = context_api.get_value(TLSpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_model(attrs: dict[str, Any], instance: Any, request_kwargs: dict[str, Any]) -> None:
    model = request_kwargs.get(MODEL_ID_KEY) or model_id_from_instance(instance)
    if model:
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = str(model)


def _set_prompt_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        if role is not None:
            attrs[f"{_GEN_AI_PROMPT_PREFIX}{index}.role"] = str(role)
        if content is not None:
            attrs[f"{_GEN_AI_PROMPT_PREFIX}{index}.content"] = to_attr_value(content)


def _set_completion_attrs(attrs: dict[str, Any], content: str) -> None:
    attrs[f"{_GEN_AI_COMPLETION_PREFIX}0.role"] = ASSISTANT_ROLE
    attrs[f"{_GEN_AI_COMPLETION_PREFIX}0.content"] = content


def _set_usage_attrs(attrs: dict[str, Any], response_or_chunks: Any) -> None:
    usage = extract_usage(response_or_chunks)
    prompt_tokens = usage.get(PROMPT_TOKENS_KEY)
    completion_tokens = usage.get(COMPLETION_TOKENS_KEY)
    total_tokens = usage.get(TOTAL_TOKENS_KEY)
    if prompt_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
        attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
        attrs[gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
    if total_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def build_text_attrs(
    *,
    instance: Any,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=WATSONX_TEXT_SPAN_NAME,
        log_type=LOG_TYPE_TEXT,
        request_type=_request_type_value("COMPLETION", "completion"),
    )
    _set_model(attrs=attrs, instance=instance, request_kwargs=request_kwargs)
    messages = normalize_text_prompts(request_kwargs.get("prompt"))
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = format_input_messages(messages)
    _set_prompt_attrs(attrs=attrs, messages=messages)
    if response_or_chunks is not None:
        output = format_text_output(response_or_chunks)
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
        _set_completion_attrs(attrs=attrs, content=output)
        _set_usage_attrs(attrs=attrs, response_or_chunks=response_or_chunks)
    return attrs


def build_chat_attrs(
    *,
    instance: Any,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=WATSONX_CHAT_SPAN_NAME,
        log_type=LOG_TYPE_CHAT,
        request_type=_request_type_value("CHAT", "chat"),
    )
    _set_model(attrs=attrs, instance=instance, request_kwargs=request_kwargs)
    messages = normalize_chat_messages(request_kwargs.get("messages"))
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = format_input_messages(messages)
    _set_prompt_attrs(attrs=attrs, messages=messages)

    tools = normalize_tools(request_kwargs.get("tools"))
    if tools:
        attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(tools)

    if response_or_chunks is not None:
        output = format_chat_output(response_or_chunks)
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
        _set_completion_attrs(attrs=attrs, content=output)

        tool_calls = extract_chat_tool_calls(response_or_chunks)
        if tool_calls:
            attrs[f"{_GEN_AI_COMPLETION_PREFIX}0.tool_calls"] = safe_json(tool_calls)

        _set_usage_attrs(attrs=attrs, response_or_chunks=response_or_chunks)
    return attrs


def build_embedding_attrs(
    *,
    instance: Any,
    request_kwargs: dict[str, Any],
    response: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=WATSONX_EMBEDDING_SPAN_NAME,
        log_type=LOG_TYPE_EMBEDDING,
        request_type=_request_type_value("EMBEDDING", "embedding"),
    )
    _set_model(attrs=attrs, instance=instance, request_kwargs=request_kwargs)
    inputs = normalize_embedding_inputs(
        request_kwargs.get("inputs")
        if request_kwargs.get("inputs") is not None
        else request_kwargs.get("texts", request_kwargs.get("text"))
    )
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(inputs)
    attrs[f"{_GEN_AI_PROMPT_PREFIX}0.content"] = safe_json(inputs)

    if response is not None:
        summary = {
            "vector_count": embedding_vector_count(response),
            "dimension": embedding_dimension(response),
        }
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(
            {key: value for key, value in summary.items() if value is not None}
        )
        _set_usage_attrs(attrs=attrs, response_or_chunks=response)
    return attrs


def emit_span(
    *,
    span_name: str,
    attrs: dict[str, Any],
    start_ns: int,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        if error_message:
            attrs["error.message"] = error_message
            attrs.setdefault("status_code", status_code if status_code >= 400 else 500)
        trace_id, parent_id = _current_trace_parent_ids()
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
    except Exception:
        logger.debug("Failed to emit Watsonx span", exc_info=True)


def emit_text_span(
    *,
    instance: Any,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    emit_span(
        span_name=WATSONX_TEXT_SPAN_NAME,
        attrs=build_text_attrs(
            instance=instance,
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        ),
        start_ns=start_ns,
        error_message=error_message,
        status_code=status_code,
    )


def emit_chat_span(
    *,
    instance: Any,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    emit_span(
        span_name=WATSONX_CHAT_SPAN_NAME,
        attrs=build_chat_attrs(
            instance=instance,
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        ),
        start_ns=start_ns,
        error_message=error_message,
        status_code=status_code,
    )


def emit_embedding_span(
    *,
    instance: Any,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    emit_span(
        span_name=WATSONX_EMBEDDING_SPAN_NAME,
        attrs=build_embedding_attrs(
            instance=instance,
            request_kwargs=request_kwargs,
            response=response,
        ),
        start_ns=start_ns,
        error_message=error_message,
        status_code=status_code,
    )
