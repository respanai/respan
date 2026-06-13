"""Emit Together SDK calls as OTEL ReadableSpan objects."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_together._constants import (
    ASSISTANT_ROLE,
    BANNED_ALIAS_ATTRS,
    CONTENT_KEY,
    MODEL_KEY,
    ROLE_KEY,
    TOGETHER_CHAT_SPAN_NAME,
    TOGETHER_COMPLETION_SPAN_NAME,
    TOGETHER_EMBEDDING_SPAN_NAME,
    TOGETHER_IMAGE_SPAN_NAME,
    TOGETHER_RERANK_SPAN_NAME,
    TOGETHER_SYSTEM_NAME,
    TOOL_CALLS_KEY,
    TOOLS_KEY,
    USER_ROLE,
)
from respan_instrumentation_together._translator import (
    chat_input,
    chat_input_messages,
    chat_output,
    embedding_input,
    embedding_summary,
    extract_tool_calls,
    extract_usage,
    finish_reason,
    image_input,
    image_output,
    normalize_tools,
    request_model,
    rerank_input,
    rerank_output,
    safe_json,
    text_completion_input,
    text_completion_output,
    to_json_attr,
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


def _base_attrs(
    *,
    span_name: str,
    log_type: str,
    request_type: str,
) -> dict[str, Any]:
    attrs = {
        SpanAttributes.LLM_SYSTEM: TOGETHER_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        RESPAN_LOG_TYPE: log_type,
    }
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_model(attrs: dict[str, Any], request_kwargs: dict[str, Any], response: Any) -> None:
    model = request_model(request_kwargs=request_kwargs, response=response)
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = model


def _set_usage(attrs: dict[str, Any], response_or_chunks: Any) -> None:
    usage = extract_usage(response_or_chunks=response_or_chunks)
    if "prompt_tokens" in usage:
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage["completion_tokens"]
    if "total_tokens" in usage:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage["total_tokens"]


def _set_chat_input_attrs(attrs: dict[str, Any], request_kwargs: dict[str, Any]) -> None:
    messages = chat_input_messages(request_kwargs=request_kwargs)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = chat_input(request_kwargs)
    for index, message in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{index}"
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        if role is not None:
            attrs[f"{prefix}.role"] = str(role)
        if content is not None:
            attrs[f"{prefix}.content"] = to_json_attr(content)
        tool_calls = message.get(TOOL_CALLS_KEY)
        if tool_calls:
            attrs[f"{prefix}.tool_calls"] = safe_json(tool_calls)

    tools = normalize_tools(request_kwargs.get(TOOLS_KEY))
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(tools)


def _set_chat_output_attrs(
    attrs: dict[str, Any],
    response_or_chunks: Any,
) -> None:
    output = chat_output(response_or_chunks=response_or_chunks)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
    completion_prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
    attrs[f"{completion_prefix}.role"] = ASSISTANT_ROLE
    attrs[f"{completion_prefix}.content"] = output

    tool_calls = extract_tool_calls(response_or_chunks=response_or_chunks)
    if tool_calls:
        attrs[f"{completion_prefix}.tool_calls"] = safe_json(tool_calls)

    reason = finish_reason(response_or_chunks=response_or_chunks)
    if reason:
        attrs[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = reason
    _set_usage(attrs=attrs, response_or_chunks=response_or_chunks)


def build_chat_attrs(
    *,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=TOGETHER_CHAT_SPAN_NAME,
        log_type=LOG_TYPE_CHAT,
        request_type=LLMRequestTypeValues.CHAT.value,
    )
    _set_model(attrs=attrs, request_kwargs=request_kwargs, response=response_or_chunks)
    _set_chat_input_attrs(attrs=attrs, request_kwargs=request_kwargs)
    if response_or_chunks is not None:
        _set_chat_output_attrs(attrs=attrs, response_or_chunks=response_or_chunks)
    _assert_no_banned_aliases(attrs)
    return attrs


def build_completion_attrs(
    *,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=TOGETHER_COMPLETION_SPAN_NAME,
        log_type=LOG_TYPE_TEXT,
        request_type=LLMRequestTypeValues.COMPLETION.value,
    )
    _set_model(attrs=attrs, request_kwargs=request_kwargs, response=response_or_chunks)
    prompt = text_completion_input(request_kwargs)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = prompt
    attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] = USER_ROLE
    attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = prompt
    if response_or_chunks is not None:
        output = text_completion_output(response_or_chunks=response_or_chunks)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = ASSISTANT_ROLE
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = output
        reason = finish_reason(response_or_chunks=response_or_chunks)
        if reason:
            attrs[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = reason
        _set_usage(attrs=attrs, response_or_chunks=response_or_chunks)
    _assert_no_banned_aliases(attrs)
    return attrs


def build_embedding_attrs(
    *,
    request_kwargs: dict[str, Any],
    response: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=TOGETHER_EMBEDDING_SPAN_NAME,
        log_type=LOG_TYPE_EMBEDDING,
        request_type=LLMRequestTypeValues.EMBEDDING.value,
    )
    _set_model(attrs=attrs, request_kwargs=request_kwargs, response=response)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = embedding_input(request_kwargs)
    attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = embedding_input(request_kwargs)
    if response is not None:
        summary = embedding_summary(response)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(summary)
    _assert_no_banned_aliases(attrs)
    return attrs


def build_rerank_attrs(
    *,
    request_kwargs: dict[str, Any],
    response: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=TOGETHER_RERANK_SPAN_NAME,
        log_type=LOG_TYPE_TEXT,
        request_type=LLMRequestTypeValues.RERANK.value,
    )
    _set_model(attrs=attrs, request_kwargs=request_kwargs, response=response)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = rerank_input(request_kwargs)
    if response is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = rerank_output(response)
        _set_usage(attrs=attrs, response_or_chunks=response)
    _assert_no_banned_aliases(attrs)
    return attrs


def build_image_attrs(
    *,
    request_kwargs: dict[str, Any],
    response: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=TOGETHER_IMAGE_SPAN_NAME,
        log_type=LOG_TYPE_TEXT,
        request_type=LLMRequestTypeValues.UNKNOWN.value,
    )
    _set_model(attrs=attrs, request_kwargs=request_kwargs, response=response)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = image_input(request_kwargs)
    if response is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = image_output(response)
    _assert_no_banned_aliases(attrs)
    return attrs


def _assert_no_banned_aliases(attrs: dict[str, Any]) -> None:
    present_aliases = BANNED_ALIAS_ATTRS.intersection(attrs)
    if present_aliases:
        raise AssertionError(
            "Together instrumentation emitted off-contract aliases: "
            f"{sorted(present_aliases)}"
        )


def build_attrs(
    *,
    operation: str,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    if operation == "chat":
        return build_chat_attrs(
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        )
    if operation == "completion":
        return build_completion_attrs(
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        )
    if operation == "embedding":
        return build_embedding_attrs(
            request_kwargs=request_kwargs,
            response=response_or_chunks,
        )
    if operation == "rerank":
        return build_rerank_attrs(
            request_kwargs=request_kwargs,
            response=response_or_chunks,
        )
    if operation == "image":
        return build_image_attrs(
            request_kwargs=request_kwargs,
            response=response_or_chunks,
        )
    raise ValueError(f"Unsupported Together operation: {operation}")


def emit_together_span(
    *,
    operation: str,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    """Build a ReadableSpan for a Together SDK call and inject it."""
    try:
        attrs = build_attrs(
            operation=operation,
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        )
        if error_message:
            attrs["error.message"] = error_message
            attrs.setdefault("status_code", status_code if status_code >= 400 else 500)

        trace_id, parent_id = _current_trace_parent_ids()
        span_name = str(attrs[SpanAttributes.TRACELOOP_ENTITY_NAME])
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
        logger.debug("Failed to emit Together span", exc_info=True)
