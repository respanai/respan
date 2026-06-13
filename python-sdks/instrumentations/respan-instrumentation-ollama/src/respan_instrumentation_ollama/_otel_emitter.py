"""Emit Ollama SDK calls as OTEL ReadableSpan objects."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_ollama._constants import (
    ASSISTANT_ROLE,
    CHAT_METHOD_NAME,
    CONTENT_KEY,
    EMBED_METHOD_NAME,
    EMBEDDINGS_METHOD_NAME,
    EVAL_COUNT_KEY,
    GENERATE_METHOD_NAME,
    MESSAGE_KEY,
    MESSAGES_KEY,
    OLLAMA_CHAT_SPAN_NAME,
    OLLAMA_EMBED_SPAN_NAME,
    OLLAMA_GENERATE_SPAN_NAME,
    OLLAMA_SYSTEM_NAME,
    PROMPT_EVAL_COUNT_KEY,
    PROMPT_KEY,
    ROLE_KEY,
    SYSTEM_KEY,
    TOOLS_KEY,
    USER_ROLE,
)
from respan_instrumentation_ollama._translator import (
    extract_chat_tool_calls,
    extract_model,
    extract_usage,
    format_chat_input,
    format_chat_output,
    format_embedding_input,
    format_generate_input,
    format_generate_output,
    normalize_chat_messages,
    normalize_generate_messages,
    normalize_tools,
    safe_json,
    to_json_attr,
)
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
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


def _base_attrs(*, span_name: str, log_type: str, request_type: str) -> dict[str, Any]:
    attrs = {
        GenAIAttributes.GEN_AI_SYSTEM: OLLAMA_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        RESPAN_LOG_TYPE: log_type,
    }
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_model_attrs(
    *,
    attrs: dict[str, Any],
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> None:
    model = extract_model(
        request_kwargs=request_kwargs,
        response_or_chunks=response_or_chunks,
    )
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = model
        attrs.setdefault(GenAIAttributes.GEN_AI_RESPONSE_MODEL, model)


def _set_usage_attrs(attrs: dict[str, Any], response_or_chunks: Any) -> None:
    usage = extract_usage(response_or_chunks=response_or_chunks)
    prompt_tokens = usage.get(PROMPT_EVAL_COUNT_KEY)
    completion_tokens = usage.get(EVAL_COUNT_KEY)
    if prompt_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if completion_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if prompt_tokens is not None or completion_tokens is not None:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = (prompt_tokens or 0) + (
            completion_tokens or 0
        )


def _set_prompt_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        tool_calls = message.get("tool_calls")
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{index}"
        if role is not None:
            attrs[f"{prefix}.role"] = str(role)
        if content is not None:
            attrs[f"{prefix}.content"] = to_json_attr(content)
        if tool_calls:
            attrs[f"{prefix}.tool_calls"] = safe_json(tool_calls)


def _set_completion_attrs(
    *,
    attrs: dict[str, Any],
    role: str = ASSISTANT_ROLE,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
) -> None:
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
    attrs[f"{prefix}.role"] = role or ASSISTANT_ROLE
    attrs[f"{prefix}.content"] = content
    if tool_calls:
        attrs[f"{prefix}.tool_calls"] = safe_json(tool_calls)


def _set_tool_definition_attrs(attrs: dict[str, Any], tools: Any) -> None:
    normalized_tools = normalize_tools(tools)
    if normalized_tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(normalized_tools)


def build_chat_attrs(
    *,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=OLLAMA_CHAT_SPAN_NAME,
        log_type=LOG_TYPE_CHAT,
        request_type=LLMRequestTypeValues.CHAT.value,
    )
    _set_model_attrs(
        attrs=attrs,
        request_kwargs=request_kwargs,
        response_or_chunks=response_or_chunks,
    )

    messages = normalize_chat_messages(request_kwargs.get(MESSAGES_KEY))
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = format_chat_input(
        messages=request_kwargs.get(MESSAGES_KEY),
        tools=request_kwargs.get(TOOLS_KEY),
    )
    _set_prompt_attrs(attrs, messages)
    _set_tool_definition_attrs(attrs, request_kwargs.get(TOOLS_KEY))

    if response_or_chunks is not None:
        output = format_chat_output(response_or_chunks)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
        tool_calls = extract_chat_tool_calls(response_or_chunks)
        role = ASSISTANT_ROLE
        if not isinstance(response_or_chunks, list):
            message = getattr(response_or_chunks, MESSAGE_KEY, None)
            role = getattr(message, ROLE_KEY, None) or ASSISTANT_ROLE
        _set_completion_attrs(
            attrs=attrs,
            role=role,
            content=output,
            tool_calls=tool_calls,
        )
        _set_usage_attrs(attrs, response_or_chunks)
    return attrs


def build_generate_attrs(
    *,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=OLLAMA_GENERATE_SPAN_NAME,
        log_type=LOG_TYPE_CHAT,
        request_type=LLMRequestTypeValues.CHAT.value,
    )
    _set_model_attrs(
        attrs=attrs,
        request_kwargs=request_kwargs,
        response_or_chunks=response_or_chunks,
    )

    messages = normalize_generate_messages(
        prompt=request_kwargs.get(PROMPT_KEY),
        system=request_kwargs.get(SYSTEM_KEY),
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = format_generate_input(
        prompt=request_kwargs.get(PROMPT_KEY),
        system=request_kwargs.get(SYSTEM_KEY),
    )
    _set_prompt_attrs(attrs, messages)

    if response_or_chunks is not None:
        output = format_generate_output(response_or_chunks)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
        _set_completion_attrs(attrs=attrs, content=output)
        _set_usage_attrs(attrs, response_or_chunks)
    return attrs


def build_embedding_attrs(
    *,
    request_kwargs: dict[str, Any],
    response: Any = None,
    method_name: str = EMBED_METHOD_NAME,
) -> dict[str, Any]:
    attrs = _base_attrs(
        span_name=OLLAMA_EMBED_SPAN_NAME,
        log_type=LOG_TYPE_EMBEDDING,
        request_type=LLMRequestTypeValues.EMBEDDING.value,
    )
    _set_model_attrs(
        attrs=attrs, request_kwargs=request_kwargs, response_or_chunks=response
    )

    input_value = request_kwargs.get("input")
    prompt = request_kwargs.get(PROMPT_KEY)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = format_embedding_input(
        prompt=prompt,
        input_value=input_value,
    )
    prompt_content = input_value if method_name == EMBED_METHOD_NAME else prompt
    if prompt_content is not None:
        attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] = USER_ROLE
        attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = to_json_attr(prompt_content)

    if response is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = ""
        _set_usage_attrs(attrs, response)
    return attrs


def _emit_span(
    *,
    span_name: str,
    attrs: dict[str, Any],
    start_ns: int,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
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


def emit_chat_span(
    *,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        attrs = build_chat_attrs(
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        )
        _emit_span(
            span_name=OLLAMA_CHAT_SPAN_NAME,
            attrs=attrs,
            start_ns=start_ns,
            error_message=error_message,
            status_code=status_code,
        )
    except Exception:
        logger.debug("Failed to emit Ollama chat span", exc_info=True)


def emit_generate_span(
    *,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    try:
        attrs = build_generate_attrs(
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
        )
        _emit_span(
            span_name=OLLAMA_GENERATE_SPAN_NAME,
            attrs=attrs,
            start_ns=start_ns,
            error_message=error_message,
            status_code=status_code,
        )
    except Exception:
        logger.debug("Failed to emit Ollama generate span", exc_info=True)


def emit_embedding_span(
    *,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
    method_name: str = EMBED_METHOD_NAME,
) -> None:
    try:
        attrs = build_embedding_attrs(
            request_kwargs=request_kwargs,
            response=response,
            method_name=method_name,
        )
        _emit_span(
            span_name=OLLAMA_EMBED_SPAN_NAME,
            attrs=attrs,
            start_ns=start_ns,
            error_message=error_message,
            status_code=status_code,
        )
    except Exception:
        logger.debug("Failed to emit Ollama embedding span", exc_info=True)


METHOD_TO_EMITTER = {
    CHAT_METHOD_NAME: emit_chat_span,
    GENERATE_METHOD_NAME: emit_generate_span,
    EMBED_METHOD_NAME: emit_embedding_span,
    EMBEDDINGS_METHOD_NAME: emit_embedding_span,
}
