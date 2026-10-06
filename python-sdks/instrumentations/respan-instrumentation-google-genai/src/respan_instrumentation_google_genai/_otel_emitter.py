"""Emit Google Gen AI SDK calls as OTEL ReadableSpan objects."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_MESSAGE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
)
from respan_sdk.utils.data_processing.id_processing import (
    format_span_id,
    format_trace_id,
)
from respan_tracing.utils.span_factory import build_readable_span, inject_span

from respan_instrumentation_google_genai._constants import (
    ASSISTANT_ROLE,
    CANDIDATES_TOKEN_COUNT_KEY,
    CONFIG_KEY,
    CONTENTS_KEY,
    GOOGLE_GENAI_CHAT_SPAN_NAME,
    GOOGLE_GENAI_SYSTEM_NAME,
    MODEL_KEY,
    PROMPT_TOKEN_COUNT_KEY,
    ROLE_KEY,
    TOTAL_TOKEN_COUNT_KEY,
)
from respan_instrumentation_google_genai._translator import (
    capture_content,
    contents_with_history,
    extract_tool_calls,
    extract_tools,
    extract_usage,
    format_input,
    format_output,
    normalize_input_messages,
    safe_json,
    to_json_attr,
)

logger = logging.getLogger(__name__)


def _current_trace_parent_ids() -> tuple[str | None, str | None]:
    try:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
    except Exception:
        logger.debug("Unable to read current trace context", exc_info=True)
        return None, None

    trace_id = getattr(span_context, "trace_id", 0) or 0
    span_id = getattr(span_context, "span_id", 0) or 0
    if trace_id == 0 or span_id == 0:
        return None, None
    return format_trace_id(trace_id=trace_id), format_span_id(span_id=span_id)


def _base_attrs(*, is_streaming: bool) -> dict[str, Any]:
    attrs = {
        SpanAttributes.LLM_SYSTEM: GOOGLE_GENAI_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: GOOGLE_GENAI_CHAT_SPAN_NAME,
        SpanAttributes.TRACELOOP_ENTITY_PATH: GOOGLE_GENAI_CHAT_SPAN_NAME,
        SpanAttributes.LLM_IS_STREAMING: is_streaming,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
    }
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_input_attrs(attrs: dict[str, Any], contents: Any, config: Any) -> None:
    messages = normalize_input_messages(contents=contents, config=config)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = format_input(
        contents=contents,
        config=config,
    )
    for index, message in enumerate(messages):
        role = message.get(ROLE_KEY)
        content = message.get("content")
        if role is not None:
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] = str(role)
        if content is not None:
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.content"] = to_json_attr(
                content
            )
        if message.get("tool_calls"):
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.tool_calls"] = safe_json(
                message["tool_calls"]
            )


def _set_output_attrs(
    attrs: dict[str, Any],
    response_or_chunks: Any,
) -> None:
    if capture_content():
        output = format_output(response_or_chunks=response_or_chunks)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = ASSISTANT_ROLE
        attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = output

        tool_calls = extract_tool_calls(response_or_chunks=response_or_chunks)
        if tool_calls:
            attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"] = safe_json(
                value=tool_calls
            )

    usage = extract_usage(response_or_chunks=response_or_chunks)
    if PROMPT_TOKEN_COUNT_KEY in usage:
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage[PROMPT_TOKEN_COUNT_KEY]
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = usage[PROMPT_TOKEN_COUNT_KEY]
    if CANDIDATES_TOKEN_COUNT_KEY in usage:
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage[
            CANDIDATES_TOKEN_COUNT_KEY
        ]
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = usage[CANDIDATES_TOKEN_COUNT_KEY]
    if TOTAL_TOKEN_COUNT_KEY in usage:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage[TOTAL_TOKEN_COUNT_KEY]


def _set_request_attrs(attrs: dict[str, Any], request_kwargs: dict[str, Any]) -> None:
    model = request_kwargs.get(MODEL_KEY)
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = model

    if not capture_content():
        return
    config = request_kwargs.get(CONFIG_KEY)
    tools = extract_tools(config=config)
    if tools:
        tools_json = safe_json(value=tools)
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = tools_json

    _set_input_attrs(
        attrs=attrs,
        contents=request_kwargs.get(CONTENTS_KEY),
        config=config,
    )


def build_generate_content_attrs(
    *,
    request_kwargs: dict[str, Any],
    response_or_chunks: Any = None,
    is_streaming: bool = False,
) -> dict[str, Any]:
    attrs = _base_attrs(is_streaming=is_streaming)
    request_with_history = {
        **request_kwargs,
        CONTENTS_KEY: contents_with_history(
            request_kwargs.get(CONTENTS_KEY), response_or_chunks
        ),
    }
    _set_request_attrs(attrs=attrs, request_kwargs=request_with_history)
    if response_or_chunks is not None:
        _set_output_attrs(attrs=attrs, response_or_chunks=response_or_chunks)
    return attrs


def emit_generate_content_span(
    *,
    request_kwargs: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
    is_streaming: bool = False,
) -> None:
    """Build a ReadableSpan for a Google Gen AI generation and inject it."""
    try:
        attrs = build_generate_content_attrs(
            request_kwargs=request_kwargs,
            response_or_chunks=response_or_chunks,
            is_streaming=is_streaming,
        )
        if error_message:
            attrs[ERROR_MESSAGE] = error_message
            attrs.setdefault("status_code", status_code if status_code >= 400 else 500)
            attrs.setdefault(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                safe_json(
                    value={
                        "error": error_message,
                        "status_code": status_code if status_code >= 400 else 500,
                    }
                ),
            )

        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name=GOOGLE_GENAI_CHAT_SPAN_NAME,
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
        logger.debug("Failed to emit Google Gen AI span", exc_info=True)
