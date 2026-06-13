"""Emit Vertex AI SDK calls as OTEL ReadableSpan objects."""

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

from respan_instrumentation_vertexai._constants import (
    ASSISTANT_ROLE,
    CANDIDATES_TOKEN_COUNT_KEY,
    PROMPT_TOKEN_COUNT_KEY,
    ROLE_KEY,
    SYSTEM_INSTRUCTION_KEY,
    TOOLS_KEY,
    TOTAL_TOKEN_COUNT_KEY,
    VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
    VERTEXAI_SYSTEM_NAME,
)
from respan_instrumentation_vertexai._translator import (
    extract_tool_calls,
    extract_tools,
    extract_usage,
    format_input,
    format_output,
    normalize_input_messages,
    safe_json,
    to_json_attr,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT
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


def _base_attrs(span_name: str) -> dict[str, Any]:
    attrs = {
        SpanAttributes.LLM_SYSTEM: VERTEXAI_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
    }
    workflow_name = context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_input_attrs(attrs: dict[str, Any], request_payload: dict[str, Any]) -> None:
    contents = request_payload.get("contents")
    system_instruction = request_payload.get(SYSTEM_INSTRUCTION_KEY)
    messages = normalize_input_messages(
        contents=contents,
        system_instruction=system_instruction,
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = format_input(
        contents=contents,
        system_instruction=system_instruction,
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


def _set_output_attrs(attrs: dict[str, Any], response_or_chunks: Any) -> None:
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
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = usage[PROMPT_TOKEN_COUNT_KEY]
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage[PROMPT_TOKEN_COUNT_KEY]
    if CANDIDATES_TOKEN_COUNT_KEY in usage:
        attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = usage[
            CANDIDATES_TOKEN_COUNT_KEY
        ]
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage[
            CANDIDATES_TOKEN_COUNT_KEY
        ]
    if TOTAL_TOKEN_COUNT_KEY in usage:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage[TOTAL_TOKEN_COUNT_KEY]


def _set_request_attrs(attrs: dict[str, Any], request_payload: dict[str, Any]) -> None:
    model = request_payload.get("model")
    if model:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = model

    tools = extract_tools(request_payload.get(TOOLS_KEY))
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(value=tools)

    _set_input_attrs(attrs=attrs, request_payload=request_payload)


def build_generate_content_attrs(
    *,
    request_payload: dict[str, Any],
    response_or_chunks: Any = None,
    span_name: str = VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
) -> dict[str, Any]:
    attrs = _base_attrs(span_name=span_name)
    _set_request_attrs(attrs=attrs, request_payload=request_payload)
    if response_or_chunks is not None:
        _set_output_attrs(attrs=attrs, response_or_chunks=response_or_chunks)
    return attrs


def emit_generate_content_span(
    *,
    request_payload: dict[str, Any],
    start_ns: int,
    response_or_chunks: Any = None,
    span_name: str = VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    """Build a ReadableSpan for a Vertex AI generation and inject it."""
    try:
        attrs = build_generate_content_attrs(
            request_payload=request_payload,
            response_or_chunks=response_or_chunks,
            span_name=span_name,
        )
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
        logger.debug("Failed to emit Vertex AI span", exc_info=True)
