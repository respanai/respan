"""Emit AWS Bedrock Runtime calls as OTEL ReadableSpan objects."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as TLSpanAttributes

from respan_instrumentation_aws_bedrock._constants import (
    AWS_BEDROCK_CHAT_SPAN_NAME,
    AWS_BEDROCK_SYSTEM_NAME,
)
from respan_instrumentation_aws_bedrock._translator import (
    BedrockResponse,
    parse_bedrock_request,
    parse_bedrock_response,
    parse_bedrock_stream_response,
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


def _base_attrs() -> dict[str, Any]:
    attrs = {
        TLSpanAttributes.LLM_SYSTEM: AWS_BEDROCK_SYSTEM_NAME,
        TLSpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        TLSpanAttributes.TRACELOOP_ENTITY_NAME: AWS_BEDROCK_CHAT_SPAN_NAME,
        TLSpanAttributes.TRACELOOP_ENTITY_PATH: AWS_BEDROCK_CHAT_SPAN_NAME,
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
    }
    workflow_name = context_api.get_value(TLSpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[TLSpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_prompt_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role is not None:
            attrs[f"{TLSpanAttributes.LLM_PROMPTS}.{index}.role"] = str(role)
        if content is not None:
            attrs[f"{TLSpanAttributes.LLM_PROMPTS}.{index}.content"] = to_json_attr(
                content
            )


def _set_usage_attrs(attrs: dict[str, Any], usage: Mapping[str, int]) -> None:
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if input_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        attrs[TLSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input_tokens
    if output_tokens is not None:
        attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        attrs[TLSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output_tokens
    if total_tokens is not None:
        attrs[TLSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def _set_response_attrs(attrs: dict[str, Any], response: BedrockResponse) -> None:
    attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.role"] = response.role
    attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.content"] = response.content
    attrs[TLSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = response.content
    if response.tool_calls:
        attrs[f"{TLSpanAttributes.LLM_COMPLETIONS}.0.tool_calls"] = safe_json(
            response.tool_calls
        )
    _set_usage_attrs(attrs=attrs, usage=response.usage)


def build_bedrock_attrs(
    *,
    operation_name: str,
    api_params: Mapping[str, Any] | None,
    response_payload: Any = None,
    stream_events: list[Any] | None = None,
) -> dict[str, Any]:
    attrs = _base_attrs()
    request = parse_bedrock_request(
        operation_name=operation_name,
        api_params=api_params,
    )

    if request.model_id:
        attrs[TLSpanAttributes.LLM_REQUEST_MODEL] = request.model_id

    if request.messages:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(request.messages)
        _set_prompt_attrs(attrs=attrs, messages=request.messages)
    elif request.raw_payload is not None:
        attrs[TLSpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(request.raw_payload)

    if request.tools:
        attrs[TLSpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(request.tools)

    response: BedrockResponse | None = None
    if stream_events is not None:
        response = parse_bedrock_stream_response(
            operation_name=operation_name,
            events=stream_events,
        )
    elif response_payload is not None:
        response = parse_bedrock_response(
            operation_name=operation_name,
            response_payload=response_payload,
        )

    if response is not None:
        _set_response_attrs(attrs=attrs, response=response)

    return attrs


def emit_bedrock_span(
    *,
    operation_name: str,
    api_params: Mapping[str, Any] | None,
    start_ns: int,
    response_payload: Any = None,
    stream_events: list[Any] | None = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    """Build a ReadableSpan for an AWS Bedrock Runtime call and inject it."""
    try:
        attrs = build_bedrock_attrs(
            operation_name=operation_name,
            api_params=api_params,
            response_payload=response_payload,
            stream_events=stream_events,
        )
        if error_message:
            attrs["error.message"] = error_message
            status_code = status_code if status_code >= 400 else 500

        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name=AWS_BEDROCK_CHAT_SPAN_NAME,
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
        logger.debug("Failed to emit AWS Bedrock span", exc_info=True)
