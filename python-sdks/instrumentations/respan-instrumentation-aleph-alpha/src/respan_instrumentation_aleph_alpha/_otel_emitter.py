"""Emit Aleph Alpha SDK calls as OTEL ReadableSpan objects."""

from __future__ import annotations

import logging
import time
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

from respan_instrumentation_aleph_alpha._constants import (
    ALEPH_ALPHA_SYSTEM_NAME,
    CONTENT_KEY,
    OPERATION_BATCH_SEMANTIC_EMBED,
    OPERATION_CHAT,
    OPERATION_CHAT_STREAM,
    OPERATION_COMPLETE,
    OPERATION_COMPLETE_STREAM,
    OPERATION_EMBED,
    OPERATION_EMBEDDINGS,
    OPERATION_EVALUATE,
    OPERATION_EXPLAIN,
    OPERATION_INSTRUCTABLE_EMBED,
    OPERATION_SEMANTIC_EMBED,
    PROMPT_KEY,
    ROLE_KEY,
)
from respan_instrumentation_aleph_alpha._translator import (
    chat_output,
    completion_texts,
    embedding_input,
    embedding_output_summary,
    first_non_empty,
    input_messages_from_chat_payload,
    input_messages_from_completion_payload,
    prompt_content,
    request_payload,
    safe_json,
    to_json_attr,
    tools_from_payload,
    usage_from_response,
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

_EMBEDDING_OPERATIONS = {
    OPERATION_EMBED,
    OPERATION_EMBEDDINGS,
    OPERATION_SEMANTIC_EMBED,
    OPERATION_BATCH_SEMANTIC_EMBED,
    OPERATION_INSTRUCTABLE_EMBED,
}

_TEXT_OPERATIONS = {
    OPERATION_COMPLETE,
    OPERATION_COMPLETE_STREAM,
    OPERATION_EVALUATE,
    OPERATION_EXPLAIN,
}

_REQUEST_ATTRS = {
    "maximum_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
    "temperature": SpanAttributes.LLM_REQUEST_TEMPERATURE,
    "top_p": SpanAttributes.LLM_REQUEST_TOP_P,
    "top_k": SpanAttributes.LLM_TOP_K,
    "presence_penalty": SpanAttributes.LLM_PRESENCE_PENALTY,
    "frequency_penalty": SpanAttributes.LLM_FREQUENCY_PENALTY,
}


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


def _span_name(operation: str) -> str:
    return f"aleph_alpha.{operation}"


def _base_attrs(operation: str) -> dict[str, Any]:
    if operation in (OPERATION_CHAT, OPERATION_CHAT_STREAM):
        log_type = LOG_TYPE_CHAT
        request_type = LLMRequestTypeValues.CHAT.value
    elif operation in _EMBEDDING_OPERATIONS:
        log_type = LOG_TYPE_EMBEDDING
        request_type = LLMRequestTypeValues.EMBEDDING.value
    else:
        log_type = LOG_TYPE_TEXT
        request_type = LLMRequestTypeValues.COMPLETION.value

    span_name = _span_name(operation=operation)
    attrs: dict[str, Any] = {
        SpanAttributes.LLM_SYSTEM: ALEPH_ALPHA_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: request_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: span_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: span_name,
        RESPAN_LOG_TYPE: log_type,
    }
    workflow_name = context_api.get_value(
        SpanAttributes.TRACELOOP_WORKFLOW_NAME
    ) or context_api.get_value(SpanAttributes.TRACELOOP_ENTITY_NAME)
    if workflow_name:
        attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflow_name
    return attrs


def _set_model(attrs: dict[str, Any], model: str | None, response: Any = None) -> None:
    model_name = first_non_empty(
        (
            model,
            getattr(response, "model", None),
            getattr(response, "model_version", None),
        )
    )
    if model_name:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = str(model_name)


def _set_usage_attrs(attrs: dict[str, Any], response_or_items: Any) -> None:
    usage = usage_from_response(response_or_items=response_or_items)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if isinstance(prompt_tokens, int):
        attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = prompt_tokens
    if isinstance(completion_tokens, int):
        attrs[gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = completion_tokens
    if isinstance(total_tokens, int):
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total_tokens


def _set_request_attrs(attrs: dict[str, Any], payload: dict[str, Any]) -> None:
    for payload_key, attr_key in _REQUEST_ATTRS.items():
        value = payload.get(payload_key)
        if value is not None:
            attrs[attr_key] = value


def _set_message_attrs(attrs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(value=messages)
    for index, message in enumerate(messages):
        prefix = f"{gen_ai_attributes.GEN_AI_PROMPT}.{index}"
        role = message.get(ROLE_KEY)
        content = message.get(CONTENT_KEY)
        tool_calls = message.get("tool_calls")
        if role is not None:
            attrs[f"{prefix}.role"] = str(role)
        if content is not None:
            attrs[f"{prefix}.content"] = to_json_attr(content)
        if tool_calls:
            attrs[f"{prefix}.tool_calls"] = safe_json(value=tool_calls)


def _set_tools_attrs(attrs: dict[str, Any], payload: dict[str, Any]) -> None:
    tools = tools_from_payload(payload=payload)
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(value=tools)


def _set_chat_attrs(
    attrs: dict[str, Any],
    *,
    payload: dict[str, Any],
    response_or_items: Any,
) -> None:
    _set_message_attrs(attrs=attrs, messages=input_messages_from_chat_payload(payload))
    _set_tools_attrs(attrs=attrs, payload=payload)
    if response_or_items is None:
        return

    content, role, tool_calls, finish_reason = chat_output(response_or_items)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = content
    completion_prefix = f"{gen_ai_attributes.GEN_AI_COMPLETION}.0"
    attrs[f"{completion_prefix}.role"] = role
    attrs[f"{completion_prefix}.content"] = content
    if tool_calls:
        attrs[f"{completion_prefix}.tool_calls"] = safe_json(value=tool_calls)
    if finish_reason:
        attrs[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = finish_reason
    _set_usage_attrs(attrs=attrs, response_or_items=response_or_items)


def _set_completion_attrs(
    attrs: dict[str, Any],
    *,
    payload: dict[str, Any],
    response_or_items: Any,
) -> None:
    _set_message_attrs(
        attrs=attrs,
        messages=input_messages_from_completion_payload(payload),
    )
    if response_or_items is None:
        return

    texts = completion_texts(response_or_items=response_or_items)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = (
        texts[0] if len(texts) == 1 else safe_json(value=texts)
    )
    for index, text in enumerate(texts):
        prefix = f"{gen_ai_attributes.GEN_AI_COMPLETION}.{index}"
        attrs[f"{prefix}.role"] = "assistant"
        attrs[f"{prefix}.content"] = text
    _set_usage_attrs(attrs=attrs, response_or_items=response_or_items)


def _set_text_model_attrs(
    attrs: dict[str, Any],
    *,
    payload: dict[str, Any],
    response: Any,
) -> None:
    input_value = (
        prompt_content(payload[PROMPT_KEY]) if PROMPT_KEY in payload else payload
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = to_json_attr(input_value)
    if response is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(value=response)
        _set_usage_attrs(attrs=attrs, response_or_items=response)


def _set_embedding_attrs(
    attrs: dict[str, Any],
    *,
    payload: dict[str, Any],
    response: Any,
) -> None:
    input_value = embedding_input(payload=payload)
    input_attr = to_json_attr(input_value)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = input_attr
    prompt_prefix = f"{gen_ai_attributes.GEN_AI_PROMPT}.0"
    attrs[f"{prompt_prefix}.role"] = "user"
    attrs[f"{prompt_prefix}.content"] = input_attr
    if response is not None:
        output_attr = safe_json(value=embedding_output_summary(response))
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output_attr
        completion_prefix = f"{gen_ai_attributes.GEN_AI_COMPLETION}.0"
        attrs[f"{completion_prefix}.role"] = "assistant"
        attrs[f"{completion_prefix}.content"] = output_attr
        _set_usage_attrs(attrs=attrs, response_or_items=response)


def build_aleph_alpha_attrs(
    *,
    operation: str,
    request: Any,
    model: str | None,
    response_or_items: Any = None,
) -> dict[str, Any]:
    attrs = _base_attrs(operation=operation)
    payload = request_payload(request)
    _set_model(attrs=attrs, model=model, response=response_or_items)
    _set_request_attrs(attrs=attrs, payload=payload)

    if operation in (OPERATION_CHAT, OPERATION_CHAT_STREAM):
        _set_chat_attrs(attrs=attrs, payload=payload, response_or_items=response_or_items)
    elif operation in (OPERATION_COMPLETE, OPERATION_COMPLETE_STREAM):
        _set_completion_attrs(
            attrs=attrs,
            payload=payload,
            response_or_items=response_or_items,
        )
    elif operation in _EMBEDDING_OPERATIONS:
        _set_embedding_attrs(attrs=attrs, payload=payload, response=response_or_items)
    elif operation in _TEXT_OPERATIONS:
        _set_text_model_attrs(attrs=attrs, payload=payload, response=response_or_items)
    return attrs


def emit_aleph_alpha_span(
    *,
    operation: str,
    request: Any,
    model: str | None,
    start_ns: int,
    response_or_items: Any = None,
    error_message: str | None = None,
    status_code: int = 200,
) -> None:
    """Build and inject a ReadableSpan for an Aleph Alpha SDK call."""
    try:
        attrs = build_aleph_alpha_attrs(
            operation=operation,
            request=request,
            model=model,
            response_or_items=response_or_items,
        )
        if error_message:
            attrs["error.message"] = error_message
            attrs.setdefault("status_code", status_code if status_code >= 400 else 500)

        trace_id, parent_id = _current_trace_parent_ids()
        span = build_readable_span(
            name=_span_name(operation=operation),
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
        logger.debug("Failed to emit Aleph Alpha span", exc_info=True)
