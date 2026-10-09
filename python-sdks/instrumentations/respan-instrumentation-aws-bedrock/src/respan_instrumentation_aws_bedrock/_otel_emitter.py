"""Translate observed Bedrock payloads into the canonical span contract."""

from __future__ import annotations

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

from respan_instrumentation_aws_bedrock._constants import AWS_BEDROCK_SYSTEM_NAME
from respan_instrumentation_aws_bedrock._privacy import value
from respan_instrumentation_aws_bedrock._translator import (
    parse_bedrock_request,
    parse_bedrock_response,
    parse_bedrock_stream_response,
    safe_json,
    to_json_attr,
)


def build_bedrock_attrs(
    *,
    operation_name,
    api_params,
    response_payload=None,
    stream_events=None,
    capture_content=True,
):
    attrs = {
        SpanAttributes.LLM_SYSTEM: AWS_BEDROCK_SYSTEM_NAME,
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value,
        SpanAttributes.TRACELOOP_ENTITY_NAME: operation_name,
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT,
    }
    params = value(api_params) if type(api_params) is dict else {}
    response_payload = value(response_payload)
    stream_events = value(stream_events)
    if type(params.get("modelId")) is str:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = params["modelId"]
    if not capture_content:
        return attrs
    request = parse_bedrock_request(operation_name=operation_name, api_params=params)
    if request.messages:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(request.raw_payload)
        for index, message in enumerate(request.messages):
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.role"] = message["role"]
            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.content"] = to_json_attr(
                message["content"]
            )
            if "tool_calls" in message:
                attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.tool_calls"] = safe_json(
                    message["tool_calls"]
                )
    elif request.raw_payload is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(request.raw_payload)
    if request.tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(request.tools)
    response = None
    if stream_events is not None:
        response = parse_bedrock_stream_response(
            operation_name=operation_name, events=stream_events
        )
    elif response_payload is not None:
        response = parse_bedrock_response(
            operation_name=operation_name, response_payload=response_payload
        )
    if response is None:
        return attrs
    payload = response.raw_payload
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(payload)
    embedding = None
    if type(payload) is dict:
        for key in ("embedding", "embeddings"):
            if key in payload:
                embedding = payload[key]
                break
    if embedding is not None:
        attrs[RESPAN_LOG_TYPE] = LOG_TYPE_EMBEDDING
        attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.EMBEDDING.value
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(embedding)
    else:
        if response.content or response.tool_calls or _has_text(payload):
            attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] = response.role
        if response.content or _has_text(payload):
            attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] = response.content
        if response.tool_calls:
            attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"] = safe_json(
                response.tool_calls
            )
    usage = response.usage
    if "input_tokens" in usage:
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = usage["input_tokens"]
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage["input_tokens"]
    if "output_tokens" in usage:
        attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = usage["output_tokens"]
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = usage["output_tokens"]
    if "total_tokens" in usage:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage["total_tokens"]
    if "cache_read_input_tokens" in usage:
        attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = usage[
            "cache_read_input_tokens"
        ]
    if "cache_creation_input_tokens" in usage:
        attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] = usage[
            "cache_creation_input_tokens"
        ]
    return attrs


def _has_text(payload):
    if type(payload) is dict:
        if payload.get("type") in ("thinking", "reasoning"):
            return False
        return any(
            type(item) is str
            and key in ("text", "outputText", "generation", "completion")
            or key
            in (
                "output",
                "message",
                "content",
                "delta",
                "contentBlockDelta",
                "chunk",
                "bytes",
                "outputs",
                "results",
            )
            and _has_text(item)
            for key, item in payload.items()
        )
    if type(payload) is list:
        return any(_has_text(item) for item in payload)
    return False
