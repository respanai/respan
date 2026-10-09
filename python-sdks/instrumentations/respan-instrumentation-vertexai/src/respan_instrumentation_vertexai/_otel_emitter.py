"""Canonical attributes from known Vertex request/response data."""

from __future__ import annotations

import math

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

from respan_instrumentation_vertexai._serialization import json_dumps
from respan_instrumentation_vertexai._translator import (
    extract_tools,
    extract_usage,
    native_value,
    normalize_input_messages,
    response_messages,
    response_values,
)


def base_attributes(embedding=False):
    return {
        SpanAttributes.LLM_SYSTEM: "google",
        SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.EMBEDDING.value
        if embedding
        else LLMRequestTypeValues.CHAT.value,
        RESPAN_LOG_TYPE: LOG_TYPE_EMBEDDING if embedding else LOG_TYPE_CHAT,
    }


def request_attributes(request):
    attrs = {}
    if request.get("model"):
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = request["model"]
    if request.get("stream"):
        attrs[SpanAttributes.LLM_IS_STREAMING] = True
    if request.get("embedding"):
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(native_value(request))
        return attrs
    messages = normalize_input_messages(
        request.get("contents"), system_instruction=request.get("system_instruction")
    )
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(
        {"messages": messages, "request": native_value(request)}
    )
    for index, message in enumerate(messages):
        for key in ("role", "content", "tool_calls"):
            if key in message:
                value = message[key]
                attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.{key}"] = (
                    value
                    if type(value) is str and key != "tool_calls"
                    else json_dumps(value)
                )
    tools = extract_tools(request.get("tools"))
    if tools:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_dumps(tools)
    config = native_value(request.get("generation_config"))
    if type(config) is dict:
        if type(config.get("temperature")) in {float, int}:
            attrs[SpanAttributes.LLM_REQUEST_TEMPERATURE] = config["temperature"]
        if type(config.get("max_output_tokens")) is int:
            attrs[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = config["max_output_tokens"]
    return attrs


def response_attributes(response, *, embedding=False):
    if embedding:
        values = native_value(response)
        if type(values) is not list:
            return {}
        vectors = [
            value["values"]
            for value in values
            if type(value) is dict and type(value.get("values")) is list
        ]
        attrs = (
            {SpanAttributes.TRACELOOP_ENTITY_OUTPUT: json_dumps(vectors)}
            if vectors
            else {}
        )
        counts = [
            value.get("statistics", {}).get("token_count")
            for value in values
            if type(value) is dict and type(value.get("statistics")) is dict
        ]
        if (
            len(counts) == len(values)
            and counts
            and all(
                type(count) in {int, float}
                and math.isfinite(count)
                and count >= 0
                and int(count) == count
                for count in counts
            )
        ):
            attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = sum(
                int(count) for count in counts
            )
        return attrs
    messages = response_messages(response)
    attrs = {}
    values = response_values(response)
    if values:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(values)
    for index, message in enumerate(messages):
        for key in ("role", "content", "tool_calls"):
            if key in message:
                value = message[key]
                attrs[f"{SpanAttributes.LLM_COMPLETIONS}.{index}.{key}"] = (
                    value
                    if type(value) is str and key != "tool_calls"
                    else json_dumps(value)
                )
    usage = extract_usage(response)
    if "prompt_token_count" in usage:
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = usage["prompt_token_count"]
        attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = usage["prompt_token_count"]
    if "candidates_token_count" in usage:
        output = usage["candidates_token_count"] + usage.get("thoughts_token_count", 0)
        attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = output
        attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output
    if "total_token_count" in usage:
        attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage["total_token_count"]
    if "cached_content_token_count" in usage:
        attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = usage[
            "cached_content_token_count"
        ]
    return attrs
