"""Canonical attribute mapping; full I/O never depends on message projection."""

from __future__ import annotations

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TEXT,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA

from respan_instrumentation_together._serialization import json_dumps, safe_text
from respan_instrumentation_together._translator import (
    completions,
    native_value,
    redact_stream_fragments,
    responses,
    usage,
)


def base_attributes(operation):
    attrs = {
        SpanAttributes.LLM_SYSTEM: "together",
        RESPAN_LOG_TYPE: LOG_TYPE_EMBEDDING
        if operation == "embedding"
        else LOG_TYPE_CHAT
        if operation == "chat"
        else LOG_TYPE_TEXT,
    }
    types = {
        "embedding": LLMRequestTypeValues.EMBEDDING,
        "chat": LLMRequestTypeValues.CHAT,
        "completion": LLMRequestTypeValues.COMPLETION,
        "rerank": LLMRequestTypeValues.RERANK,
    }
    if operation in types:
        attrs[SpanAttributes.LLM_REQUEST_TYPE] = types[operation].value
    return attrs


def request_attributes(kwargs, operation):
    native = native_value(
        {key: value for key, value in kwargs.items() if key != "extra_headers"}
    )
    effective = dict(native)
    extra = effective.pop("extra_body", None)
    if type(extra) is dict:
        effective.update(extra)
    attrs = {}
    if operation == "chat":
        messages = effective.get("messages")
        if type(messages) is list:
            for index, message in enumerate(messages):
                if type(message) is dict:
                    for key in ("role", "content", "tool_calls"):
                        if key in message and message[key] is not None:
                            item = message[key]
                            attrs[f"{SpanAttributes.LLM_PROMPTS}.{index}.{key}"] = (
                                safe_text(item)
                                if type(item) is str and key != "tool_calls"
                                else json_dumps(item)
                            )
    elif operation == "completion" and "prompt" in effective:
        attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] = "user"
        attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] = (
            safe_text(effective["prompt"])
            if type(effective["prompt"]) is str
            else json_dumps(effective["prompt"])
        )
    if type(effective.get("model")) is str:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = safe_text(effective["model"])
    if effective.get("stream") is True:
        attrs[SpanAttributes.LLM_IS_STREAMING] = True
    for key, target in (
        ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
        ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
    ):
        if type(effective.get(key)) in {int, float}:
            attrs[target] = effective[key]
    if type(effective.get("tools")) is list:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_dumps(effective["tools"])
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_dumps(native)
    return attrs


def response_attributes(response, operation):
    values = responses(response)
    attrs = {}
    if not values:
        return attrs
    if operation == "embedding":
        vectors = []
        for value in values:
            if type(value) is dict:
                for entry in value.get("data") or []:
                    if type(entry) is dict and type(entry.get("embedding")) is list:
                        vectors.append(entry["embedding"])
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(vectors)
        envelope = [
            {
                key: (
                    [
                        {k: v for k, v in entry.items() if k != "embedding"}
                        for entry in item
                    ]
                    if key == "data" and type(item) is list
                    else item
                )
                for key, item in source.items()
            }
            for source in values
            if type(source) is dict
        ]
        attrs[f"{RESPAN_METADATA}.together.result"] = json_dumps(envelope)
    else:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_dumps(values)
    if operation in {"chat", "completion"}:
        for index, message in completions(values, operation).items():
            for key in ("role", "content", "tool_calls"):
                if key in message:
                    item = message[key]
                    attrs[f"{SpanAttributes.LLM_COMPLETIONS}.{index}.{key}"] = (
                        safe_text(item)
                        if type(item) is str and key != "tool_calls"
                        else json_dumps(item)
                    )
            if "finish_reason" in message:
                attrs[SpanAttributes.LLM_RESPONSE_FINISH_REASON] = message[
                    "finish_reason"
                ]
    for source in reversed(values):
        if type(source) is dict and type(source.get("model")) is str:
            attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = safe_text(source["model"])
            break
    counts = usage(values)
    for source, targets in (
        (
            "prompt_tokens",
            (
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            ),
        ),
        (
            "completion_tokens",
            (
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            ),
        ),
        ("total_tokens", (SpanAttributes.LLM_USAGE_TOTAL_TOKENS,)),
    ):
        if source in counts:
            for target in targets:
                attrs[target] = counts[source]
    for value in reversed(values):
        raw = value.get("usage") if type(value) is dict else None
        if type(raw) is dict:
            nested = raw.get("prompt_tokens_details")
            if type(nested) is dict and type(nested.get("cached_tokens")) is int:
                attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = nested[
                    "cached_tokens"
                ]
            break
    # Full I/O is last so native OTel bounds can only limit convenience fields.
    if SpanAttributes.TRACELOOP_ENTITY_OUTPUT in attrs:
        value = attrs.pop(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = (
            json_dumps(redact_stream_fragments(values))
            if operation in {"chat", "completion"}
            else value
        )
    return attrs
