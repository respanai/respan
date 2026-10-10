"""Canonical Writer attributes from full native request and response values."""

from __future__ import annotations

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as S
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_TEXT, LOG_TYPE_TOOL
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
)

from respan_instrumentation_writer._serialization import json_dumps, safe_text
from respan_instrumentation_writer._translator import (
    completions,
    native_value,
    redact_stream_fragments,
    redact_text_fragments,
    responses,
    text_chunks,
    usage,
)

_TOOLS = {"web_search", "parse_pdf"}


def base_attributes(operation):
    if operation in _TOOLS:
        return {
            RESPAN_LOG_TYPE: LOG_TYPE_TOOL,
        }
    return {
        S.LLM_SYSTEM: "writer",
        RESPAN_LOG_TYPE: LOG_TYPE_CHAT if operation == "chat" else LOG_TYPE_TEXT,
        S.LLM_REQUEST_TYPE: (
            LLMRequestTypeValues.CHAT
            if operation == "chat"
            else LLMRequestTypeValues.COMPLETION
        ).value,
    }


def request_attributes(kwargs, operation):
    native = native_value({k: v for k, v in kwargs.items() if k != "extra_headers"})
    effective = dict(native)
    extra = effective.pop("extra_body", None)
    if type(extra) is dict:
        effective.update(extra)
    attrs = {}
    messages = effective.get("messages")
    if operation == "chat" and type(messages) is list:
        for index, message in enumerate(messages):
            if type(message) is dict:
                for key in ("role", "content", "tool_calls"):
                    if key in message and message[key] is not None:
                        item = message[key]
                        attrs[f"{S.LLM_PROMPTS}.{index}.{key}"] = (
                            safe_text(item)
                            if type(item) is str and key != "tool_calls"
                            else json_dumps(item)
                        )
    elif operation not in _TOOLS:
        key = {
            "completion": "prompt",
            "graph": "question",
            "vision": "prompt",
            "translation": "text",
        }.get(operation)
        if key in effective:
            attrs[f"{S.LLM_PROMPTS}.0.role"] = "user"
            attrs[f"{S.LLM_PROMPTS}.0.content"] = (
                safe_text(effective[key])
                if type(effective[key]) is str
                else json_dumps(effective[key])
            )
    if type(effective.get("model")) is str:
        attrs[S.LLM_REQUEST_MODEL] = safe_text(effective["model"])
    if effective.get("stream") is True:
        attrs[S.LLM_IS_STREAMING] = True
    for key, target in (
        ("temperature", S.LLM_REQUEST_TEMPERATURE),
        ("max_tokens", S.LLM_REQUEST_MAX_TOKENS),
    ):
        if any(
            type(effective.get(key)) is item
            for item in (
                int,
                float,
            )
        ):
            attrs[target] = effective[key]
    if type(effective.get("tools")) is list:
        attrs[S.LLM_REQUEST_FUNCTIONS] = json_dumps(effective["tools"])
    attrs[S.TRACELOOP_ENTITY_INPUT] = json_dumps(native)
    return attrs


def response_attributes(response, operation):
    values = responses(response)
    attrs = {}
    if not values:
        return attrs
    if operation in {"chat", "completion"}:
        for index, message in completions(values, operation).items():
            for key in ("role", "content", "tool_calls"):
                if key in message:
                    item = message[key]
                    attrs[f"{S.LLM_COMPLETIONS}.{index}.{key}"] = (
                        safe_text(item)
                        if type(item) is str and key != "tool_calls"
                        else json_dumps(item)
                    )
            if "finish_reason" in message:
                attrs[S.LLM_RESPONSE_FINISH_REASON] = message["finish_reason"]
    if operation != "chat" and operation not in _TOOLS:
        pieces = text_chunks(values, operation)
        if pieces:
            attrs[f"{S.LLM_COMPLETIONS}.0.content"] = safe_text(
                "".join(item[2] for item in pieces)
            )
    for value in reversed(values):
        if type(value) is dict and type(value.get("model")) is str:
            attrs[GenAI.GEN_AI_RESPONSE_MODEL] = safe_text(value["model"])
            break
    if operation not in _TOOLS:
        counts = usage(values)
        for field, targets in (
            (
                "prompt_tokens",
                (GenAI.GEN_AI_USAGE_INPUT_TOKENS, S.LLM_USAGE_PROMPT_TOKENS),
            ),
            (
                "completion_tokens",
                (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, S.LLM_USAGE_COMPLETION_TOKENS),
            ),
            ("total_tokens", (S.LLM_USAGE_TOTAL_TOKENS,)),
        ):
            if field in counts:
                for target in targets:
                    attrs[target] = counts[field]
        for value in reversed(values):
            raw = value.get("usage") if type(value) is dict else None
            nested = raw.get("prompt_tokens_details") if type(raw) is dict else None
            if type(nested) is dict and type(nested.get("cached_tokens")) is int:
                attrs[S.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = nested["cached_tokens"]
                break
    # Full actual native response is independent of text/message projection.
    clean = (
        redact_stream_fragments(values)
        if operation == "chat"
        else redact_text_fragments(values, operation)
    )
    attrs[S.TRACELOOP_ENTITY_OUTPUT] = json_dumps(clean)
    return attrs
