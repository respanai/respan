"""Project native source fields while retaining full canonical JSON bodies."""

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

from respan_instrumentation_ollama._privacy import json_text, text, value
from respan_instrumentation_ollama._translator import messages, response_projection


def base_attributes(mode):
    return {
        SpanAttributes.LLM_SYSTEM: "ollama",
        SpanAttributes.LLM_REQUEST_TYPE: {
            "chat": LLMRequestTypeValues.CHAT.value,
            "generate": LLMRequestTypeValues.COMPLETION.value,
            "embed": LLMRequestTypeValues.EMBEDDING.value,
            "embeddings": LLMRequestTypeValues.EMBEDDING.value,
        }[mode],
        RESPAN_LOG_TYPE: {
            "chat": LOG_TYPE_CHAT,
            "generate": LOG_TYPE_TEXT,
            "embed": LOG_TYPE_EMBEDDING,
            "embeddings": LOG_TYPE_EMBEDDING,
        }[mode],
    }


def build_attributes(
    *, mode, request, payload=None, stream=False, capture_content=True, metadata=None
):
    attrs = {}
    if capture_content:
        for i, message in enumerate(messages(request, mode)):
            for key in ("role", "content", "tool_calls"):
                if key in message:
                    v = message[key]
                    attrs[f"{SpanAttributes.LLM_PROMPTS}.{i}.{key}"] = (
                        v if type(v) is str and key != "tool_calls" else json_text(v)
                    )
        if payload is not None and mode in ("chat", "generate"):
            projection = response_projection(payload, mode, stream)
            for key in ("role", "content", "tool_calls"):
                v = projection[key]
                if v is not None and (key != "tool_calls" or v):
                    attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.{key}"] = (
                        json_text(v)
                        if key == "tool_calls"
                        else text(v)
                        if type(v) is str
                        else v
                    )
    # Required canonical data is written last, surviving the SDK-owned bound on
    # indexed convenience fields without limiting histories/vectors ourselves.
    attrs.update(base_attributes(mode))
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = "ollama." + mode
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = ""
    if type(request.get("model")) is str:
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = request["model"]
    if stream:
        attrs[SpanAttributes.LLM_IS_STREAMING] = True
    if not capture_content:
        return attrs
    options = request.get("options")
    if type(options) is dict:
        if type(options.get("temperature")) in (int, float):
            attrs[SpanAttributes.LLM_REQUEST_TEMPERATURE] = options["temperature"]
        if type(options.get("num_predict")) is int and options["num_predict"] >= 0:
            attrs[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = options["num_predict"]
    if payload is not None:
        p = response_projection(payload, mode, stream)
        if p["model"] is not None:
            attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = p["model"]
        for name, key, legacy in (
            (
                "input_tokens",
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            ),
            (
                "output_tokens",
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            ),
        ):
            if name in p["usage"]:
                attrs[key] = p["usage"][name]
                attrs[legacy] = p["usage"][name]
        if "cache_read_input_tokens" in p["usage"]:
            attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] = p["usage"][
                "cache_read_input_tokens"
            ]
        if mode in ("embed", "embeddings") and type(payload) is dict:
            key = "embeddings" if mode == "embed" else "embedding"
            if key in payload:
                attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_text(payload[key])
                envelope = {name: item for name, item in payload.items() if name != key}
                if envelope:
                    attrs[RESPAN_METADATA + ".ollama.result"] = json_text(envelope)
            else:
                attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_text(payload)
        else:
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_text(
                _stream_payload(payload, mode) if stream else payload
            )
    tools = request.get("tools")
    if type(tools) is list:
        attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_text(tools)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_text(request)
    if metadata is not None:
        attrs[RESPAN_METADATA] = json_text(metadata)
    return attrs


def _stream_payload(payload, mode):
    cleaned = value(payload)
    if type(payload) is not list or type(cleaned) is not list:
        return cleaned
    keys = ("content", "thinking") if mode == "chat" else ("response", "thinking")
    for key in keys:
        pieces = []
        for index, frame in enumerate(payload):
            source = (
                frame.get("message")
                if mode == "chat" and type(frame) is dict
                else frame
            )
            if type(source) is dict and type(source.get(key)) is str:
                pieces.append((index, source[key]))
        joined = "".join(piece for _, piece in pieces)
        if text(joined) != joined:
            for index, _ in pieces:
                target = (
                    cleaned[index].get("message") if mode == "chat" else cleaned[index]
                )
                if type(target) is dict:
                    target[key] = "[REDACTED]"
    return cleaned
