"""Full canonical source I/O with source-only native model/usage descriptors."""

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as A
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TEXT,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA

from ._privacy import json_text, text
from ._translator import messages, project, stream_payload


def base_attributes(mode):
    return {
        A.LLM_SYSTEM: "watsonx",
        A.LLM_REQUEST_TYPE: {
            "chat": LLMRequestTypeValues.CHAT,
            "generate": LLMRequestTypeValues.COMPLETION,
            "embed": LLMRequestTypeValues.EMBEDDING,
        }[mode].value,
        RESPAN_LOG_TYPE: {
            "chat": LOG_TYPE_CHAT,
            "generate": LOG_TYPE_TEXT,
            "embed": LOG_TYPE_EMBEDDING,
        }[mode],
    }


def build_attributes(
    *, mode, request, payload=None, stream=False, capture_content=True
):
    attrs = {}
    if capture_content:
        for i, message in enumerate(messages(request, mode)):
            for key in ("role", "content", "tool_calls"):
                if key in message:
                    attrs[f"{A.LLM_PROMPTS}.{i}.{key}"] = (
                        text(message[key])
                        if type(message[key]) is str and key != "tool_calls"
                        else json_text(message[key])
                    )
        if payload is not None and mode != "embed":
            projected, _, _ = project(
                payload, mode, stream=stream and request.get("async_mode") is not True
            )
            for i, message in projected.items():
                for key, v in message.items():
                    attrs[f"{A.LLM_COMPLETIONS}.{i}.{key}"] = (
                        text(v)
                        if type(v) is str and key != "tool_calls"
                        else json_text(v)
                    )
    attrs.update(base_attributes(mode))
    attrs[A.TRACELOOP_ENTITY_NAME] = "watsonx." + mode
    attrs[A.TRACELOOP_ENTITY_PATH] = ""
    if type(request.get("model_id")) is str:
        attrs[A.LLM_REQUEST_MODEL] = request["model_id"]
    if stream:
        attrs[A.LLM_IS_STREAMING] = True
    if not capture_content:
        return attrs
    params = request.get("parameters", request.get("params"))
    if type(params) is dict:
        for src, dest in (
            ("temperature", A.LLM_REQUEST_TEMPERATURE),
            ("max_tokens", A.LLM_REQUEST_MAX_TOKENS),
            ("max_new_tokens", A.LLM_REQUEST_MAX_TOKENS),
        ):
            if any(type(params.get(src)) is kind for kind in (int, float)):
                attrs[dest] = params[src]
    if payload is not None:
        _, counts, model = project(
            payload, mode, stream=stream and request.get("async_mode") is not True
        )
        if model is not None:
            attrs[G.GEN_AI_RESPONSE_MODEL] = model
        for src, targets in (
            ("prompt_tokens", (G.GEN_AI_USAGE_INPUT_TOKENS, A.LLM_USAGE_PROMPT_TOKENS)),
            (
                "completion_tokens",
                (G.GEN_AI_USAGE_OUTPUT_TOKENS, A.LLM_USAGE_COMPLETION_TOKENS),
            ),
            ("total_tokens", (A.LLM_USAGE_TOTAL_TOKENS,)),
        ):
            if src in counts:
                for target in targets:
                    attrs[target] = counts[src]
        if mode == "embed" and type(payload) is dict:
            results = payload.get("results")
            vectors = (
                [
                    item["embedding"]
                    for item in results
                    if type(item) is dict and "embedding" in item
                ]
                if type(results) is list
                else None
            )
            if vectors is not None:
                attrs[A.TRACELOOP_ENTITY_OUTPUT] = json_text(vectors)
                envelope = {
                    k: (
                        [
                            {x: y for x, y in item.items() if x != "embedding"}
                            for item in v
                        ]
                        if k == "results" and type(v) is list
                        else v
                    )
                    for k, v in payload.items()
                }
                attrs[RESPAN_METADATA + ".watsonx.result"] = json_text(envelope)
            else:
                attrs[A.TRACELOOP_ENTITY_OUTPUT] = json_text(payload)
        else:
            attrs[A.TRACELOOP_ENTITY_OUTPUT] = json_text(
                stream_payload(payload, mode) if stream else payload
            )
    if type(request.get("tools")) is list:
        attrs[A.LLM_REQUEST_FUNCTIONS] = json_text(request["tools"])
    attrs[A.TRACELOOP_ENTITY_INPUT] = json_text(request)
    return attrs
