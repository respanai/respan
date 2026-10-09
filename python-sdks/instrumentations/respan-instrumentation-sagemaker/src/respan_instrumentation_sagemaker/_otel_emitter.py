"""Canonical attributes from actual request, body and SDK response fields."""

from __future__ import annotations

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv_ai import SpanAttributes as AI
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA

from ._serialization import json_dumps
from ._translator import embedding, usage


def build_sagemaker_attrs(
    *,
    operation_name,
    params,
    body=None,
    payload=None,
    response_fields=None,
    streaming=False,
):
    attrs = {
        RESPAN_LOG_TYPE: "task",
        AI.TRACELOOP_ENTITY_NAME: "sagemaker." + operation_name,
        AI.TRACELOOP_ENTITY_PATH: "",
    }
    metadata = {
        "operation": operation_name,
        "request": {k: v for k, v in params.items() if k != "Body"},
        "response": response_fields or {},
    }
    attrs[RESPAN_METADATA + ".sagemaker"] = json_dumps(metadata)
    attrs[AI.TRACELOOP_ENTITY_INPUT] = json_dumps(
        params if operation_name == "InvokeEndpointAsync" else body
    )
    kind = "task"
    model = None
    if type(body) is dict:
        if type(body.get("model")) is str:
            model = body["model"]
        messages = body.get("messages")
        if type(messages) is list:
            kind = "chat"
            for index, message in enumerate(messages):
                if type(message) is not dict:
                    continue
                role = message.get("role")
                content = message.get("content")
                if type(role) is str:
                    attrs[f"{AI.LLM_PROMPTS}.{index}.role"] = role
                if "content" in message:
                    attrs[f"{AI.LLM_PROMPTS}.{index}.content"] = (
                        content if type(content) is str else json_dumps(content)
                    )
                if "tool_calls" in message:
                    attrs[f"{AI.LLM_PROMPTS}.{index}.tool_calls"] = json_dumps(
                        message["tool_calls"]
                    )
                if "tool_call_id" in message:
                    attrs[f"{AI.LLM_PROMPTS}.{index}.content"] = json_dumps(message)
        elif any(type(body.get(k)) is str for k in ("inputs", "prompt", "text")):
            kind = "text"
        tools = body.get("tools", body.get("functions"))
        if type(tools) is list:
            attrs[AI.LLM_REQUEST_FUNCTIONS] = json_dumps(tools)
    if type(payload) is dict and type(payload.get("model")) is str:
        model = payload["model"]
    vectors = embedding(payload)
    if vectors is not None:
        kind = "embedding"
    if operation_name == "InvokeEndpointAsync":
        kind = "task"
    attrs[RESPAN_LOG_TYPE] = kind
    if kind != "task":
        attrs[AI.LLM_SYSTEM] = "sagemaker"
        attrs[GenAI.GEN_AI_PROVIDER_NAME] = "sagemaker"
        attrs[AI.LLM_REQUEST_TYPE] = (
            "chat"
            if kind == "chat"
            else ("embedding" if kind == "embedding" else "completion")
        )
        if model:
            attrs[AI.LLM_REQUEST_MODEL] = model
    if streaming:
        attrs[AI.LLM_IS_STREAMING] = True
    if payload is not None:
        attrs[AI.TRACELOOP_ENTITY_OUTPUT] = json_dumps(
            vectors if vectors is not None else payload
        )
        if type(payload) is dict:
            for index, choice in enumerate(
                payload.get("choices", [])
                if type(payload.get("choices")) is list
                else []
            ):
                if type(choice) is not dict:
                    continue
                message = choice.get("message")
                if type(message) is dict:
                    if type(message.get("role")) is str:
                        attrs[f"{AI.LLM_COMPLETIONS}.{index}.role"] = message["role"]
                    if "content" in message:
                        attrs[f"{AI.LLM_COMPLETIONS}.{index}.content"] = (
                            message["content"]
                            if type(message["content"]) is str
                            else json_dumps(message["content"])
                        )
                    if "tool_calls" in message:
                        attrs[f"{AI.LLM_COMPLETIONS}.{index}.tool_calls"] = json_dumps(
                            message["tool_calls"]
                        )
            counts = usage(payload)
            for field, keys in [
                (
                    "input_tokens",
                    (GenAI.GEN_AI_USAGE_INPUT_TOKENS, AI.LLM_USAGE_PROMPT_TOKENS),
                ),
                (
                    "output_tokens",
                    (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, AI.LLM_USAGE_COMPLETION_TOKENS),
                ),
                ("total_tokens", (AI.LLM_USAGE_TOTAL_TOKENS,)),
                ("cache_read", (AI.LLM_USAGE_CACHE_READ_INPUT_TOKENS,)),
                ("cache_creation", (AI.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,)),
                ("reasoning", (AI.LLM_USAGE_REASONING_TOKENS,)),
            ]:
                if field in counts:
                    for key in keys:
                        attrs[key] = counts[field]
    # Native OTel limits can bound convenience indexed attributes. Write the
    # complete canonical bodies last so they survive those configured limits.
    priority = (
        RESPAN_LOG_TYPE,
        AI.TRACELOOP_ENTITY_INPUT,
        AI.TRACELOOP_ENTITY_OUTPUT,
        AI.LLM_REQUEST_FUNCTIONS,
        RESPAN_METADATA + ".sagemaker",
    )
    for name in priority:
        if name in attrs:
            attrs[name] = attrs.pop(name)
    return attrs
