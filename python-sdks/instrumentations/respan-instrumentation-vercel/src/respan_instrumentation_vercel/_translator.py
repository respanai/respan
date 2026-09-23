"""Translate the AI SDK's typed telemetry into the Respan span contract."""

from __future__ import annotations

import json
from typing import Any

from ai import experimental_telemetry as telemetry
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import ConfigDict, TypeAdapter
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_EMBEDDING,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_KIND,
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
)
from respan_sdk.utils.serialization import serialize_value

from ._constants import AI_OPERATION_CONTENT

_json_adapter = TypeAdapter(Any, config=ConfigDict(ser_json_bytes="base64"))


def json_value(value: Any) -> str:
    return json.dumps(
        _json_adapter.dump_python(value, mode="json", fallback=serialize_value),
        ensure_ascii=False,
    )


def usage_attributes(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    attrs = {
        gen_ai.GEN_AI_USAGE_INPUT_TOKENS: usage.input_tokens,
        gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS: usage.output_tokens,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS: usage.input_tokens,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS: usage.output_tokens,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS: usage.total_tokens,
    }
    for field, key in (
        ("cache_read_tokens", SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS),
        ("cache_write_tokens", SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS),
        ("reasoning_tokens", SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS),
    ):
        value = getattr(usage, field, None)
        if value is not None:
            attrs[key] = value
    return attrs


def message_attributes(message: Any, prefix: str) -> dict[str, Any]:
    parts = message.model_dump(mode="json")["parts"]
    calls = [
        {
            "id": part["tool_call_id"],
            "type": "function",
            "function": {"name": part["tool_name"], "arguments": part["tool_args"]},
        }
        for part in parts
        if part["kind"] in {"tool_call", "builtin_tool_call"}
    ]
    content = [p for p in parts if p["kind"] not in {"tool_call", "builtin_tool_call"}]
    attrs = {
        f"{prefix}.role": message.role,
        f"{prefix}.content": (
            "".join(p["text"] for p in content)
            if all(p["kind"] == "text" for p in content)
            else json_value(content)
        ),
    }
    if calls:
        attrs[f"{prefix}.tool_calls"] = json_value(calls)
    return attrs


_KINDS = {
    "run": LOG_TYPE_AGENT,
    "ai_stream": LOG_TYPE_CHAT,
    "ai_generate": LOG_TYPE_CHAT,
    "tool_execution": LOG_TYPE_TOOL,
    "embed": LOG_TYPE_EMBEDDING,
}

_OPERATION_KINDS = {
    "embed",
    "evaluate",
    "generate_audio",
    "generate_image",
    "generate_video",
    "rerank",
    "transcribe",
}


def span_attributes(span: Any, capture_content: bool) -> dict[str, Any]:
    data = span.data
    kind = data.kind
    log_type = _KINDS.get(kind, LOG_TYPE_TASK)
    name = getattr(data, "agent", None) or getattr(data, "tool_name", None) or kind
    attrs: dict[str, Any] = {
        RESPAN_LOG_TYPE: log_type,
        SpanAttributes.TRACELOOP_ENTITY_NAME: name,
    }
    metadata = (
        {
            key: value
            for key, value in span.trace_attrs.items()
            if key != AI_OPERATION_CONTENT
        }
        if capture_content
        else {}
    )
    if kind in _OPERATION_KINDS:
        metadata["operation"] = kind
        if kind == "embed":
            attrs[SpanAttributes.LLM_REQUEST_TYPE] = "embedding"
            attrs[SpanAttributes.LLM_REQUEST_MODEL] = data.model
            attrs[SpanAttributes.LLM_SYSTEM] = data.provider or "unknown"
            attrs.update(usage_attributes(data.usage))
        else:
            metadata.update({"model": data.model, "provider": data.provider})
            if data.usage is not None:
                metadata["usage"] = data.usage.model_dump(exclude_none=True)
        if kind in {"generate_audio", "transcribe"}:
            attrs[RESPAN_INTERNAL_SPAN_NAME_KIND] = (
                "speech" if kind == "generate_audio" else "transcribe"
            )
        content = span.trace_attrs.get(AI_OPERATION_CONTENT)
        if (
            capture_content
            and isinstance(content, dict)
            and content.get("span_id") == span.id
            and content.get("kind") == kind
        ):
            for field, key in (
                ("input", SpanAttributes.TRACELOOP_ENTITY_INPUT),
                ("output", SpanAttributes.TRACELOOP_ENTITY_OUTPUT),
            ):
                if isinstance(content.get(field), str):
                    attrs[key] = content[field]
    if metadata:
        attrs[RESPAN_METADATA] = json_value(metadata)
    if kind in {"ai_stream", "ai_generate"}:
        attrs.update(
            {
                SpanAttributes.LLM_REQUEST_TYPE: "chat",
                SpanAttributes.LLM_REQUEST_MODEL: data.model,
                SpanAttributes.LLM_SYSTEM: data.provider or "unknown",
                SpanAttributes.GEN_AI_IS_STREAMING: kind == "ai_stream",
            }
        )
        attrs.update(usage_attributes(data.usage))
        params = data.params
        sampling = getattr(params, "sampling", None)
        for sampler in sampling.values() if isinstance(sampling, dict) else ():
            for field, key in (
                ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
                ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
                ("top_k", gen_ai.GEN_AI_REQUEST_TOP_K),
            ):
                value = getattr(sampler, field, None)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    attrs[key] = value
        max_tokens = getattr(getattr(params, "output", None), "max_tokens", None)
        if isinstance(max_tokens, int):
            attrs[SpanAttributes.LLM_REQUEST_MAX_TOKENS] = max_tokens
        if kind == "ai_stream" and span.started_at is not None:
            for event in span.events:
                if event.name == telemetry.FIRST_TOKEN:
                    attrs[gen_ai.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = (
                        event.time_ns - span.started_at
                    ) / 1e9
                    break
        for field, key in (
            ("response_id", gen_ai.GEN_AI_RESPONSE_ID),
            ("response_model", SpanAttributes.LLM_RESPONSE_MODEL),
            ("finish_reason", gen_ai.GEN_AI_RESPONSE_FINISH_REASONS),
        ):
            value = getattr(data, field, None)
            if value is not None:
                attrs[key] = [value] if field == "finish_reason" else value
        if data.output_type is not None:
            attrs[gen_ai.GEN_AI_OUTPUT_TYPE] = "json"
        if capture_content:
            for index, message in enumerate(data.messages):
                attrs.update(
                    message_attributes(message, f"{SpanAttributes.LLM_PROMPTS}.{index}")
                )
            if data.message is not None:
                attrs.update(
                    message_attributes(
                        data.message, f"{SpanAttributes.LLM_COMPLETIONS}.0"
                    )
                )
            if data.tool_names:
                attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json_value(
                    [{"name": name} for name in data.tool_names]
                )
    elif kind == "run" and capture_content:
        # Agent usage is a rollup of its child calls, so never emit it twice.
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_value(data.messages)
        if data.final_message is not None:
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_value(
                data.final_message
            )
    elif kind == "tool_execution" and capture_content:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_value(
            {
                "name": data.tool_name,
                "arguments": data.args,
                "tool_call_id": data.tool_call_id,
            }
        )
        if data.result is not None:
            attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_value(data.result)
    elif kind == "hook" and capture_content:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json_value(
            {
                "label": data.label,
                "hook_type": data.hook_type,
                "tool_call_id": data.tool_call_id,
                "metadata": data.metadata,
            }
        )
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json_value(
            {
                "status": data.status,
                "resolution": data.resolution,
            }
        )
    return attrs
