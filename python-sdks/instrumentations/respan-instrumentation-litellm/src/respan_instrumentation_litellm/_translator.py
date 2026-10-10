"""Canonical translation of observed LiteLLM request and provider fields."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.llm_logging import LOG_TYPE_CHAT, LOG_TYPE_EMBEDDING
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_SPAN_ATTRIBUTES_MAP,
)

from ._serialization import json_string, redact_text


def get(value, key, default=None):
    try:
        return (
            value.get(key, default)
            if isinstance(value, Mapping)
            else getattr(value, key, default)
        )
    except Exception:  # noqa: BLE001 - telemetry cannot replace native behavior
        return default


def plain(value, depth=0):
    if depth > 64:
        return {"type": type(value).__name__, "recursive": True}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(k): plain(v, depth + 1)
            for k, v in value.items()
            if isinstance(k, str) or type(k) is int
        }
    if isinstance(value, list | tuple):
        return [plain(v, depth + 1) for v in value]
    if type(value).__module__.startswith(("litellm.", "openai.")):
        fields = get(type(value), "model_fields", {})
        if isinstance(fields, Mapping):
            return {k: plain(get(value, k), depth + 1) for k in fields}
    return {"type": type(value).__name__}


def safe_json(value, *, schema=False):
    return json_string(plain(value), schema=schema) or "null"


def calls(values):
    result = []
    for call in values or ():
        function = get(call, "function")
        name = get(function, "name") if function is not None else get(call, "name")
        arguments = (
            get(function, "arguments")
            if function is not None
            else get(call, "arguments")
        )
        item = {
            "type": get(call, "type", "function"),
            "function": {
                "name": name,
                "arguments": redact_text(arguments)
                if isinstance(arguments, str)
                else safe_json(arguments),
            },
        }
        identifier = (
            get(call, "id")
            if function is not None
            else get(call, "call_id", get(call, "id"))
        )
        if isinstance(identifier, str):
            item["id"] = identifier
        result.append(item)
    return result


def message(value):
    if get(value, "type") == "function_call":
        return {"role": "assistant", "tool_calls": calls([value])}
    if get(value, "type") == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": get(value, "call_id"),
            "content": plain(get(value, "output")),
        }
    item = {"role": get(value, "role", "assistant")}
    if (content := get(value, "content")) is not None:
        item["content"] = plain(content)
    if tool_calls := get(value, "tool_calls"):
        item["tool_calls"] = calls(tool_calls)
    if isinstance(identifier := get(value, "tool_call_id"), str):
        item["tool_call_id"] = identifier
    return item


def set_messages(attrs, prefix, messages):
    # Preserve every message in entity I/O. Bound only indexed projections so
    # OTel's default128-attribute budget cannot evict identity/usage/run fields.
    for index, item in enumerate(messages[:8]):
        base = f"{prefix}.{index}"
        attrs[f"{base}.role"] = item["role"]
        if "content" in item:
            attrs[f"{base}.content"] = (
                redact_text(item["content"])
                if isinstance(item["content"], str)
                else safe_json(item["content"])
            )
        if item.get("tool_calls"):
            attrs[f"{base}.tool_calls"] = safe_json(item["tool_calls"])
        if item.get("tool_call_id"):
            attrs[f"{base}.tool_call_id"] = item["tool_call_id"]


_COUNTERS = (
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)
_DETAILS = (
    "prompt_tokens_details",
    "input_tokens_details",
    "completion_tokens_details",
    "output_tokens_details",
)


def usage_fields(source):
    result = {}
    fields = get(source, "model_fields_set")
    for key in _COUNTERS:
        if isinstance(fields, set) and key not in fields:
            continue
        value = get(source, key)
        if type(value) is int and value >= 0:
            result[key] = value
    for key in _DETAILS:
        detail = get(source, key)
        values = {
            name: v
            for name in (
                "cached_tokens",
                "cache_write_tokens",
                "cache_creation_tokens",
                "reasoning_tokens",
            )
            if type(v := get(detail, name)) is int and v >= 0
        }
        if values:
            result[key] = values
    return result


def _raw_counters(text, *, detail=False):
    result = {}
    names = (
        (
            "cached_tokens",
            "cache_write_tokens",
            "cache_creation_tokens",
            "reasoning_tokens",
        )
        if detail
        else _COUNTERS
    )
    decoder = json.JSONDecoder()
    level = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character == '"':
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            key = text[start + 1 : index - 1]
            remaining = text[index:].lstrip()
            if level == 1 and remaining.startswith(":"):
                raw = remaining[1:].lstrip()
                if key in names and raw and raw[0] in "-0123456789":
                    try:
                        value, _ = decoder.raw_decode(raw)
                        if type(value) is int and value >= 0:
                            result[key] = value
                    except (ValueError, TypeError):
                        pass
                elif not detail and key in _DETAILS and raw.startswith("{"):
                    values = _raw_counters(raw, detail=True)
                    if values:
                        result[key] = values
            continue
        if character in "{[":
            level += 1
        elif character in "}]":
            level -= 1
            if level == 0:
                break
        index += 1
    return result


def raw_usage(value):
    """Decode only known actual counters; other provider values stay undecoded."""
    if not isinstance(value, str):
        return usage_fields(get(value, "usage"))
    level = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == '"':
            start = index
            index += 1
            while index < len(value):
                if value[index] == "\\":
                    index += 2
                elif value[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            if level == 1 and value[start:index] == '"usage"':
                remaining = value[index:].lstrip()
                if remaining.startswith(":"):
                    raw = remaining[1:].lstrip()
                    return _raw_counters(raw) if raw.startswith("{") else {}
            continue
        if character in "{[":
            level += 1
        elif character in "}]":
            level -= 1
        index += 1
    return {}


def add_usage(attrs, usage):
    def count(*keys):
        return next(
            (
                get(usage, key)
                for key in keys
                if type(get(usage, key)) is int and get(usage, key) >= 0
            ),
            None,
        )

    read = count("cache_read_tokens", "cache_read_input_tokens")
    write = count("cache_write_tokens", "cache_creation_input_tokens")
    reasoning = count("reasoning_tokens")
    for detail in ("prompt_tokens_details", "input_tokens_details"):
        data = get(usage, detail)
        if read is None:
            read = get(data, "cached_tokens")
        if write is None:
            write = get(data, "cache_write_tokens", get(data, "cache_creation_tokens"))
    for detail in ("completion_tokens_details", "output_tokens_details"):
        if reasoning is None:
            reasoning = get(get(usage, detail), "reasoning_tokens")
    for value, keys in (
        (
            count("input_tokens", "prompt_tokens"),
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
        ),
        (
            count("output_tokens", "completion_tokens"),
            (
                GenAI.GEN_AI_USAGE_OUTPUT_TOKENS,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            ),
        ),
        (count("total_tokens"), (SpanAttributes.LLM_USAGE_TOTAL_TOKENS,)),
        (
            read,
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            ),
        ),
        (
            write,
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            ),
        ),
        (
            reasoning,
            (
                SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                SpanAttributes.LLM_USAGE_REASONING_TOKENS,
            ),
        ),
    ):
        if type(value) is int and value >= 0:
            for key in keys:
                attrs[key] = value


def build_litellm_span_data(
    *,
    kwargs: Mapping[str, Any],
    response_obj: Any,
    error: BaseException | None = None,
    include_content=True,
    usage=None,
):
    kind = (
        "embedding"
        if "embedding" in str(kwargs.get("call_type", "")).lower()
        else "chat"
    )
    attrs = {
        RESPAN_LOG_TYPE: LOG_TYPE_EMBEDDING if kind == "embedding" else LOG_TYPE_CHAT,
        SpanAttributes.LLM_REQUEST_TYPE: kind,
        SpanAttributes.TRACELOOP_ENTITY_NAME: f"litellm.{kind}",
        SpanAttributes.TRACELOOP_ENTITY_PATH: "",
    }
    params = get(kwargs, "litellm_params", {})
    optional = get(kwargs, "optional_params", {})
    if isinstance(model := get(kwargs, "model"), str):
        attrs[SpanAttributes.LLM_REQUEST_MODEL] = redact_text(model)
    if isinstance(model := get(response_obj, "model"), str):
        attrs[GenAI.GEN_AI_RESPONSE_MODEL] = redact_text(model)
    if isinstance(
        provider := get(
            params, "custom_llm_provider", get(kwargs, "custom_llm_provider")
        ),
        str,
    ):
        attrs[SpanAttributes.LLM_SYSTEM] = provider
        attrs[GenAI.GEN_AI_PROVIDER_NAME] = provider
    if type(stream := get(kwargs, "stream", get(optional, "stream"))) is bool:
        attrs[SpanAttributes.GEN_AI_IS_STREAMING] = stream
    for source, target in [
        ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
        ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
        ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
    ]:
        if type(value := get(kwargs, source, get(optional, source))) in (int, float):
            attrs[target] = value
    metadata = get(kwargs, "metadata", get(params, "metadata", {}))
    attribution = get(metadata, "respan_params", {})
    for key, target in RESPAN_SPAN_ATTRIBUTES_MAP.items():
        value = get(attribution, key)
        if value is None or target in (RESPAN_METADATA, RESPAN_LOG_TYPE):
            continue
        if isinstance(value, str | int | bool | float):
            attrs[target] = redact_text(value) if isinstance(value, str) else value
    span_name = get(attribution, "span_name", f"litellm.{kind}")
    if not isinstance(span_name, str):
        span_name = f"litellm.{kind}"
    attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = redact_text(span_name)
    if include_content:
        if kind == "embedding":
            if (inputs := get(kwargs, "input", get(kwargs, "messages"))) is not None:
                attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(inputs)
            if isinstance(data := get(response_obj, "data"), list | tuple):
                vectors = [
                    plain(get(item, "embedding"))
                    for item in data
                    if get(item, "embedding") is not None
                ]
                if vectors:
                    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(vectors)
        else:
            inputs = get(kwargs, "messages", get(kwargs, "input"))
            messages = (
                [message(item) for item in inputs]
                if isinstance(inputs, list | tuple)
                else [{"role": "user", "content": plain(inputs)}]
                if inputs is not None
                else []
            )
            if messages:
                attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safe_json(messages)
                set_messages(attrs, SpanAttributes.LLM_PROMPTS, messages)
            outputs = [
                message(get(choice, "message"))
                for choice in get(response_obj, "choices", ()) or ()
                if get(choice, "message") is not None
            ]
            response_output = get(response_obj, "output")
            if isinstance(response_output, list | tuple):
                outputs = []
                current_calls = []
                for item in response_output:
                    if get(item, "type") == "function_call":
                        current_calls.extend(calls([item]))
                    elif get(item, "type") == "message":
                        outputs.append(message(item))
                if current_calls:
                    outputs.append({"role": "assistant", "tool_calls": current_calls})
            if outputs:
                attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(outputs)
                set_messages(attrs, SpanAttributes.LLM_COMPLETIONS, outputs)
            tools = get(kwargs, "tools", get(optional, "tools"))
            if (
                tools is None
                and (functions := get(kwargs, "functions", get(optional, "functions")))
                is not None
            ):
                tools = [{"type": "function", "function": plain(f)} for f in functions]
            if tools is not None:
                attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(
                    tools, schema=True
                )
        if get(attribution, "metadata") is not None:
            attrs[RESPAN_METADATA] = safe_json(get(attribution, "metadata"))
    add_usage(
        attrs, usage if usage is not None else usage_fields(get(response_obj, "usage"))
    )
    return redact_text(span_name), attrs
