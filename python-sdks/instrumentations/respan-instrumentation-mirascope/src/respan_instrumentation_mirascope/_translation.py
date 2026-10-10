"""Canonical attributes from observed Mirascope model and toolkit values."""

from __future__ import annotations

import json
from collections.abc import Mapping

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.semconv_ai import SpanAttributes as A
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_CHAT,
    LOG_TYPE_TOOL,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE

from ._serialization import json_string, json_value, safe_text


def get(value, key, default=None):
    try:
        return (
            value.get(key, default)
            if isinstance(value, Mapping)
            else getattr(value, key, default)
        )
    except Exception:  # noqa: BLE001 - optional observations preserve native values.
        return default


def identity(model, response=None):
    raw = get(response, "model_id") or get(model, "model_id")
    if not isinstance(raw, str):
        return None, None
    provider, name = raw.split("/", 1) if "/" in raw else (None, raw)
    actual = get(response, "provider_id")
    return actual if isinstance(actual, str) else provider, name


def arguments(value):
    if isinstance(value, str):
        try:
            return json_value(json.loads(value), complete=True)
        except ValueError:
            return safe_text(value, complete=True)
    return json_value(value, complete=True)


def calls(values):
    result = []
    for call in values or ():
        value = get(call, "args", get(call, "arguments"))
        if isinstance(value, str):
            try:
                encoded = json_string(json.loads(value), complete=True)
            except ValueError:
                encoded = safe_text(value, complete=True)
        else:
            encoded = json_string(value, complete=True)
        item = {
            "type": "function",
            "function": {
                "name": safe_text(get(call, "name"), complete=True),
                "arguments": encoded,
            },
        }
        if isinstance(identifier := get(call, "id"), str):
            item["id"] = safe_text(identifier, complete=True)
        result.append(item)
    return result


def messages(content):
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    values = content if isinstance(content, list | tuple) else [content]
    result = []
    for message in values:
        if isinstance(message, str):
            result.append({"role": "user", "content": message})
            continue
        role = get(message, "role")
        if not isinstance(role, str):
            result.append(
                {"role": "user", "content": json_value(message, complete=True)}
            )
            continue
        parts = get(message, "content")
        parts = parts if isinstance(parts, list | tuple) else [parts]
        ordinary, tool_calls = [], []
        for part in parts:
            kind = get(part, "type")
            if kind == "tool_call":
                tool_calls.append(part)
            elif kind == "tool_output":
                item = {
                    "role": "tool",
                    "content": json_value(get(part, "result"), complete=True),
                }
                if isinstance(identifier := get(part, "id"), str):
                    item["tool_call_id"] = identifier
                result.append(item)
            elif kind == "text":
                ordinary.append(get(part, "text"))
            else:
                ordinary.append(json_value(part, complete=True))
        item = {"role": role}
        if ordinary:
            item["content"] = ordinary[0] if len(ordinary) == 1 else ordinary
        if tool_calls:
            item["tool_calls"] = calls(tool_calls)
        if ordinary or tool_calls:
            result.append(item)
    return result


def indexed(attrs, prefix, values):
    for index, item in enumerate(values[:8]):
        base = f"{prefix}.{index}"
        attrs[f"{base}.role"] = item["role"]
        if "content" in item:
            value = item["content"]
            attrs[f"{base}.content"] = (
                safe_text(value, complete=True)
                if isinstance(value, str)
                else json_string(value, complete=True)
            )
        if item.get("tool_calls"):
            attrs[f"{base}.tool_calls"] = json_string(item["tool_calls"], complete=True)
        if item.get("tool_call_id"):
            attrs[f"{base}.tool_call_id"] = item["tool_call_id"]


def definitions(value):
    if value is None:
        return []
    values = get(value, "tools", value)
    values = values if isinstance(values, list | tuple) else [values]
    result = []
    for tool in values:
        params = get(tool, "parameters")
        # Provider tools have provider-specific wire schemas, not function schemas.
        if params is None:
            continue
        params = json_value(params, complete=True, schema=True)
        if isinstance(params, dict):
            params.setdefault("type", "object")
        function = {
            "name": safe_text(get(tool, "name"), complete=True),
            "parameters": params,
        }
        if type(strict := get(tool, "strict")) is bool:
            function["strict"] = strict
        if isinstance(description := get(tool, "description"), str):
            function["description"] = safe_text(description, complete=True)
        result.append({"type": "function", "function": function})
    return result


def prepare(state):
    attrs = {
        RESPAN_LOG_METHOD: LogMethodChoices.TRACING_INTEGRATION.value,
        RESPAN_LOG_TYPE: LOG_TYPE_TOOL if state.tool else LOG_TYPE_CHAT,
        A.TRACELOOP_ENTITY_PATH: "",
    }
    if state.tool:
        name = get(state.item, "name")
        if isinstance(name, str):
            attrs[A.TRACELOOP_ENTITY_NAME] = safe_text(name)
        if isinstance(identifier := get(state.item, "id"), str):
            attrs[G.GEN_AI_TOOL_CALL_ID] = safe_text(identifier, complete=True)
        if state.capture:
            attrs[A.TRACELOOP_ENTITY_INPUT] = json_string(
                {"name": name, "arguments": arguments(get(state.item, "args"))},
                complete=True,
            )
    else:
        provider, model = identity(state.item)
        attrs[A.TRACELOOP_ENTITY_NAME] = "mirascope.model"
        attrs[A.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT.value
        attrs[A.GEN_AI_IS_STREAMING] = state.stream
        if provider:
            attrs[A.LLM_SYSTEM] = safe_text(provider)
            attrs[G.GEN_AI_PROVIDER_NAME] = safe_text(provider)
        if model:
            attrs[A.LLM_REQUEST_MODEL] = safe_text(model)
        params = get(state.item, "params", {})
        for key, attr in (
            ("temperature", A.LLM_REQUEST_TEMPERATURE),
            ("max_tokens", A.LLM_REQUEST_MAX_TOKENS),
        ):
            value = get(params, key)
            if type(value) in (int, float):
                attrs[attr] = value
        if state.capture:
            values = messages(state.content)
            attrs[A.TRACELOOP_ENTITY_INPUT] = json_string(values, complete=True)
            indexed(attrs, A.LLM_PROMPTS, values)
            if tools := definitions(state.kwargs.get("tools")):
                attrs[A.LLM_REQUEST_FUNCTIONS] = json_string(
                    tools, complete=True, schema=True
                )
    return attrs


_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
)
_DETAILS = (
    "input_tokens_details",
    "output_tokens_details",
    "prompt_tokens_details",
    "completion_tokens_details",
)


def usage_fields(source):
    result = {}
    fields = get(source, "model_fields_set")
    for key in _COUNTERS:
        if isinstance(fields, set) and key not in fields:
            continue
        if type(value := get(source, key)) is int and value >= 0:
            result[key] = value
    for key in _DETAILS:
        detail = get(source, key)
        fields = get(detail, "model_fields_set")
        values = {
            name: value
            for name in (
                "cached_tokens",
                "cache_write_tokens",
                "cache_creation_tokens",
                "reasoning_tokens",
            )
            if (not isinstance(fields, set) or name in fields)
            and type(value := get(detail, name)) is int
            and value >= 0
        }
        if values:
            result[key] = values
    return result


def _raw_counters(text, *, detail=False):
    result, level, index = {}, 0, 0
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
                    except ValueError:
                        pass
                elif (
                    not detail
                    and key in _DETAILS
                    and raw.startswith("{")
                    and (values := _raw_counters(raw, detail=True))
                ):
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
    if not isinstance(value, str):
        return usage_fields(get(value, "usage"))
    level, index = 0, 0
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
                rest = value[index:].lstrip()
                if rest.startswith(":"):
                    raw = rest[1:].lstrip()
                    return _raw_counters(raw) if raw.startswith("{") else {}
            continue
        if character in "{[":
            level += 1
        elif character in "}]":
            level -= 1
        index += 1
    return {}


def add_usage(attrs, values):
    def count(*keys):
        return next(
            (
                get(values, key)
                for key in keys
                if type(get(values, key)) is int and get(values, key) >= 0
            ),
            None,
        )

    pairs = [
        (
            (G.GEN_AI_USAGE_INPUT_TOKENS, A.LLM_USAGE_PROMPT_TOKENS),
            count("input_tokens", "prompt_tokens"),
        ),
        (
            (G.GEN_AI_USAGE_OUTPUT_TOKENS, A.LLM_USAGE_COMPLETION_TOKENS),
            count("output_tokens", "completion_tokens"),
        ),
        ((A.LLM_USAGE_TOTAL_TOKENS,), count("total_tokens")),
    ]
    input_details = get(
        values, "input_tokens_details", get(values, "prompt_tokens_details", {})
    )
    output_details = get(
        values, "output_tokens_details", get(values, "completion_tokens_details", {})
    )
    for keys, value in [
        (
            (
                A.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                A.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            ),
            count("cache_read_tokens", "cache_read_input_tokens")
            if count("cache_read_tokens", "cache_read_input_tokens") is not None
            else get(input_details, "cached_tokens"),
        ),
        (
            (
                A.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                A.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            ),
            count("cache_write_tokens", "cache_creation_input_tokens")
            if count("cache_write_tokens", "cache_creation_input_tokens") is not None
            else get(
                input_details,
                "cache_write_tokens",
                get(input_details, "cache_creation_tokens"),
            ),
        ),
        (
            (A.GEN_AI_USAGE_REASONING_TOKENS, A.LLM_USAGE_REASONING_TOKENS),
            count("reasoning_tokens")
            if count("reasoning_tokens") is not None
            else get(output_details, "reasoning_tokens"),
        ),
    ]:
        pairs.append((keys, value))
    for keys, value in pairs:
        if type(value) is int and value >= 0:
            for key in keys:
                attrs[key] = value


def finish(state, response):
    attrs = {}
    if state.tool:
        if state.capture and response is not None:
            attrs[A.TRACELOOP_ENTITY_OUTPUT] = json_string(
                get(response, "result", response), complete=True
            )
        return attrs
    provider, name = identity(None, response)
    if name:
        attrs[A.LLM_RESPONSE_MODEL] = safe_text(name)
    if provider:
        attrs[A.LLM_SYSTEM] = safe_text(provider)
        attrs[G.GEN_AI_PROVIDER_NAME] = safe_text(provider)
    usage = get(response, "usage")
    if state.usage_seen:
        values = state.usage
    elif get(usage, "raw") is not None:
        values = usage_fields(get(usage, "raw"))
    else:
        # Mirascope's dataclass defaults zero for every unreported field. Without
        # a retained raw source, only actual nonzero native counters are known.
        values = {
            key: value
            for key, value in usage_fields(usage).items()
            if key != "total_tokens" and type(value) is int and value > 0
        }
        if all(key in values for key in ("input_tokens", "output_tokens")):
            values["total_tokens"] = get(usage, "total_tokens")
    add_usage(attrs, values)
    if (
        state.capture
        and response is not None
        and (not state.stream or state.observed_content)
    ):
        history = get(response, "messages", [])
        values = (
            messages([get(response, "assistant_message")])
            if get(response, "assistant_message") is not None
            else messages([history[-1]])
            if history
            else []
        )
        if values:
            attrs[A.TRACELOOP_ENTITY_OUTPUT] = json_string(values, complete=True)
            indexed(attrs, A.LLM_COMPLETIONS, values)
    return attrs
