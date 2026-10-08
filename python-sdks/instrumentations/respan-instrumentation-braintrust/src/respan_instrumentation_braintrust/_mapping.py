"""Canonical Braintrust native record translation; no provider guesses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA

from ._constants import BRAINTRUST_SPAN_TYPE_TO_LOG_TYPE
from ._serialization import json_string, redact_text


def get(value, key, default=None):
    try:
        return (
            value.get(key, default)
            if isinstance(value, Mapping)
            else getattr(value, key, default)
        )
    except Exception:  # noqa: BLE001 - telemetry must preserve native behavior
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
    return {"type": type(value).__name__}


def safe_json(value):
    return json_string(plain(value)) or "null"


def content_to_string(value):
    return redact_text(value) if isinstance(value, str) else safe_json(value)


def calls(values: Any) -> list[dict]:
    result = []
    for call in values or ():
        function = get(call, "function")
        name = get(function, "name") if function is not None else get(call, "name")
        arguments = (
            get(function, "arguments")
            if function is not None
            else get(call, "input", get(call, "args"))
        )
        item = {
            "type": "function",
            "function": {
                "name": name,
                "arguments": redact_text(arguments)
                if isinstance(arguments, str)
                else safe_json(arguments),
            },
        }
        if identifier := get(call, "id"):
            item["id"] = identifier
        result.append(item)
    return result


def message(value: Any) -> dict:
    role = get(value, "role", "assistant")
    item = {"role": role}
    parts = get(value, "parts")
    if parts is not None:
        contents = []
        tool_calls = []
        for part in parts:
            kind = get(part, "type")
            if kind == "tool_call":
                tool_calls.extend(calls([part]))
            elif kind == "tool_result":
                item["tool_call_id"] = get(part, "id")
                contents.extend(plain(get(part, "content", ())))
            elif kind == "text":
                contents.append(get(part, "text"))
            else:
                contents.append(plain(part))
        if contents:
            item["content"] = (
                "".join(contents)
                if all(isinstance(x, str) for x in contents)
                else contents
            )
        if tool_calls:
            item["tool_calls"] = tool_calls
    else:
        content = get(value, "content", get(value, "text"))
        if content is not None:
            item["content"] = plain(content)
        tool_calls = calls(get(value, "tool_calls"))
        if tool_calls:
            item["tool_calls"] = tool_calls
        if identifier := get(value, "tool_call_id"):
            item["tool_call_id"] = identifier
    return item


def normalize_messages(prompt: Any, messages: Any) -> list[dict]:
    if get(prompt, "messages") is not None:
        messages = get(prompt, "messages")
        result = []
        if (system := get(prompt, "system")) is not None:
            result.append({"role": "system", "content": plain(system)})
        return result + [message(item) for item in messages]
    if messages is not None:
        return [message(item) for item in messages]
    if prompt is not None:
        return [{"role": "user", "content": plain(prompt)}]
    return []


def set_messages(attrs: dict, prefix: str, messages: list[dict]) -> None:
    for index, item in enumerate(messages):
        root = f"{prefix}.{index}"
        attrs[f"{root}.role"] = item["role"]
        if "content" in item:
            attrs[f"{root}.content"] = content_to_string(item["content"])
        if item.get("tool_calls"):
            attrs[f"{root}.tool_calls"] = safe_json(item["tool_calls"])
        if item.get("tool_call_id"):
            attrs[f"{root}.tool_call_id"] = item["tool_call_id"]


def add_lm_usage_attributes(attributes: dict, usage: Any) -> None:
    def counter(*keys):
        for key in keys:
            value = get(usage, key)
            if type(value) is int and value >= 0:
                return value
        return None

    for value, keys in (
        (
            counter("input_tokens", "prompt_tokens"),
            (
                gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            ),
        ),
        (
            counter("output_tokens", "completion_tokens"),
            (
                gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            ),
        ),
        (counter("total_tokens"), (SpanAttributes.LLM_USAGE_TOTAL_TOKENS,)),
        (
            counter("cache_read_tokens"),
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            ),
        ),
        (
            counter("cache_write_tokens"),
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            ),
        ),
        (
            counter("reasoning_tokens"),
            (
                SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                SpanAttributes.LLM_USAGE_REASONING_TOKENS,
            ),
        ),
    ):
        if value is not None:
            for key in keys:
                attributes[key] = value
    for details, field, keys in (
        (
            ("prompt_tokens_details", "input_tokens_details"),
            ("cached_tokens",),
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            ),
        ),
        (
            ("prompt_tokens_details", "input_tokens_details"),
            ("cache_write_tokens", "cache_creation_tokens"),
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            ),
        ),
        (
            ("completion_tokens_details", "output_tokens_details"),
            ("reasoning_tokens",),
            (
                SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                SpanAttributes.LLM_USAGE_REASONING_TOKENS,
            ),
        ),
    ):
        for detail in details:
            for name in field:
                value = get(get(usage, detail), name)
                if type(value) is int and value >= 0:
                    for key in keys:
                        attributes[key] = value
                    break


def attributes(record, *, capture, masking=None):
    native = record.get("span_attributes") or {}
    kind = BRAINTRUST_SPAN_TYPE_TO_LOG_TYPE.get(get(native, "type"), "task")
    name = get(native, "name")
    attrs = {RESPAN_LOG_TYPE: kind, SpanAttributes.TRACELOOP_ENTITY_PATH: ""}
    if isinstance(name, str):
        attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = redact_text(name)
    metadata = record.get("metadata") or {}
    model = get(record, "model") or get(metadata, "model") or get(native, "model")
    provider = get(metadata, "provider") or get(metadata, "system")
    if kind in {"chat", "embedding"}:
        attrs[SpanAttributes.LLM_REQUEST_TYPE] = (
            "embedding" if kind == "embedding" else "chat"
        )
        if isinstance(model, str):
            attrs[SpanAttributes.LLM_REQUEST_MODEL] = redact_text(model)
        if isinstance(provider, str):
            attrs[SpanAttributes.LLM_SYSTEM] = redact_text(provider).lower()
        usage = record.get("metrics") or {}
        normalized = dict(usage)
        for original, target in (
            ("tokens", "total_tokens"),
            ("prompt_cached_tokens", "cache_read_tokens"),
            ("prompt_cache_write_tokens", "cache_write_tokens"),
            ("completion_reasoning_tokens", "reasoning_tokens"),
        ):
            if original in usage:
                normalized[target] = usage[original]
        add_lm_usage_attributes(attrs, normalized)
        for nested in [
            get(metadata, "usage"),
            get(usage, "usage"),
            get(usage, "token_usage"),
        ]:
            if isinstance(nested, Mapping):
                add_lm_usage_attributes(attrs, nested)
    if not capture:
        return attrs

    def content(field):
        value = record.get(field)
        if value is None:
            return None
        if masking is not None:
            try:
                value = masking(value)
            except Exception:  # noqa: BLE001 - telemetry must preserve native behavior
                return None
        return value

    input_value = content("input")
    output_value = content("output")
    if input_value is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = (
            safe_json({"name": name, "arguments": input_value})
            if kind == "tool"
            else safe_json(input_value)
        )
    if output_value is not None:
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safe_json(output_value)
    if kind == "chat":
        parsed = input_value
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except ValueError:
                pass
        if isinstance(parsed, Mapping):
            parsed = parsed.get("messages", parsed)
        prompts = (
            [message(v) for v in parsed]
            if isinstance(parsed, list)
            else [{"role": "user", "content": parsed}]
            if parsed is not None
            else []
        )
        set_messages(attrs, SpanAttributes.LLM_PROMPTS, prompts)
        if output_value is not None:
            value = output_value
            if isinstance(value, Mapping) and get(value, "choices"):
                completions = [
                    message(
                        get(
                            choice,
                            "message",
                            {"role": "assistant", "content": get(choice, "text")},
                        )
                    )
                    for choice in value["choices"]
                ]
            elif isinstance(value, list) and all(
                isinstance(choice, Mapping) for choice in value
            ):
                completions = [
                    message(get(choice, "message", choice)) for choice in value
                ]
            elif isinstance(value, Mapping) and any(
                k in value for k in ["role", "content", "tool_calls"]
            ):
                completions = [message(value)]
            else:
                completions = [{"role": "assistant", "content": value}]
            set_messages(attrs, SpanAttributes.LLM_COMPLETIONS, completions)
        definitions = (
            get(metadata, "tools") or get(metadata, "functions") or get(native, "tools")
        )
        if definitions:
            attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safe_json(definitions)
    if kind == "tool":
        identifier = get(metadata, "tool_call_id") or get(native, "tool_call_id")
        if isinstance(identifier, str):
            attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] = redact_text(identifier)
    clean_metadata = content("metadata")
    bucket = dict(plain(clean_metadata)) if isinstance(clean_metadata, Mapping) else {}
    for key in ["scores", "tags", "expected"]:
        if record.get(key) is not None:
            bucket["braintrust_" + key] = plain(content(key))
    bucket = {
        k: v
        for k, v in bucket.items()
        if k
        not in {"org_id", "organization_id", "account_id", "api_key", "authorization"}
    }
    bucket = json.loads(safe_json(bucket))
    if bucket:
        attrs[RESPAN_METADATA] = safe_json(bucket)
        for k, v in bucket.items():
            attrs[f"{RESPAN_METADATA}.{k}"] = (
                redact_text(v)
                if isinstance(v, str)
                else v
                if type(v) in (int, float, bool)
                else safe_json(v)
            )
    return attrs


def private_usage(value):
    """Copy only actual numeric counters, never arbitrary private usage content."""
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "prompt_cached_tokens",
        "prompt_cache_write_tokens",
        "completion_reasoning_tokens",
    ):
        item = get(value, key)
        if type(item) is int and item >= 0:
            result[key] = item
    for key, names in (
        (
            "prompt_tokens_details",
            ("cached_tokens", "cache_write_tokens", "cache_creation_tokens"),
        ),
        ("input_tokens_details", ("cached_tokens", "cache_write_tokens")),
        ("completion_tokens_details", ("reasoning_tokens",)),
        ("output_tokens_details", ("reasoning_tokens",)),
    ):
        detail = get(value, key)
        if not isinstance(detail, Mapping):
            continue
        counters = {}
        for name in names:
            item = get(detail, name)
            if type(item) is int and item >= 0:
                counters[name] = item
        if counters:
            result[key] = counters
    return result
