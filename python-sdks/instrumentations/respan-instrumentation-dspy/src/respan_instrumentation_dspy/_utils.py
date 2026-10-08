"""Translate actual DSPy/lm15 response data into canonical attributes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_dspy._serialization import json_string, redact_text


def get(value: Any, key: str, default: Any = None) -> Any:
    try:
        return (
            value.get(key, default)
            if isinstance(value, Mapping)
            else getattr(value, key, default)
        )
    except Exception:  # noqa: BLE001 - telemetry access cannot alter SDK behavior
        return default


def plain(value: Any, depth: int = 0) -> Any:
    if depth > 64:
        return {"type": type(value).__name__, "recursive": True}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            key: plain(item, depth + 1)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, list | tuple):
        return [plain(item, depth + 1) for item in value]
    if type(value).__module__ == "numpy" and type(value).__name__ == "ndarray":
        return value.tolist()
    module = type(value).__module__
    if module.startswith("dspy."):
        if is_dataclass(value):
            return {
                field.name: plain(get(value, field.name), depth + 1)
                for field in fields(value)
                if field.name not in {"provider_data", "raw", "adaptations"}
            }
        store = get(value, "_store")
        if isinstance(store, dict):
            return plain(store, depth + 1)
        # SDK ToolCalls / typed decisions are Pydantic objects. Read validated
        # fields without invoking model_dump, repr, or arbitrary application hooks.
        namespace = get(value, "__dict__", {})
        if isinstance(namespace, dict) and namespace:
            return {
                key: plain(item, depth + 1)
                for key, item in namespace.items()
                if not key.startswith("_")
                and key not in {"kwargs", "history", "callbacks", "lm", "func"}
            }
    if isinstance(value, type) and getattr(value, "__module__", "").startswith("dspy."):
        return {"type": value.__name__}
    return {"type": type(value).__name__}


def safe_json(value: Any) -> str:
    return json_string(plain(value)) or "null"


def content_to_string(value: Any) -> str:
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


def add_lm_request_attributes(attributes: dict, instance: Any, inputs: Mapping) -> None:
    prompt = inputs.get("prompt")
    model = get(prompt, "model", get(instance, "model"))
    if isinstance(model, str) and model:
        attributes[SpanAttributes.LLM_REQUEST_MODEL] = model
        if "/" in model:
            attributes[SpanAttributes.LLM_SYSTEM] = {
                "gemini": "google",
                "azure": "openai",
            }.get(model.split("/")[0], model.split("/")[0])
    attributes[SpanAttributes.LLM_REQUEST_TYPE] = "chat"
    kwargs = dict(get(instance, "kwargs", {}) or {})
    kwargs.update(inputs.get("kwargs", {}))
    config = get(prompt, "config")
    for name, key in (
        ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
        ("max_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
        ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
        ("reasoning_effort", SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT),
    ):
        value = get(
            config,
            "max_output_tokens" if name == "max_tokens" else name,
            kwargs.get(name),
        )
        if value is None and name == "max_tokens":
            value = kwargs.get("max_completion_tokens")
        if isinstance(value, str | int | float) and not isinstance(value, bool):
            attributes[key] = redact_text(value) if isinstance(value, str) else value


def tool_definitions(tools: Any) -> list[dict]:
    """Keep validated definitions complete in the canonical function shape."""
    result = []
    for tool in tools:
        if get(tool, "type") == "function" and get(tool, "function") is None:
            definition = {
                "name": get(tool, "name"),
                "parameters": plain(get(tool, "parameters")),
            }
            if get(tool, "description") is not None:
                definition["description"] = get(tool, "description")
            result.append({"type": "function", "function": definition})
        else:
            result.append(plain(tool))
    return result
