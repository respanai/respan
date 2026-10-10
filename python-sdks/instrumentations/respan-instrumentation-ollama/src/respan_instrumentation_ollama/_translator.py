"""Canonical projections from the request/JSON actually handled by Ollama."""

from __future__ import annotations

import importlib
import inspect

from respan_instrumentation_ollama._privacy import json_text, text, value


def native_value(data):
    """Fallback for exact installed native models; never call model_dump hooks."""
    cls = type(data)
    if cls in (dict, list, tuple):
        if cls is dict:
            return {k: native_value(v) for k, v in data.items() if type(k) is str}
        return [native_value(v) for v in data]
    if data is None or cls in (str, bool, int, float, bytes, bytearray):
        return value(data)
    if type.__getattribute__(cls, "__module__") != "ollama._types":
        return None
    actual = importlib.import_module("ollama._types")
    try:
        for name in type.__getattribute__(cls, "__qualname__").split("."):
            actual = inspect.getattr_static(actual, name)
    except AttributeError:
        return None
    if actual is not cls:
        return None
    state = object.__getattribute__(data, "__dict__")
    fields = object.__getattribute__(data, "__pydantic_fields_set__")
    if type(state) is not dict or type(fields) is not set:
        return None
    return {key: native_value(item) for key, item in state.items() if key in fields}


def tool_calls(calls):
    if type(calls) is not list:
        return []
    result = []
    for call in calls:
        if type(call) is not dict or type(call.get("function")) is not dict:
            continue
        source = call["function"]
        function = {}
        if type(source.get("name")) is str:
            function["name"] = source["name"]
        if "arguments" in source:
            args = source["arguments"]
            function["arguments"] = text(args) if type(args) is str else json_text(args)
        item = {"type": "function", "function": function}
        for key in ("id", "index"):
            if key in call:
                item[key] = call[key]
        result.append(item)
    return result


def messages(request, mode):
    if mode == "chat":
        source = request.get("messages")
        if type(source) is not list:
            return []
        output = []
        for message in source:
            if type(message) is not dict:
                continue
            item = dict(message)
            if "tool_calls" in item:
                item["tool_calls"] = tool_calls(item["tool_calls"])
            output.append(item)
        return output
    if mode == "generate":
        return [
            {"role": role, "content": request[key]}
            for key, role in (("system", "system"), ("prompt", "user"))
            if key in request
        ]
    return []


def response_projection(payload, mode, stream=False):
    frames = payload if stream and type(payload) is list else [payload]
    content = []
    role = None
    calls = []
    usage = {}
    model = None
    for frame in frames:
        if type(frame) is not dict:
            continue
        if type(frame.get("model")) is str:
            model = frame["model"]
        for source, target in (
            ("prompt_eval_count", "input_tokens"),
            ("eval_count", "output_tokens"),
            ("prompt_eval_cached_count", "cache_read_input_tokens"),
        ):
            if type(frame.get(source)) is int:
                usage[target] = frame[source]
        message = frame.get("message") if mode == "chat" else None
        if type(message) is dict:
            if "role" in message:
                role = message["role"]
            if type(message.get("content")) is str:
                content.append(message["content"])
            calls.extend(tool_calls(message.get("tool_calls")))
        elif mode == "generate" and type(frame.get("response")) is str:
            role = "assistant"
            content.append(frame["response"])
    return {
        "content": "".join(content) if content else None,
        "role": role,
        "tool_calls": calls,
        "usage": usage,
        "model": model,
    }
