"""Builtin/installed-native values and lossless source projections."""

from __future__ import annotations

import importlib
import inspect

from ._privacy import json_text, text, value


def native_value(data, seen=None):
    cls = type(data)
    if data is None or any(
        cls is kind for kind in (str, bool, int, float, bytes, bytearray)
    ):
        return value(data)
    active = set() if seen is None else seen
    if id(data) in active:
        return None
    if any(cls is kind for kind in (dict, list, tuple)):
        active.add(id(data))
        try:
            return (
                {k: native_value(v, active) for k, v in data.items() if type(k) is str}
                if cls is dict
                else [native_value(v, active) for v in data]
            )
        finally:
            active.remove(id(data))
    module = type.__getattribute__(cls, "__module__")
    if type(module) is not str or not module.startswith(
        "ibm_watsonx_ai.foundation_models.schema"
    ):
        return None
    actual = importlib.import_module(module)
    try:
        for name in type.__getattribute__(cls, "__qualname__").split("."):
            actual = inspect.getattr_static(actual, name)
    except AttributeError:
        return None
    if actual is not cls:
        return None
    state = object.__getattribute__(data, "__dict__")
    return native_value(state, active) if type(state) is dict else None


def tool_calls(calls):
    if type(calls) is not list:
        return []
    result = []
    for call in calls:
        if type(call) is not dict or type(call.get("function")) is not dict:
            continue
        source = call["function"]
        function = {k: v for k, v in source.items() if k != "arguments"}
        if "arguments" in source:
            function["arguments"] = (
                text(source["arguments"])
                if type(source["arguments"]) is str
                else json_text(source["arguments"])
            )
        result.append({**call, "function": function})
    return result


def messages(request, mode):
    source = request.get("messages")
    if mode == "chat" and type(source) is list:
        return [
            {
                **m,
                **(
                    {"tool_calls": tool_calls(m["tool_calls"])}
                    if "tool_calls" in m
                    else {}
                ),
            }
            for m in source
            if type(m) is dict
        ]
    prompt = request.get("input", request.get("prompt"))
    if mode == "generate" and prompt is not None:
        return (
            [{"role": "user", "content": v} for v in prompt]
            if type(prompt) is list
            else [{"role": "user", "content": prompt}]
        )
    return []


def project(payload, mode, *, stream=False):
    values = payload if type(payload) is list else [payload]
    result = {}
    counts = {}
    model = None
    for frame_offset, frame in enumerate(values):
        if type(frame) is not dict:
            continue
        if type(frame.get("model_id")) is str:
            model = frame["model_id"]
        usage = frame.get("usage")
        if type(usage) is dict:
            counts.update({k: v for k, v in usage.items() if type(v) is int})
        for index, item in enumerate(frame.get("results") or []):
            if type(item) is not dict:
                continue
            key = index if stream else (frame_offset, index)
            target = result.setdefault(key, {})
            if type(item.get("generated_text")) is str:
                target["content"] = target.get("content", "") + item["generated_text"]
                target["role"] = "assistant"
            for src, key in (
                ("input_token_count", "prompt_tokens"),
                ("generated_token_count", "completion_tokens"),
            ):
                if type(item.get(src)) is int:
                    counts[key] = item[src]
        if type(frame.get("input_token_count")) is int:
            counts["prompt_tokens"] = frame["input_token_count"]
        for offset, choice in enumerate(frame.get("choices") or []):
            if type(choice) is not dict:
                continue
            index = choice.get("index", offset)
            index = index if type(index) is int else offset
            source = choice.get("message") or choice.get("delta") or {}
            key = index if stream else (frame_offset, index)
            target = result.setdefault(key, {})
            if type(source) is not dict:
                continue
            if type(source.get("role")) is str:
                target["role"] = source["role"]
            if type(source.get("content")) is str:
                target["content"] = target.get("content", "") + source["content"]
            for call_offset, call in enumerate(source.get("tool_calls") or []):
                if type(call) is not dict:
                    continue
                key = call.get("index", call_offset)
                key = key if type(key) is int else call_offset
                merged = target.setdefault("tool_calls", {}).setdefault(
                    key, {"function": {}}
                )
                for field in ("id", "type"):
                    if field in call:
                        merged[field] = call[field]
                function = call.get("function")
                if type(function) is dict:
                    for field, v in function.items():
                        merged["function"][field] = (
                            merged["function"].get(field, "") + v
                            if type(v) is str
                            else v
                        )
    for target in result.values():
        if "tool_calls" in target:
            target["tool_calls"] = tool_calls(list(target["tool_calls"].values()))
    if type(payload) is list and len(payload) > 1 and not stream:
        counts = {}
    return dict(enumerate(result.values())), counts, model


def stream_payload(payload, mode):
    clean = value(payload)
    if type(payload) is not list:
        return clean
    groups = {}
    for fi, frame in enumerate(payload):
        if type(frame) is not dict:
            continue
        for ci, choice in enumerate(frame.get("choices") or []):
            delta = choice.get("delta") if type(choice) is dict else None
            if type(delta) is not dict:
                continue
            for key in ("content", "reasoning", "reasoning_content"):
                if type(delta.get(key)) is str:
                    groups.setdefault(("choice", ci, key), []).append(
                        (fi, ci, key, delta[key])
                    )
            for ti, call in enumerate(delta.get("tool_calls") or []):
                function = call.get("function") if type(call) is dict else None
                if type(function) is dict and type(function.get("arguments")) is str:
                    groups.setdefault(("tool", ci, call.get("index", ti)), []).append(
                        (fi, ci, ti, function["arguments"])
                    )
        for ri, item in enumerate(frame.get("results") or []):
            if type(item) is dict and type(item.get("generated_text")) is str:
                groups.setdefault(("result", ri), []).append(
                    (fi, ri, "generated_text", item["generated_text"])
                )
    for key, pieces in groups.items():
        joined = "".join(p[-1] for p in pieces)
        if text(joined) != joined:
            for fi, ci, field, _ in pieces:
                if key[0] == "choice":
                    clean[fi]["choices"][ci]["delta"][field] = "[REDACTED]"
                elif key[0] == "result":
                    clean[fi]["results"][ci][field] = "[REDACTED]"
                else:
                    clean[fi]["choices"][ci]["delta"]["tool_calls"][field]["function"][
                        "arguments"
                    ] = "[REDACTED]"
    return clean
