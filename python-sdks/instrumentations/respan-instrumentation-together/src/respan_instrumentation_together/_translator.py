"""Complete builtin and exact installed Together model values without hooks."""

from __future__ import annotations

import importlib

from respan_instrumentation_together._serialization import (
    safe_text,
    safe_type_name,
    to_jsonable,
)


def _known_model(value):
    cls = type(value)
    module = type.__dict__["__dict__"].__get__(cls).get("__module__", "")
    name = type.__dict__["__name__"].__get__(cls)
    if type(module) is not str or not module.startswith("together.types."):
        return False
    return getattr(importlib.import_module(module), name, None) is cls


def native_value(value, seen=None):
    kind = type(value)
    types = importlib.import_module("together._types")
    if any(kind is getattr(types, name, None) for name in ("Omit", "NotGiven")):
        return _OMITTED
    if kind is str:
        return value
    active = seen if seen is not None else set()
    if kind in {dict, list, tuple} or _known_model(value):
        identity = id(value)
        if identity in active:
            return "<cycle>"
        active.add(identity)
        try:
            if kind in {list, tuple}:
                result = [native_value(item, active) for item in value]
                return [item for item in result if item is not _OMITTED]
            if kind is dict:
                state = value
            else:
                state = dict(object.__getattribute__(value, "__dict__"))
                extra = (
                    object.__getattribute__(value, "__pydantic_extra__")
                    if hasattr(type(value), "model_fields")
                    else None
                )
                if type(extra) is dict:
                    state.update(extra)
                try:
                    fields = object.__getattribute__(value, "__pydantic_fields_set__")
                except AttributeError:
                    fields = object.__getattribute__(value, "__fields_set__")
                if type(fields) is set:
                    state = {
                        key: item
                        for key, item in state.items()
                        if key in fields or (type(extra) is dict and key in extra)
                    }
            result = {}
            for key, item in state.items():
                converted = native_value(item, active)
                if converted is not _OMITTED:
                    name = key if type(key) is str else "<" + safe_type_name(key) + ">"
                    result[name] = converted
            return result
        finally:
            active.remove(identity)
    return to_jsonable(value)


_OMITTED = object()


def responses(value):
    values = value if type(value) is list else [value]
    return [native_value(item) for item in values]


def usage(values):
    for value in reversed(values):
        native = value.get("usage") if type(value) is dict else None
        if type(native) is dict:
            return {key: item for key, item in native.items() if type(item) is int}
    return {}


def completions(values, operation):
    result = {}
    for response in values:
        if type(response) is not dict:
            continue
        for offset, choice in enumerate(response.get("choices") or []):
            if type(choice) is not dict:
                continue
            index = choice.get("index", offset)
            if type(index) is not int:
                index = offset
            message = choice.get("message") or choice.get("delta")
            if type(message) is not dict:
                message = (
                    {"content": choice["text"]}
                    if "text" in choice and choice["text"] is not None
                    else {}
                )
            target = result.setdefault(index, {})
            if type(message.get("role")) is str:
                target["role"] = message["role"]
            content = message.get("content")
            if type(content) is str:
                target["content"] = target.get("content", "") + content
            for key in ("tool_calls", "function_call"):
                calls = message.get(key)
                if not calls:
                    continue
                if key == "function_call":
                    calls = [{"type": "function", "function": calls}]
                if type(calls) is not list:
                    continue
                stored = target.setdefault("tool_calls", {})
                for call_offset, call in enumerate(calls):
                    if type(call) is not dict:
                        continue
                    call_index = call.get("index", call_offset)
                    if type(call_index) is not int:
                        call_index = call_offset
                    merged = stored.setdefault(call_index, {"function": {}})
                    for field in ("id", "type"):
                        if field in call and call[field] is not None:
                            merged[field] = call[field]
                    function = call.get("function")
                    if type(function) is dict:
                        for field in ("name", "arguments"):
                            item = function.get(field)
                            if type(item) is str:
                                merged["function"][field] = (
                                    merged["function"].get(field, "") + item
                                )
                            elif item is not None:
                                merged["function"][field] = item
            if choice.get("finish_reason") is not None:
                target["finish_reason"] = choice["finish_reason"]
    for target in result.values():
        if "tool_calls" in target:
            target["tool_calls"] = list(target["tool_calls"].values())
    return result


def redact_stream_fragments(values):
    # Preserve complete merged argument/text projections, and redact every raw
    # fragment when a credential straddles native SSE chunk boundaries.
    groups = {}
    for response in values:
        if type(response) is not dict:
            continue
        for offset, choice in enumerate(response.get("choices") or []):
            if type(choice) is not dict:
                continue
            candidate = choice.get("index", offset)
            if type(candidate) is not int:
                candidate = offset
            delta = choice.get("delta")
            if type(delta) is not dict:
                continue
            for key in ("content", "reasoning", "reasoning_content"):
                if type(delta.get(key)) is str:
                    groups.setdefault((candidate, key), []).append(
                        (delta, key, delta[key])
                    )
            for offset, call in enumerate(delta.get("tool_calls") or []):
                if type(call) is not dict:
                    continue
                index = call.get("index", offset)
                if type(index) is not int:
                    index = offset
                function = call.get("function")
                if type(function) is dict and type(function.get("arguments")) is str:
                    groups.setdefault((candidate, "tool", index), []).append(
                        (function, "arguments", function["arguments"])
                    )
            function = delta.get("function_call")
            if type(function) is dict and type(function.get("arguments")) is str:
                groups.setdefault((candidate, "function"), []).append(
                    (function, "arguments", function["arguments"])
                )
    for pieces in groups.values():
        text = "".join(item[2] for item in pieces)
        if safe_text(text) != text:
            for target, key, _ in pieces:
                target[key] = "[REDACTED]"
    return values
