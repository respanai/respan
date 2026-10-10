"""Complete builtin native payloads without unknown conversion hooks."""

from __future__ import annotations

import json
import math
import re
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE = re.compile(
    r"(?:apikey|authorization|password|secret|sessiontoken|accesstoken|refreshtoken|token|credentials?|privatekey|cookie)$"
)
_ASSIGNMENT = re.compile(
    r"""(?ix)(["']?(?:api[_-]?key|authorization|password|secret|session[_-]?token|access[_-]?token|refresh[_-]?token|credentials?|private[_-]?key|token)["']?)(\s*[:=]\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}]+)"""
)
_AUTH = re.compile(
    r"""(?i)\b(?:bearer|basic)\s+(?!\[REDACTED\])(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[a-z0-9._~+/=-]+)"""
)
_URL = re.compile(
    r"(?i)\b(https?://)([^\s/?#@\"\']+@)?([^\s/?#\"\']+)([^\s?#\"\']*)(?:\?[^\s#\"\']*)?(?:#[^\s\"\']*)?"
)


def _type(value: Any) -> str:
    return type.__getattribute__(type(value), "__name__")


def sensitive_key(key: Any) -> bool:
    return type(key) is str and bool(
        _SENSITIVE.search(re.sub(r"[^a-z0-9]", "", key.lower()))
    )


def safe_text(value: Any, *, default: str = "") -> str:
    if type(value) is str:
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = None
            if type(parsed) in (dict, list):
                converted = to_jsonable(parsed)
                return (
                    value
                    if converted == parsed
                    else json.dumps(
                        converted,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )

        def assignment(match):
            raw = match[3]
            replacement = (
                json.dumps(REDACTED)
                if raw.startswith('"')
                else ("'" + REDACTED + "'" if raw.startswith("'") else REDACTED)
            )
            return match[1] + match[2] + replacement

        return _ASSIGNMENT.sub(
            assignment,
            _AUTH.sub(REDACTED, _URL.sub(lambda m: m[1] + m[3] + m[4], value)),
        )
    if value is None:
        return default
    if type(value) in (bool, int):
        return str(value).lower()
    if type(value) is float and math.isfinite(value):
        return str(value)
    return "<" + _type(value) + ">"


def to_jsonable(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
    schema: bool = False,
    property_map: bool = False,
    secret_property: bool = False,
) -> Any:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is str:
        return safe_text(value)
    if type(value) in (bytes, bytearray, memoryview):
        return {"length": len(value), "type": _type(value)}
    active = set() if seen is None else seen
    if id(value) in active:
        return "<cycle>"
    active.add(id(value))
    try:
        if type(value) is dict:
            is_schema = schema or (
                type(value.get("properties")) is dict and value.get("type") == "object"
            )
            result = {}
            for key, item in value.items():
                text = safe_text(key)
                secret = sensitive_key(key)
                if (
                    key == "token"
                    and type(item) is dict
                    and type(item.get("text")) is str
                    and set(item).issubset({"id", "text", "special", "logprob"})
                ):
                    secret = False
                if (secret and not property_map) or (
                    secret_property
                    and key in ("default", "const", "enum", "examples", "example")
                ):
                    result[text] = REDACTED
                else:
                    result[text] = to_jsonable(
                        item,
                        depth=depth + 1,
                        seen=active,
                        schema=is_schema,
                        property_map=is_schema
                        and key in ("properties", "$defs", "definitions"),
                        secret_property=property_map and secret,
                    )
            return result
        if type(value) in (list, tuple):
            return [
                to_jsonable(v, depth=depth + 1, seen=active, schema=schema)
                for v in value
            ]
        return {"type": _type(value)}
    finally:
        active.remove(id(value))


def json_dumps(value: Any, *, max_bytes: int | None = None) -> str:
    encoded = json.dumps(
        to_jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if max_bytes is None or len(encoded.encode("utf-8")) <= max_bytes:
        return encoded
    # Only callers explicitly requesting a cap receive a valid JSON summary.
    return json.dumps(
        {"original_bytes": len(encoded.encode("utf-8")), "truncated": True},
        separators=(",", ":"),
    )


# Only exact released native types are traversed; never invoke customer dump hooks.
_NATIVE_TYPES = set()
_EVENT_TYPES = set()


def register_native_types():
    import importlib

    for module_name, class_name in [
        ("prediction", "Prediction"),
        ("model", "Model"),
        ("version", "Version"),
        ("deployment", "Deployment"),
        ("pagination", "Page"),
        ("stream", "ServerSentEvent"),
        ("helpers", "FileOutput"),
    ]:
        module = importlib.import_module("replicate." + module_name)
        _NATIVE_TYPES.add(getattr(module, class_name))
    from replicate.stream import ServerSentEvent

    _EVENT_TYPES.add(ServerSentEvent.EventType)


def native_value(value, seen=None):
    from datetime import date, datetime

    active = set() if seen is None else seen
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) in (date, datetime):
        return value.isoformat()
    if id(value) in active:
        return "<cycle>"
    active.add(id(value))
    try:
        if type(value) in _EVENT_TYPES:
            return object.__getattribute__(value, "_value_")
        if type(value) in _NATIVE_TYPES:
            fields = object.__getattribute__(value, "__dict__")
            return {
                k: native_value(v, active)
                for k, v in fields.items()
                if type(k) is str and not k.startswith("_") and k != "client"
            }
        if type(value) is dict:
            return {
                k: native_value(v, active)
                for k, v in value.items()
                if type(k) in (str, int, bool, float)
            }
        if type(value) in (list, tuple):
            return [native_value(v, active) for v in value]
        return {"type": _type(value)}
    finally:
        active.remove(id(value))


def json_string(value):
    return json_dumps(native_value(value))


def prediction_summary(value):
    return to_jsonable(native_value(value))
