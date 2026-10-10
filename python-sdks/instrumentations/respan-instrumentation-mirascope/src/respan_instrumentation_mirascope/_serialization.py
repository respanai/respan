"""Safe capture of known Mirascope values, including complete tool payloads."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

_MAX_DEPTH = 64
_MAX_ITEMS = 50
_MAX_STRING_LENGTH = 8_000
_MAX_JSON_LENGTH = 16_000
_REDACTED = "[REDACTED]"
_SENSITIVE = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|authorization|cookie|password|secret|token|credential)(?:$|[._-])"
    r"|(?:api[_-]?key|authorization|cookie|password|secret|token|credential)$",
    re.IGNORECASE,
)
_QUOTED = re.compile(
    r"(?i)([\"']?(?:[a-z0-9_-]*[_-])?(?:api[_-]?key|authorization|cookie|password|secret|token|credential)[\"']?\s*[:=]\s*)([\"'])((?:\\.|(?!\2).)*)\2",
    re.DOTALL,
)
_UNFINISHED_QUOTED = re.compile(
    r"(?i)([\"']?(?:[a-z0-9_-]*[_-])?(?:api[_-]?key|authorization|cookie|password|secret|token|credential)[\"']?\s*[:=]\s*)([\"'])((?:\\.|(?!\2).)*)\Z",
    re.DOTALL,
)
_BARE = re.compile(
    r"(?i)([\"']?(?:[a-z0-9_-]*[_-])?(?:api[_-]?key|authorization|cookie|password|secret|token|credential)[\"']?\s*[:=]\s*)(?!['\"\[{])[^,;)\s}\"']+"
)


def safe_text(value: Any, *, complete: bool = False) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        return ""
    elif type(value) in (bool, int, float):
        text = str(value)
    else:
        return f"<{type(value).__name__}>"
    text = _QUOTED.sub(lambda m: f"{m[1]}{m[2]}{_REDACTED}{m[2]}", text)
    text = _UNFINISHED_QUOTED.sub(lambda m: f"{m[1]}{m[2]}{_REDACTED}", text)
    text = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"(?i)\b(bearer|basic)\s+[^\s,;\"'\\]+", r"\1 [REDACTED]", text)
    text = _BARE.sub(r"\1[REDACTED]", text)
    return (
        text
        if complete or len(text) <= _MAX_STRING_LENGTH
        else text[:_MAX_STRING_LENGTH] + "...[truncated]"
    )


def safe_exception_text(exc: BaseException) -> str:
    try:
        args = BaseException.args.__get__(exc)
    except Exception:  # noqa: BLE001 - diagnostics must preserve native exceptions.
        args = ()
    return safe_text(": ".join(v for v in args[:4] if isinstance(v, str)))


def _vector(value: Any) -> bool:
    if isinstance(value, list | tuple) and len(value) > _MAX_ITEMS:
        return all(type(v) in (int, float) for v in value)
    if type(value) is dict and len(value) > _MAX_ITEMS:
        return all(
            (type(k) is int and k >= 0 or isinstance(k, str) and k.isdecimal())
            and type(v) in (int, float)
            for k, v in value.items()
        )
    return False


def json_value(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
    complete: bool = False,
    schema: bool = False,
    _properties: bool = False,
    _sensitive_property: bool = False,
) -> Any:
    complete = complete or _vector(value)
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return safe_text(value, complete=complete)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if _depth >= _MAX_DEPTH:
        return f"<{type(value).__name__}:max-depth>"
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return "<cycle>"
    seen.add(identity)
    try:
        if type(value).__module__.startswith(("mirascope.", "openai.")):
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                value = {
                    f.name: getattr(value, f.name)
                    for f in dataclasses.fields(value)
                    if not f.name.startswith("_")
                }
            else:
                fields = getattr(type(value), "model_fields", {})
                if isinstance(fields, Mapping) and fields:
                    value = {
                        getattr(f, "alias", None) or name: field_value
                        for name, f in fields.items()
                        if (field_value := getattr(value, name, None)) is not None
                        or not schema
                    }
                else:
                    return f"<{type(value).__name__}>"
        if isinstance(value, Mapping):
            result = {}
            iterator = value.items() if complete else islice(value.items(), _MAX_ITEMS)
            for key, item in iterator:
                if not isinstance(key, str) and type(key) is not int:
                    continue
                key = str(key)
                sensitive = bool(_SENSITIVE.search(key))
                structural = schema and _properties and isinstance(item, dict | bool)
                credential_default = (
                    schema
                    and _sensitive_property
                    and key in {"default", "example", "examples", "const", "enum"}
                )
                if credential_default or sensitive and not structural:
                    result[key] = (
                        [_REDACTED for _ in item]
                        if credential_default and isinstance(item, list | tuple)
                        else _REDACTED
                    )
                else:
                    result[key] = json_value(
                        item,
                        _depth=_depth + 1,
                        _seen=seen,
                        complete=complete,
                        schema=schema
                        and key
                        not in {"default", "example", "examples", "const", "enum"},
                        _properties=schema
                        and key
                        in {"properties", "patternProperties", "$defs", "definitions"},
                        _sensitive_property=_sensitive_property
                        or _properties
                        and sensitive,
                    )
            return result
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            items = value if complete else islice(value, _MAX_ITEMS)
            return [
                json_value(
                    v,
                    _depth=_depth + 1,
                    _seen=seen,
                    complete=complete,
                    schema=schema,
                    _sensitive_property=_sensitive_property,
                )
                for v in items
            ]
        return f"<{type(value).__name__}>"
    except Exception:  # noqa: BLE001 - telemetry cannot replace native results.
        return f"<{type(value).__name__}:unserializable>"
    finally:
        seen.discard(identity)


def json_string(value: Any, *, complete: bool = False, schema: bool = False) -> str:
    complete = complete or _vector(value)
    encoded = json.dumps(
        json_value(value, complete=complete, schema=schema),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if complete or len(encoded) <= _MAX_JSON_LENGTH:
        return encoded
    return json.dumps(
        {"preview": encoded[: _MAX_JSON_LENGTH - 200], "truncated": True},
        ensure_ascii=False,
        sort_keys=True,
    )
