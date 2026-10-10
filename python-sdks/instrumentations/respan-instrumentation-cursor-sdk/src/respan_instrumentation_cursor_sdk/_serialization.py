"""Serialize only JSON values and known native SDK dataclasses."""

from __future__ import annotations

import dataclasses
import json
import math
import re

_SECRET = re.compile(
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|password|credential|secret|token)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)\b(?:Bearer|Basic)\s+(?:\[REDACTED\]|[^\s,;\"'{}\[\]\\]+)")
_ASSIGNMENT = re.compile(
    r"(?P<prefix>['\"]?(?:api[_-]?key|authorization|access[_-]?token|password|secret|token)['\"]?\s*[:=]\s*)(?:(?P<quote>['\"])(?:\\.|(?!(?P=quote)|\\).)*(?P=quote)|\[REDACTED\]|[^\s,;&?#}\]\[\"']+)",
    re.IGNORECASE,
)
_KEY = re.compile(r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{8,}|crsr_[A-Za-z0-9_-]{8,})\b")


def redact_text(value):
    value = re.sub(r"(https?://)[^/\s?\"\']+@", r"\1[REDACTED]@", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _KEY.sub("[REDACTED]", value)
    return _ASSIGNMENT.sub(
        lambda m: m["prefix"] + (m["quote"] or "") + "[REDACTED]" + (m["quote"] or ""),
        value,
    )


def safe(value, seen=None, *, schema=False):
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is str:
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return redact_text(value)
        if isinstance(parsed, (dict, list)):
            return json.dumps(safe(parsed, seen, schema=schema), separators=(",", ":"))
        return redact_text(value)
    seen = set() if seen is None else seen
    if id(value) in seen:
        return "[Circular]"
    original_id = id(value)
    seen.add(original_id)
    try:
        if dataclasses.is_dataclass(value) and type(value).__module__.startswith(
            "cursor_sdk"
        ):
            value = {
                f.name: getattr(value, f.name)
                for f in dataclasses.fields(value)
                if not f.name.startswith("_") and not callable(getattr(value, f.name))
            }
        if type(value) is dict:
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    continue
                if key in ("properties", "$defs", "definitions"):
                    result[key] = (
                        {
                            k: safe(
                                {
                                    nk: "[REDACTED]"
                                    if _SECRET.fullmatch(k)
                                    and nk in ("default", "enum", "examples", "const")
                                    else nv
                                    for nk, nv in v.items()
                                },
                                seen,
                                schema=True,
                            )
                            if type(v) is dict
                            else safe(v, seen, schema=True)
                            for k, v in item.items()
                        }
                        if type(item) is dict
                        else safe(item, seen)
                    )
                elif (
                    _SECRET.fullmatch(key)
                    or key.lower() in ("env", "env_vars", "envvars", "headers", "auth")
                    or key.lower().endswith(
                        ("_token", "_secret", "_password", "_api_key")
                    )
                ):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = safe(item, seen, schema=schema)
            return result
        if type(value) in (list, tuple):
            return [safe(item, seen, schema=schema) for item in value]
        return None
    finally:
        seen.discard(original_id)


def dumps(value):
    return json.dumps(safe(value), separators=(",", ":"), allow_nan=False)
