"""Redacted JSON values without application repr or serializer hooks."""

from __future__ import annotations

import json
import math
import re
from dataclasses import fields, is_dataclass
from typing import Any

from pydantic import BaseModel

MAX_CHARS = 16_000
_MAX_ITEMS = 4096
_MAX_DEPTH = 8
_SECRET = re.compile(r"(?i)\b(?:bearer|basic)\s+[\w.+/=-]{8,}|\b(?:sk|rk|pk)-[\w-]{8,}")
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/\s@]+@")
_ASSIGNMENT = re.compile(
    r"(?i)\b(api[-_]?key|authorization|password|secret|(?:access|refresh|auth)?[-_]?token)(\s*[\"\']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)


def text(value: Any, limit: int | None = MAX_CHARS) -> str:
    if not isinstance(value, str):
        return f"[UNSUPPORTED:{type(value).__name__}]"
    value = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    value = _SECRET.sub("[REDACTED]", value)

    def assignment(match):
        quote = match[3][0] if match[3][0] in {'"', "'"} else ""
        return match[1] + match[2] + quote + "[REDACTED]" + quote

    value = _ASSIGNMENT.sub(assignment, value)
    return (
        value
        if limit is None or len(value) <= limit
        else value[: limit - 14] + "...[TRUNCATED]"
    )


def data(
    value: Any, *, complete: bool = False, depth: int = 0, seen: set[int] | None = None
) -> Any:
    if depth == 0:
        complete = complete or _contains_vector(value)
    if not complete and depth >= _MAX_DEPTH:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE]"
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")) and (
            complete or len(value) <= MAX_CHARS * 4
        ):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                pass
            else:
                return text(
                    json.dumps(
                        data(parsed, complete=complete, depth=depth + 1),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    None if complete else MAX_CHARS,
                )
        return text(value, None if complete else MAX_CHARS)
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    seen = seen or set()
    if id(value) in seen:
        return "[CYCLE]"
    seen = {*seen, id(value)}
    if isinstance(value, BaseModel):
        value = vars(value)
    elif is_dataclass(value) and not isinstance(value, type):
        value = {
            field.name: object.__getattribute__(value, field.name)
            for field in fields(value)
        }
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if not complete and index >= _MAX_ITEMS:
                result["__truncated_items__"] = True
                break
            if type(key) is int:
                key = str(key)
            if not isinstance(key, str) or key.startswith("_"):
                continue
            normalized = key.lower().replace("-", "_")
            secret = normalized in {
                "authorization",
                "cookie",
                "credentials",
                "password",
                "secret",
                "token",
                "api_key",
            } or normalized.endswith(
                ("_api_key", "_secret", "_password", "_access_token", "_refresh_token")
            )
            result[key] = (
                "[REDACTED]"
                if secret
                else data(item, complete=complete, depth=depth + 1, seen=seen)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            data(item, complete=complete, depth=depth + 1, seen=seen)
            for item in (value if complete else value[:_MAX_ITEMS])
        ]
    return f"[UNSUPPORTED:{type(value).__name__}]"


def _contains_vector(value: Any, seen: set[int] | None = None) -> bool:
    """Keep actual numeric vectors complete, including retriever/tool payloads."""
    seen = seen or set()
    if id(value) in seen:
        return False
    seen = {*seen, id(value)}
    if isinstance(value, BaseModel):
        value = vars(value)
    if isinstance(value, (list, tuple)):
        if len(value) > 50 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        ):
            return True
        return any(_contains_vector(v, seen) for v in value)
    if isinstance(value, dict):
        if len(value) > 50 and all(
            (type(k) is int and k >= 0 or type(k) is str and k.isdecimal())
            and type(v) in (int, float)
            for k, v in value.items()
        ):
            return True
        if value.get("role") == "tool" or any(
            k in value
            for k in ("tool_calls", "function", "inputSchema", "tool_call_id")
        ):
            return True
        return any(_contains_vector(v, seen) for v in value.values())
    return False


def json_value(value: Any, *, complete: bool = False) -> str:
    complete = complete or _contains_vector(value)
    value = json.dumps(
        data(value, complete=complete), ensure_ascii=False, separators=(",", ":")
    )
    if complete or len(value) <= MAX_CHARS:
        return value
    return json.dumps(
        {"truncated": True, "preview": value[: (MAX_CHARS - 100) // 6]},
        ensure_ascii=False,
    )
