"""Bounded serialization that never calls arbitrary application hooks."""

import json
import math
import re
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

    def redact_assignment(match: Any) -> str:
        quote = match[3][0] if match[3][0] in {'"', "'"} else ""
        return match[1] + match[2] + quote + "[REDACTED]" + quote

    value = _ASSIGNMENT.sub(redact_assignment, value)
    return (
        value
        if limit is None or len(value) <= limit
        else value[: limit - 14] + "...[TRUNCATED]"
    )


def data(value: Any, depth: int = 0, *, complete: bool = False) -> Any:
    if not complete and depth >= _MAX_DEPTH:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE]"
    if isinstance(value, str):
        # SDK tool arguments often arrive as JSON strings. Redact their keys,
        # then keep the argument value as a single JSON-encoded string.
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
                        data(parsed, depth + 1, complete=complete),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    None if complete else MAX_CHARS,
                )
        return text(value, None if complete else MAX_CHARS)
    if isinstance(value, BaseModel):
        value = vars(value)
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if not complete and index >= _MAX_ITEMS:
                result["__truncated_items__"] = True
                break
            if not isinstance(key, str) or key.startswith("_"):
                continue
            normalized = key.lower().replace("-", "_")
            sensitive = normalized in {
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
                "[REDACTED]" if sensitive else data(item, depth + 1, complete=complete)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            data(item, depth + 1, complete=complete)
            for item in (value if complete else value[:_MAX_ITEMS])
        ]
    return f"[UNSUPPORTED:{type(value).__name__}]"


def json_value(value: Any, *, complete: bool = False) -> str:
    value = json.dumps(
        data(value, complete=complete), ensure_ascii=False, separators=(",", ":")
    )
    if complete or len(value) <= MAX_CHARS:
        return value
    # Escape expansion is at most sixfold; keep the preview envelope valid JSON.
    return json.dumps(
        {"truncated": True, "preview": value[: (MAX_CHARS - 100) // 6]},
        ensure_ascii=False,
    )
