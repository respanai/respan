"""Bounded, privacy-safe serialization for OpenInference attributes."""

from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_ATTRIBUTE_CHARS = 16_000
MAX_LABEL_CHARS = 512
_MAX_DEPTH = 8
_MAX_ITEMS = 1_000
_REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credentials",
    "password",
    "private-key",
    "private_key",
    "refresh-token",
    "refresh_token",
    "secret",
    "token",
    "x-api-key",
}
_SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "-api-key",
    "_password",
    "-password",
    "_secret",
    "-secret",
    "_access_token",
    "-access-token",
    "_refresh_token",
    "-refresh-token",
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}")
_BASIC_AUTH_RE = re.compile(r"(?i)\bbasic\s+[a-z0-9+/=]{8,}")
_API_KEY_RE = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b("
    r"api[-_]?key|authorization|password|passwd|secret|"
    r"(?:access|refresh|session|auth)?[-_]?token"
    r")(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def sanitize_text(value: str) -> str:
    """Redact common credential shapes without stringifying arbitrary objects."""
    value = re.sub(r"(https?://)[^/@\s]+@", r"\1[REDACTED]@", value)
    value = _BEARER_RE.sub("Bearer [REDACTED]", value)
    value = _BASIC_AUTH_RE.sub("Basic [REDACTED]", value)
    value = _API_KEY_RE.sub(_REDACTED, value)
    return _ASSIGNMENT_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        value,
    )


def bounded_text(value: Any, *, max_chars: int = MAX_LABEL_CHARS) -> str:
    """Return a redacted bounded scalar without invoking user string hooks."""
    if not isinstance(value, str):
        return f"[UNSUPPORTED:{type(value).__name__}]"
    sanitized = sanitize_text(value)
    if len(sanitized) <= max_chars:
        return sanitized
    suffix = "...[TRUNCATED]"
    return f"{sanitized[: max(0, max_chars - len(suffix))]}{suffix}"


def to_jsonable(value: Any, *, depth: int = 0, complete: bool = False) -> Any:
    """Convert supported values to JSON data without calling user ``repr``/``str``."""
    if depth > _MAX_DEPTH and not complete:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE_FLOAT]"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, bytes):
        return sanitize_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS and not complete:
                result["__truncated_items__"] = True
                break
            key = (
                str(raw_key) if type(raw_key) in (str, int) else type(raw_key).__name__
            )
            result[key] = (
                _REDACTED
                if _is_sensitive_key(key)
                else to_jsonable(item, depth=depth + 1, complete=complete)
            )
        return result
    if isinstance(value, (list, tuple)):
        result = [
            to_jsonable(item, depth=depth + 1, complete=complete)
            for item in (value if complete else value[:_MAX_ITEMS])
        ]
        if len(value) > _MAX_ITEMS and not complete:
            result.append("[TRUNCATED_ITEMS]")
        return result
    return f"[UNSUPPORTED:{type(value).__name__}]"


def parse_json(value: Any) -> Any:
    """Parse JSON strings and leave all other supported values unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def bounded_json(value: Any, *, max_chars: int = MAX_ATTRIBUTE_CHARS) -> str:
    """Return redacted valid JSON no longer than ``max_chars``."""
    parsed = parse_json(value)
    if _contains_vector(parsed):
        return complete_json(parsed)
    normalized = to_jsonable(parsed)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(serialized) <= max_chars:
        return serialized

    low = 0
    high = len(serialized)
    best = json.dumps(
        {"preview": "", "truncated": True},
        separators=(",", ":"),
        sort_keys=True,
    )
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {"preview": serialized[:midpoint], "truncated": True},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(candidate) <= max_chars:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def content_value(value: Any) -> str:
    """Keep scalar message text readable; JSON-encode structured content."""
    parsed = parse_json(value)
    if isinstance(parsed, str):
        text = sanitize_text(parsed)
        if len(text) <= MAX_ATTRIBUTE_CHARS:
            return text
    return bounded_json(parsed)


def complete_json(value: Any) -> str:
    """Serialize full tool/vector payloads; storage limits belong to ingestion."""
    return json.dumps(
        to_jsonable(parse_json(value), complete=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def complete_value(value: str) -> str:
    """Redact an actual invocation ID/signature without changing its length."""
    return sanitize_text(value)


def _contains_vector(value: Any, seen: set[int] | None = None) -> bool:
    """Do not preview dense/sparse vectors nested in message/document payloads."""
    seen = set() if seen is None else seen
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, (list, tuple)):
            if value and all(type(item) in (int, float) for item in value):
                return True
            return any(_contains_vector(item, seen) for item in value)
        if value and all(
            (type(key) is int or (isinstance(key, str) and key.isdigit()))
            and type(item) in (int, float)
            for key, item in value.items()
        ):
            return True
        return any(_contains_vector(item, seen) for item in value.values())
    return False
