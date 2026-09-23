"""Bounded, secret-safe serialization for Helicone payloads."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from numbers import Integral, Real
from typing import Any

from respan_instrumentation_helicone._constants import (
    MAX_ATTRIBUTE_BYTES,
    MAX_COLLECTION_ITEMS,
    MAX_DEPTH,
    MAX_TEXT_BYTES,
)

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credentials",
        "credential",
        "helicone_auth",
        "password",
        "passwd",
        "private_key",
        "secret",
        "set_cookie",
        "token",
    }
)
_SENSITIVE_SUFFIXES = (
    "access_key_id",
    "api_key",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "cookie",
    "credential",
    "credentials",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session_token",
    "secret_access_key",
    "secret_key",
    "token",
    "access_token",
    "id_token",
)
_KEY_VALUE = re.compile(
    r"(?i)(?P<prefix>[\"']?(?P<key>[A-Za-z][A-Za-z0-9_-]*)[\"']?\s*[:=]\s*)"
    r'(?:"(?P<double>(?:\\.|[^"\\])*)"|'
    r"'(?P<single>(?:\\.|[^'\\])*)'|"
    r"(?P<bare>[^\s}\[{]+))"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_AUTHORIZATION_VALUE = re.compile(
    r'(?im)(?P<prefix>(?<!["\'])\bauthorization\s*[:=]\s*)[^\r\n}]*'
)


def truncate_utf8(value: str, limit: int = MAX_TEXT_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    budget = max(0, limit - len(suffix.encode("utf-8")))
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


_SENSITIVE_PREFIXES = ("password_", "secret_")


def _redact_json_value(value: Any, *, depth: int) -> Any:
    if depth >= MAX_DEPTH:
        return {"type": safe_type_name(value), "truncated": True}
    if isinstance(value, Mapping):
        result = {
            key: (
                "<redacted>"
                if isinstance(key, str) and is_sensitive_key(key)
                else _redact_json_value(item, depth=depth + 1)
            )
            for key, item in islice(value.items(), MAX_COLLECTION_ITEMS)
        }
        if len(value) > MAX_COLLECTION_ITEMS:
            result["_truncated"] = True
        return result
    if isinstance(value, list):
        result = [
            _redact_json_value(item, depth=depth + 1)
            for item in islice(value, MAX_COLLECTION_ITEMS)
        ]
        if len(value) > MAX_COLLECTION_ITEMS:
            result.append({"truncated": True})
        return result
    if isinstance(value, str):
        return safe_text(value, _depth=depth + 1)
    return value


def _redact_free_text(value: str, *, depth: int = 0) -> str:
    redacted = _AUTHORIZATION_VALUE.sub(
        lambda match: f"{match.group('prefix')}<redacted>", value
    )
    redacted = _BEARER.sub("Bearer <redacted>", redacted)
    if depth >= MAX_DEPTH:
        return "<redacted>" if _KEY_VALUE.search(redacted) else redacted

    def replace_secret(match: re.Match[str]) -> str:
        is_sensitive = is_sensitive_key(match.group("key"))
        if match.group("double") is not None:
            inner = (
                "<redacted>"
                if is_sensitive
                else _redact_free_text(match.group("double"), depth=depth + 1)
            )
            replacement = f'"{inner}"'
        elif match.group("single") is not None:
            inner = (
                "<redacted>"
                if is_sensitive
                else _redact_free_text(match.group("single"), depth=depth + 1)
            )
            replacement = f"'{inner}'"
        else:
            replacement = (
                "<redacted>"
                if is_sensitive
                else _redact_free_text(match.group("bare"), depth=depth + 1)
            )
        return f"{match.group('prefix')}{replacement}"

    return _KEY_VALUE.sub(replace_secret, redacted)


def safe_text(value: str, *, _depth: int = 0) -> str:
    if _depth >= MAX_DEPTH:
        return truncate_utf8(_redact_free_text(value, depth=_depth))
    try:
        parsed = json.loads(value)
    except (RecursionError, TypeError, ValueError):
        parsed = None
    if isinstance(parsed, Mapping | list):
        redacted = json.dumps(
            _redact_json_value(parsed, depth=_depth),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    elif isinstance(parsed, str):
        redacted = json.dumps(
            _redact_free_text(parsed, depth=_depth + 1),
            ensure_ascii=False,
        )
    else:
        redacted = _redact_free_text(value, depth=_depth)
    return truncate_utf8(redacted)


def safe_type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    module = getattr(value_type, "__module__", "")
    name = getattr(value_type, "__qualname__", None) or getattr(
        value_type, "__name__", "object"
    )
    resolved = f"{module}.{name}" if module and module != "builtins" else name
    return re.sub(r"[^A-Za-z0-9_.-]+", ".", resolved).strip(".")[:256] or "object"


def is_sensitive_key(value: str) -> bool:
    normalized = _CAMEL_BOUNDARY.sub("_", value.strip()).lower().replace("-", "_")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.startswith(_SENSITIVE_PREFIXES)
        or any(
            normalized == suffix or normalized.endswith(f"_{suffix}")
            for suffix in _SENSITIVE_SUFFIXES
        )
    )


def _is_numeric_vector(value: Sequence[Any]) -> bool:
    if not value:
        return False
    return all(isinstance(item, Real) and not isinstance(item, bool) for item in value)


def _numeric_vector(value: Sequence[Any]) -> list[float | None]:
    return [float(item) if math.isfinite(float(item)) else None for item in value]


def _budgeted_text(value: str, budget: list[int]) -> str:
    safe_value = safe_text(value)
    available = max(0, min(MAX_TEXT_BYTES, budget[0]))
    if available == 0:
        return ""
    result = truncate_utf8(safe_value, available)
    budget[0] = max(0, budget[0] - len(result.encode("utf-8")))
    return result


def _sanitize_preserving_vectors(
    value: Any,
    *,
    budget: list[int],
    depth: int = 0,
    allow_numeric_vector: bool = False,
) -> Any:
    """Preserve numeric vectors while bounding all surrounding context."""

    if depth >= MAX_DEPTH:
        return {"type": safe_type_name(value), "truncated": True}
    if value is None or isinstance(value, bool):
        budget[0] = max(0, budget[0] - 5)
        return value
    if isinstance(value, str):
        return _budgeted_text(value, budget)
    if isinstance(value, Integral):
        budget[0] = max(0, budget[0] - 24)
        return int(value)
    if isinstance(value, Real):
        budget[0] = max(0, budget[0] - 32)
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        consumed = 0
        for key, item in islice(value.items(), MAX_COLLECTION_ITEMS):
            if budget[0] <= 0:
                break
            raw_key_text = key if isinstance(key, str) else safe_type_name(key)
            is_sensitive = is_sensitive_key(raw_key_text)
            key_text = safe_text(raw_key_text) if isinstance(key, str) else raw_key_text
            key_text = _budgeted_text(key_text, budget)
            if not key_text:
                break
            result[key_text] = (
                "<redacted>"
                if is_sensitive
                else _sanitize_preserving_vectors(
                    item,
                    budget=budget,
                    depth=depth + 1,
                    allow_numeric_vector=key_text.lower()
                    in {"vector", "vectors", "embedding", "embeddings"},
                )
            )
            consumed += 1
        if consumed < len(value):
            result["_truncated"] = True
        return result
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        if allow_numeric_vector and _is_numeric_vector(value):
            return _numeric_vector(value)
        is_vector_batch = allow_numeric_vector and all(
            isinstance(item, Sequence)
            and not isinstance(item, str | bytes | bytearray)
            and _is_numeric_vector(item)
            for item in value
        )
        items: list[Any] = []
        consumed = 0
        limit = len(value) if is_vector_batch else MAX_COLLECTION_ITEMS
        for item in islice(value, limit):
            if budget[0] <= 0 and not is_vector_batch:
                break
            items.append(
                _sanitize_preserving_vectors(
                    item,
                    budget=budget,
                    depth=depth + 1,
                    allow_numeric_vector=allow_numeric_vector,
                )
            )
            consumed += 1
        if consumed < len(value):
            items.append({"truncated": True})
        return items
    return {"type": _budgeted_text(safe_type_name(value), budget)}


def sanitize(
    value: Any, *, depth: int = 0, preserve_large_sequences: bool = False
) -> Any:
    if depth >= MAX_DEPTH:
        return {"type": safe_type_name(value), "truncated": True}
    if value is None or isinstance(value, bool | str):
        return safe_text(value) if isinstance(value, str) else value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in islice(value.items(), MAX_COLLECTION_ITEMS):
            raw_key_text = key if isinstance(key, str) else safe_type_name(key)
            is_sensitive = is_sensitive_key(raw_key_text)
            key_text = safe_text(raw_key_text) if isinstance(key, str) else raw_key_text
            result[key_text] = (
                "<redacted>"
                if is_sensitive
                else sanitize(
                    item,
                    depth=depth + 1,
                    preserve_large_sequences=preserve_large_sequences,
                )
            )
        if len(value) > MAX_COLLECTION_ITEMS:
            result["_truncated"] = True
        return result
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        if preserve_large_sequences and _is_numeric_vector(value):
            return _numeric_vector(value)
        items = [
            sanitize(
                item,
                depth=depth + 1,
                preserve_large_sequences=preserve_large_sequences,
            )
            for item in islice(value, MAX_COLLECTION_ITEMS)
        ]
        if len(value) > MAX_COLLECTION_ITEMS:
            items.append({"truncated": True})
        return items
    return {"type": safe_type_name(value)}


def json_dumps(value: Any, *, preserve_large_vectors: bool = False) -> str:
    sanitized = (
        _sanitize_preserving_vectors(
            value,
            budget=[MAX_ATTRIBUTE_BYTES // 2],
            allow_numeric_vector=not isinstance(value, Mapping),
        )
        if preserve_large_vectors
        else sanitize(value)
    )
    encoded = json.dumps(
        sanitized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if preserve_large_vectors or len(encoded.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES:
        return encoded
    preview = truncate_utf8(encoded, MAX_ATTRIBUTE_BYTES - 64)
    return json.dumps(
        {"preview": preview, "truncated": True},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def exception_message(error: BaseException) -> str:
    for argument in getattr(error, "args", ()):
        if isinstance(argument, str) and argument:
            return safe_text(argument)
    return safe_type_name(error)
