"""Redact builtin JSON without invoking methods on unknown customer objects."""

from __future__ import annotations

import json
import math
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_DEPTH = 64
MAX_STRING_BYTES = 4_000
REDACTED = "[REDACTED]"
_SENSITIVE_SUFFIXES = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "sessiontoken",
    "token",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|(?:client[_-]?)?secret|session[_-]?token|token)[\"']?)"
    r"(\s*[:=]\s*)(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\[REDACTED\]|[^\s,;}\]]+)"
)
_AUTHORIZATION = re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]+")
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")
_SCHEMA_KEYS = frozenset(
    {"type", "$ref", "properties", "items", "anyOf", "allOf", "oneOf"}
)


def _type_name(value):
    return type.__getattribute__(type(value), "__name__")


def _truncate_utf8(value, limit=MAX_STRING_BYTES):
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "...[truncated]"
    return (
        encoded[: max(0, limit - len(suffix))].decode("utf-8", errors="ignore") + suffix
    )


def sanitize_url(value):
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return "<redacted-endpoint>"
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "<redacted-endpoint>"


def _redact_assignment(match):
    value = match.group(3)
    quote = value[0] if value[:1] in {'"', "'"} else ""
    return f"{match.group(1)}{match.group(2)}{quote}{REDACTED}{quote}"


def safe_text(value: Any, *, default="", truncate=True):
    if type(value) is str:
        # A JSON-encoded argument remains valid JSON after redaction. Repeated
        # sanitation is idempotent, including escaped quoted credential values.
        stripped = value.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                pass
            else:
                sanitized = json_value(decoded)
                if sanitized != decoded:
                    value = json.dumps(
                        sanitized, ensure_ascii=False, separators=(",", ":")
                    )
                return _truncate_utf8(value) if truncate else value
        value = _URL.sub(lambda match: sanitize_url(match.group()), value)
        value = _AUTHORIZATION.sub(REDACTED, value)
        value = _SECRET_ASSIGNMENT.sub(_redact_assignment, value)
        return _truncate_utf8(value) if truncate else value
    if value is None:
        return default
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return str(value)
    return f"<{_type_name(value)}>"


def exception_message(exc):
    try:
        arguments = exc.args
    except Exception:  # noqa: BLE001 - diagnostics never alter native errors.
        arguments = ()
    if type(arguments) is tuple:
        for argument in arguments:
            if type(argument) in {str, bool, int, float}:
                return safe_text(argument)
    return _type_name(exc)


def exception_status(exc, *, default=500):
    try:
        response = getattr(exc, "response", None)
    except Exception:  # noqa: BLE001 - errors may have hostile properties.
        response = None
    for candidate in (exc, response):
        for name in ("status_code", "status"):
            try:
                value = getattr(candidate, name, None)
            except Exception:  # noqa: BLE001 - preserve the original error.
                value = None
            if type(value) is int and 400 <= value <= 599:
                return value
    return default


def sensitive_key(value):
    if type(value) is not str:
        return False
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return any(normalized.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def json_value(
    value, *, depth=0, seen=None, schema_properties=False, schema_secret=False
):
    kind = type(value)
    if value is None or kind in {bool, int}:
        return value
    if kind is str:
        return safe_text(value, truncate=False)
    if kind is float:
        return value if math.isfinite(value) else None
    if kind in {bytes, bytearray, memoryview}:
        return {"length": len(value), "type": _type_name(value)}
    if kind not in {dict, list, tuple}:
        return {"type": _type_name(value)}
    if depth >= MAX_DEPTH:
        return {"truncated": "max_depth", "type": _type_name(value)}
    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        return "<cycle>"
    active.add(identity)
    try:
        if kind is dict:
            result = {}
            for key, item in value.items():
                name = safe_text(key, truncate=False)
                secret = sensitive_key(key)
                definition = (
                    schema_properties
                    and type(item) is dict
                    and any(type(k) is str and k in _SCHEMA_KEYS for k in item)
                )
                if (secret and not definition) or (
                    schema_secret
                    and name in {"default", "examples", "example", "const", "enum"}
                ):
                    result[name] = REDACTED
                else:
                    result[name] = json_value(
                        item,
                        depth=depth + 1,
                        seen=active,
                        schema_properties=name == "properties",
                        schema_secret=secret if definition else schema_secret,
                    )
            return result
        return [json_value(item, depth=depth + 1, seen=active) for item in value]
    finally:
        active.discard(identity)


def json_string(value):
    return json.dumps(
        json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
