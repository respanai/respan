"""Bounded, privacy-safe serialization for Pinecone spans."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from itertools import islice
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_ATTRIBUTE_BYTES = 16_000
MAX_ITEMS = 50
MAX_DEPTH = 8
MAX_TEXT_BYTES = 8_000

_SENSITIVE_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "password",
    "secret",
    "token",
)
_INLINE_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|client[_-]?secret|password|secret|token)"
    r"(\s*[\"']?\s*[:=]\s*[\"']?\s*)([^\s,;&\"'}]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,;&]+")
_MEMORY_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")


def safe_type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    module = getattr(value_type, "__module__", "")
    qualname = getattr(value_type, "__qualname__", None) or getattr(
        value_type, "__name__", "object"
    )
    candidate = f"{module}.{qualname}" if module and module != "builtins" else qualname
    stable = re.sub(r"[^A-Za-z0-9_.-]+", ".", candidate).strip(".")
    return (stable or "object")[:256]


def truncate_utf8(value: str, max_bytes: int = MAX_TEXT_BYTES) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = "...[truncated]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    return encoded[:budget].decode("utf-8", errors="ignore") + suffix


def sanitize_endpoint(value: str) -> str:
    """Strip endpoint credentials, query parameters, and fragments."""

    try:
        candidate = value if "://" in value else f"//{value}"
        parsed = urlsplit(candidate)
        host = parsed.hostname
        if not host:
            return "<redacted-endpoint>"
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{host}{port}"
        scheme = parsed.scheme if "://" in value else ""
        sanitized = urlunsplit((scheme, netloc, parsed.path, "", ""))
        return sanitized if scheme else sanitized.removeprefix("//")
    except Exception:  # noqa: BLE001 - malformed endpoints are redacted
        return "<redacted-endpoint>"


def safe_text(value: str, *, endpoint: bool = False) -> str:
    if endpoint:
        value = sanitize_endpoint(value)
    value = _BEARER_TOKEN.sub("Bearer <redacted>", value)
    value = _INLINE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value
    )
    value = _MEMORY_ADDRESS.sub("0x<redacted>", value)
    return truncate_utf8(value)


def is_sensitive_key(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return any(
        normalized == part or normalized.endswith(f"_{part}") or part in normalized
        for part in _SENSITIVE_PARTS
    )


def _stable_key(value: Any) -> str:
    if isinstance(value, Enum):
        return _stable_key(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return safe_text(str(value))[:256]
    if isinstance(value, float) and math.isfinite(value):
        return str(value)[:256]
    return f"<{safe_type_name(value)}>"


def _summary(value: Any, *, reason: str | None = None) -> dict[str, str]:
    result = {"type": safe_type_name(value)}
    if reason:
        result["truncated"] = reason
    return result


def _trusted_pinecone_mapping(value: Any) -> Any | None:
    value_type = type(value)
    if not getattr(value_type, "__module__", "").startswith("pinecone"):
        return None
    for method_name in ("to_dict", "model_dump"):
        try:
            method = getattr(value, method_name, None)
        except Exception:  # noqa: BLE001, S112 - hostile provider descriptors
            continue
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001, S112 - malformed provider models
                continue
    # Pinecone 10 exposes msgspec response structs without to_dict/model_dump.
    if (
        value_type.__module__ == "pinecone.models.pagination"
        and value_type.__name__ == "Page"
    ):
        return {"items": value.items, "pagination_token": value.pagination_token}
    fields = getattr(value_type, "__struct_fields__", ())
    if (
        fields
        and isinstance(fields, tuple)
        and all(isinstance(name, str) for name in fields)
    ):
        try:
            return {name: getattr(value, name) for name in fields[:MAX_ITEMS]}
        except Exception:  # noqa: BLE001 - malformed provider models
            return None
    return None


def jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        return _summary(value, reason="max_depth")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Enum):
        return jsonable(value.value, depth=depth + 1)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            items = list(islice(value.items(), MAX_ITEMS + 1))
        except Exception:  # noqa: BLE001 - hostile mappings become type summaries
            return _summary(value)
        for key, item in items[:MAX_ITEMS]:
            key_text = _stable_key(key)
            result[key_text] = (
                "<redacted>"
                if is_sensitive_key(key_text)
                else jsonable(item, depth=depth + 1)
            )
        if len(items) > MAX_ITEMS:
            result["__truncated__"] = True
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            source = list(islice(iter(value), MAX_ITEMS + 1))
        except Exception:  # noqa: BLE001 - hostile sequences become type summaries
            return _summary(value)
        items = [jsonable(item, depth=depth + 1) for item in source[:MAX_ITEMS]]
        if len(source) > MAX_ITEMS:
            return {
                "items": items,
                "truncated": True,
                "count": f">{MAX_ITEMS}",
            }
        return items
    trusted = _trusted_pinecone_mapping(value)
    if trusted is not None:
        return jsonable(trusted, depth=depth + 1)
    return _summary(value)


def json_dumps(value: Any) -> str:
    text = json.dumps(
        jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    original_bytes = len(text.encode("utf-8"))
    if original_bytes <= MAX_ATTRIBUTE_BYTES:
        return text
    low, high = 0, min(len(text), MAX_ATTRIBUTE_BYTES)
    result = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {
                "original_bytes": original_bytes,
                "preview": text[:midpoint],
                "truncated": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(candidate.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES:
            result = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return result


def exception_message(exc: BaseException) -> str:
    try:
        arguments = exc.args
    except Exception:  # noqa: BLE001 - hostile exception args are ignored
        arguments = ()
    for argument in arguments:
        if isinstance(argument, str):
            return safe_text(argument)
        if argument is None or isinstance(argument, (bool, int)):
            return safe_text(str(argument))
        if isinstance(argument, float) and math.isfinite(argument):
            return safe_text(str(argument))
    # Legacy generated Pinecone exceptions leave args empty. Read only the
    # bounded provider message, never __str__, which includes response headers.
    if type(exc).__module__.startswith("pinecone."):
        try:
            body = getattr(exc, "body", None)
            if isinstance(body, bytes):
                body = body[:MAX_ATTRIBUTE_BYTES].decode("utf-8", errors="replace")
            if isinstance(body, str) and len(body) <= MAX_ATTRIBUTE_BYTES:
                payload = json.loads(body)
                if isinstance(payload, Mapping) and isinstance(
                    payload.get("message"), str
                ):
                    return safe_text(payload["message"])
            reason = getattr(exc, "reason", None)
            if isinstance(reason, str):
                return safe_text(reason)
        except Exception:  # noqa: BLE001, S110 - malformed provider errors are non-fatal
            pass
    return safe_type_name(exc)


def exception_status_code(exc: BaseException) -> int:
    for owner, attribute in ((exc, "status_code"), (exc, "status")):
        try:
            value = getattr(owner, attribute, None)
        except Exception:  # noqa: BLE001 - hostile status properties are ignored
            value = None
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    try:
        response = getattr(exc, "response", None)
    except Exception:  # noqa: BLE001 - hostile response properties are ignored
        response = None
    if response is not None:
        try:
            value = getattr(response, "status_code", None)
        except Exception:  # noqa: BLE001 - hostile status properties are ignored
            value = None
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    return 500
