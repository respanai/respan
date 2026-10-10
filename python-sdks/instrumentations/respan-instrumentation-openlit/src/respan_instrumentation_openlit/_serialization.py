"""Bounded, redacting JSON helpers for OpenLIT request enrichment."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_ATTRIBUTE_BYTES = 16_000
MAX_COLLECTION_ITEMS = 50
MAX_DEPTH = 6
MAX_STRING_CHARS = 4_000
REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "apikey",
    "authorization",
    "authtoken",
    "clientsecret",
    "cookie",
    "credential",
    "credentials",
    "password",
    "passwd",
    "privatekey",
    "refreshtoken",
    "secret",
    "sessioncookie",
    "sessiontoken",
    "token",
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
_QUOTED_DOUBLE = re.compile(
    r'''(?i)(["']?[a-z0-9_-]*(?:api[_-]?key|authorization|password|secret|session[_-]?token|token)["']?\s*[:=]\s*")((?:\\.|[^"\\])*)"'''
)
_QUOTED_SINGLE = re.compile(
    r"""(?i)(["']?[a-z0-9_-]*(?:api[_-]?key|authorization|password|secret|session[_-]?token|token)["']?\s*[:=]\s*')((?:\\.|[^'\\])*)'"""
)

_CREDENTIAL_URI = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/@\s]+)@")
_MEMORY_ADDRESS = re.compile(r"(?i)\b0x[0-9a-f]{6,}\b")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?<![a-z0-9])"
    r"([a-z0-9_-]*(?:api[_-]?key|authorization|password|secret|session[_-]?token|token))"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def _redact_text(value: str, *, complete: bool = False) -> str:
    redacted = _QUOTED_DOUBLE.sub(lambda m: f'{m.group(1)}{REDACTED}"', value)
    redacted = _QUOTED_SINGLE.sub(lambda m: f"{m.group(1)}{REDACTED}'", redacted)
    redacted = _CREDENTIAL_URI.sub(
        lambda match: f"{match.group(1)}{REDACTED}@", redacted
    )
    redacted = _MEMORY_ADDRESS.sub("<memory-address>", redacted)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    redacted = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", redacted
    )
    if complete or len(redacted) <= MAX_STRING_CHARS:
        return redacted
    return f"{redacted[:MAX_STRING_CHARS]}...[truncated]"


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        (
            "apikey",
            "credential",
            "credentials",
            "password",
            "passwd",
            "privatekey",
            "secret",
            "secretkey",
            "token",
        )
    )


def safe_text(value: Any, *, default: str = "", limit: int | None = 512) -> str:
    """Return bounded scalar text without arbitrary ``str``/``repr`` calls."""

    if isinstance(value, str):
        return _redact_text(value, complete=limit is None)[:limit]
    if value is None:
        return default
    if isinstance(value, bool):
        return ("true" if value else "false")[:limit]
    if isinstance(value, int):
        return str(value)[:limit]
    if isinstance(value, float) and math.isfinite(value):
        return str(value)[:limit]
    return f"<{type(value).__name__}>"[:limit]


def safe_url(value: Any, *, limit: int = 2_048) -> str:
    """Return a bounded URL without credentials, query, or fragment."""

    if not isinstance(value, str):
        return ""
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.netloc or parsed.hostname is None:
            return safe_text(value.split("?", 1)[0].split("#", 1)[0], limit=limit)
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = f"{host}:{port}" if port is not None else host
        sanitized = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        sanitized = value.split("?", 1)[0].split("#", 1)[0]
    return safe_text(sanitized, limit=limit)


def _safe_key(key: Any) -> str:
    if isinstance(key, str):
        return _redact_text(key)
    return safe_text(key, default="null", limit=128)


def _public_dataclass_fields(value: Any) -> Mapping[str, Any] | None:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        return None
    result: dict[str, Any] = {}
    try:
        fields = dataclasses.fields(value)
        for field in fields[:MAX_COLLECTION_ITEMS]:
            result[field.name] = getattr(value, field.name)
    except Exception:  # noqa: BLE001 - user dataclasses can expose descriptors.
        return None
    return result


def requires_complete_payload(value: Any, _seen=None) -> bool:
    _seen = set() if _seen is None else _seen
    if id(value) in _seen:
        return False
    _seen.add(id(value))
    if isinstance(value, (list, tuple)):
        return any(requires_complete_payload(v, _seen) for v in value) or (
            len(value) > MAX_COLLECTION_ITEMS
            and all(
                type(v) in (int, float)
                and (not isinstance(v, float) or math.isfinite(v))
                for v in value
            )
        )
    if isinstance(value, dict):
        if len(value) > MAX_COLLECTION_ITEMS and all(
            ((type(k) is int and k >= 0) or (type(k) is str and k.isdecimal()))
            and type(v) in (int, float)
            and (not isinstance(v, float) or math.isfinite(v))
            for k, v in value.items()
        ):
            return True
        return any(requires_complete_payload(v, _seen) for v in value.values()) or any(
            k in value
            for k in (
                "tool_calls",
                "function",
                "parameters",
                "tool_call_id",
                "arguments",
                "dense",
                "sparse",
                "vector",
                "embedding",
            )
        )
    return False


def json_value(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _max_items: int = MAX_COLLECTION_ITEMS,
    complete: bool = False,
    schema: bool = False,
    _property_map: bool = False,
    _sensitive_property: bool = False,
) -> Any:
    """Convert to finite JSON data without custom string or dump hooks."""

    complete = complete or requires_complete_payload(value)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact_text(value, complete=complete)
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{type(value).__name__}:{len(value)} bytes>"
    if _depth >= MAX_DEPTH and not complete:
        return f"<{type(value).__name__}:max-depth>"

    seen = _seen if _seen is not None else set()
    object_id = id(value)
    if object_id in seen:
        return "<cycle>"
    seen.add(object_id)
    try:
        mapping: Mapping[Any, Any] | None = None
        sequence: Sequence[Any] | None = None
        if isinstance(value, Mapping):
            mapping = value
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            sequence = value
        else:
            mapping = _public_dataclass_fields(value)

        if mapping is not None:
            result: dict[str, Any] = {}
            try:
                iterator = mapping.items()
                for index, (key, item) in enumerate(iterator):
                    if not complete and index >= _max_items:
                        result["__truncated_items__"] = True
                        break
                    key_text = _safe_key(key)
                    sensitive_property = (
                        _is_sensitive_key(key) if _property_map else _sensitive_property
                    )
                    redacted_default = (
                        schema
                        and _sensitive_property
                        and key_text
                        in {"default", "example", "examples", "const", "enum"}
                    )
                    property_schema = (
                        schema and _property_map and isinstance(item, (dict, bool))
                    )
                    result[key_text] = (
                        (
                            [REDACTED for _ in item]
                            if isinstance(item, list)
                            else REDACTED
                        )
                        if redacted_default
                        or (_is_sensitive_key(key) and not property_schema)
                        else json_value(
                            item,
                            _depth=_depth + 1,
                            _seen=seen,
                            _max_items=_max_items,
                            complete=complete,
                            schema=schema
                            and key_text
                            not in {"default", "example", "examples", "const", "enum"},
                            _property_map=schema
                            and not _property_map
                            and key_text == "properties",
                            _sensitive_property=sensitive_property,
                        )
                    )
            except Exception:  # noqa: BLE001 - mappings can have hostile iterators.
                result["__serialization_error__"] = type(value).__name__
            return result

        if sequence is not None:
            try:
                items = (
                    list(sequence)
                    if complete
                    else list(islice(iter(sequence), _max_items + 1))
                )
            except Exception:  # noqa: BLE001 - sequences can have hostile iterators.
                return f"<{type(value).__name__}:unserializable>"
            result = [
                json_value(
                    item,
                    _depth=_depth + 1,
                    _seen=seen,
                    _max_items=_max_items,
                    complete=complete,
                    schema=schema,
                    _sensitive_property=_sensitive_property,
                )
                for item in (items if complete else items[:_max_items])
            ]
            if not complete and len(items) > _max_items:
                result.append("<truncated:more items>")
            return result

        return f"<{type(value).__name__}>"
    except Exception:  # noqa: BLE001 - tracing must not break provider calls.
        return f"<{type(value).__name__}:unserializable>"
    finally:
        seen.discard(object_id)


def json_string(
    value: Any,
    *,
    max_bytes: int = MAX_ATTRIBUTE_BYTES,
    max_collection_items: int = MAX_COLLECTION_ITEMS,
    complete: bool = False,
    schema: bool = False,
) -> str:
    """Return redacted valid JSON capped by an exact UTF-8 byte budget."""

    max_bytes = max(128, min(int(max_bytes), MAX_ATTRIBUTE_BYTES))
    max_collection_items = max(1, min(int(max_collection_items), 8_192))
    complete = complete or requires_complete_payload(value)
    normalized = json_value(
        value, _max_items=max_collection_items, complete=complete, schema=schema
    )
    encoded = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    encoded_bytes = len(encoded.encode("utf-8"))
    if complete or encoded_bytes <= max_bytes:
        return encoded

    preview = encoded[: max_bytes // 2]
    while True:
        bounded = json.dumps(
            {
                "original_bytes": encoded_bytes,
                "preview": preview,
                "truncated": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(bounded.encode("utf-8")) <= max_bytes:
            return bounded
        preview = preview[: max(0, len(preview) - 128)]


def input_messages(value: Any) -> list[dict[str, Any]]:
    """Normalize common OpenAI message inputs to finite GenAI messages."""

    normalized = json_value(value, complete=True)
    raw_messages = normalized if isinstance(normalized, list) else [normalized]
    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if isinstance(item, Mapping):
            role = safe_text(item.get("role"), default="user", limit=64)
            if isinstance(item.get("parts"), list):
                parts = item["parts"]
            else:
                content = item.get("content")
                parts = [{"type": "text", "content": content}]
            message: dict[str, Any] = {"role": role, "parts": parts}
            tool_calls = item.get("tool_calls")
            if tool_calls:
                message["tool_calls"] = tool_calls
            for key in ("tool_call_id", "id"):
                if item.get(key) is not None:
                    message[key] = item[key]
            messages.append(message)
        else:
            messages.append(
                {"role": "user", "parts": [{"type": "text", "content": item}]}
            )
    return messages


def tool_definitions(value: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI function tools to the OTel GenAI definition shape."""

    normalized = json_value(value, complete=True, schema=True)
    raw_tools = normalized if isinstance(normalized, list) else [normalized]
    definitions: list[dict[str, Any]] = []
    for item in raw_tools:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        source = function if isinstance(function, Mapping) else item
        name = source.get("name")
        if not name:
            continue
        definition: dict[str, Any] = {
            "type": safe_text(item.get("type"), default="function", limit=64),
            "name": safe_text(name, default="unknown", limit=None),
        }
        for key in ("description", "parameters"):
            if source.get(key) is not None:
                definition[key] = source[key]
        definitions.append(definition)
    return definitions
