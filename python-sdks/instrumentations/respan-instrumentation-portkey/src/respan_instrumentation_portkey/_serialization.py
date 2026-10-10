"""Serialize known SDK values without consuming iterators or arbitrary hooks."""

from __future__ import annotations

import json
import math
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel

MAX_ITEMS = 50
MAX_BYTES = 16_000
MAX_DEPTH = 8
_SECRET = re.compile(
    r"(?i)(api[_-]?key|virtual[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)"
)
_INLINE = re.compile(
    r"""(?i)(["']?(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)["']?\s*[:=]\s*["']?)([^\s,;&"'}]+)"""
)
_AUTH = re.compile(r"(?i)\b(bearer|basic)\s+[^\s,;&]+")
_QUOTED_DOUBLE = re.compile(
    r'''(?i)(["']?(?:api[_-]?key|virtual[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)["']?\s*[:=]\s*")((?:\\.|[^"\\])*)"'''
)
_QUOTED_SINGLE = re.compile(
    r"""(?i)(["']?(?:api[_-]?key|virtual[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)["']?\s*[:=]\s*')((?:\\.|[^'\\])*)' """.rstrip()
)
_URL = re.compile(r'https?://[^\s"<>]+')


def sanitize_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if not parsed.hostname:
            return "<redacted-endpoint>"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (ValueError, TypeError):
        return "<redacted-endpoint>"


def safe_text(value: str, *, complete: bool = False) -> str:
    value = _QUOTED_DOUBLE.sub(lambda m: m.group(1) + '<redacted>"', value)
    value = _QUOTED_SINGLE.sub(lambda m: m.group(1) + "<redacted>'", value)
    value = _AUTH.sub(lambda m: m.group(1) + " <redacted>", value)
    value = _INLINE.sub(lambda m: m.group(1) + "<redacted>", value)
    value = _URL.sub(lambda m: sanitize_endpoint(m.group()), value)
    if complete or len(value.encode()) <= MAX_BYTES:
        return value
    return value.encode()[:MAX_BYTES].decode(errors="ignore") + "...[truncated]"


def safe_type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"[:256]


def fields(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        raw = object.__getattribute__(value, "__dict__")
        extra = object.__getattribute__(value, "__pydantic_extra__")
        return {**raw, **(extra or {})}
    return None


def jsonable(
    value: Any,
    *,
    complete: bool = False,
    depth: int = 0,
    schema=False,
    properties=False,
    credential=False,
    tool_definitions=False,
) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return safe_text(value, complete=complete)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth > MAX_DEPTH and not complete:
        return {"type": safe_type_name(value), "truncated": "max_depth"}
    if depth > 64:
        return {"type": safe_type_name(value), "truncated": "max_depth"}
    data = fields(value)
    if data is not None:
        pairs = list(data.items())
        result = {}
        for key, item in pairs if complete else pairs[:MAX_ITEMS]:
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                continue
            name = str(key)
            if type(item).__name__ in {"Omit", "NotGiven"}:
                continue
            sensitive = bool(_SECRET.search(name) or name.lower() == "key")
            if (sensitive and not properties) or (
                credential
                and name in {"default", "example", "examples", "const", "enum"}
            ):
                result[name] = "<redacted>"
            else:
                nested_schema = (
                    schema
                    or (tool_definitions and name in {"parameters", "input_schema"})
                    or name in {"json_schema"}
                )
                result[name] = jsonable(
                    item,
                    complete=complete,
                    depth=depth + 1,
                    schema=nested_schema,
                    properties=nested_schema
                    and name
                    in {"properties", "patternProperties", "$defs", "definitions"},
                    credential=credential or (properties and sensitive),
                    tool_definitions=tool_definitions or name in {"tools", "functions"},
                )
        if not complete and len(pairs) > MAX_ITEMS:
            result["__truncated__"] = True
        return result
    if isinstance(value, (list, tuple)):
        full = complete or bool(
            value
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
            )
        )
        result = [
            jsonable(
                v,
                complete=full,
                depth=depth + 1,
                schema=schema,
                credential=credential,
                tool_definitions=tool_definitions,
            )
            for v in (value if full else value[:MAX_ITEMS])
        ]
        return (
            result
            if full or len(value) <= MAX_ITEMS
            else {"items": result, "truncated": True}
        )
    return {"type": safe_type_name(value)}


def json_dumps(value: Any, *, complete: bool = False, tool_definitions=False) -> str:
    text = json.dumps(
        jsonable(value, complete=complete, tool_definitions=tool_definitions),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    if complete or len(text.encode()) <= MAX_BYTES:
        return text
    return json.dumps(
        {
            "truncated": True,
            "original_bytes": len(text.encode()),
            "preview": text[:3000],
        },
        ensure_ascii=False,
    )


def parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def exception_message(exc: BaseException) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str):
            return safe_text(message)
    for value in exc.args:
        if isinstance(value, str):
            return safe_text(value)
    return None


def exception_status(exc: BaseException) -> int | None:
    from portkey_ai._vendor.openai import APIStatusError
    from portkey_ai.api_resources.exceptions import (
        APIStatusError as PortkeyAPIStatusError,
    )

    if isinstance(exc, (APIStatusError, PortkeyAPIStatusError)):
        value = exc.status_code
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 100 <= value <= 599
        ):
            return value
    return None
