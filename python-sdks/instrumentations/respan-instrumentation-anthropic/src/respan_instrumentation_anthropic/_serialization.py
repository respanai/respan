"""Serialize builtins and released Anthropic models without invoking user hooks."""

from __future__ import annotations

import json
import math
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token|session[_-]?token|token|credential|cookie)$"
)
_AUTH = re.compile(r"(?i)\b(Bearer|Basic)\s+(?!\[REDACTED\])[A-Za-z0-9._~+/=-]+")
_ASSIGN = re.compile(
    r"""(?ix)(["']?(?:api[_-]?key|authorization|password|secret|session[_-]?token|access[_-]?token|token|credential|cookie)["']?)(\s*[:=]\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}]+)"""
)
_URL = re.compile(r"https?://[^\s\"'<>]+")


def redact_text(value: str) -> str:
    if value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            pass
        else:
            return json.dumps(
                json_value(parsed),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )

    def url(match: re.Match[str]) -> str:
        try:
            parsed = urlsplit(match.group())
            host = parsed.netloc.rsplit("@", 1)[-1]
            query = urlencode(
                [
                    (key, REDACTED if _SECRET.search(key) else item)
                    for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                ]
            )
            return urlunsplit(
                (parsed.scheme, host, parsed.path, query, parsed.fragment)
            )
        except ValueError:
            return REDACTED

    value = _URL.sub(url, value)
    value = _AUTH.sub(lambda m: f"{m.group(1)} {REDACTED}", value)

    def assignment(match: re.Match[str]) -> str:
        raw = match.group(3)
        replacement = (
            json.dumps(REDACTED)
            if raw.startswith('"')
            else ("'" + REDACTED + "'" if raw.startswith("'") else REDACTED)
        )
        return match.group(1) + match.group(2) + replacement

    return _ASSIGN.sub(assignment, value)


def native_dict(value: Any) -> dict[str, Any] | None:
    if type(value) is dict:
        return value
    if type(value).__module__.startswith("anthropic."):
        from anthropic._models import BaseModel

        if isinstance(value, BaseModel):
            data = object.__getattribute__(value, "__dict__")
            try:
                extras = object.__getattribute__(value, "__pydantic_extra__")
            except AttributeError:
                extras = None
            return {**data, **extras} if type(extras) is dict else data
    return None


def json_value(
    value: Any,
    *,
    seen: set[int] | None = None,
    schema_names: bool = False,
    credential_schema: bool = False,
) -> Any:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is str:
        return redact_text(value)
    active = seen if seen is not None else set()
    if id(value) in active:
        return None
    active.add(id(value))
    try:
        data = native_dict(value)
        if data is not None:
            result = {}
            for key, item in data.items():
                if type(key) is not str:
                    continue
                sensitive = bool(_SECRET.search(key))
                if (sensitive and not schema_names) or (
                    credential_schema
                    and key in ("default", "const", "enum", "examples")
                ):
                    result[key] = REDACTED
                else:
                    result[key] = json_value(
                        item,
                        seen=active,
                        schema_names=key == "properties",
                        credential_schema=schema_names and sensitive,
                    )
            return result
        if type(value) in (list, tuple):
            return [
                json_value(item, seen=active, credential_schema=credential_schema)
                for item in value
            ]
        return None
    finally:
        active.discard(id(value))


def json_string(value: Any) -> str:
    return json.dumps(
        json_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
