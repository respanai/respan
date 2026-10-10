"""Serialize complete builtins and known native storage without user conversion hooks."""

from __future__ import annotations

import json
import math
import re
import types
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token|session[_-]?token|token|credentials?|private[_-]?key|cookie)$"
)
_AUTH = re.compile(
    r"""(?i)\b(Bearer|Basic)\s+(?!\[REDACTED\])(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z0-9._~+/=-]+)"""
)
_ASSIGN = re.compile(
    r"""(?ix)(["']?(?:api[_-]?key|authorization|password|secret|session[_-]?token|access[_-]?token|refresh[_-]?token|token|credentials?|private[_-]?key|cookie)["']?)(\s*[:=]\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}]+)"""
)
_URL = re.compile(r"https?://[^\s\"'<>]+")


def redact_text(value: str) -> str:
    if value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            pass
        else:
            converted = json_value(parsed)
            if converted == parsed:
                return value
            return json.dumps(
                converted,
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
    value = re.sub(
        r"""(?i)\b(Bearer|Basic)\s+(\\+)(["'])(?:[\s\S]*?)(?<!\\)\2\3""",
        lambda m: f"{m.group(1)} {REDACTED}",
        value,
    )
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


def native_storage(value: Any, base: type) -> dict | None:
    """Read a known native base's C storage descriptor, bypassing subclass getters."""
    if not any(
        item is base
        for item in type.__dict__["__mro__"].__get__(type(value), type(type(value)))
    ):
        return None
    for ancestor in type.__dict__["__mro__"].__get__(base, type(base)):
        descriptor = (
            type.__dict__["__dict__"].__get__(ancestor, type(ancestor)).get("__dict__")
        )
        if any(
            type(descriptor) is t
            for t in (types.GetSetDescriptorType, types.MemberDescriptorType)
        ):
            storage = descriptor.__get__(value, base)
            return storage if type(storage) is dict else None
    return None


def native_dict(value):
    from enum import Enum

    from pydantic import BaseModel
    from requests import Response

    if type(value) is dict:
        return value
    for base in (BaseModel, Enum):
        data = native_storage(value, base)
        if data is not None:
            if base is Enum:
                return {"value": data.get("_value_")}
            return dict(data)
    # Non-JSON native HTTP responses (e.g. empty 204) keep structural facts.
    data = native_storage(value, Response)
    if data is not None:
        return {"status_code": data.get("status_code")}
    return None


def _schema(data):
    valid_types = {"object", "array", "string", "integer", "number", "boolean", "null"}
    if type(data.get("type")) is str and data["type"] in valid_types:
        return True
    if type(data.get("$schema")) is str:
        return True
    properties = data.get("properties")
    return (
        type(properties) is dict
        and bool(properties)
        and all(
            type(item) is bool
            or (
                type(item) is dict
                and any(
                    key in item
                    for key in (
                        "type",
                        "$ref",
                        "enum",
                        "const",
                        "allOf",
                        "anyOf",
                        "oneOf",
                        "properties",
                        "default",
                        "example",
                        "examples",
                        "items",
                    )
                )
            )
            for item in properties.values()
        )
    )


def json_value(
    value: Any,
    *,
    seen: set[int] | None = None,
    schema_names: bool = False,
    schema: bool = False,
    credential_schema: bool = False,
) -> Any:
    if value is None or any(type(value) is t for t in (bool, int)):
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
        from uuid import UUID

        if type(value) is UUID:
            return UUID.__str__(value)
        from enum import Enum

        enum_data = native_storage(value, Enum)
        if enum_data is not None:
            return json_value(enum_data.get("_value_"), seen=active)
        data = native_dict(value)
        if data is not None:
            is_schema = schema or _schema(data)
            result = {}
            for key, item in data.items():
                if type(key) is not str:
                    continue
                sensitive = bool(_SECRET.search(key))
                if (
                    sensitive
                    and (
                        not schema_names
                        or not any(type(item) is t for t in (dict, bool))
                    )
                ) or (
                    credential_schema
                    and key in ("default", "const", "enum", "examples", "example")
                ):
                    result[key] = REDACTED
                else:
                    result[key] = json_value(
                        item,
                        seen=active,
                        schema_names=is_schema
                        and key in ("properties", "$defs", "definitions"),
                        schema=is_schema,
                        credential_schema=schema_names and sensitive,
                    )
            return result
        if any(type(value) is t for t in (list, tuple)):
            return [
                json_value(item, seen=active, credential_schema=credential_schema)
                for item in value
            ]
        return (
            "<"
            + type.__dict__["__name__"].__get__(type(value), type(type(value)))
            + ">"
        )
    finally:
        active.discard(id(value))


def json_string(value: Any) -> str:
    return json.dumps(
        json_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
