"""Serialize known SDK values without arbitrary object hooks or payload truncation."""

from __future__ import annotations

import json
import math
import re
from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from pydantic import BaseModel

_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|cookie|password|secret|access[_-]?token|refresh[_-]?token)$"
)
_QUOTED = re.compile(
    r"""(?i)(["']?(?:api[_-]?key|authorization|cookie|password|secret|access[_-]?token|refresh[_-]?token)["']?\s*[:=]\s*)(["'])((?:\\.|(?!\2).)*)\2""",
    re.DOTALL,
)
_ASSIGN = re.compile(
    r"""(?i)(["']?(?:api[_-]?key|authorization|cookie|password|secret|access[_-]?token|refresh[_-]?token)["']?\s*[:=]\s*)(?!["'\[{])[^\s,;}&"'\\]+"""
)
_URL = re.compile(r'https?://[^\s"<>\\]+')


def text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = _QUOTED.sub(
        lambda m: m.group(1) + m.group(2) + "[REDACTED]" + m.group(2), value
    )
    value = re.sub(r"""(?i)\b(bearer|basic)\s+[^\s,;"'\\]+""", r"\1 [REDACTED]", value)

    def url(m):
        try:
            p = urlsplit(m.group())
            host = p.hostname or ""
            host = f"[{host}]" if ":" in host else host
            if p.port:
                host += f":{p.port}"

            def scrub(parameters):
                parts = []
                for raw in parameters.split("&"):
                    name, separator, _value = raw.partition("=")
                    if separator and _SECRET.search(unquote_plus(name)):
                        parts.append(name + "=[REDACTED]")
                    else:
                        parts.append(raw)
                return "&".join(parts)

            return urlunsplit(
                (p.scheme, host, p.path, scrub(p.query), scrub(p.fragment))
            )
        except (ValueError, TypeError):
            return "[REDACTED URL]"

    # Parse credential query values before the plain assignment matcher so a
    # URL's fragment cannot be swallowed as part of its credential value.
    return _ASSIGN.sub(r"\1[REDACTED]", _URL.sub(url, value))


def to_mapping(value: Any):
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return object.__getattribute__(value, "__dict__")
    return None


def get_value(value: Any, key: str, default=None):
    data = to_mapping(value)
    if data is not None:
        return data.get(key, default)
    try:
        return getattr(value, key, default)
    except Exception:  # noqa: BLE001 - telemetry observes without replacing native behavior.
        return default


def data(value: Any, *, schema=False, properties=False, credential=False, seen=None):
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, str):
        return text(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    seen = set() if seen is None else seen
    if id(value) in seen:
        return "<cycle>"
    mapping = to_mapping(value)
    if mapping is not None:
        seen.add(id(value))
        result = {}
        for key, item in mapping.items():
            if type(key) not in (str, int):
                continue
            name = str(key)
            sensitive = bool(_SECRET.search(name))
            if (sensitive and not properties) or (
                credential
                and name in {"default", "example", "examples", "enum", "const"}
            ):
                result[name] = (
                    ["[REDACTED]" for _ in item]
                    if isinstance(item, list)
                    else "[REDACTED]"
                )
            else:
                result[name] = data(
                    item,
                    schema=schema,
                    properties=schema
                    and name
                    in {"properties", "patternProperties", "$defs", "definitions"},
                    credential=credential or (properties and sensitive),
                    seen=seen,
                )
        seen.remove(id(value))
        return result
    if isinstance(value, (list, tuple)):
        seen.add(id(value))
        result = [
            data(i, schema=schema, credential=credential, seen=seen) for i in value
        ]
        seen.remove(id(value))
        return result
    return {"type": type(value).__name__}


def safe_json(value: Any, *, schema=False) -> str:
    return json.dumps(
        data(value, schema=schema),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def parse_jsonish(value: Any):
    if isinstance(value, str) and value.lstrip().startswith(("{", "[", '"')):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return value


def json_string_or_value(value: Any):
    return text(value) if isinstance(value, str) else safe_json(value)
