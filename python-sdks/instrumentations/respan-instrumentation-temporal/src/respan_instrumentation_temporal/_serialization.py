"""Stable, bounded, privacy-safe Temporal attribute serialization."""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import re
from datetime import date, datetime, time, timedelta
from enum import Enum
from types import MemberDescriptorType
from typing import Any
from uuid import UUID

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from pydantic import BaseModel

_SENSITIVE = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|headers?|password|rpc[_-]?metadata|secret|token)([._-]|$)",
    re.IGNORECASE,
)
_TEXT_SECRET = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|cookie|password|(?:client[_-]?)?secret|token)[\"']?)"
    r"(\s*[:=]\s*)(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\[REDACTED\]|[^\s,;}\]]+)"
)
_AUTHORIZATION = re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]+")
_SCHEMA_KEYS = frozenset(
    {"type", "$ref", "properties", "items", "anyOf", "allOf", "oneOf"}
)


def _assignment(match: Any) -> str:
    value = match.group(3)
    quote = value[0] if value[:1] in {'"', "'"} else ""
    return f"{match.group(1)}{match.group(2)}{quote}[REDACTED]{quote}"


def _redact_text(value: str) -> str:
    if value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            pass
        else:
            sanitized = to_jsonable(decoded)
            if sanitized != decoded:
                return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
            return value
    return _TEXT_SECRET.sub(_assignment, _AUTHORIZATION.sub("[REDACTED]", value))


def _type_name(value: Any) -> str:
    return type.__getattribute__(type(value), "__name__")[:120]


def _bounded_text(value: str, *, max_bytes: int = 4_000) -> str:
    value = _redact_text(value)
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    low, high, best = 0, len(value), ""
    suffix = "…[truncated]"
    while low <= high:
        middle = (low + high) // 2
        candidate = f"{value[:middle]}{suffix}"
        if len(candidate.encode("utf-8")) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or "[truncated]"


def safe_baggage_value(key: str, value: Any) -> str:
    if _SENSITIVE.search(key):
        return "[REDACTED]"
    if type(value) is str:
        return _bounded_text(value, max_bytes=2_000)
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int or type(value) is float and math.isfinite(value):
        return str(value)
    return json_dumps(value, max_bytes=2_000)


def safe_error_message(error: BaseException, *, capture_content: bool) -> str:
    if not capture_content:
        return _type_name(error)
    pieces: list[str] = []
    try:
        for item in error.args:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, (int, float, bool)):
                pieces.append(json.dumps(item))
    except BaseException:  # noqa: BLE001 - hostile exception args must fail closed
        pieces = []
    return _bounded_text(" ".join(pieces) or _type_name(error))


def _inherits(value: Any, cls: type) -> bool:
    return cls in type.__getattribute__(type(value), "__mro__")


def to_jsonable(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int | None = None,
    max_items: int | None = None,
    schema_properties: bool = False,
    schema_secret: bool = False,
) -> Any:
    if _inherits(value, Enum):
        return to_jsonable(
            object.__getattribute__(value, "_value_"),
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
        )
    if _inherits(value, datetime):
        return datetime.isoformat(value)
    if _inherits(value, date):
        return date.isoformat(value)
    if _inherits(value, time):
        return time.isoformat(value)
    if _inherits(value, timedelta):
        return timedelta.total_seconds(value)
    if _inherits(value, UUID):
        return UUID.__str__(value)
    if _inherits(value, Message):
        return to_jsonable(
            MessageToDict(value, preserving_proto_field_name=True),
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
        )
    if type(value) in {set, frozenset}:
        return [
            to_jsonable(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for item in value
        ]
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else str(value)
    if type(value) is str:
        return _redact_text(value)
    if max_depth is not None and depth >= max_depth:
        return {"type": _type_name(value), "truncated": True}
    if type(value) in {bytes, bytearray, memoryview}:
        return {
            "type": "bytes",
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if type(value) is dict:
        result: dict[str, Any] = {}
        try:
            for index, (key, item) in enumerate(value.items()):
                if max_items is not None and index == max_items:
                    result["_respan_truncated_items"] = True
                    break
                safe_key = key if isinstance(key, str) else f"<{_type_name(key)}>"
                safe_key = safe_key[:256]
                secret = bool(_SENSITIVE.search(safe_key))
                definition = (
                    schema_properties
                    and type(item) is dict
                    and any(key in _SCHEMA_KEYS for key in item if type(key) is str)
                )
                if (
                    secret
                    and not definition
                    or schema_secret
                    and safe_key in {"default", "examples", "example", "const", "enum"}
                ):
                    result[safe_key] = "[REDACTED]"
                else:
                    result[safe_key] = to_jsonable(
                        item,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                        schema_properties=safe_key == "properties",
                        schema_secret=secret if definition else schema_secret,
                    )

        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _type_name(value), "unavailable": True}
        return result
    if type(value) in {list, tuple}:
        result: list[Any] = []
        try:
            for index, item in enumerate(value):
                if max_items is not None and index == max_items:
                    result.append({"_respan_truncated_items": True})
                    break
                result.append(
                    to_jsonable(
                        item,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                    )
                )
        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _type_name(value), "unavailable": True}
        return result
    if type(type(value)) is type and "__dataclass_fields__" in type.__getattribute__(
        type(value), "__dict__"
    ):
        values: dict[str, Any] = {}
        state = (
            object.__getattribute__(value, "__dict__")
            if hasattr(type(value), "__dict__")
            and not hasattr(type(value), "__slots__")
            else {}
        )
        for field in dataclasses.fields(type(value)):
            if field.name in state:
                values[field.name] = state[field.name]
            else:
                descriptor = type.__getattribute__(type(value), "__dict__").get(
                    field.name
                )
                if isinstance(descriptor, MemberDescriptorType):
                    values[field.name] = descriptor.__get__(value, type(value))
                else:
                    values[field.name] = {"unavailable": True}
        return to_jsonable(
            values, depth=depth + 1, max_depth=max_depth, max_items=max_items
        )
    if _inherits(value, BaseModel):
        # Read the native field storage without invoking user model_dump or
        # custom field serializers a second time for observability.
        state = dict(object.__getattribute__(value, "__dict__"))
        extra = object.__getattribute__(value, "__pydantic_extra__")
        if type(extra) is dict:
            state.update(extra)
        return to_jsonable(
            state, depth=depth + 1, max_depth=max_depth, max_items=max_items
        )
    return {"type": _type_name(value)}


def json_dumps(value: Any, *, max_bytes: int | None = None) -> str:
    serialized = json.dumps(
        to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if max_bytes is None or len(serialized.encode("utf-8")) <= max_bytes:
        return serialized
    low, high, best = 0, min(len(serialized), max_bytes), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {"preview": serialized[:middle], "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(candidate.encode("utf-8")) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or '{"truncated":true}'
