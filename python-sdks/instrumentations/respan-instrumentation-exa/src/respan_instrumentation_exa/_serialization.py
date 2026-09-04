"""Safe serialization for Exa request and response objects."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

_MAX_DEPTH = 16
_SENSITIVE_MARKERS = frozenset(
    {
        "apikey",
        "apitoken",
        "authorization",
        "authtoken",
        "bearertoken",
        "cookie",
        "credential",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "sessiontoken",
        "accesstoken",
        "xapikey",
    }
)


def is_sensitive_key(key: object) -> bool:
    compact = "".join(char for char in str(key).lower() if char.isalnum())
    return compact == "token" or any(marker in compact for marker in _SENSITIVE_MARKERS)


def type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def to_jsonable(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    """Convert Exa/Pydantic values to JSON data without leaking credentials."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth > _MAX_DEPTH:
        return {"type": type_name(value), "truncated": True}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_jsonable(value.value, depth=depth + 1, seen=seen)

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return {"type": type_name(value), "recursive": True}

    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            return {
                str(key): (
                    "<redacted>"
                    if is_sensitive_key(key)
                    else to_jsonable(item, depth=depth + 1, seen=seen)
                )
                for key, item in value.items()
                if not callable(item)
            }
        finally:
            seen.discard(value_id)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen.add(value_id)
        try:
            return [to_jsonable(item, depth=depth + 1, seen=seen) for item in value]
        finally:
            seen.discard(value_id)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_jsonable(
                model_dump(mode="json", exclude_none=True),
                depth=depth + 1,
                seen=seen,
            )
        except TypeError:
            return to_jsonable(model_dump(), depth=depth + 1, seen=seen)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value), depth=depth + 1, seen=seen)

    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            return to_jsonable(as_dict(), depth=depth + 1, seen=seen)
        except Exception:  # noqa: BLE001, S110 - third-party serializers are best effort.
            pass

    if hasattr(value, "__dict__"):
        public = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
        if public:
            return to_jsonable(public, depth=depth + 1, seen=seen)

    return {"type": type_name(value), "value": str(value)}


def json_dumps(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def value_at(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
