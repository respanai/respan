"""Serialization helpers for LiveKit instrumentation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from respan_sdk.utils.serialization import serialize_value


def safe_json(value: Any) -> str:
    """Serialize arbitrary LiveKit values into an OTEL-safe JSON string."""
    try:
        return json.dumps(
            serialize_value(value=value),
            default=str,
            separators=(",", ":"),
        )
    except Exception:
        return json.dumps(str(value), separators=(",", ":"))


def to_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method()
            except Exception:
                continue
            if isinstance(converted, Mapping):
                return converted

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return value_dict
    return None


def get_value(value: Any, key: str, default: Any = None) -> Any:
    mapping = to_mapping(value)
    if mapping is not None:
        return mapping.get(key, default)
    return getattr(value, key, default)


def json_string_or_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return safe_json(value)


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value
