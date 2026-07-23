"""Bounded JSON serialization for Ragas values."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_DEPTH = 5
_MAX_ITEMS = 25
_MAX_STRING = 8_000


def _value(value: Any, *, depth: int, seen: set[int]) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_STRING]
    if depth >= _MAX_DEPTH:
        return repr(value)[:_MAX_STRING]

    value_id = id(value)
    if value_id in seen:
        return "<cycle>"
    seen.add(value_id)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        elif callable(getattr(value, "model_dump", None)):
            value = value.model_dump()
        elif callable(getattr(value, "to_dict", None)):
            value = value.to_dict()
        elif callable(getattr(value, "dict", None)):
            value = value.dict()

        if isinstance(value, Mapping):
            return {
                str(key): _value(item, depth=depth + 1, seen=seen)
                for key, item in list(value.items())[:_MAX_ITEMS]
            }
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [
                _value(item, depth=depth + 1, seen=seen)
                for item in list(value)[:_MAX_ITEMS]
            ]
        if hasattr(value, "__dict__"):
            public = {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
            if public:
                return _value(public, depth=depth + 1, seen=seen)
        return repr(value)[:_MAX_STRING]
    except Exception:
        return repr(value)[:_MAX_STRING]
    finally:
        seen.discard(value_id)


def json_string(value: Any) -> str:
    """Return a deterministic, bounded JSON representation."""
    return json.dumps(_value(value, depth=0, seen=set()), default=str, sort_keys=True)
