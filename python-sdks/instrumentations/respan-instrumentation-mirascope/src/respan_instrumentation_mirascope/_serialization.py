"""Small, defensive serializers for Mirascope messages and response values."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any


def json_value(value: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if _depth >= 6:
        return repr(value)[:8_000]
    seen = _seen if _seen is not None else set()
    if id(value) in seen:
        return "<cycle>"
    seen.add(id(value))
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        elif callable(getattr(value, "model_dump", None)):
            value = value.model_dump()
        elif callable(getattr(value, "to_dict", None)):
            value = value.to_dict()
        elif isinstance(value, Mapping):
            value = dict(value)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            value = list(value)
        elif hasattr(value, "__dict__"):
            value = {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        elif callable(value):
            return {
                "name": getattr(value, "__name__", value.__class__.__name__),
                "description": getattr(value, "__doc__", None),
            }
        else:
            return repr(value)[:8_000]

        if isinstance(value, Mapping):
            return {
                str(key): json_value(item, _depth=_depth + 1, _seen=seen)
                for key, item in list(value.items())[:50]
            }
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [
                json_value(item, _depth=_depth + 1, _seen=seen)
                for item in list(value)[:50]
            ]
        return value
    except Exception:
        return repr(value)[:8_000]
    finally:
        seen.discard(id(value))


def json_string(value: Any) -> str:
    return json.dumps(json_value(value), default=str, sort_keys=True)
