"""Bounded content serialization for provider and user payloads."""

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from itertools import islice
from typing import Any

MAX_ITEMS = 64
_SECRET = re.compile(
    r"authorization|api[_-]?key|access[_-]?token|password|secret", re.IGNORECASE
)
_INLINE = re.compile(
    r"(?i)\b(bearer\s+|(?:api[_-]?key|password|secret)\s*[:=]\s*)[^\s,;]+"
)


def safe_text(value: str) -> str:
    return _INLINE.sub(lambda m: m.group(1) + "[REDACTED]", value[:8192])


def safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return safe_value(value.value, depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if _SECRET.search(str(key))
            else safe_value(item, depth + 1)
            for key, item in islice(value.items(), MAX_ITEMS)
            if isinstance(key, (str, int, bool))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [safe_value(item, depth + 1) for item in islice(value, MAX_ITEMS)]
    if is_dataclass(value) and not isinstance(value, type):
        return safe_value(
            {f.name: getattr(value, f.name) for f in fields(value)[:MAX_ITEMS]},
            depth + 1,
        )
    if type(value).__module__.startswith("agno."):
        method = getattr(value, "to_dict", None)
        if callable(method):
            return safe_value(method(), depth + 1)
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return safe_value(value.model_dump(), depth + 1)
    return {"type": type(value).__name__}


def json_string(value: Any) -> str:
    try:
        result = json.dumps(
            safe_value(value), ensure_ascii=False, separators=(",", ":")
        )
    except Exception:  # noqa: BLE001 - telemetry must not break user serialization
        return json.dumps({"type": type(value).__name__})
    if len(result.encode("utf-8")) > 16384:
        return json.dumps(
            {"preview": result[:2048], "truncated": True}, ensure_ascii=False
        )
    return result


def exception_text(exception: BaseException) -> str:
    for arg in exception.args:
        if isinstance(arg, str):
            return safe_text(arg)
    return type(exception).__name__
