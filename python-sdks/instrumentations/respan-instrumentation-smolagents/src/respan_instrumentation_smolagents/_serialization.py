"""Safe JSON capture without truncating tool identifiers or known payloads."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_DEPTH = 64
_SENSITIVE_KEY = re.compile(
    r"(?:^|[._-])(api[_-]?key|authorization|cookie|password|secret|token)(?:$|[._-])"
    r"|(?:api[_-]?key|authorization|cookie|password|secret|token)$",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)((?:['\"]?)(?:[a-z0-9_-]*[_-])?"
    r"(?:api[_-]?key|authorization|cookie|password|secret|token)"
    r"(?:['\"]?)\s*[:=]\s*)(?:['\"]?)[^,;)\s}]+(?:['\"]?)"
)


_QUOTED_SECRET = re.compile(
    r"(?i)([\"'](?:[a-z0-9_-]*[_-])?(?:api[_-]?key|authorization|cookie|password|secret|token)[\"']\s*[:=]\s*)([\"'])(.*?)\2",
    re.DOTALL,
)


def redact_text(value: str) -> str:
    normalized = "".join(ch if ch >= " " or ch in "\n\t" else " " for ch in value)
    normalized = _QUOTED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}",
        normalized,
    )
    normalized = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[REDACTED]@", normalized)
    normalized = re.sub(
        r"(?i)\b(bearer|basic)\s+[^\s,;\"']+", r"\1 [REDACTED]", normalized
    )
    normalized = _ASSIGNMENT_SECRET.sub(r"\1[REDACTED]", normalized)
    return normalized


def jsonable(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes", "length": len(value)}
    if depth >= _MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            iterator = iter(value.items())
            for key, item in iterator:
                key_text = key if isinstance(key, str) else f"<{type(key).__name__}>"
                result[key_text[:128]] = (
                    "[REDACTED]"
                    if _SENSITIVE_KEY.search(key_text)
                    else jsonable(item, depth=depth + 1)
                )
        except Exception:  # noqa: BLE001 - telemetry must fail open
            return {"type": type(value).__name__, "serialization_error": True}
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        result = []
        try:
            iterator = iter(value)
            for item in iterator:
                result.append(jsonable(item, depth=depth + 1))
        except Exception:  # noqa: BLE001 - telemetry must fail open
            return {"type": type(value).__name__, "serialization_error": True}
        return result
    return {"type": type(value).__name__}


def json_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        encoded = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001 - telemetry must fail open
        encoded = json.dumps(
            {"type": type(value).__name__, "serialization_error": True},
            sort_keys=True,
        )
    return encoded
