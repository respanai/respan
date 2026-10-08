"""Bounded, privacy-safe serialization for Strands telemetry."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

MAX_JSON_BYTES = 16_000
MAX_ITEMS = 50
MAX_DEPTH = 8
_SENSITIVE_KEY = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|password|secret|token)([._-]|$)",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|cookie|password|secret|token)\s*[:=]\s*([^\s,;]+)"
)
_QUOTED_SECRET = re.compile(
    r"""(?i)(["'](?:api[_-]?key|authorization|cookie|password|secret|token)["']\s*:\s*)(["'])(.*?)\2"""
)


def _safe_type(value: Any) -> str:
    return type(value).__name__[:120]


def redact_text(value: str) -> str:
    value = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)\b(bearer)\s+[^\s,;]+", r"\1 [REDACTED]", value)
    value = re.sub(
        r"(?i)\b(basic)\s+(?:[A-Za-z0-9+/]{4})+(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?=$|[\s,;])",
        r"\1 [REDACTED]",
        value,
    )
    value = _QUOTED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}",
        value,
    )
    return _ASSIGNMENT_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)


def safe_text(value: Any, *, max_bytes: int = 4_000) -> str:
    if isinstance(value, str):
        text = redact_text(value)
    elif value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int) or isinstance(value, float) and math.isfinite(value):
        text = str(value)
    else:
        text = f"<{_safe_type(value)}>"
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    low, high, best = 0, len(text), ""
    suffix = "…[truncated]"
    while low <= high:
        middle = (low + high) // 2
        candidate = f"{text[:middle]}{suffix}"
        if len(candidate.encode("utf-8")) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or "[truncated]"


def _safe_key(value: Any) -> str:
    if isinstance(value, str):
        return redact_text(value)[:256]
    if isinstance(value, (int, bool)) or value is None:
        return json.dumps(value)
    return f"<{_safe_type(value)}>"


def requires_complete_payload(value: Any, *, depth: int = 0) -> bool:
    if depth >= MAX_DEPTH:
        return False
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return requires_complete_payload(json.loads(value), depth=depth + 1)
        except (ValueError, TypeError):
            return False
    if type(value) in (list, tuple):
        if value and all(type(item) in (int, float) for item in value):
            return True
        return any(requires_complete_payload(item, depth=depth + 1) for item in value)
    if type(value) is dict:
        if any(
            key in value
            for key in (
                "toolUse",
                "toolResult",
                "tool_calls",
                "function",
                "inputSchema",
                "embedding",
                "embeddings",
                "vector",
                "vectors",
            )
        ):
            return True
        return any(
            requires_complete_payload(item, depth=depth + 1) for item in value.values()
        )
    return False


def payload_text(value: Any, *, complete: bool = False) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return safe_text(value, max_bytes=MAX_JSON_BYTES)
    return json_dumps(value, complete=complete)


def to_jsonable(
    value: Any,
    *,
    depth: int = 0,
    complete: bool = False,
    _seen: frozenset[int] = frozenset(),
) -> Any:
    complete = complete or requires_complete_payload(value)
    if id(value) in _seen:
        return {"type": _safe_type(value), "circular": True}
    if isinstance(value, (Mapping, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        _seen = _seen | {id(value)}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return redact_text(value)
    if depth >= MAX_DEPTH and not complete:
        return {"type": _safe_type(value), "truncated": True}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        truncated = False
        try:
            iterator = iter(value.items())
            for key, item in iterator if complete else islice(iterator, MAX_ITEMS + 1):
                if not complete and len(result) >= MAX_ITEMS:
                    truncated = True
                    break
                safe_key = _safe_key(key)
                result[safe_key] = (
                    "[REDACTED]"
                    if _SENSITIVE_KEY.search(safe_key)
                    else to_jsonable(
                        item, depth=depth + 1, complete=complete, _seen=_seen
                    )
                )
        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _safe_type(value), "unavailable": True}
        if truncated:
            result["_respan_truncated_items"] = True
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[Any] = []
        truncated = False
        try:
            for item in iter(value) if complete else islice(iter(value), MAX_ITEMS + 1):
                if not complete and len(result) >= MAX_ITEMS:
                    truncated = True
                    break
                result.append(
                    to_jsonable(item, depth=depth + 1, complete=complete, _seen=_seen)
                )
        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _safe_type(value), "unavailable": True}
        if truncated:
            result.append({"_respan_truncated_items": True})
        return result
    try:
        model_dump = getattr(value, "model_dump", None)
    except BaseException:  # noqa: BLE001 - hostile objects must fail closed
        return {"type": _safe_type(value), "unavailable": True}
    if callable(model_dump):
        try:
            return to_jsonable(
                model_dump(mode="json"), depth=depth + 1, complete=complete, _seen=_seen
            )
        except BaseException:  # noqa: BLE001 - vendor hooks must not break tracing
            return {"type": _safe_type(value), "unavailable": True}
    return {"type": _safe_type(value)}


def json_dumps(
    value: Any, *, max_bytes: int = MAX_JSON_BYTES, complete: bool = False
) -> str:
    complete = complete or requires_complete_payload(value)
    safe = to_jsonable(value, complete=complete)
    serialized = json.dumps(
        safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    encoded = serialized.encode("utf-8")
    if complete or len(encoded) <= max_bytes:
        return serialized

    low, high = 0, min(len(serialized), max_bytes)
    best = ""
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
