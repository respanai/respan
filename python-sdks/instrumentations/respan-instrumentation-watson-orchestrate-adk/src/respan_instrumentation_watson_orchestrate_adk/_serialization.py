"""Bounded, privacy-safe serialization for Watson Orchestrate ADK telemetry."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MAX_JSON_BYTES = 16_000
MAX_TEXT_BYTES = 4_000
MAX_ITEMS = 50
MAX_DEPTH = 8

_SENSITIVE_KEY = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|password|secret|token)([._-]|$)",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)([a-z0-9_.-]*(?:api[_-]?key|authorization|cookie|password|secret|token))"
    r"\s*[:=]\s*([^\s,;]+)"
)
_QUOTED_SECRET = re.compile(
    r"""(?i)(["'][^"']*(?:api[_-]?key|authorization|cookie|password|secret|token)["']\s*:\s*)(["'])(.*?)\2"""
)
_BEARER_SECRET = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")


def safe_type_name(value: Any) -> str:
    return type(value).__name__[:120]


def redact_text(value: str) -> str:
    value = _BEARER_SECRET.sub(lambda match: f"{match.group(1)} [REDACTED]", value)
    value = _QUOTED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}",
        value,
    )
    value = _ASSIGNMENT_SECRET.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        value,
    )
    return value


def safe_text(value: Any, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, str):
        text = redact_text(value)
    elif value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int) or isinstance(value, float) and math.isfinite(value):
        text = str(value)
    else:
        text = f"<{safe_type_name(value)}>"
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


def safe_exception_message(exc: BaseException) -> str:
    try:
        raw_args = exc.args
    except BaseException:  # noqa: BLE001
        raw_args = ()
    parts: list[str] = []
    if isinstance(raw_args, tuple):
        for value in raw_args[:4]:
            if isinstance(value, (str, bool, int, float)) or value is None:
                rendered = safe_text(value, max_bytes=1_000)
                if rendered:
                    parts.append(rendered)
    detail = "; ".join(parts)
    name = safe_type_name(exc)
    return safe_text(f"{name}: {detail}" if detail else name)


def provider_status_code(exc: BaseException) -> int | None:
    candidates: list[Any] = []
    for name in ("status_code", "status"):
        try:
            candidates.append(getattr(exc, name, None))
        except BaseException:  # noqa: BLE001, S112
            continue
    try:
        response = getattr(exc, "response", None)
    except BaseException:  # noqa: BLE001
        response = None
    if response is not None:
        for name in ("status_code", "status"):
            try:
                candidates.append(getattr(response, name, None))
            except BaseException:  # noqa: BLE001, S112
                continue
    for value in candidates:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 400 <= value <= 599
        ):
            return value
    return None


def _safe_key(value: Any) -> str:
    if isinstance(value, str):
        return safe_text(value, max_bytes=256)
    if isinstance(value, (int, bool)) or value is None:
        return json.dumps(value)
    return f"<{safe_type_name(value)}>"


def to_jsonable(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else safe_text(value)
    if isinstance(value, str):
        if value.startswith(("https://", "http://")):
            value = sanitize_url(value) or "[invalid URL]"
        return redact_text(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            length = len(value)
        except BaseException:  # noqa: BLE001
            length = None
        return {"type": "bytes", "length": length}
    if depth >= MAX_DEPTH:
        return {"type": safe_type_name(value), "truncated": True}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        truncated = False
        try:
            for key, item in islice(iter(value.items()), MAX_ITEMS + 1):
                if len(result) >= MAX_ITEMS:
                    truncated = True
                    break
                rendered_key = _safe_key(key)
                result[rendered_key] = (
                    "[REDACTED]"
                    if _SENSITIVE_KEY.search(rendered_key)
                    else to_jsonable(item, depth=depth + 1)
                )
        except BaseException:  # noqa: BLE001
            return {"type": safe_type_name(value), "unavailable": True}
        if truncated:
            result["_respan_truncated_items"] = True
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[Any] = []
        truncated = False
        try:
            for item in islice(iter(value), MAX_ITEMS + 1):
                if len(items) >= MAX_ITEMS:
                    truncated = True
                    break
                items.append(to_jsonable(item, depth=depth + 1))
        except BaseException:  # noqa: BLE001
            return {"type": safe_type_name(value), "unavailable": True}
        if truncated:
            items.append({"_respan_truncated_items": True})
        return items
    for method_name in ("model_dump", "to_dict"):
        try:
            method = getattr(value, method_name, None)
        except BaseException:  # noqa: BLE001
            return {"type": safe_type_name(value), "unavailable": True}
        if callable(method):
            try:
                dumped = (
                    method(mode="json") if method_name == "model_dump" else method()
                )
            except TypeError:
                try:
                    dumped = method()
                except BaseException:  # noqa: BLE001, S112
                    continue
            except BaseException:  # noqa: BLE001, S112
                continue
            return to_jsonable(dumped, depth=depth + 1)
    return {"type": safe_type_name(value)}


def json_dumps(value: Any, *, max_bytes: int = MAX_JSON_BYTES) -> str:
    serialized = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(serialized.encode("utf-8")) <= max_bytes:
        return serialized
    low, high, best = 0, len(serialized), ""
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


def sanitize_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    text = safe_text(value)
    try:
        parsed = urlsplit(text if "://" in text else f"//{text}")
        host = parsed.hostname or ""
        if not host:
            return "[REDACTED-ENDPOINT]"
        netloc = host
        if ":" in host and not host.startswith("["):
            netloc = f"[{host}]"
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "[REDACTED-ENDPOINT]"
