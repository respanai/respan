"""Bounded and privacy-safe Superagent serialization helpers."""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

from respan_instrumentation_superagent._constants import INPUT_KEY, MODEL_KEY, REPO_KEY

MAX_JSON_BYTES = 16_000
MAX_ITEMS = 50
MAX_DEPTH = 8
_SENSITIVE_KEY = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|password|secret|token)([._-]|$)",
    re.IGNORECASE,
)
_TEXT_SECRET = re.compile(
    r"(?i)(api[_-]?key|authorization|cookie|password|secret|token)\s*[:=]\s*((?:(?:bearer|basic)\s+)?[^\s,;]+)"
)
_QUOTED_SECRET = re.compile(
    r"""(?i)(["'](?:api[_-]?key|authorization|cookie|password|secret|token)["']\s*:\s*)(["'])(.*?)\2"""
)


def _redact_text(value: str) -> str:
    value = re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^/\s@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)\b(bearer)\s+[^\s,;]+", r"\1 [REDACTED]", value)

    def basic(match):
        try:
            decoded = base64.b64decode(match.group(2), validate=True)
        except ValueError:
            return match.group(0)
        return match.group(1) + "[REDACTED]" if b":" in decoded else match.group(0)

    value = re.sub(r"(?i)\b(Basic\s+)([A-Za-z0-9+/]+={0,2})", basic, value)
    value = _QUOTED_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]{match.group(2)}",
        value,
    )
    return _TEXT_SECRET.sub(r"\1=[REDACTED]", value)


def _type_name(value: Any) -> str:
    return type(value).__name__[:120]


def safe_text(value: Any, *, max_bytes: int = 4_000) -> str:
    if isinstance(value, str):
        text = _redact_text(value)
    elif value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int) or isinstance(value, float) and math.isfinite(value):
        text = str(value)
    else:
        text = f"<{_type_name(value)}>"
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


def safe_error_message(error: BaseException) -> str:
    pieces: list[str] = []
    try:
        for item in islice(iter(error.args), 4):
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, (int, float, bool)):
                pieces.append(json.dumps(item))
    except BaseException:  # noqa: BLE001 - hostile exception args must fail closed
        pieces = []
    return safe_text(" ".join(pieces) or _type_name(error))


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _redact_text(value)
    if depth >= MAX_DEPTH:
        return {"type": _type_name(value), "truncated": True}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "type": "bytes",
            "length": len(value),
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            for index, (key, item) in enumerate(
                islice(iter(value.items()), MAX_ITEMS + 1)
            ):
                if index == MAX_ITEMS:
                    result["_respan_truncated_items"] = True
                    break
                safe_key = key if isinstance(key, str) else f"<{_type_name(key)}>"
                safe_key = safe_key[:256]
                result[safe_key] = (
                    "[REDACTED]"
                    if _SENSITIVE_KEY.search(safe_key)
                    else _jsonable(item, depth=depth + 1)
                )
        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _type_name(value), "unavailable": True}
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[Any] = []
        try:
            for index, item in enumerate(islice(iter(value), MAX_ITEMS + 1)):
                if index == MAX_ITEMS:
                    result.append({"_respan_truncated_items": True})
                    break
                result.append(_jsonable(item, depth=depth + 1))
        except BaseException:  # noqa: BLE001 - hostile containers must fail closed
            return {"type": _type_name(value), "unavailable": True}
        return result
    try:
        model_dump = getattr(value, "model_dump", None)
    except BaseException:  # noqa: BLE001 - hostile objects must fail closed
        return {"type": _type_name(value), "unavailable": True}
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"), depth=depth + 1)
        except BaseException:  # noqa: BLE001 - vendor hooks must not break tracing
            return {"type": _type_name(value), "unavailable": True}
    try:
        object_values = vars(value)
    except BaseException:  # noqa: BLE001 - hostile objects must fail closed
        object_values = None
    if isinstance(object_values, Mapping):
        return _jsonable(object_values, depth=depth + 1)
    return {"type": _type_name(value)}


def safe_json_dumps(value: Any) -> str:
    serialized = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(serialized.encode("utf-8")) <= MAX_JSON_BYTES:
        return serialized
    low, high, best = 0, min(len(serialized), MAX_JSON_BYTES), ""
    while low <= high:
        middle = (low + high) // 2
        candidate = json.dumps(
            {"preview": serialized[:middle], "truncated": True},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(candidate.encode("utf-8")) <= MAX_JSON_BYTES:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best or '{"truncated":true}'


def normalize_call_input(
    *, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {"method": method_name}
    primary = extract_primary_input(method_name=method_name, args=args, kwargs=kwargs)
    model = extract_model(args=args, kwargs=kwargs)
    if primary is not None:
        payload[INPUT_KEY if method_name != "scan" else REPO_KEY] = primary
    if model:
        payload[MODEL_KEY] = model
    for key in (
        "entities",
        "chunk_size",
        "system_prompt",
        "fallback_model",
        "rewrite",
        "branch",
    ):
        option = args[0] if args else kwargs.get(INPUT_KEY, kwargs.get(REPO_KEY))
        if option is not None and key not in kwargs:
            try:
                value = getattr(option, key, None)
            except BaseException:  # noqa: BLE001 - option serialization is best effort
                value = None
            if value is not None:
                payload[key] = value
        if key in kwargs:
            payload[key] = kwargs[key]
    return payload


def extract_model(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    model = kwargs.get(MODEL_KEY)
    if isinstance(model, str) and model:
        return safe_text(model)
    option = args[0] if args else kwargs.get(INPUT_KEY, kwargs.get(REPO_KEY))
    try:
        option_model = getattr(option, MODEL_KEY, None)
    except BaseException:  # noqa: BLE001 - hostile option objects must fail closed
        return None
    return (
        safe_text(option_model)
        if isinstance(option_model, str) and option_model
        else None
    )


def extract_primary_input(
    *, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    for field_name in (INPUT_KEY, REPO_KEY):
        if field_name in kwargs:
            candidate = kwargs[field_name]
            try:
                value = getattr(candidate, field_name, None)
            except BaseException:  # noqa: BLE001 - hostile options must not alter the SDK
                value = None
            return value if value is not None else candidate
    if not args:
        return None
    first_arg = args[0]
    if isinstance(first_arg, (str, bytes)):
        return first_arg
    for field_name in (INPUT_KEY, REPO_KEY):
        try:
            field_value = getattr(first_arg, field_name, None)
        except BaseException:  # noqa: BLE001, S112 - try the next safe field
            continue
        if field_value is not None:
            return field_value
    return None
