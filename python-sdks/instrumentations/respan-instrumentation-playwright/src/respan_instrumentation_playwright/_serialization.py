"""Turn Playwright call arguments and results into the JSON Respan renders."""

from __future__ import annotations

import inspect
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Any, Optional

from respan_instrumentation_playwright._constants import (
    OUTPUT_TEXT_LIMIT,
    SCREENSHOT_METHOD,
    TRUNCATED_SUFFIX,
)

_MAX_DEPTH = 8


def _text(value: str) -> str:
    if len(value) <= OUTPUT_TEXT_LIMIT:
        return value
    return value[:OUTPUT_TEXT_LIMIT] + TRUNCATED_SUFFIX.format(size=len(value))


def _describe(impl: Any) -> dict[str, Any]:
    """A Playwright object, from attributes it keeps locally (no browser round trip)."""
    kind = type(impl).__name__
    described: dict[str, Any] = {"type": kind}
    if selector := getattr(impl, "_selector", None):
        described["selector"] = selector
    for name in ("url", "status", "method", "suggested_filename"):
        value = getattr(type(impl), name, None)
        if isinstance(value, property):
            try:
                described[name] = getattr(impl, name)
            except Exception:  # noqa: BLE001 - telemetry must fail open
                continue
    return described


def jsonable(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, re.Pattern):
        return value.pattern
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if depth >= _MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if (impl := getattr(value, "_impl_obj", None)) is not None:
        return _describe(impl)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [jsonable(v, depth=depth + 1) for v in value]
    if callable(value):
        return {"type": "function", "name": getattr(value, "__name__", type(value).__name__)}
    return {"type": type(value).__name__}


def json_string(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, default=str)


@cache
def _signature(function: Callable[..., Any]) -> inspect.Signature:
    return inspect.signature(function)


def call_arguments(
    function: Callable[..., Any], public: Any, args: tuple, kwargs: dict
) -> dict[str, Any]:
    """The arguments the caller passed, by name (defaults and ``None`` left out)."""
    try:
        bound = _signature(function).bind(public, *args, **kwargs)
        arguments = {
            name: jsonable(value)
            for name, value in list(bound.arguments.items())[1:]
            if value is not None
        }
    except Exception:  # noqa: BLE001 - telemetry must fail open
        return {}
    # A Locator call acts on the locator's own selector, like the trace viewer shows.
    if (selector := getattr(getattr(public, "_impl_obj", None), "_selector", None)) and (
        "selector" not in arguments
    ):
        arguments = {"locator": selector, **arguments}
    return arguments


def call_result(
    result: Any, *, screenshot: Callable[[bytes], Optional[dict[str, Any]]]
) -> Any:
    """The call's return value; image bytes become an inline JPEG image part."""
    if result is None:
        return None
    if isinstance(result, bytes) and _is_image(result):
        part = screenshot(result)
        return {SCREENSHOT_METHOD: part} if part else jsonable(result)
    return jsonable(result)


def _is_image(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"
