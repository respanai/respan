"""Serialization helpers for Arize SDK operation spans."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

MAX_ITEMS = 6
MAX_STRING_LENGTH = 2000


def _truncate(value: str) -> str:
    if len(value) <= MAX_STRING_LENGTH:
        return value
    return f"{value[:MAX_STRING_LENGTH]}...<truncated>"


def _summarize_dataframe(value: Any) -> dict[str, Any] | None:
    module_name = type(value).__module__
    class_name = type(value).__name__
    if module_name.startswith("pandas.") and class_name == "DataFrame":
        columns = [str(column) for column in getattr(value, "columns", [])]
        return {
            "type": "pandas.DataFrame",
            "rows": len(value),
            "columns": columns[:MAX_ITEMS],
            "truncated_columns": max(0, len(columns) - MAX_ITEMS),
        }
    return None



def _summarize_future(value: Any, *, depth: int, seen: set[int]) -> dict[str, Any] | None:
    if not isinstance(value, concurrent.futures.Future):
        return None

    result: dict[str, Any] = {
        "type": type(value).__name__,
        "done": value.done(),
        "cancelled": value.cancelled(),
    }
    if value.done() and not value.cancelled():
        exception = value.exception()
        if exception is not None:
            result["exception"] = str(exception)
            result["exception_type"] = type(exception).__name__
        else:
            result["result"] = _summarize_value(value.result(), depth=depth + 1, seen=seen)
    return result


def _summarize_response(value: Any) -> dict[str, Any] | None:
    status_code = getattr(value, "status_code", None)
    if status_code is None:
        return None
    result: dict[str, Any] = {
        "type": type(value).__name__,
        "status_code": status_code,
    }
    url = getattr(value, "url", None)
    if url:
        result["url"] = str(url)
    text = getattr(value, "text", None)
    if isinstance(text, str) and text:
        result["text"] = _truncate(text)
    return result


def _summarize_mapping(value: Mapping[Any, Any], *, depth: int, seen: set[int]) -> dict[str, Any]:
    return {
        str(key): _summarize_value(child, depth=depth + 1, seen=seen)
        for key, child in list(value.items())[:MAX_ITEMS]
    }


def _summarize_sequence(value: Sequence[Any], *, depth: int, seen: set[int]) -> list[Any]:
    return [_summarize_value(child, depth=depth + 1, seen=seen) for child in value[:MAX_ITEMS]]


def _summarize_value(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()

    if value is None or isinstance(value, bool | int | str):
        return _truncate(value) if isinstance(value, str) else value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, bytes):
        return _truncate(value.decode("utf-8", errors="replace"))

    dataframe_summary = _summarize_dataframe(value)
    if dataframe_summary is not None:
        return dataframe_summary

    response_summary = _summarize_response(value)
    if response_summary is not None:
        return response_summary

    future_summary = _summarize_future(value, depth=depth, seen=seen)
    if future_summary is not None:
        return future_summary

    object_id = id(value)
    if object_id in seen:
        return "[CYCLE]"
    seen.add(object_id)

    if depth >= 4:
        return str(value)
    if isinstance(value, Mapping):
        return _summarize_mapping(value, depth=depth, seen=seen)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _summarize_mapping(dataclasses.asdict(value), depth=depth, seen=seen)
    if isinstance(value, tuple):
        return _summarize_sequence(value, depth=depth, seen=seen)
    if isinstance(value, list):
        return _summarize_sequence(value, depth=depth, seen=seen)
    if isinstance(value, set | frozenset):
        return _summarize_sequence(list(value), depth=depth, seen=seen)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _summarize_value(to_dict(), depth=depth + 1, seen=seen)
        except Exception:
            pass

    return str(value)


def safe_json_dumps(value: Any) -> str:
    """JSON serialize *value* into an OTel-safe string attribute."""
    return json.dumps(_summarize_value(value), default=str, sort_keys=True)
