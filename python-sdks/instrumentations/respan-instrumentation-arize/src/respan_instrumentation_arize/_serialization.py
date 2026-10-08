"""Serialize known SDK data without consuming opaque iterators or user hooks."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import re
from enum import Enum
from itertools import islice

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel
from requests import Response

MAX_ITEMS = 50
MAX_JSON_BYTES = 16000
MAX_DEPTH = 8
_SENSITIVE = re.compile(
    r"(^|[._-])(api[_-]?key|authorization|cookie|password|secret|token|credential|access[_-]?key)([._-]|$)|^key$",
    re.IGNORECASE,
)


def redact_text(value):
    value = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)\b(bearer)\s+[^\s,;]+", r"\1 [REDACTED]", value)
    value = re.sub(
        r"(?i)\b(basic)\s+(?:[A-Za-z0-9+/]{4})+(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?=$|[\s,;])",
        r"\1 [REDACTED]",
        value,
    )
    value = re.sub(
        r"""(?i)(["'](?:[^"']*[._-])?(?:api[_-]?key|authorization|cookie|password|secret|token|credential|access[_-]?key|key)["']\s*:\s*)(["'])(.*?)\2""",
        lambda m: m[1] + m[2] + "[REDACTED]" + m[2],
        value,
    )
    return re.sub(
        r"(?i)(api[_-]?key|authorization|cookie|password|secret|token|credential)\s*[:=]\s*([^\s,;]+)",
        lambda m: m[1] + "=[REDACTED]",
        value,
    )


def safe_text(value, max_bytes=MAX_JSON_BYTES):
    text = redact_text(value) if isinstance(value, str) else ""
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    return (
        text.encode("utf-8")[: max_bytes - 16].decode("utf-8", errors="ignore")
        + "...[truncated]"
    )


def _type_name(value):
    return type(value).__name__[:120]


def _raw_fields(value):
    if type(value).__module__.startswith("arize.") and isinstance(value, tuple):
        fields = getattr(type(value), "_fields", None)
        if type(fields) is tuple:
            return dict(zip(fields, tuple.__iter__(value), strict=True))
    if BaseModel in type(value).__mro__:
        fields = type(value).model_fields
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {field.name: None for field in dataclasses.fields(type(value))}
    else:
        return None
    try:
        data = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return None
    return {key: data[key] for key in fields if key in data}


def requires_complete_payload(value, depth=0):
    if depth >= MAX_DEPTH:
        return False
    if type(value) is str and value.lstrip().startswith(("{", "[")):
        try:
            return requires_complete_payload(json.loads(value), depth + 1)
        except (ValueError, TypeError):
            return False
    if type(value) is np.ndarray:
        return np.issubdtype(value.dtype, np.number)
    if type(value) is pd.DataFrame:
        return requires_complete_payload(
            pd.DataFrame.to_dict(value, orient="records"), depth + 1
        )
    fields = _raw_fields(value)
    if fields is not None:
        return requires_complete_payload(fields, depth + 1)
    if type(value) in (tuple, list):
        return (
            bool(value)
            and all(type(v) in (int, float) for v in value)
            or any(requires_complete_payload(v, depth + 1) for v in value)
        )
    if type(value) is dict:
        if value and all(
            (type(k) is int and k >= 0 or type(k) is str and k.isdecimal())
            and type(v) in (int, float)
            for k, v in value.items()
        ):
            return True
        if any(
            k in value
            for k in (
                "tool_calls",
                "function",
                "inputSchema",
                "toolUse",
                "toolResult",
                "embedding",
                "embeddings",
                "vector",
                "vectors",
                "vector_values",
            )
        ):
            return True
        return any(requires_complete_payload(v, depth + 1) for v in value.values())
    return False


def _safe_key(value):
    if type(value) is str:
        return redact_text(value)[:256]
    if type(value) in (int, bool) or value is None:
        return json.dumps(value)
    return "<" + _type_name(value) + ">"


def to_jsonable(value, depth=0, complete=False, seen=frozenset()):
    complete = complete or requires_complete_payload(value)
    if id(value) in seen:
        return {"type": _type_name(value), "circular": True}
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else str(value)
    if type(value) is str:
        return redact_text(value)
    if isinstance(value, Enum):
        return to_jsonable(
            object.__getattribute__(value, "__dict__").get("_value_"),
            depth,
            complete,
            seen,
        )
    if type(value) in (dt.datetime, dt.date, dt.time):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "length": len(value)}
    if depth >= MAX_DEPTH and not complete:
        return {"type": _type_name(value), "truncated": True}
    seen = seen | {id(value)}
    if type(value) is np.ndarray:
        return to_jsonable(np.ndarray.tolist(value), depth, complete, seen)
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), depth, complete, seen)
    if type(value) is pd.DataFrame:
        return to_jsonable(
            {
                "type": "DataFrame",
                "columns": list(value.columns),
                "rows": len(value),
                "records": pd.DataFrame.to_dict(value, orient="records"),
            },
            depth,
            complete,
            seen,
        )
    if type(value) in (pa.Table, pa.RecordBatch):
        return to_jsonable(
            {
                "type": _type_name(value),
                "rows": value.num_rows,
                "records": value.to_pylist(),
            },
            depth,
            complete,
            seen,
        )
    if isinstance(value, Response):
        content = value.__dict__.get("_content")
        payload = {"type": "Response"}
        if isinstance(content, bytes):
            try:
                payload["body"] = json.loads(content)
            except (ValueError, UnicodeDecodeError):
                payload["body"] = safe_text(content.decode("utf-8", errors="replace"))
        return to_jsonable(payload, depth, complete, seen)
    fields = _raw_fields(value)
    if fields is not None:
        return to_jsonable(fields, depth, complete, seen)
    if type(value) is dict:
        output = {}
        for key, item in (
            value.items() if complete else islice(value.items(), MAX_ITEMS + 1)
        ):
            if not complete and len(output) >= MAX_ITEMS:
                output["_respan_truncated_items"] = True
                break
            key = _safe_key(key)
            output[key] = (
                "[REDACTED]"
                if _SENSITIVE.search(key)
                else to_jsonable(item, depth + 1, complete, seen)
            )
        return output
    if type(value) in (list, tuple):
        output = [
            to_jsonable(item, depth + 1, complete, seen)
            for item in (value if complete else value[:MAX_ITEMS])
        ]
        if not complete and len(value) > MAX_ITEMS:
            output.append({"_respan_truncated_items": True})
        return output
    return {"type": _type_name(value)}


def safe_json_dumps(value):
    normalized = to_jsonable(value)
    serialized = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if (
        requires_complete_payload(normalized)
        or len(serialized.encode("utf-8")) <= MAX_JSON_BYTES
    ):
        return serialized
    preview = safe_text(serialized, max_bytes=MAX_JSON_BYTES // 3)
    return json.dumps(
        {"_respan_truncated_bytes": True, "preview": preview},
        ensure_ascii=False,
        separators=(",", ":"),
    )
