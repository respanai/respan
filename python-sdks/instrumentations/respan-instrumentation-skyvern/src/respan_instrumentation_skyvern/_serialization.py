"""Convert Skyvern / litellm objects into the JSON shapes Respan renders."""

from __future__ import annotations

import base64
import dataclasses
import enum
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from respan_tracing.utils.agent_instrumentation import (
    DATA_URL_PREFIX,
    compact_image_data_url,
)

from respan_instrumentation_skyvern._constants import (
    IMAGE_PLACEHOLDER,
    PNG_DATA_URL_PREFIX,
)

_MAX_DEPTH = 32
_IMAGE_PART = "image_url"


def jsonable(value: Any, *, depth: int = 0) -> Any:
    """A JSON-safe copy: pydantic models and dataclasses are dumped, bytes summarised."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, enum.Enum):
        return jsonable(value.value, depth=depth)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes", "length": len(value)}
    if depth >= _MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json", exclude_none=True), depth=depth + 1)
        except Exception:  # noqa: BLE001 - telemetry must fail open
            return {"type": type(value).__name__, "serialization_error": True}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        return jsonable({k: v for k, v in fields.items() if v is not None}, depth=depth + 1)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [jsonable(v, depth=depth + 1) for v in value]
    return {"type": type(value).__name__}


def json_string(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, default=str)


def screenshot_part(png: bytes) -> dict[str, Any]:
    """A screenshot as an OpenAI-style image part (Respan lifts it into media storage)."""
    url = PNG_DATA_URL_PREFIX + base64.b64encode(png).decode()
    return {"type": _IMAGE_PART, _IMAGE_PART: {"url": compact_image_data_url(url)}}


def _content(content: Any, *, keep_images: bool) -> Any:
    if not isinstance(content, list):
        return content
    parts = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != _IMAGE_PART:
            parts.append(part)
            continue
        image_url = part.get(_IMAGE_PART) or {}
        url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
        url = url or ""
        if not url.startswith(DATA_URL_PREFIX):
            parts.append({"type": _IMAGE_PART, _IMAGE_PART: {"url": url}})
        elif keep_images:
            parts.append({"type": _IMAGE_PART, _IMAGE_PART: {"url": compact_image_data_url(url)}})
        else:
            parts.append({"type": "text", "text": IMAGE_PLACEHOLDER.format(size=len(url))})
    return parts


def prompt_messages(messages: Any, *, keep_images: bool) -> list[dict[str, Any]]:
    """litellm messages are OpenAI-shaped dicts; shrink or drop their inline screenshots."""
    result = []
    for message in messages or ():
        data = jsonable(message)
        if isinstance(data, dict):
            data["content"] = _content(data.get("content"), keep_images=keep_images)
            result.append(data)
    return result


def content_text(content: Any) -> str:
    """Flatten a message content for the gen_ai.prompt.N.content attribute."""
    if content is None or isinstance(content, str):
        return content or ""
    return json.dumps(content, ensure_ascii=False, default=str)
