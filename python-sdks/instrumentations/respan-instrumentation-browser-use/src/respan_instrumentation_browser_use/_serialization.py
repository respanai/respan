"""Convert browser-use objects into the JSON shapes Respan renders."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from respan_tracing.utils.agent_instrumentation import (  # noqa: F401 - re-exported
    DATA_URL_PREFIX,
    compact_image_data_url,
    json_string,
    jsonable,
)

from respan_instrumentation_browser_use._constants import IMAGE_PLACEHOLDER

_MAX_DEPTH = 32
_IMAGE_PART = "image_url"


def _image_placeholder(image_url: Mapping[str, Any]) -> str:
    url = image_url.get("url") or ""
    return IMAGE_PLACEHOLDER.format(
        media_type=image_url.get("media_type") or "image", size=len(url)
    )


def _content(content: Any, *, keep_images: bool) -> Any:
    if not isinstance(content, list):
        return content
    parts = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != _IMAGE_PART:
            parts.append(part)
            continue
        image_url = part.get(_IMAGE_PART) or {}
        url = image_url.get("url") or ""
        if not url.startswith(DATA_URL_PREFIX):
            # Remote image URLs are small and render as-is.
            parts.append({"type": _IMAGE_PART, _IMAGE_PART: {"url": url}})
        elif keep_images:
            parts.append({"type": _IMAGE_PART, _IMAGE_PART: {"url": compact_image_data_url(url)}})
        else:
            parts.append({"type": "text", "text": _image_placeholder(image_url)})
    return parts


def message_dicts(messages: Any, *, keep_images: bool) -> list[dict[str, Any]]:
    """browser-use messages are OpenAI-shaped pydantic models; dump them as such."""
    result = []
    for message in messages or ():
        data = jsonable(message)
        if not isinstance(data, dict):
            continue
        data.pop("cache", None)  # browser-use's prompt-cache hint, not message content
        data["content"] = _content(data.get("content"), keep_images=keep_images)
        if not data.get("tool_calls"):
            data.pop("tool_calls", None)
        result.append(data)
    return result


def content_text(content: Any) -> str:
    """Flatten a message content for the gen_ai.prompt.N.content attribute."""
    if content is None or isinstance(content, str):
        return content or ""
    return json.dumps(content, ensure_ascii=False, default=str)
