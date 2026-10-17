"""Turn Kernel call arguments and responses into the JSON Respan renders."""

from __future__ import annotations

import base64
import binascii
import inspect
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, Optional

from respan_tracing.utils.agent_instrumentation import (
    compact_image_data_url,
    jsonable,
)

from respan_instrumentation_kernel._constants import (
    B64_OUTPUT_FIELDS,
    OUTPUT_TEXT_LIMIT,
    TRANSPORT_ARGS,
    TRUNCATED_SUFFIX,
    UNSET_TYPES,
)

_IMAGE_PART = "image_url"


def capped(text: str) -> str:
    if len(text) <= OUTPUT_TEXT_LIMIT:
        return text
    return text[:OUTPUT_TEXT_LIMIT] + TRUNCATED_SUFFIX.format(size=len(text))


def decode_b64(value: Any) -> Any:
    """Kernel returns process output base64-encoded; show it as text."""
    if not isinstance(value, str):
        return value
    try:
        return capped(base64.b64decode(value, validate=True).decode("utf-8", "replace"))
    except (binascii.Error, ValueError):
        return value


def is_unset(value: Any) -> bool:
    return value is None or type(value).__name__ in UNSET_TYPES


@cache
def _signature(function: Callable[..., Any]) -> inspect.Signature:
    return inspect.signature(function)


def call_arguments(function: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    """The arguments the caller passed, by name (unset, ``None`` and transport options left out)."""
    try:
        bound = _signature(function).bind(None, *args, **kwargs)
    except TypeError:
        return {}
    return {
        name: value
        for name, value in list(bound.arguments.items())[1:]
        if name not in TRANSPORT_ARGS and not is_unset(value)
    }


def response_data(result: Any) -> Any:
    """A parsed Kernel response as JSON, with base64 process output decoded."""
    data = jsonable(result)
    if isinstance(data, Mapping):
        return {
            B64_OUTPUT_FIELDS.get(k, k): decode_b64(v) if k in B64_OUTPUT_FIELDS else v
            for k, v in data.items()
        }
    return data


def response_bytes(result: Any) -> Optional[bytes]:
    """The body of a binary response, only if it is already in memory.

    Kernel's non-streaming responses are read before they are returned, so this
    never consumes or blocks on what the caller will read.
    """
    http_response = getattr(result, "http_response", None)
    try:
        content = getattr(http_response, "content", None)
    except Exception:  # noqa: BLE001 - httpx raises ResponseNotRead for a stream
        return None
    return content if isinstance(content, bytes) else None


def image_part(image: bytes) -> dict[str, Any]:
    """A screenshot as an OpenAI-style image part (Respan lifts it into media storage)."""
    media = "jpeg" if image[:3] == b"\xff\xd8\xff" else "png"
    url = f"data:image/{media};base64,{base64.b64encode(image).decode()}"
    return {"type": _IMAGE_PART, _IMAGE_PART: {"url": compact_image_data_url(url)}}
