"""Translate OpenAI SDK request/response objects into span-ready primitives.

Pure functions, no OTEL imports — everything here turns OpenAI's pydantic
objects (or dicts) into JSON-safe Python values that ``_otel_emitter`` maps
onto span attributes.
"""

from __future__ import annotations

import json
from typing import Any

from respan_sdk.utils.serialization import serialize_value

from respan_instrumentation_openai._constants import (
    ASSISTANT_ROLE,
    CONTENT_KEY,
    ROLE_KEY,
    USER_ROLE,
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _dump(value: Any) -> Any:
    """Best-effort convert an OpenAI pydantic object / dict to plain data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return serialize_value(value=value)


def safe_json(value: Any) -> str:
    try:
        return json.dumps(_dump(value), default=str)
    except Exception:
        return str(value)


def to_attr_value(value: Any) -> str:
    return value if isinstance(value, str) else safe_json(value)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return safe_json(value)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or a pydantic/attr object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def request_model(request_kwargs: dict[str, Any]) -> str | None:
    model = request_kwargs.get("model")
    return str(model) if model else None


def response_model(response: Any) -> str | None:
    model = _get(response, "model")
    return str(model) if model else None


def response_id(response: Any) -> str | None:
    rid = _get(response, "id")
    return str(rid) if rid else None


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def extract_usage(response: Any) -> dict[str, int]:
    """Return ``{prompt, completion, total}`` tokens from any OpenAI response.

    Handles both Chat/Completions usage (``prompt_tokens``/``completion_tokens``)
    and Responses-API usage (``input_tokens``/``output_tokens``).
    """
    usage = _get(response, "usage")
    if usage is None:
        return {}
    prompt = _get(usage, "prompt_tokens")
    completion = _get(usage, "completion_tokens")
    if prompt is None:
        prompt = _get(usage, "input_tokens")
    if completion is None:
        completion = _get(usage, "output_tokens")
    total = _get(usage, "total_tokens")
    out: dict[str, int] = {}
    if prompt is not None:
        out["prompt"] = int(prompt)
    if completion is not None:
        out["completion"] = int(completion)
    if total is not None:
        out["total"] = int(total)
    return out


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def normalize_chat_messages(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, dict):
        messages = [messages]
    normalized: list[dict[str, Any]] = []
    for msg in messages or []:
        role = _get(msg, ROLE_KEY) or USER_ROLE
        content = _get(msg, CONTENT_KEY)
        entry: dict[str, Any] = {ROLE_KEY: role, CONTENT_KEY: _dump(content)}
        tool_calls = _get(msg, "tool_calls")
        if tool_calls:
            entry["tool_calls"] = _dump(tool_calls)
        normalized.append(entry)
    return normalized


def format_input_messages(messages: list[dict[str, Any]]) -> str:
    return safe_json(messages)


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices") or []
    if isinstance(choices, (list, tuple)) and choices:
        return choices[0]
    return None


def format_chat_output(response: Any) -> str:
    choice = _first_choice(response)
    message = _get(choice, "message") if choice is not None else None
    content = _get(message, CONTENT_KEY) if message is not None else None
    return _coerce_text(content)


def extract_chat_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    choice = _first_choice(response)
    message = _get(choice, "message") if choice is not None else None
    tool_calls = _get(message, "tool_calls") if message is not None else None
    if not tool_calls:
        return None
    return _dump(tool_calls)


def normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return _dump(tools)


# ---------------------------------------------------------------------------
# Legacy completions
# ---------------------------------------------------------------------------


def normalize_text_prompts(prompt: Any) -> list[dict[str, Any]]:
    if prompt is None:
        return []
    if isinstance(prompt, (list, tuple)):
        return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _coerce_text(p)} for p in prompt]
    return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: _coerce_text(prompt)}]


def format_completion_output(response: Any) -> str:
    choice = _first_choice(response)
    return _coerce_text(_get(choice, "text") if choice is not None else None)


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


def normalize_responses_input(value: Any) -> list[dict[str, Any]]:
    """Normalize the Responses-API ``input`` (str or message list)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{ROLE_KEY: USER_ROLE, CONTENT_KEY: value}]
    if isinstance(value, dict):
        return [{ROLE_KEY: _get(value, ROLE_KEY) or USER_ROLE, CONTENT_KEY: _dump(_get(value, CONTENT_KEY))}]
    normalized: list[dict[str, Any]] = []
    for item in value or []:
        if isinstance(item, (dict,)) or hasattr(item, "role"):
            normalized.append({ROLE_KEY: _get(item, ROLE_KEY) or USER_ROLE, CONTENT_KEY: _dump(_get(item, CONTENT_KEY))})
        else:
            normalized.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: _coerce_text(item)})
    return normalized


def format_responses_output(response: Any) -> str:
    """Prefer ``output_text``; fall back to serializing ``output``."""
    text = _get(response, "output_text")
    if isinstance(text, str) and text:
        return text
    output = _get(response, "output")
    if output is None:
        return ""
    return safe_json(output)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def normalize_embedding_inputs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [_coerce_text(v) for v in value]
    return [_coerce_text(value)]


def embedding_summary(response: Any) -> dict[str, Any]:
    data = _get(response, "data") or []
    summary: dict[str, Any] = {}
    if isinstance(data, (list, tuple)):
        summary["vector_count"] = len(data)
        first = data[0] if data else None
        emb = _get(first, "embedding") if first is not None else None
        if isinstance(emb, (list, tuple)):
            summary["dimension"] = len(emb)
    return summary


# re-export so the emitter doesn't need to know the role constant
ASSISTANT = ASSISTANT_ROLE
