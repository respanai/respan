"""Serialization and extraction helpers for LlamaIndex payloads."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from itertools import islice
from typing import Any

from respan_instrumentation_llama_index._constants import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_SYSTEM,
    MESSAGE_ROLE_USER,
)

_REACT_OBSERVATION_PREFIX = "Observation:"
_CONTEXT_PROMPT_PREFIX = "Context information is below."
_QUERY_MARKER = "\nQuery:"
_ANSWER_MARKER = "\nAnswer:"


def enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


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
        if value and all(
            (type(key) is int and key >= 0 or type(key) is str and key.isdecimal())
            and type(item) in (int, float)
            for key, item in value.items()
        ):
            return True
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
    return safe_json(value, complete=complete)


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


def safe_json(
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


def get_model_name(model_dict: dict[str, Any] | None) -> str | None:
    if not model_dict:
        return None
    for key in ("model_name", "model", "model_id", "deployment_name", "name"):
        value = model_dict.get(key)
        if value:
            return str(value)
    return None


def get_model_system(model_dict: dict[str, Any] | None) -> str | None:
    if not model_dict:
        return None
    candidates = (
        model_dict.get("class_name"),
        model_dict.get("provider"),
        model_dict.get("model_provider"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        normalized = str(candidate).lower()
        if "openai" in normalized:
            return "openai"
        if "anthropic" in normalized or "claude" in normalized:
            return "anthropic"
        if "gemini" in normalized or "google" in normalized:
            return "google"
        if "bedrock" in normalized:
            return "bedrock"
        if candidate in (model_dict.get("provider"), model_dict.get("model_provider")):
            return safe_text(normalized).replace(" ", "_")
    return None


def message_to_dict(message: Any) -> dict[str, Any]:
    role = enum_value(getattr(message, "role", None)) or MESSAGE_ROLE_USER
    content = getattr(message, "content", None)
    if content is None and hasattr(message, "blocks"):
        content = [to_jsonable(block) for block in getattr(message, "blocks", [])]

    result: dict[str, Any] = {
        "role": str(role),
        "content": to_jsonable(content),
    }
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        result["additional_kwargs"] = to_jsonable(additional_kwargs)
    calls = (
        getattr(message, "additional_kwargs", {}).get("tool_calls")
        if isinstance(getattr(message, "additional_kwargs", {}), dict)
        else None
    )
    if calls:
        result["tool_calls"] = to_jsonable(calls, complete=True)
    return normalize_message_dict(result)


def chat_messages_to_dicts(messages: Any) -> list[dict[str, Any]]:
    if messages is None:
        return []
    return normalize_message_sequence(
        [message_to_dict(message) for message in messages]
    )


def normalize_message_sequence(messages: list[Any]) -> list[Any]:
    if not all(isinstance(message, dict) for message in messages):
        return messages

    result: list[dict[str, Any]] = []
    for message in messages:
        for candidate in split_generated_context_message(
            message=normalize_message_dict(message)
        ):
            result.append(
                normalize_react_observation_message(
                    message=candidate,
                    previous_messages=result,
                )
            )
    return result


def normalize_message_dict(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize LlamaIndex-generated messages before export."""
    role = message.get("role")
    if role != MESSAGE_ROLE_USER:
        return message

    text = _message_text(message.get("content"))
    if text is None:
        text = _message_text(message.get("blocks"))
    if text is None:
        return message

    if _is_llama_index_context_prompt(text=text):
        normalized = dict(message)
        normalized["role"] = MESSAGE_ROLE_SYSTEM
        return normalized
    return message


def normalize_react_observation_message(
    *,
    message: dict[str, Any],
    previous_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    if not _is_react_observation_message(message=message):
        return message
    if not _previous_message_is_react_action(previous_messages=previous_messages):
        return message

    normalized = dict(message)
    normalized["role"] = MESSAGE_ROLE_SYSTEM
    return normalized


def split_generated_context_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Separate LlamaIndex-generated retrieval context from the user query."""
    text = _message_text(message.get("content"))
    if text is None or not _is_llama_index_context_prompt(text=text):
        return [message]

    query = _extract_generated_query(text=text)
    if not query:
        return [message]

    context_message = dict(message)
    context_message["role"] = MESSAGE_ROLE_SYSTEM
    context_message["content"] = text.split(_QUERY_MARKER, maxsplit=1)[0].rstrip()
    user_message = {
        "role": MESSAGE_ROLE_USER,
        "content": query,
    }
    return [context_message, user_message]


def chat_response_to_message_dict(response: Any) -> dict[str, Any] | None:
    message = getattr(response, "message", None)
    if message is not None:
        return message_to_dict(message)
    content = getattr(response, "text", None)
    if content is None:
        content = getattr(response, "response", None)
    if content is None:
        return None
    return {"role": MESSAGE_ROLE_ASSISTANT, "content": to_jsonable(content)}


def completion_response_to_text(response: Any) -> str | None:
    text = getattr(response, "text", None)
    if text is not None:
        return safe_text(text, max_bytes=16_000) if isinstance(text, str) else None
    response_text = getattr(response, "response", None)
    if response_text is not None:
        return (
            safe_text(response_text, max_bytes=16_000)
            if isinstance(response_text, str)
            else None
        )
    return None


def extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage_candidates = [
        getattr(response, "raw", None),
        getattr(response, "additional_kwargs", None),
        response,
    ]
    for candidate in usage_candidates:
        usage = _find_usage_dict(candidate)
        if usage:
            prompt_tokens = _get_int(
                usage,
                "prompt_tokens",
                "input_tokens",
                "total_prompt_tokens",
            )
            completion_tokens = _get_int(
                usage,
                "completion_tokens",
                "output_tokens",
                "total_completion_tokens",
            )
            total_tokens = _get_int(usage, "total_tokens")
            if total_tokens is None and (
                prompt_tokens is not None and completion_tokens is not None
            ):
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            return prompt_tokens, completion_tokens, total_tokens
    return None, None, None


def _find_usage_dict(value: Any) -> Any:
    if value is None:
        return None
    getter = (
        value.get
        if isinstance(value, Mapping)
        else lambda key, default=None: getattr(value, key, default)
    )
    for key in ("usage", "token_usage", "usage_metadata"):
        nested = getter(key)
        if nested is not None:
            return nested
    if any(
        getter(key) is not None
        for key in (
            "prompt_tokens",
            "input_tokens",
            "completion_tokens",
            "output_tokens",
            "total_tokens",
        )
    ):
        return value
    return None


def usage_attributes(response: Any) -> dict[str, int]:
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
    from opentelemetry.semconv_ai import SpanAttributes

    attributes = {}
    usage = None
    for candidate in (
        getattr(response, "raw", None),
        getattr(response, "additional_kwargs", None),
        response,
    ):
        usage = _find_usage_dict(candidate)
        if usage is not None:
            break
    if usage is None:
        return attributes
    prompt, completion, total = extract_usage(response)
    for value, keys in (
        (
            prompt,
            (gen_ai.GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_USAGE_PROMPT_TOKENS),
        ),
        (
            completion,
            (
                gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            ),
        ),
        (
            total,
            (
                SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            ),
        ),
    ):
        if value is not None:
            for key in keys:
                attributes[key] = value
    getter = (
        usage.get
        if isinstance(usage, Mapping)
        else lambda key, default=None: getattr(usage, key, default)
    )
    for detail, field, keys in (
        (
            getter("prompt_tokens_details") or getter("input_tokens_details"),
            "cached_tokens",
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            ),
        ),
        (
            getter("prompt_tokens_details") or getter("input_tokens_details"),
            "cache_write_tokens",
            (
                SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            ),
        ),
        (
            getter("completion_tokens_details") or getter("output_tokens_details"),
            "reasoning_tokens",
            (
                SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
                SpanAttributes.LLM_USAGE_REASONING_TOKENS,
            ),
        ),
    ):
        if detail is not None:
            value = (
                detail.get(field)
                if isinstance(detail, Mapping)
                else getattr(detail, field, None)
            )
            if type(value) is int and value >= 0:
                for key in keys:
                    attributes[key] = value
    return attributes


def _message_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        segments: list[str] = []
        for item in value:
            if isinstance(item, str):
                segments.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    segments.append(text)
        return "\n".join(segments) if segments else None
    return None


def _is_react_observation_message(*, message: dict[str, Any]) -> bool:
    if message.get("role") != MESSAGE_ROLE_USER:
        return False

    text = _message_text(message.get("content"))
    if text is None:
        text = _message_text(message.get("blocks"))
    return bool(text and text.lstrip().startswith(_REACT_OBSERVATION_PREFIX))


def _previous_message_is_react_action(
    *,
    previous_messages: list[dict[str, Any]],
) -> bool:
    for previous_message in reversed(previous_messages):
        role = previous_message.get("role")
        if role == MESSAGE_ROLE_SYSTEM:
            continue
        if role != MESSAGE_ROLE_ASSISTANT:
            return False

        text = _message_text(previous_message.get("content"))
        if text is None:
            text = _message_text(previous_message.get("blocks"))
        return bool(text and "Action:" in text)
    return False


def _is_llama_index_context_prompt(*, text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(_CONTEXT_PROMPT_PREFIX) and _QUERY_MARKER in stripped


def _extract_generated_query(*, text: str) -> str | None:
    _, query_part = text.split(_QUERY_MARKER, maxsplit=1)
    if _ANSWER_MARKER in query_part:
        query_part = query_part.split(_ANSWER_MARKER, maxsplit=1)[0]
    query = query_part.strip()
    return query or None


def _get_int(value: Any, *keys: str) -> int | None:
    for key in keys:
        candidate = (
            value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
        )
        if type(candidate) is int and candidate >= 0:
            return candidate
    return None
