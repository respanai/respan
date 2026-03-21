"""Utility functions for the Google ADK → Respan span exporter."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe attribute access
# ---------------------------------------------------------------------------

def get_attr(span, key: str, default=None):
    """Safely retrieve a span attribute, returning *default* if missing."""
    attrs = getattr(span, "attributes", None)
    if attrs is None:
        return default
    return attrs.get(key, default)


def coerce_int(value) -> Optional[int]:
    """Cast *value* to ``int``, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_json_parse(value) -> Any:
    """``json.loads(value)`` with fallback to the raw value."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def serialize(obj) -> Any:
    """Recursively convert *obj* to plain JSON-serializable Python types.

    Handles Pydantic v2 models, datetimes, and arbitrary objects.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return {
                k: serialize(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


# ---------------------------------------------------------------------------
# ID / time helpers
# ---------------------------------------------------------------------------

def trace_id_hex(trace_id: int) -> str:
    """Format an OTel trace ID as a 32-char zero-padded hex string."""
    return format(trace_id, "032x")


def span_id_hex(span_id: int) -> str:
    """Format an OTel span ID as a 16-char zero-padded hex string."""
    return format(span_id, "016x")


def ns_to_datetime(ns: int) -> datetime:
    """Convert nanosecond epoch timestamp to a UTC ``datetime``."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def ns_to_seconds(start_ns: int, end_ns: int) -> float:
    """Return the duration in seconds between two nanosecond timestamps."""
    return (end_ns - start_ns) / 1e9


# ---------------------------------------------------------------------------
# Gemini message conversion
# ---------------------------------------------------------------------------

_ROLE_MAP = {"user": "user", "model": "assistant"}


def extract_tool_calls_from_parts(parts: list) -> Optional[List[dict]]:
    """Convert Gemini ``functionCall`` parts to OpenAI-style tool_calls."""
    if not parts:
        return None
    tool_calls = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        fc = part.get("functionCall") or part.get("function_call")
        if fc:
            args = fc.get("args") or fc.get("arguments") or {}
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })
    return tool_calls if tool_calls else None


def _parts_to_text(parts: list) -> Optional[str]:
    """Extract concatenated text from a list of Gemini parts."""
    if not parts:
        return None
    texts = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(p["text"])
    return "\n".join(texts) if texts else None


def gemini_request_to_prompt_messages(llm_request: dict) -> Optional[List[dict]]:
    """Convert a Gemini ``llm_request`` dict to a list of prompt messages.

    Extracts ``config.systemInstruction`` as a system message and converts
    ``contents[]`` entries to user/assistant messages with optional tool_calls.
    """
    if not isinstance(llm_request, dict):
        return None

    messages: List[dict] = []

    # System instruction
    config = llm_request.get("config") or {}
    sys_instr = config.get("systemInstruction") or config.get("system_instruction")
    if sys_instr:
        parts = sys_instr.get("parts") if isinstance(sys_instr, dict) else None
        text = _parts_to_text(parts) if parts else (sys_instr if isinstance(sys_instr, str) else None)
        if text:
            messages.append({"role": "system", "content": text})

    # Contents
    contents = llm_request.get("contents") or []
    for entry in contents:
        if not isinstance(entry, dict):
            continue
        raw_role = entry.get("role", "user")
        role = _ROLE_MAP.get(raw_role, raw_role)
        parts = entry.get("parts") or []
        text = _parts_to_text(parts)
        tool_calls = extract_tool_calls_from_parts(parts)

        msg: Dict[str, Any] = {"role": role}
        if text:
            msg["content"] = text
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)

    return messages if messages else None


def gemini_request_to_input_text(llm_request: dict) -> Optional[str]:
    """Extract only user-role text content from an ADK/Gemini request."""
    if not isinstance(llm_request, dict):
        return None

    texts: List[str] = []
    for entry in llm_request.get("contents") or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "user")
        if _ROLE_MAP.get(role, role) != "user":
            continue
        text = _parts_to_text(entry.get("parts") or [])
        if text:
            texts.append(text)

    return "\n".join(texts) if texts else None


def gemini_response_to_completion_message(llm_response: dict) -> Optional[dict]:
    """Convert an ADK/Gemini ``llm_response`` dict to a completion message."""
    if not isinstance(llm_response, dict):
        return None

    # Google ADK stores a serialized LlmResponse with top-level ``content``.
    if isinstance(llm_response.get("content"), dict):
        content_obj = llm_response.get("content") or {}
    else:
        candidates = llm_response.get("candidates") or []
        if not candidates:
            return None
        content_obj = candidates[0].get("content") or {}
    parts = content_obj.get("parts") or []

    text = _parts_to_text(parts)
    tool_calls = extract_tool_calls_from_parts(parts)

    msg: Dict[str, Any] = {"role": "assistant"}
    if text:
        msg["content"] = text
    if tool_calls:
        msg["tool_calls"] = tool_calls

    return msg if (text or tool_calls) else None


def messages_to_text(messages: List[dict]) -> Optional[str]:
    """Convert prompt/completion messages into a plain-text representation."""
    if not messages:
        return None

    parts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is None:
            continue
        if isinstance(content, (dict, list)):
            parts.append(json.dumps(content, default=str))
        else:
            parts.append(str(content))

    return "\n".join(parts) if parts else None


def message_to_text(message: dict) -> Optional[str]:
    """Convert a single prompt/completion message into text."""
    if not isinstance(message, dict):
        return None
    return messages_to_text([message])


# ---------------------------------------------------------------------------
# Span type detection
# ---------------------------------------------------------------------------

def extract_span_type(span) -> str:
    """Determine the logical span type from attributes and span name.

    Checks ``gen_ai.operation_name`` first (authoritative), then falls back
    to span name prefix matching.
    """
    op = get_attr(span, "gen_ai.operation_name")
    if op:
        return op.lower().strip()

    name = getattr(span, "name", "") or ""
    name_lower = name.lower().strip()

    prefixes = [
        ("invocation", "invocation"),
        ("call_llm", "call_llm"),
        ("send_data", "send_data"),
        ("handle_context_caching", "handle_context_caching"),
        ("create_cache", "create_cache"),
        ("execute_tool", "execute_tool"),
        ("invoke_agent", "invoke_agent"),
        ("generate_content", "generate_content"),
    ]
    for prefix, span_type in prefixes:
        if name_lower.startswith(prefix):
            return span_type

    return "unknown"


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_metadata(span) -> dict:
    """Build a metadata dict from common ADK span attributes."""
    meta: Dict[str, Any] = {}

    conv_id = get_attr(span, "gen_ai.conversation.id")
    if conv_id is not None:
        meta["conversation_id"] = conv_id

    session_id = get_attr(span, "gcp.vertex.agent.session_id")
    if session_id is not None:
        meta["session_id"] = session_id

    inv_id = get_attr(span, "gcp.vertex.agent.invocation_id")
    if inv_id is not None:
        meta["invocation_id"] = inv_id

    event_id = get_attr(span, "gcp.vertex.agent.event_id")
    if event_id is not None:
        meta["event_id"] = event_id

    return meta
