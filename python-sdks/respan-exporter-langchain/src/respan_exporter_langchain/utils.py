"""Utility functions for Respan LangChain exporter."""
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from respan_sdk.utils.data_processing.id_processing import is_hex_string


def normalize_trace_id(trace_id: str) -> str:
    """Normalize trace ID to 32-char hex string."""
    if is_hex_string(value=trace_id, length=32):
        return trace_id.lower()
    return uuid.uuid5(uuid.NAMESPACE_DNS, trace_id).hex


def normalize_span_id(span_id: str, trace_id: str) -> str:
    """Normalize span ID to 16-char hex string."""
    if is_hex_string(value=span_id, length=16):
        return span_id.lower()
    stable_seed = f"{trace_id}:{span_id}"
    return uuid.uuid5(uuid.NAMESPACE_DNS, stable_seed).hex[:16]


def serialize_value(value: Any) -> Optional[str]:
    """Serialize value to JSON string."""
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            try:
                json.loads(trimmed)
                return trimmed
            except Exception:
                return json.dumps(value)
        return json.dumps(value)
    try:
        return json.dumps(value, default=str)
    except Exception:
        return json.dumps(str(value))


def format_rfc3339(value: Optional[datetime]) -> Optional[str]:
    """Format datetime as RFC3339 string."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None, empty dict, and empty list values from payload."""
    return {key: value for key, value in payload.items() if value not in (None, {}, [])}


def is_blank_value(value: Any) -> bool:
    """Check if value is blank (None, empty, or null-like)."""
    if value is None:
        return True
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return True
        if trimmed in ("[]", "{}", "null"):
            return True
        try:
            parsed = json.loads(trimmed)
            return parsed in (None, [], {})
        except Exception:
            return False
    if isinstance(value, (list, tuple, dict)) and not value:
        return True
    return False


def to_prompt_messages(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Convert value to prompt messages format."""
    parsed = _parse_json_value(value=value)
    if isinstance(parsed, list) and parsed and all(isinstance(item, dict) for item in parsed):
        if all("role" in item and "content" in item for item in parsed):
            return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("messages"), list):
            messages = parsed.get("messages") or []
            if messages and all(isinstance(item, dict) for item in messages):
                return messages
        if "role" in parsed and "content" in parsed:
            return [parsed]
    return None


def to_completion_message(value: Any) -> Optional[Dict[str, Any]]:
    """Convert value to completion message format."""
    parsed = _parse_json_value(value=value)
    if isinstance(parsed, dict):
        if "role" in parsed and "content" in parsed:
            return parsed
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return message
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict) and "content" in first:
            return first
    return None


def _parse_json_value(value: Any) -> Any:
    """Parse JSON string value if possible."""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            try:
                return json.loads(trimmed)
            except Exception:
                return value
        return value
    return value


def langchain_messages_to_dicts(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert LangChain BaseMessage objects to dicts."""
    result: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
            continue
        role = getattr(msg, "type", "unknown")
        # Map LangChain message types to standard roles
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "function": "function",
            "tool": "tool",
        }
        content = getattr(msg, "content", str(msg))
        entry: Dict[str, Any] = {
            "role": role_map.get(role, role),
            "content": content,
        }
        # Include tool_calls if present
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                tc if isinstance(tc, dict) else tc.dict()
                for tc in tool_calls
            ]
        result.append(entry)
    return result
