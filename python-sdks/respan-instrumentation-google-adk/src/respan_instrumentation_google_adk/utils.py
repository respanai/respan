"""Utility functions for Respan Google ADK instrumentation."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence


def ns_to_datetime(value: Optional[int]) -> Optional[datetime]:
    """Convert nanoseconds timestamp to datetime."""
    if not value:
        return None
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)


def format_trace_id(trace_id: int) -> str:
    """Format trace ID as 32-char hex string."""
    return format(trace_id, "032x")


def format_span_id(span_id: int) -> str:
    """Format span ID as 16-char hex string."""
    return format(span_id, "016x")


_ADK_SCOPE_NAMES = {"gcp.vertex.agent", "google_adk", "google-adk"}
_ADK_SPAN_NAMES = {"invocation", "agent_run", "call_llm", "execute_tool", "invoke_agent"}


def is_adk_span(span: object) -> bool:
    """Check if span is from Google ADK instrumentation."""
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_library", None
    )
    scope_name = getattr(scope, "name", "") or ""
    if scope_name in _ADK_SCOPE_NAMES:
        return True
    # Fallback: check span name prefix + gen_ai attributes
    span_name = (getattr(span, "name", "") or "").split(" ")[0]
    if span_name in _ADK_SPAN_NAMES:
        attributes = getattr(span, "attributes", None) or {}
        if any(key.startswith("gen_ai.") for key in attributes):
            return True
    return False


def otel_span_to_dict(span: object) -> Dict[str, object]:
    """Convert OpenTelemetry span to dict format for ADK spans."""
    attributes = dict(getattr(span, "attributes", None) or {})
    span_context = getattr(span, "context", None)
    trace_id = format_trace_id(trace_id=span_context.trace_id) if span_context else None
    span_id = format_span_id(span_id=span_context.span_id) if span_context else None

    parent = getattr(span, "parent", None)
    parent_id = None
    if parent is not None and getattr(parent, "span_id", None) is not None:
        parent_id = format_span_id(span_id=parent.span_id)

    span_name = getattr(span, "name", None)

    raw_kind = getattr(span, "kind", None)
    span_kind = getattr(raw_kind, "name", None) if raw_kind is not None else None
    if span_kind is None and raw_kind is not None:
        span_kind = str(raw_kind)

    status = getattr(span, "status", None)
    status_code = None
    error_message = None
    if status is not None:
        status_enum = getattr(status, "status_code", None)
        status_name = getattr(status_enum, "name", None)
        if status_name == "ERROR":
            status_code = 500
            error_message = getattr(status, "description", None) or "error"
        elif status_name == "OK":
            status_code = 200

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_id": parent_id,
        "name": span_name,
        "span_type": span_kind,
        "kind": span_kind,
        "start_time": ns_to_datetime(value=getattr(span, "start_time", None)),
        "end_time": ns_to_datetime(value=getattr(span, "end_time", None)),
        "attributes": attributes,
        "status_code": status_code,
        "error": error_message,
    }


def group_spans_by_trace(
    spans: Sequence[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    """Group spans by trace ID."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for span in spans:
        trace_id = span.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            continue
        grouped.setdefault(trace_id, []).append(span)
    return grouped


def get_attr(obj: Any, *keys: str, default: Any = None) -> Any:
    """Get attribute from object or dict by trying multiple keys."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
    for key in keys:
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def as_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Convert value to dict if possible."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return None
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            return None
    return None


def pick_metadata_value(metadata: Optional[Dict[str, Any]], *keys: str) -> Any:
    """Pick first available value from metadata by trying multiple keys."""
    if not metadata:
        return None
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def coerce_datetime(value: Any, reference: Optional[datetime] = None) -> Optional[datetime]:
    """Coerce various value types to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if reference and numeric_value < 1_000_000_000:
            return reference + timedelta(seconds=numeric_value)
        if numeric_value > 100_000_000_000:
            return datetime.fromtimestamp(numeric_value / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(numeric_value, tz=timezone.utc)
    if isinstance(value, str):
        trimmed = value.strip()
        try:
            return datetime.fromisoformat(trimmed.replace("Z", "+00:00"))
        except ValueError:
            try:
                numeric_value = float(trimmed)
                if reference and numeric_value < 1_000_000_000:
                    return reference + timedelta(seconds=numeric_value)
                if numeric_value > 100_000_000_000:
                    return datetime.fromtimestamp(numeric_value / 1000, tz=timezone.utc)
                return datetime.fromtimestamp(numeric_value, tz=timezone.utc)
            except ValueError:
                return None
    return None


def coerce_token_count(value: Any) -> Optional[int]:
    """Coerce value to token count integer."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def infer_trace_start_time(spans: Sequence[Any]) -> Optional[datetime]:
    """Infer the earliest start time from a sequence of spans."""
    earliest: Optional[datetime] = None
    for span in spans:
        raw_value = get_attr(span, "start_time", "started_at", "start", "start_timestamp")
        if isinstance(raw_value, (int, float)) and float(raw_value) < 1_000_000_000:
            continue
        if isinstance(raw_value, str):
            trimmed = raw_value.strip()
            if trimmed:
                try:
                    numeric_value = float(trimmed)
                    if numeric_value < 1_000_000_000:
                        continue
                except ValueError:
                    pass
        candidate = coerce_datetime(value=raw_value)
        if candidate and candidate.year < 2001:
            continue
        if candidate and (earliest is None or candidate < earliest):
            earliest = candidate
    return earliest


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


def is_hex_string(value: str, length: int) -> bool:
    """Check if string is a valid hex string of given length."""
    if len(value) != length:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


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


def clean_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None, empty dict, and empty list values from payload."""
    return {key: value for key, value in payload.items() if value not in (None, {}, [])}


def parse_json_value(value: Any) -> Any:
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


def find_root_span(spans: Sequence[Any]) -> Optional[Any]:
    """Find the root span (span without parent in the trace)."""
    if not spans:
        return None
    span_ids = set()
    for span in spans:
        span_id = get_attr(span, "span_id", "id", "uid")
        if span_id is not None:
            span_ids.add(str(span_id))
    for span in spans:
        parent_id = get_attr(span, "parent_id", "parent_span_id", "parentId")
        if not parent_id or str(parent_id) not in span_ids:
            return span
    return spans[0]


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
    parsed = parse_json_value(value=value)
    if isinstance(parsed, list) and parsed and all(isinstance(item, dict) for item in parsed):
        if all("role" in item and "content" in item for item in parsed):
            return [_normalize_message_role(m) for m in parsed]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("messages"), list):
            messages = parsed.get("messages") or []
            if messages and all(isinstance(item, dict) for item in messages):
                return [_normalize_message_role(m) for m in messages]
        if "role" in parsed and "content" in parsed:
            return [_normalize_message_role(parsed)]
    return None


def to_completion_message(value: Any) -> Optional[Dict[str, Any]]:
    """Convert value to completion message format."""
    parsed = parse_json_value(value=value)
    result: Optional[Dict[str, Any]] = None
    if isinstance(parsed, dict):
        if "role" in parsed and "content" in parsed:
            result = parsed
        else:
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict) and "content" in message:
                        result = message
    if result is None and isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict) and "content" in first:
            result = first
    if result is not None:
        return _normalize_message_role(result)
    return None


def messages_to_text(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Convert messages to text format."""
    if not messages:
        return None
    parts: List[str] = []
    for message in messages:
        content = message.get("content")
        if content is None:
            continue
        if isinstance(content, (dict, list)):
            parts.append(json.dumps(content, default=str))
        else:
            parts.append(str(content))
    if not parts:
        return None
    return "\n".join(parts)


def _normalize_message_role(message: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Gemini ``"model"`` role to ``"assistant"`` for OpenAI compatibility."""
    if message.get("role") == "model":
        message = {**message, "role": "assistant"}
    return message


def extract_genai_messages(
    attributes: Dict[str, Any],
    key: str,
) -> Optional[List[Dict[str, Any]]]:
    """Extract messages from GenAI semantic convention attributes.

    ADK stores messages as JSON strings in attributes like
    ``gen_ai.input.messages`` and ``gen_ai.output.messages``.
    """
    raw = attributes.get(key)
    if raw is None:
        return None
    parsed = parse_json_value(value=raw)
    if isinstance(parsed, list):
        messages = []
        for item in parsed:
            if isinstance(item, dict):
                messages.append(_normalize_message_role(item))
        if messages:
            return messages
    if isinstance(parsed, dict) and ("role" in parsed or "content" in parsed):
        return [_normalize_message_role(parsed)]
    return None
