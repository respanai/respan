"""Utility functions for Respan CrewAI exporter."""
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

def ns_to_datetime(value: Optional[int]) -> Optional[datetime]:
    """Convert nanoseconds timestamp to datetime."""
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)


def format_trace_id(trace_id: int) -> str:
    """Format trace ID as 32-char hex string."""
    return format(trace_id, "032x")


def format_span_id(span_id: int) -> str:
    """Format span ID as 16-char hex string."""
    return format(span_id, "016x")


def is_crewai_span(span: object) -> bool:
    """Check if span is from CrewAI instrumentation.

    Only returns True for CrewAI-specific signals (scope name or crewai.* attributes)
    so that we do not intercept spans from other OpenInference instrumentors (e.g.
    LlamaIndex) which also set openinference.span.kind and graph.node.id.
    """
    scope = getattr(span, "instrumentation_scope", None) or getattr(
        span, "instrumentation_library", None
    )
    scope_name = getattr(scope, "name", "") or ""
    if "crewai" in scope_name.lower():
        return True
    attributes = getattr(span, "attributes", None) or {}
    if any(key.startswith("crewai.") for key in attributes):
        return True
    return False


def otel_span_to_dict(span: object) -> Dict[str, object]:
    """Convert OpenTelemetry span to dict format."""
    attributes = dict(getattr(span, "attributes", None) or {})
    span_context = getattr(span, "context", None)
    trace_id = None
    span_id = None
    if span_context is not None:
        tid = getattr(span_context, "trace_id", None)
        sid = getattr(span_context, "span_id", None)
        if tid == 0 or sid == 0:
            logger.debug(
                "Span with invalid OTel context (zero trace_id or span_id); treating as missing"
            )
        else:
            trace_id = format_trace_id(trace_id=tid) if tid is not None else None
            span_id = format_span_id(span_id=sid) if sid is not None else None

    parent = getattr(span, "parent", None)
    parent_id = None
    if parent is not None and getattr(parent, "span_id", None) is not None:
        parent_id = format_span_id(span_id=parent.span_id)

    span_kind = attributes.get("openinference.span.kind")
    if not span_kind:
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

    span_path = attributes.get("graph.node.id")

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_id": parent_id,
        "name": getattr(span, "name", None),
        "span_type": span_kind,
        "kind": span_kind,
        "span_path": span_path,
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
            logger.debug(
                "group_spans_by_trace: skipping span with missing trace_id: %r",
                span,
            )
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
    return uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=trace_id).hex


def normalize_span_id(span_id: str, trace_id: str) -> str:
    """Normalize span ID to 16-char hex string."""
    if is_hex_string(value=span_id, length=16):
        return span_id.lower()
    stable_seed = f"{trace_id}:{span_id}"
    return uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=stable_seed).hex[:16]


def build_traces_ingest_url(
    base_url: Optional[str],
    default_endpoint: str = "https://api.respan.ai/api/v1/traces/ingest",
) -> str:
    """Build Respan traces ingest URL from a base URL (shared normalization).

    Handles base URLs that already end with /v1/traces/ingest, /v1/traces, /api,
    or a bare host so callers do not need to branch locally.
    """
    if not base_url:
        return default_endpoint
    base = base_url.rstrip("/")
    if base.endswith("/v1/traces/ingest"):
        return base
    if base.endswith("/v1/traces"):
        return f"{base}/ingest"
    if base.endswith("/api"):
        return f"{base}/v1/traces/ingest"
    return f"{base}/api/v1/traces/ingest"


def normalize_respan_base_url_for_gateway(url: str) -> str:
    """Strip traces path from a Respan URL to get the API base for gateway use.

    E.g. https://api.respan.ai/api/v1/traces/ingest -> https://api.respan.ai/api.
    """
    base = url.rstrip("/")
    for suffix in ("/v1/traces/ingest", "/v1/traces", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if not base.endswith("/api"):
        base = f"{base}/api"
    return base


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


def extract_metadata_payload(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract nested metadata payload from metadata dict."""
    if not metadata:
        return {}
    raw_meta = metadata.get("metadata")
    if isinstance(raw_meta, dict):
        return dict(raw_meta)
    if isinstance(raw_meta, str):
        parsed = parse_json_value(value=raw_meta)
        if isinstance(parsed, dict):
            return parsed
    return {}


def merge_openinference_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge OpenInference metadata format with extracted nested metadata."""
    if not metadata:
        return {}
    merged = dict(metadata)
    extracted = extract_metadata_payload(metadata=merged)
    if extracted:
        merged.pop("metadata", None)
        merged.update(extracted)
    return merged


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
        if not parent_id:
            return span
    for span in spans:
        parent_id = get_attr(span, "parent_id", "parent_span_id", "parentId")
        if str(parent_id) not in span_ids:
            return span
    return spans[0]


def extract_span_metadata(span: Any) -> Dict[str, Any]:
    """Extract and normalize metadata from span."""
    raw_metadata = as_dict(value=get_attr(span, "metadata", "attributes", "tags", "data")) or {}
    return merge_openinference_metadata(metadata=raw_metadata)


def to_prompt_messages(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Convert value to prompt messages format."""
    parsed = parse_json_value(value=value)
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
    parsed = parse_json_value(value=value)
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


def extract_openinference_messages(
    metadata: Optional[Dict[str, Any]],
    prefix: str,
) -> Optional[List[Dict[str, Any]]]:
    """Extract messages from OpenInference format metadata."""
    if not metadata:
        return None
    prefix_token = f"{prefix}."
    messages: Dict[int, Dict[str, Any]] = {}
    message_contents: Dict[int, List[str]] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.startswith(prefix_token):
            continue
        remainder = key[len(prefix_token):]
        parts = remainder.split(".")
        if len(parts) < 3:
            continue
        index_str, section, field = parts[0], parts[1], parts[2]
        if not index_str.isdigit() or section != "message":
            continue
        index = int(index_str)
        if field == "role":
            messages.setdefault(index, {})["role"] = value
        elif field == "content":
            messages.setdefault(index, {})["content"] = value
        elif field == "contents" and len(parts) >= 6:
            if parts[-2] == "message_content" and parts[-1] == "text":
                message_contents.setdefault(index, []).append(str(value))
    if not messages:
        if not message_contents:
            return None
        messages = {index: {} for index in message_contents.keys()}
    if message_contents:
        for index, contents in message_contents.items():
            if contents and not messages.setdefault(index, {}).get("content"):
                messages[index]["content"] = "\n".join(contents)
    return [messages[index] for index in sorted(messages)]


def extract_openinference_choice_texts(
    metadata: Optional[Dict[str, Any]],
) -> Optional[List[str]]:
    """Extract choice texts from OpenInference format metadata."""
    if not metadata:
        return None
    prefix_token = "llm.choices."
    choices: Dict[int, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.startswith(prefix_token):
            continue
        parts = key.split(".")
        if len(parts) < 5:
            continue
        # Key format: llm.choices.<index>.completion.text
        index_str = parts[2]
        if not index_str.isdigit():
            continue
        if parts[3] != "completion" or parts[4] != "text":
            continue
        choices[int(index_str)] = str(value)
    if not choices:
        return None
    return [choices[index] for index in sorted(choices)]


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
