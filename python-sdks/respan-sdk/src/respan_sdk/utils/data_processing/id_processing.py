"""ID processing utilities for OTEL trace/span IDs."""

import hashlib
import uuid
from typing import Optional

TRACE_ID_HEX_LENGTH = 32
SPAN_ID_HEX_LENGTH = 16


def generate_unique_id() -> str:
    """Generate a unique hex ID."""
    return uuid.uuid4().hex


def format_trace_id(trace_id: int) -> str:
    """Format an integer trace ID as a zero-padded 32-char hex string."""
    return format(trace_id, "032x")


def format_span_id(span_id: int) -> str:
    """Format an integer span ID as a zero-padded 16-char hex string."""
    return format(span_id, "016x")


def is_hex_string(value: str, length: int) -> bool:
    """Check whether *value* is a valid hex string of exactly *length* chars."""
    if len(value) != length:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def normalize_hex_id(identifier: str, expected_length: int, field_name: str) -> str:
    """Normalize a hex trace/span identifier and validate its shape.

    Accepts optional ``0x`` prefix, lowercases, and validates length + hex
    characters.  Returns the cleaned hex string (no prefix).

    Raises:
        TypeError: if *identifier* is not a string.
        ValueError: if length or hex check fails.
    """
    if not isinstance(identifier, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = identifier.lower().removeprefix("0x")
    if len(normalized) != expected_length:
        raise ValueError(
            f"{field_name} must be {expected_length} hex characters, got {len(normalized)}"
        )

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a hexadecimal string") from exc

    return normalized


def _id_to_int(val: str, bits: int) -> int:
    """Convert a string ID to int.

    If *val* is valid hex, parse it directly.  Otherwise, use a deterministic
    hash so that non-hex trace IDs (e.g. UUIDs with hyphens, or arbitrary
    strings from external SDKs) still produce a stable numeric ID.
    """
    cleaned = val.replace("-", "")
    try:
        return int(cleaned, 16) & ((1 << bits) - 1)
    except ValueError:
        h = hashlib.md5(cleaned.encode(), usedforsecurity=False).hexdigest()
        return int(h, 16) & ((1 << bits) - 1)


def ensure_trace_id(val: Optional[str] = None) -> int:
    """Return a 128-bit trace ID as int.  Generates one if *val* is ``None``."""
    if val:
        return _id_to_int(val, 128)
    return uuid.uuid4().int & ((1 << 128) - 1)


def ensure_span_id(val: Optional[str] = None) -> int:
    """Return a 64-bit span ID as int.  Generates one if *val* is ``None``."""
    if val:
        return _id_to_int(val, 64)
    return uuid.uuid4().int & ((1 << 64) - 1)
