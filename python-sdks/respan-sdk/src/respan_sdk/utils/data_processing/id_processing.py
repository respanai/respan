TRACE_ID_HEX_LENGTH = 32
SPAN_ID_HEX_LENGTH = 16


def generate_unique_id():
    """Generate a unique ID - placeholder implementation"""
    import uuid

    return str(uuid.uuid4()).replace("-", "")


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
