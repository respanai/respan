import logging
from datetime import datetime, timezone
from typing import Optional, Union

logger = logging.getLogger(__name__)


def parse_datetime(v: Union[str, datetime]) -> datetime:
    if isinstance(v, str):
        # Lazy import to improve import speed
        from dateparser import parse

        try:
            value = datetime.fromisoformat(v)
            return value
        except Exception as e:
            try:
                value = parse(v)
                return value
            except Exception as e:
                raise ValueError(
                    "timestamp has to be a valid ISO 8601 formatted date-string YYYY-MM-DD"
                )
    return v


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def format_timestamp(ts: Optional[float]) -> str:
    """Format Unix timestamp as ISO-8601 string."""
    if ts is None:
        return now_iso()
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError) as exc:
        logger.warning(
            "Invalid timestamp %r, falling back to now(): %s",
            ts,
            exc,
        )
        return now_iso()
