from datetime import datetime
from datetime import timezone
from typing import Union


def now_utc() -> datetime:
    """Return current UTC time. Use for start_time/end_time in export payloads."""
    return datetime.now(timezone.utc)


# NOTE: PR #46 adds now_iso and format_timestamp; merge order may cause conflicts here.


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