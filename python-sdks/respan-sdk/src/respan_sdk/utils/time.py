from datetime import datetime
from typing import Optional, Union


def iso_to_ns(iso_str: Optional[str]) -> Optional[int]:
    """Convert an ISO-8601 timestamp to nanoseconds since epoch.

    Returns ``None`` if *iso_str* is falsy or unparseable.
    """
    if not iso_str:
        return None
    try:
        return int(datetime.fromisoformat(iso_str).timestamp() * 1e9)
    except Exception:
        return None


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