import json
from datetime import date, datetime
from typing import Any


def safe_json_dumps(value: Any) -> str:
    """Serialize value to JSON; fallback to str() for non-JSON-serializable types."""
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")