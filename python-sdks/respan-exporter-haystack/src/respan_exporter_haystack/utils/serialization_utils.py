"""Serialization helper utilities.

Note: This mirrors the safe_json_dumps pattern used in other exporters (e.g. SuperAgent,
Dify). When respan-sdk exposes a shared helper (e.g. in respan_sdk.utils.export or
respan_sdk.utils.serialization), prefer using that for consistency.
"""

import json
from typing import Any


def serialize_data(data: Any) -> str:
    """Serialize values for trace payload logging fields."""
    try:
        if isinstance(data, (str, int, float, bool)):
            return str(data)
        return json.dumps(data, default=str)
    except Exception:
        return str(data)
