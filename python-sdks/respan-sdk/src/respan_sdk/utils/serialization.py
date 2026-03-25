from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Dict, List


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code."""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def serialize_value(value: Any) -> Any:
    """Convert complex values into JSON-safe Python structures.

    This preserves dict/list structure for downstream instrumentations that
    need to shape messages before JSON-encoding them for span attributes.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        normalized_dict: Dict[str, Any] = {}
        for key, nested_value in value.items():
            normalized_dict[str(key)] = serialize_value(nested_value)
        return normalized_dict

    if isinstance(value, (list, tuple, set)):
        normalized_list: List[Any] = []
        for nested_value in value:
            normalized_list.append(serialize_value(nested_value))
        return normalized_list

    if is_dataclass(value):
        return serialize_value(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return serialize_value(model_dump())
        except TypeError:
            pass

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return serialize_value(to_dict())
        except TypeError:
            pass

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return serialize_value(dict_method())
        except TypeError:
            pass

    if hasattr(value, "__dict__"):
        return serialize_value(value.__dict__)

    return str(value)
