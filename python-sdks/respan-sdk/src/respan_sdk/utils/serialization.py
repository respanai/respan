from datetime import date, datetime
from typing import Any


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))


def safe_serialize(obj: Any) -> Any:
    """Recursively convert *obj* to plain JSON-serializable Python types.

    Handles Pydantic models (v2 ``model_dump``), dicts, lists, tuples,
    datetime objects, and arbitrary types (via ``str()``).

    Pydantic v2 defers serializer construction for models with forward
    references (``MockValSer``).  The deferred rebuild uses
    ``sys._getframe(5)`` which fails in shallow call stacks (Celery
    workers, asyncio callbacks).  By pre-serializing foreign Pydantic
    model instances before assigning them to SDK param objects, callers
    sidestep the issue entirely.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return {
                k: safe_serialize(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def safe_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get an attribute from a Pydantic model or dict, with a fallback default.

    Tries ``getattr`` first (for Pydantic models / objects), then falls
    back to ``dict.get`` if *obj* is a dict.  Returns *default* when the
    value is ``None``.
    """
    val = getattr(obj, key, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(key, default)
    return val if val is not None else default