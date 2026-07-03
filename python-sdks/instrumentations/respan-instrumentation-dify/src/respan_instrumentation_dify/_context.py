"""Package-local context for Dify-specific call options."""

from __future__ import annotations

import contextvars
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any


_RESPAN_PARAMS: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "respan_dify_params",
    default={},
)


def _to_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                converted = method()
            except Exception:
                continue
            if isinstance(converted, Mapping):
                return converted
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return value_dict
    return None


@contextmanager
def use_respan_params(value: Any):
    """Temporarily attach optional Respan params to the current Dify call."""
    mapping = _to_mapping(value) or {}
    parent = _RESPAN_PARAMS.get()
    merged = {**parent, **dict(mapping)}
    token = _RESPAN_PARAMS.set(merged)
    try:
        yield
    finally:
        _RESPAN_PARAMS.reset(token)


def read_respan_params() -> dict[str, Any]:
    return dict(_RESPAN_PARAMS.get())
