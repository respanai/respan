"""Runtime-controlled automatic flush helpers."""

from __future__ import annotations

import os
from threading import Lock, Timer
from typing import Literal

from opentelemetry import trace

AutoFlushPolicy = Literal["off", "root", "always"]

_VALID_POLICIES = {"off", "root", "always"}
_DEFAULT_POLICY: AutoFlushPolicy = "root"
_policy: AutoFlushPolicy = _DEFAULT_POLICY
_timer: Timer | None = None
_timer_lock = Lock()
_DEBOUNCE_SECONDS = 0.05


def _normalize_policy(policy: str | bool | None) -> AutoFlushPolicy:
    if policy is None:
        policy = os.getenv("RESPAN_AUTO_FLUSH", _DEFAULT_POLICY)
    if isinstance(policy, bool):
        return "root" if policy else "off"
    normalized = policy.lower()
    if normalized not in _VALID_POLICIES:
        raise ValueError("auto_flush must be one of 'off', 'root', or 'always'")
    return normalized  # type: ignore[return-value]


def configure_auto_flush(policy: str | bool | None) -> AutoFlushPolicy:
    """Set the process-wide auto-flush policy used by decorators and injection."""
    global _policy
    _policy = _normalize_policy(policy)
    return _policy


def get_auto_flush_policy() -> AutoFlushPolicy:
    return _policy


def has_recording_parent_span() -> bool:
    return trace.get_current_span().is_recording()


def should_flush_after_span(is_root_boundary: bool) -> bool:
    if _policy == "off":
        return False
    if _policy == "always":
        return True
    return is_root_boundary


def flush_now() -> bool:
    provider = _get_active_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if force_flush is None:
        return False
    force_flush()
    return True


def _get_active_tracer_provider():
    try:
        from respan_tracing.core.tracer import RespanTracer

        instance = RespanTracer._instance
        if instance is not None and hasattr(instance, "tracer_provider"):
            return instance.tracer_provider
    except Exception:
        pass
    return trace.get_tracer_provider()


def flush_after_span(is_root_boundary: bool) -> None:
    if should_flush_after_span(is_root_boundary):
        flush_now()


def flush_after_injected_span() -> None:
    if _policy == "off":
        return
    if _policy == "always":
        flush_now()
        return
    _schedule_debounced_flush()


def _schedule_debounced_flush() -> None:
    global _timer
    with _timer_lock:
        if _timer is not None and _timer.is_alive():
            return
        _timer = Timer(_DEBOUNCE_SECONDS, flush_now)
        _timer.start()
