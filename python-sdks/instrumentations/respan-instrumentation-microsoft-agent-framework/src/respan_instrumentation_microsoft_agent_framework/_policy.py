"""Shared, reversible native telemetry policy for Agent Framework."""

import os
from functools import wraps
from threading import RLock

from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

_lock = RLock()
_users = 0
_epoch = 0
_capture = True
_settings = None
_original_values = {}
_owned_values = {}
_patches = []


def capture_content(parent_context=None):
    if not _capture:
        return False
    if (
        context.get_value(ENABLE_CONTENT_TRACING_KEY) is False
        or context.get_value(ENABLE_CONTENT_TRACING_KEY, parent_context) is False
    ):
        return False
    return os.getenv("TRACELOOP_TRACE_CONTENT", "true").strip().lower() not in {
        "false",
        "0",
        "no",
        "off",
    }


def _restore():
    global _settings
    for owner, name, original, replacement in reversed(_patches):
        if getattr(owner, name, None) is replacement:
            setattr(owner, name, original)
    _patches.clear()
    if _settings is not None and not getattr(_settings, "is_user_disabled", False):
        for name, value in _original_values.items():
            if getattr(_settings, name, None) == _owned_values.get(name):
                setattr(_settings, name, value)
    _original_values.clear()
    _owned_values.clear()
    _settings = None


def acquire(observability, capture):
    global _users, _capture, _settings, _epoch
    with _lock:
        settings = getattr(observability, "OBSERVABILITY_SETTINGS", None)
        if getattr(settings, "is_user_disabled", False):
            return False
        if _users:
            if settings is not _settings:
                raise RuntimeError(
                    "Agent Framework observability settings changed while active"
                )
            if capture != _capture:
                raise ValueError(
                    "Active Agent Framework instrumentors must use the same capture_content policy"
                )
            _users += 1
            return True
        _settings = settings
        _capture = capture
        _epoch += 1
        epoch = _epoch
        try:
            for name in ("enable_instrumentation", "enable_sensitive_data"):
                if hasattr(settings, name):
                    _original_values[name] = getattr(settings, name)
            enable = getattr(observability, "enable_instrumentation", None)
            if callable(enable):
                enable(enable_sensitive_data=capture)
            for name in _original_values:
                _owned_values[name] = getattr(settings, name)
            cls = type(settings)
            for name in ("ENABLED", "SENSITIVE_DATA_ENABLED"):
                original = getattr(cls, name, None)
                if not isinstance(original, property):
                    continue

                def getter(self, _original=original, _name=name):
                    value = _original.__get__(self, type(self))
                    if self is not _settings or not _users or epoch != _epoch:
                        return value
                    return (
                        value
                        and is_instrumentation_enabled()
                        and (_name != "SENSITIVE_DATA_ENABLED" or capture_content())
                    )

                replacement = property(
                    getter, original.fset, original.fdel, original.__doc__
                )
                setattr(cls, name, replacement)
                _patches.append((cls, name, original, replacement))
            original_tracer = getattr(observability, "get_tracer", None)
            if callable(original_tracer):

                @wraps(original_tracer)
                def get_tracer(*args, **kwargs):
                    if _users and epoch == _epoch and not is_instrumentation_enabled():
                        return trace.NoOpTracer()
                    return original_tracer(*args, **kwargs)

                observability.get_tracer = get_tracer
                _patches.append(
                    (observability, "get_tracer", original_tracer, get_tracer)
                )
            from respan_instrumentation_microsoft_agent_framework._embeddings import (
                install,
            )

            embedding_patch = install(
                observability, lambda: _users > 0 and epoch == _epoch
            )
            if embedding_patch is not None:
                _patches.append(embedding_patch)
            _users = 1
        except BaseException:
            # A native hook may change a setting before raising.
            for name in _original_values:
                _owned_values[name] = getattr(settings, name)
            _restore()
            _capture = True
            raise
        return True


def release():
    global _users, _capture
    with _lock:
        if not _users:
            return
        _users -= 1
        if not _users:
            _restore()
            _capture = True
