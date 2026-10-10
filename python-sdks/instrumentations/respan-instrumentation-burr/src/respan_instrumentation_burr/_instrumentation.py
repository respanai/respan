"""Owned builder hooks for Burr's synchronous and asynchronous applications."""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import Any

from respan_tracing.core.tracer import RespanTracer
from wrapt import FunctionWrapper

from respan_instrumentation_burr._adapter import _CUSTOM_EXCEPTION, BurrLifecycleAdapter
from respan_instrumentation_burr._constants import (
    BURR_APPLICATION_MODULE,
    BURR_INSTRUMENTATION_NAME,
)

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_ADAPTER: BurrLifecycleAdapter | None = None
_PATCHES: list[tuple[Any, str, Any, Any]] = []


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    return tracer is None or bool(getattr(tracer, "is_enabled", True))


def _ensure_adapter(builder: Any) -> tuple[Any, bool]:
    original = builder.lifecycle_adapters
    if _ADAPTER is None or any(adapter is _ADAPTER for adapter in original):
        return original, False
    builder.lifecycle_adapters = [*original, _ADAPTER]
    return original, True


def _prepare(builder: Any) -> tuple[Any, bool]:
    try:
        return _ensure_adapter(builder)
    except Exception:  # noqa: BLE001 - telemetry attachment cannot replace native build.
        logger.debug("Could not attach Burr telemetry adapter")
        return None, False


def _build_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    original, added = _prepare(instance)
    try:
        return wrapped(*args, **kwargs)
    finally:
        if added:
            instance.lifecycle_adapters = original


async def _abuild_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    original, added = _prepare(instance)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        if added:
            instance.lifecycle_adapters = original


def _span_exit_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    token = _CUSTOM_EXCEPTION.set(args[1] if len(args) > 1 else None)
    try:
        return wrapped(*args, **kwargs)
    finally:
        _CUSTOM_EXCEPTION.reset(token)


async def _async_span_exit_wrapper(
    wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    token = _CUSTOM_EXCEPTION.set(args[1] if len(args) > 1 else None)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _CUSTOM_EXCEPTION.reset(token)


class BurrInstrumentor:
    """Attach a shared lifecycle adapter while retaining builder wrapper ownership."""

    name = BURR_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        global _ACTIVATION_COUNT, _ADAPTER
        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            module = importlib.import_module(BURR_APPLICATION_MODULE)
        except ImportError:
            logger.debug("Burr instrumentation unavailable")
            return
        with _LOCK:
            if _ACTIVATION_COUNT:
                if (
                    _ADAPTER is not None
                    and _ADAPTER.capture_content != self._capture_content
                ):
                    raise RuntimeError(
                        "Burr is already active with a different capture_content setting"
                    )
            else:
                adapter = None
                pending: list[tuple[Any, str, Any, Any]] = []
                try:
                    adapter = BurrLifecycleAdapter(
                        capture_content=self._capture_content
                    )
                    from burr.visibility.tracing import ActionSpanTracer

                    for owner, name, callback in (
                        (module.ApplicationBuilder, "build", _build_wrapper),
                        (module.ApplicationBuilder, "abuild", _abuild_wrapper),
                        (ActionSpanTracer, "__exit__", _span_exit_wrapper),
                        (ActionSpanTracer, "__aexit__", _async_span_exit_wrapper),
                    ):
                        original = inspect.getattr_static(owner, name)
                        wrapper = FunctionWrapper(original, callback)
                        setattr(owner, name, wrapper)
                        pending.append((owner, name, original, wrapper))
                    _ADAPTER = adapter
                    _PATCHES.extend(pending)
                except Exception:
                    for owner, name, original, wrapper in reversed(pending):
                        if inspect.getattr_static(owner, name) is wrapper:
                            setattr(owner, name, original)
                    if adapter is not None:
                        adapter.close()
                    raise
            _ACTIVATION_COUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        global _ACTIVATION_COUNT, _ADAPTER
        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _ACTIVATION_COUNT -= 1
            if _ACTIVATION_COUNT:
                return
            if _ADAPTER is not None:
                _ADAPTER.close()
            for owner, name, original, wrapper in reversed(_PATCHES):
                if inspect.getattr_static(owner, name) is wrapper:
                    setattr(owner, name, original)
            _PATCHES.clear()
            _ADAPTER = None
