"""Lifecycle management for Burr instrumentation."""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from respan_instrumentation_burr._adapter import BurrLifecycleAdapter
from respan_instrumentation_burr._constants import (
    BURR_ADAPTER_MARKER,
    BURR_APPLICATION_BUILD_TARGET,
    BURR_APPLICATION_MODULE,
    BURR_INSTRUMENTATION_NAME,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_ACTIVATION_COUNT = 0
_PATCHED = False
_ADAPTER: BurrLifecycleAdapter | None = None


def _is_respan_tracing_enabled() -> bool:
    tracer = getattr(RespanTracer, "_instance", None)
    if tracer is None:
        return True
    return bool(getattr(tracer, "is_enabled", True))


def _ensure_adapter(builder: Any) -> None:
    if _ADAPTER is None:
        return
    adapters = list(getattr(builder, "lifecycle_adapters", None) or ())
    if not any(
        bool(getattr(adapter, BURR_ADAPTER_MARKER, False)) for adapter in adapters
    ):
        adapters.append(_ADAPTER)
        builder.lifecycle_adapters = adapters


def _build_wrapper(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    _ensure_adapter(instance)
    return wrapped(*args, **kwargs)


class BurrInstrumentor:
    """Attach a Respan lifecycle adapter to Burr applications at build time."""

    name = BURR_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._is_instrumented = False

    def activate(self) -> None:
        """Patch `ApplicationBuilder.build` to add the Burr lifecycle adapter."""
        global _ACTIVATION_COUNT, _ADAPTER, _PATCHED

        if self._is_instrumented or not _is_respan_tracing_enabled():
            return
        try:
            importlib.import_module(BURR_APPLICATION_MODULE)
        except ImportError as exc:
            logger.warning("Burr instrumentation unavailable: %s", exc)
            return

        with _LOCK:
            if _ACTIVATION_COUNT == 0:
                _ADAPTER = BurrLifecycleAdapter(capture_content=self._capture_content)
                wrap_function_wrapper(
                    BURR_APPLICATION_MODULE,
                    BURR_APPLICATION_BUILD_TARGET,
                    _build_wrapper,
                )
                _PATCHED = True
            elif _ADAPTER is not None and (
                _ADAPTER.capture_content != self._capture_content
            ):
                logger.warning(
                    "Burr is already active; the first capture_content setting wins"
                )
            _ACTIVATION_COUNT += 1
            self._is_instrumented = True

    def deactivate(self) -> None:
        """Restore the Burr builder; built applications retain a disabled hook."""
        global _ACTIVATION_COUNT, _ADAPTER, _PATCHED

        if not self._is_instrumented:
            return
        with _LOCK:
            self._is_instrumented = False
            _ACTIVATION_COUNT = max(0, _ACTIVATION_COUNT - 1)
            if _ACTIVATION_COUNT:
                return
            if _ADAPTER is not None:
                _ADAPTER.enabled = False
            if _PATCHED:
                try:
                    unwrap(
                        BURR_APPLICATION_MODULE,
                        BURR_APPLICATION_BUILD_TARGET,
                    )
                except Exception:
                    logger.debug("Failed to unwrap Burr builder", exc_info=True)
            _PATCHED = False
            _ADAPTER = None
