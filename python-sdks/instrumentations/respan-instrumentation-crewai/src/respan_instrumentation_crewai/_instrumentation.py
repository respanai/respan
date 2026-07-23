"""Native CrewAI instrumentation plugin for Respan."""

from __future__ import annotations

import logging
import threading
from typing import Any, ClassVar

from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_crewai._constants import CREWAI_INSTRUMENTATION_NAME

logger = logging.getLogger(__name__)


class CrewAIInstrumentor:
    """Subscribe to CrewAI lifecycle events and emit canonical Respan spans."""

    name = CREWAI_INSTRUMENTATION_NAME

    _activation_lock: ClassVar[threading.RLock] = threading.RLock()
    _active_owner: ClassVar[CrewAIInstrumentor | None] = None

    def __init__(self) -> None:
        self._listener: Any = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Activate the package-owned CrewAI event listener exactly once."""
        with self._activation_lock:
            if self._is_instrumented:
                return
            active_owner = type(self)._active_owner
            if active_owner is not None and active_owner._is_instrumented:
                logger.info("CrewAI instrumentation is already active")
                return
            if not self._is_respan_tracing_enabled():
                logger.info(
                    "CrewAI instrumentation skipped because Respan tracing is disabled"
                )
                return

            listener = None
            try:
                from respan_instrumentation_crewai._event_listener import (
                    CrewAIEventListener,
                )

                listener = CrewAIEventListener()
            except ImportError as exc:
                logger.warning(
                    "Failed to activate CrewAI instrumentation — missing dependency: %s",
                    exc,
                )
                return
            except Exception:
                if listener is not None:
                    try:
                        listener.shutdown()
                    except Exception:
                        logger.exception("Failed to clean up CrewAI instrumentation")
                logger.exception("Failed to activate CrewAI instrumentation")
                return

            self._listener = listener
            self._is_instrumented = True
            type(self)._active_owner = self
            logger.info("CrewAI instrumentation activated")

    def deactivate(self) -> None:
        """Unsubscribe and restore CrewAI's original runtime methods."""
        with self._activation_lock:
            if not self._is_instrumented:
                return
            listener = self._listener
            self._listener = None
            self._is_instrumented = False
            if type(self)._active_owner is self:
                type(self)._active_owner = None

            if listener is not None:
                try:
                    listener.shutdown()
                except Exception:
                    logger.exception("Failed to deactivate CrewAI instrumentation")
            logger.info("CrewAI instrumentation deactivated")
