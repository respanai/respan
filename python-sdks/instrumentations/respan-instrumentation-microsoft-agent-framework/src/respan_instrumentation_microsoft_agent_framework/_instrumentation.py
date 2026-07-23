"""Microsoft Agent Framework OTEL instrumentation plugin for Respan."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from opentelemetry import trace

from respan_instrumentation_microsoft_agent_framework._constants import (
    AGENT_FRAMEWORK_INSTRUMENTATION_NAME,
)
from respan_instrumentation_microsoft_agent_framework._processor import (
    AgentFrameworkSpanProcessor,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)


def _active_span_processors(tracer_provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    return active_span_processor, processors


def _register_processor(
    tracer_provider: Any,
    processor: AgentFrameworkSpanProcessor,
) -> None:
    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
        return

    remaining_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )
    active_span_processor._span_processors = (processor, *remaining_processors)


def _unregister_processor(
    tracer_provider: Any,
    processor: AgentFrameworkSpanProcessor,
) -> None:
    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        return
    active_span_processor._span_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )


class MicrosoftAgentFrameworkInstrumentor:
    """Respan instrumentor for Microsoft Agent Framework native OTEL spans."""

    name = AGENT_FRAMEWORK_INSTRUMENTATION_NAME

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content
        self._processor: AgentFrameworkSpanProcessor | None = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def _enable_agent_framework_observability(self) -> bool:
        try:
            observability = importlib.import_module("agent_framework.observability")
        except ImportError as exc:
            logger.warning(
                "Failed to activate Microsoft Agent Framework instrumentation - "
                "missing dependency: %s",
                exc,
            )
            return False

        settings = getattr(observability, "OBSERVABILITY_SETTINGS", None)
        if bool(getattr(settings, "is_user_disabled", False)):
            logger.info(
                "Microsoft Agent Framework instrumentation skipped because "
                "Agent Framework observability is user-disabled"
            )
            return False

        enable_instrumentation = getattr(observability, "enable_instrumentation", None)
        if callable(enable_instrumentation):
            try:
                enable_instrumentation(enable_sensitive_data=self._capture_content)
                return True
            except TypeError:
                enable_instrumentation()

        if self._capture_content:
            enable_sensitive_telemetry = getattr(
                observability,
                "enable_sensitive_telemetry",
                None,
            )
            if callable(enable_sensitive_telemetry):
                enable_sensitive_telemetry()
            elif settings is not None and hasattr(settings, "enable_sensitive_data"):
                settings.enable_sensitive_data = True
        return True

    def activate(self) -> None:
        """Activate Respan normalization for Microsoft Agent Framework spans."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Microsoft Agent Framework instrumentation skipped because "
                "Respan tracing is disabled"
            )
            return

        if not self._enable_agent_framework_observability():
            return

        if self._processor is None:
            self._processor = AgentFrameworkSpanProcessor()

        _register_processor(
            tracer_provider=trace.get_tracer_provider(),
            processor=self._processor,
        )
        self._is_instrumented = True
        logger.info("Microsoft Agent Framework instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate Respan normalization for Microsoft Agent Framework spans."""
        if not self._is_instrumented:
            return

        if self._processor is not None:
            _unregister_processor(
                tracer_provider=trace.get_tracer_provider(),
                processor=self._processor,
            )
        self._is_instrumented = False
        logger.info("Microsoft Agent Framework instrumentation deactivated")
