"""Respan plugin for direct PydanticAI span instrumentation."""

import logging
from typing import Any

from opentelemetry import trace

from respan_instrumentation_pydantic_ai._processor import PydanticAISpanProcessor

logger = logging.getLogger(__name__)

_UNSET = object()


class PydanticAIInstrumentor:
    """Respan instrumentor for PydanticAI.

    Enables PydanticAI's OpenTelemetry emission and registers a local
    processor that maps native PydanticAI attributes directly into the
    Respan/Traceloop conventions used by the OTLP pipeline.
    """

    name = "pydantic-ai"

    def __init__(
        self,
        agent: Any | None = None,
        *,
        include_content: bool = True,
        include_binary_content: bool = True,
        version: int = 5,
    ) -> None:
        self._agent = agent
        self._include_content = include_content
        self._include_binary_content = include_binary_content
        self._version = version
        self._agent_class = None
        self._processor = None
        self._is_instrumented = False
        self._previous_agent_instrument: Any = _UNSET
        self._previous_global_instrument: Any = _UNSET

    @staticmethod
    def _register_processor(
        tracer_provider, processor: PydanticAISpanProcessor
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )

        if processors is not None:
            active_span_processor._span_processors = tuple(
                existing_processor
                for existing_processor in processors
                if existing_processor is not processor
            )
            active_span_processor._span_processors = (
                *active_span_processor._span_processors,
                processor,
            )
            return

        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)

    @staticmethod
    def _unregister_processor(
        tracer_provider, processor: PydanticAISpanProcessor
    ) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )

    def activate(self) -> None:
        if self._is_instrumented:
            return

        try:
            from pydantic_ai.agent import Agent
            from pydantic_ai.models.instrumented import InstrumentationSettings
        except ImportError as exc:
            logger.warning(
                "Failed to activate PydanticAI instrumentation — missing dependency: %s",
                exc,
            )
            return

        if self._processor is None:
            self._processor = PydanticAISpanProcessor()
        self._register_processor(trace.get_tracer_provider(), self._processor)

        self._agent_class = Agent
        settings = InstrumentationSettings(
            tracer_provider=trace.get_tracer_provider(),
            include_content=self._include_content,
            include_binary_content=self._include_binary_content,
            version=self._version,
        )

        if self._agent is None:
            self._previous_global_instrument = getattr(Agent, "_instrument_default", _UNSET)
            Agent.instrument_all(instrument=settings)
        else:
            self._previous_agent_instrument = getattr(self._agent, "instrument", _UNSET)
            self._agent.instrument = settings

        self._is_instrumented = True
        logger.info("PydanticAI instrumentation activated")

    def deactivate(self) -> None:
        if not self._is_instrumented:
            return

        try:
            if self._agent is None and self._agent_class is not None:
                previous_instrument = (
                    False
                    if self._previous_global_instrument is _UNSET
                    else self._previous_global_instrument
                )
                self._agent_class.instrument_all(instrument=previous_instrument)
            elif self._agent is not None:
                previous_instrument = (
                    None
                    if self._previous_agent_instrument is _UNSET
                    else self._previous_agent_instrument
                )
                self._agent.instrument = previous_instrument
        finally:
            if self._processor is not None:
                self._unregister_processor(trace.get_tracer_provider(), self._processor)

        self._is_instrumented = False
        logger.info("PydanticAI instrumentation deactivated")
