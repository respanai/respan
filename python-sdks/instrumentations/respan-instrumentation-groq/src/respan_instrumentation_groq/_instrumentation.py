"""Groq instrumentation plugin for Respan."""

import importlib
import logging
from typing import Any

from opentelemetry import trace

from respan_instrumentation_groq._processor import GroqSpanProcessor
from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

GROQ_INSTRUMENTATION_NAME = "groq"
OPENINFERENCE_GROQ_MODULE = "openinference.instrumentation.groq"
OPENINFERENCE_GROQ_INSTRUMENTOR_CLASS_NAME = "GroqInstrumentor"


def _load_openinference_groq_class() -> type:
    groq_module = importlib.import_module(OPENINFERENCE_GROQ_MODULE)
    return getattr(groq_module, OPENINFERENCE_GROQ_INSTRUMENTOR_CLASS_NAME)


class GroqInstrumentor:
    """Respan instrumentor for the Groq Python SDK.

    The Groq SDK is instrumented by the upstream OpenInference package. This
    adapter only wires that instrumentor into Respan's plugin lifecycle and
    OpenInference translation pipeline.
    """

    name = GROQ_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._processor: GroqSpanProcessor | None = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    @staticmethod
    def _register_processor(tracer_provider: Any, processor: GroqSpanProcessor) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if active_span_processor is None or processors is None:
            if hasattr(tracer_provider, "add_span_processor"):
                tracer_provider.add_span_processor(processor)
            return

        remaining_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )
        translator = getattr(OpenInferenceInstrumentor, "_translator", None)

        processor_chain = list(remaining_processors)
        insert_index = 0
        if translator is not None:
            for index, existing_processor in enumerate(processor_chain):
                if existing_processor is translator:
                    insert_index = index + 1
                    break
        processor_chain.insert(insert_index, processor)
        active_span_processor._span_processors = tuple(processor_chain)

    @staticmethod
    def _unregister_processor(tracer_provider: Any, processor: GroqSpanProcessor) -> None:
        active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
        processors = (
            getattr(active_span_processor, "_span_processors", None)
            if active_span_processor is not None
            else None
        )
        if active_span_processor is None or processors is None:
            return
        active_span_processor._span_processors = tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        )

    def activate(self) -> None:
        """Instrument Groq through OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Groq instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            groq_instrumentor_class = _load_openinference_groq_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Groq instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            self._delegate = OpenInferenceInstrumentor(
                groq_instrumentor_class,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            self._processor = GroqSpanProcessor()
            self._register_processor(trace.get_tracer_provider(), self._processor)
            self._is_instrumented = True
            logger.info("Groq instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up Groq instrumentation")
            self._delegate = None
            self._processor = None
            self._is_instrumented = False
            logger.exception("Failed to activate Groq instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        if self._processor is not None:
            self._unregister_processor(trace.get_tracer_provider(), self._processor)
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate Groq instrumentation")
        self._delegate = None
        self._processor = None
        self._is_instrumented = False
        logger.info("Groq instrumentation deactivated")
