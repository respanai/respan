"""OpenRouter instrumentation plugin for Respan."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from respan_instrumentation_openai import OpenAIInstrumentor
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_openrouter._constants import (
    OPENROUTER_INSTRUMENTATION_NAME,
)
from respan_instrumentation_openrouter._processor import OpenRouterSpanProcessor

logger = logging.getLogger(__name__)


def _active_span_processors(tracer_provider: Any) -> tuple[Any, tuple[Any, ...] | None]:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    if processors is None:
        return active_span_processor, None
    return active_span_processor, tuple(processors)


def _register_processor_before_exporters(
    tracer_provider: Any,
    processor: OpenRouterSpanProcessor,
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
        and not isinstance(existing_processor, OpenRouterSpanProcessor)
    )
    active_span_processor._span_processors = (processor, *remaining_processors)


def _unregister_processor(
    tracer_provider: Any,
    processor: OpenRouterSpanProcessor | None,
) -> None:
    if processor is None:
        return

    active_span_processor, processors = _active_span_processors(tracer_provider)
    if active_span_processor is None or processors is None:
        return
    active_span_processor._span_processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )


class OpenRouterInstrumentor:
    """Respan instrumentor for OpenRouter's OpenAI-compatible Python usage."""

    name = OPENROUTER_INSTRUMENTATION_NAME

    def __init__(self, *, normalize_all_openai_spans: bool = True) -> None:
        self._normalize_all_openai_spans = normalize_all_openai_spans
        self._delegate = None
        self._processor: OpenRouterSpanProcessor | None = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument OpenRouter calls made through the OpenAI Python client."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "OpenRouter instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            self._delegate = OpenAIInstrumentor()
            self._delegate.activate()
            if getattr(self._delegate, "_is_instrumented", True) is False:
                self._delegate = None
                return

            self._processor = OpenRouterSpanProcessor(
                normalize_all_openai_spans=self._normalize_all_openai_spans,
            )
            _register_processor_before_exporters(
                tracer_provider=trace.get_tracer_provider(),
                processor=self._processor,
            )
            self._is_instrumented = True
            logger.info("OpenRouter instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up OpenRouter instrumentation")
            self._delegate = None
            self._processor = None
            self._is_instrumented = False
            logger.exception("Failed to activate OpenRouter instrumentation")

    def deactivate(self) -> None:
        """Deactivate OpenRouter instrumentation."""
        _unregister_processor(
            tracer_provider=trace.get_tracer_provider(),
            processor=self._processor,
        )
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate OpenRouter instrumentation")
        self._delegate = None
        self._processor = None
        self._is_instrumented = False
        logger.info("OpenRouter instrumentation deactivated")
