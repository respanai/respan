"""Portkey instrumentation plugin for Respan."""

import importlib
import logging
from typing import Any

from opentelemetry import trace

from respan_instrumentation_openinference import OpenInferenceInstrumentor
from respan_instrumentation_openinference._translator import OpenInferenceTranslator
from respan_tracing.core.tracer import RespanTracer

from respan_instrumentation_portkey._constants import (
    OPENINFERENCE_PORTKEY_MODULE,
    PORTKEY_INSTRUMENTATION_NAME,
)
from respan_instrumentation_portkey._processor import PortkeySpanContractProcessor

logger = logging.getLogger(__name__)


def _load_openinference_portkey_class() -> type:
    portkey_module = importlib.import_module(OPENINFERENCE_PORTKEY_MODULE)
    return portkey_module.PortkeyInstrumentor


def _get_active_span_processors(tracer_provider) -> tuple[Any, ...] | None:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    processors = (
        getattr(active_span_processor, "_span_processors", None)
        if active_span_processor is not None
        else None
    )
    if processors is None:
        return None
    return tuple(processors)


def _set_active_span_processors(tracer_provider, processors: tuple[Any, ...]) -> bool:
    active_span_processor = getattr(tracer_provider, "_active_span_processor", None)
    if active_span_processor is None:
        return False
    if getattr(active_span_processor, "_span_processors", None) is None:
        return False
    active_span_processor._span_processors = processors
    return True


def _register_processor_after_translator(tracer_provider, processor) -> None:
    processors = _get_active_span_processors(tracer_provider)
    if processors is None:
        if hasattr(tracer_provider, "add_span_processor"):
            tracer_provider.add_span_processor(processor)
        return

    processors = tuple(
        existing_processor
        for existing_processor in processors
        if existing_processor is not processor
    )

    for index, existing_processor in enumerate(processors):
        if isinstance(existing_processor, OpenInferenceTranslator):
            _set_active_span_processors(
                tracer_provider=tracer_provider,
                processors=(
                    *processors[: index + 1],
                    processor,
                    *processors[index + 1 :],
                ),
            )
            return

    _set_active_span_processors(
        tracer_provider=tracer_provider,
        processors=(*processors, processor),
    )


def _unregister_processor(tracer_provider, processor) -> None:
    processors = _get_active_span_processors(tracer_provider)
    if processors is None:
        return
    _set_active_span_processors(
        tracer_provider=tracer_provider,
        processors=tuple(
            existing_processor
            for existing_processor in processors
            if existing_processor is not processor
        ),
    )


class PortkeyInstrumentor:
    """Respan instrumentor for Portkey.

    Activates the OpenInference Portkey instrumentor and registers Respan's
    OpenInference translator so Portkey spans reach the Respan OTLP pipeline
    with canonical tracing fields.
    """

    name = PORTKEY_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = instrumentor_kwargs
        self._delegate = None
        self._contract_processor = PortkeySpanContractProcessor()
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Portkey through OpenInference and Respan's translator."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info("Portkey instrumentation skipped because Respan tracing is disabled")
            return

        try:
            portkey_instrumentor_class = _load_openinference_portkey_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Portkey instrumentation - missing dependency: %s",
                exc,
            )
            return

        try:
            self._delegate = OpenInferenceInstrumentor(
                portkey_instrumentor_class,
                **self._instrumentor_kwargs,
            )
            self._delegate.activate()
            _register_processor_after_translator(
                tracer_provider=trace.get_tracer_provider(),
                processor=self._contract_processor,
            )
            self._is_instrumented = True
            logger.info("Portkey instrumentation activated")
        except Exception:
            if self._delegate is not None:
                try:
                    self._delegate.deactivate()
                except Exception:
                    logger.exception("Failed to clean up Portkey instrumentation")
            self._delegate = None
            self._is_instrumented = False
            logger.exception("Failed to activate Portkey instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        _unregister_processor(
            tracer_provider=trace.get_tracer_provider(),
            processor=self._contract_processor,
        )
        if self._is_instrumented and self._delegate is not None:
            try:
                self._delegate.deactivate()
            except Exception:
                logger.exception("Failed to deactivate Portkey instrumentation")
        self._delegate = None
        self._is_instrumented = False
        logger.info("Portkey instrumentation deactivated")
