"""Cohere instrumentation plugin for Respan."""

import importlib
import logging
from typing import Any

from opentelemetry import trace
from respan_instrumentation_cohere._processor import (
    CohereSpanProcessor,
    insert_span_processor_before_export,
    remove_span_processor,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

COHERE_INSTRUMENTATION_NAME = "cohere"
OTEL_COHERE_MODULE = "opentelemetry.instrumentation.cohere"


def _load_otel_cohere_class() -> type:
    cohere_module = importlib.import_module(OTEL_COHERE_MODULE)
    return cohere_module.CohereInstrumentor


class CohereInstrumentor:
    """Respan instrumentor for the Cohere Python SDK."""

    name = COHERE_INSTRUMENTATION_NAME

    def __init__(
        self,
        *,
        exception_logger: Any | None = None,
        use_legacy_attributes: bool = True,
        **instrumentor_kwargs: Any,
    ) -> None:
        self._constructor_kwargs = {
            "exception_logger": exception_logger,
            "use_legacy_attributes": use_legacy_attributes,
        }
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._instrumentor = None
        self._processor = None
        self._is_instrumented = False
        self._owns_instrumentation = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Cohere via OTEL and add Respan contract normalization."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Cohere instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            cohere_instrumentor_class = _load_otel_cohere_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Cohere instrumentation - missing dependency: %s",
                exc,
            )
            return

        tracer_provider = trace.get_tracer_provider()
        try:
            self._processor = CohereSpanProcessor()
            insert_span_processor_before_export(tracer_provider, self._processor)

            self._instrumentor = cohere_instrumentor_class(**self._constructor_kwargs)
            already_instrumented = bool(
                getattr(
                    self._instrumentor,
                    "is_instrumented_by_opentelemetry",
                    False,
                )
            )
            if not already_instrumented:
                self._instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    **self._instrumentor_kwargs,
                )
                if not bool(
                    getattr(
                        self._instrumentor,
                        "is_instrumented_by_opentelemetry",
                        False,
                    )
                ):
                    remove_span_processor(tracer_provider, self._processor)
                    self._instrumentor = None
                    self._processor = None
                    logger.warning(
                        "Cohere instrumentation skipped because the upstream "
                        "instrumentor did not activate"
                    )
                    return
                self._owns_instrumentation = True
            self._is_instrumented = True
            logger.info("Cohere instrumentation activated")
        except Exception:
            if self._instrumentor is not None and self._owns_instrumentation:
                try:
                    self._instrumentor.uninstrument()
                except Exception:
                    logger.exception("Failed to clean up Cohere instrumentation")
            if self._processor is not None:
                remove_span_processor(tracer_provider, self._processor)
            self._instrumentor = None
            self._processor = None
            self._is_instrumented = False
            self._owns_instrumentation = False
            logger.exception("Failed to activate Cohere instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        tracer_provider = trace.get_tracer_provider()
        if (
            self._is_instrumented
            and self._instrumentor is not None
            and self._owns_instrumentation
        ):
            try:
                self._instrumentor.uninstrument()
            except Exception:
                logger.exception("Failed to deactivate Cohere instrumentation")
        if self._processor is not None:
            remove_span_processor(tracer_provider, self._processor)
        self._instrumentor = None
        self._processor = None
        self._is_instrumented = False
        self._owns_instrumentation = False
        logger.info("Cohere instrumentation deactivated")
