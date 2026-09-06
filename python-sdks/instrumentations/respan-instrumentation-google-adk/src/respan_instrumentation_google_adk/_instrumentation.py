"""Google ADK instrumentation plugin for Respan."""

import importlib
import logging
from typing import Any

from opentelemetry import trace
from respan_instrumentation_google_adk._compat import patch_legacy_agent_iterator
from respan_instrumentation_google_adk._processor import (
    GoogleADKSpanProcessor,
    insert_span_processor_before_export,
    remove_span_processor,
)
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

GOOGLE_ADK_INSTRUMENTATION_NAME = "google-adk"
OPENINFERENCE_GOOGLE_ADK_MODULE = "openinference.instrumentation.google_adk"


def _load_openinference_google_adk_class() -> type:
    google_adk_module = importlib.import_module(OPENINFERENCE_GOOGLE_ADK_MODULE)
    return google_adk_module.GoogleADKInstrumentor


class GoogleADKInstrumentor:
    """Respan instrumentor for Google ADK.

    Activates the OpenInference Google ADK instrumentor and registers a
    Google-ADK-specific span processor so ADK spans reach the Respan OTEL
    pipeline with canonical ``traceloop.*`` and ``gen_ai.*`` fields.
    """

    name = GOOGLE_ADK_INSTRUMENTATION_NAME

    def __init__(self, **instrumentor_kwargs: Any) -> None:
        self._instrumentor_kwargs = dict(instrumentor_kwargs)
        self._instrumentor = None
        self._processor = None
        self._undo_legacy_iterator = None
        self._is_instrumented = False

    @staticmethod
    def _is_respan_tracing_enabled() -> bool:
        tracer = getattr(RespanTracer, "_instance", None)
        if tracer is None:
            return True
        return bool(getattr(tracer, "is_enabled", True))

    def activate(self) -> None:
        """Instrument Google ADK via OpenInference and Respan's ADK processor."""
        if self._is_instrumented:
            return

        if not self._is_respan_tracing_enabled():
            logger.info(
                "Google ADK instrumentation skipped because Respan tracing is disabled"
            )
            return

        try:
            google_adk_instrumentor_class = _load_openinference_google_adk_class()
        except ImportError as exc:
            logger.warning(
                "Failed to activate Google ADK instrumentation - missing dependency: %s",
                exc,
            )
            return

        tracer_provider = trace.get_tracer_provider()
        try:
            upstream = google_adk_instrumentor_class()
            if getattr(upstream, "is_instrumented_by_opentelemetry", False):
                logger.warning(
                    "Google ADK instrumentation is already active under another "
                    "owner; deactivate that owner before activating this adapter"
                )
                return
            self._processor = GoogleADKSpanProcessor()
            insert_span_processor_before_export(tracer_provider, self._processor)
            self._instrumentor = upstream
            self._instrumentor.instrument(
                tracer_provider=tracer_provider,
                **self._instrumentor_kwargs,
            )
            # OTel instrumentors may log dependency conflicts and return without
            # raising. Do not report an active adapter or retain its processor.
            if not getattr(
                self._instrumentor, "is_instrumented_by_opentelemetry", True
            ):
                raise RuntimeError(
                    "OpenInference Google ADK instrumentation did not activate"
                )
            self._undo_legacy_iterator = patch_legacy_agent_iterator()
            self._is_instrumented = True
            logger.info("Google ADK instrumentation activated")
        except Exception:
            if self._undo_legacy_iterator is not None:
                self._undo_legacy_iterator()
                self._undo_legacy_iterator = None
            if self._instrumentor is not None:
                try:
                    self._instrumentor.uninstrument()
                except Exception:
                    logger.exception("Failed to clean up Google ADK instrumentation")
            if self._processor is not None:
                remove_span_processor(tracer_provider, self._processor)
            self._instrumentor = None
            self._processor = None
            self._is_instrumented = False
            logger.exception("Failed to activate Google ADK instrumentation")

    def deactivate(self) -> None:
        """Deactivate the instrumentation."""
        tracer_provider = trace.get_tracer_provider()
        if self._undo_legacy_iterator is not None:
            self._undo_legacy_iterator()
            self._undo_legacy_iterator = None
        if self._is_instrumented and self._instrumentor is not None:
            try:
                self._instrumentor.uninstrument()
            except Exception:
                logger.exception("Failed to deactivate Google ADK instrumentation")
        if self._processor is not None:
            remove_span_processor(tracer_provider, self._processor)
        self._instrumentor = None
        self._processor = None
        self._is_instrumented = False
        logger.info("Google ADK instrumentation deactivated")
