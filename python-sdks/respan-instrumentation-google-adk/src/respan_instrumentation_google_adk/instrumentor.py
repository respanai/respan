"""Respan instrumentation for Google ADK traces.

Patches OTel span processors to intercept ADK spans, enrich them with
traceloop attributes, and pass enriched versions through the normal
OTEL pipeline (RespanSpanExporter -> /v2/traces).

Follows the same wrapt-based processor patching pattern as
respan-exporter-agno.
"""
from __future__ import annotations

from respan_tracing.utils.logging import get_respan_logger

from .processor import patch_span_processors

logger = get_respan_logger("instrumentation.google_adk")


class GoogleAdkInstrumentor:
    """Instrument OTel span processors to enrich Google ADK traces.

    Patches BatchSpanProcessor._export() and SimpleSpanProcessor.on_end()
    to intercept ADK spans, add traceloop attributes, and pass enriched
    versions through the normal OTEL pipeline.

    Usage::

        from respan import Respan
        from respan_instrumentation_google_adk import GoogleAdkInstrumentor

        respan = Respan(
            instrumentations=[GoogleAdkInstrumentor()],
            environment="development",
            customer_identifier="demo-user",
        )
    """

    name = "google-adk"

    def activate(self) -> None:
        """Patch span processors for ADK enrichment."""
        patch_span_processors()
        logger.info("Google ADK instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate instrumentation (wrapt patches are permanent)."""
        logger.info("Google ADK instrumentation deactivated")
