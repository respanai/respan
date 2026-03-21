"""Respan instrumentation for Google ADK traces.

Patches OTel span processors to intercept ADK spans, enrich them with
traceloop attributes, and pass enriched versions through the normal
OTEL pipeline (RespanSpanExporter -> /v2/traces).

Follows the same wrapt-based processor patching pattern as
respan-exporter-agno.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .processor import patch_span_processors

logger = logging.getLogger(__name__)


class GoogleAdkInstrumentor:
    """Instrument OTel span processors to enrich Google ADK traces.

    Patches BatchSpanProcessor._export() and SimpleSpanProcessor.on_end()
    to intercept ADK spans, add traceloop attributes, and pass enriched
    versions through the normal OTEL pipeline.

    Usage::

        from respan import Respan
        from respan_instrumentation_google_adk import GoogleAdkInstrumentor

        respan = Respan(instrumentations=[GoogleAdkInstrumentor()])
    """

    name = "google-adk"

    def __init__(
        self,
        environment: Optional[str] = None,
        customer_identifier: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.environment = environment
        self.customer_identifier = customer_identifier

    def activate(self) -> None:
        """Patch span processors for ADK enrichment."""
        patch_span_processors()
        logger.info("Google ADK instrumentation activated")

    def deactivate(self) -> None:
        """Deactivate instrumentation (wrapt patches are permanent)."""
        logger.info("Google ADK instrumentation deactivated")
