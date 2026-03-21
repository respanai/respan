"""Respan instrumentation for Google ADK traces.

ADK spans are standard OTel ReadableSpan objects. They flow through the
standard OTel pipeline (RespanSpanProcessor -> BatchSpanProcessor ->
RespanSpanExporter) without any custom interception or conversion.
"""
from __future__ import annotations

from typing import Any, Optional


class GoogleAdkInstrumentor:
    """No-op marker for Google ADK instrumentation.

    ADK spans are OTel ReadableSpans handled natively by RespanSpanProcessor.
    This class exists to satisfy the Instrumentation protocol so that users
    can register it via ``Respan(instrumentations=[GoogleAdkInstrumentor()])``.
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

    def activate(self, exporter: Any = None) -> None:
        pass  # No-op: ADK spans are OTel ReadableSpans, handled by RespanSpanProcessor

    def deactivate(self) -> None:
        pass
