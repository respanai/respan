"""Test utilities for respan-tracing.

Provides InMemorySpanExporter for verifying spans in integration tests
without making real HTTP calls to the ingest endpoint.
"""

import threading
from typing import List, Sequence, Tuple

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class InMemorySpanExporter(SpanExporter):
    """SpanExporter that stores spans in memory for test assertions.

    Usage::

        from respan_tracing.testing import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        telemetry = RespanTelemetry(
            app_name="test",
            api_key="test-key",
            custom_exporter=exporter,
            is_enabled=True,
        )

        # ... run instrumented code ...

        telemetry.flush()
        spans = exporter.get_finished_spans()
        assert len(spans) > 0
    """

    def __init__(self) -> None:
        self._finished_spans: List[ReadableSpan] = []
        self._stopped = False
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._finished_spans.clear()

    def get_finished_spans(self) -> Tuple[ReadableSpan, ...]:
        with self._lock:
            return tuple(self._finished_spans)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self._stopped:
            return SpanExportResult.FAILURE
        with self._lock:
            self._finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._stopped = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
