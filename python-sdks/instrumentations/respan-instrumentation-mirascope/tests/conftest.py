from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_mirascope import MirascopeInstrumentor
from respan_tracing.core.tracer import RespanTracer


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(RespanTracer, "_instance", None)
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = MirascopeInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    yield instrumentor, provider, exporter
    instrumentor.deactivate()
    provider.shutdown()
