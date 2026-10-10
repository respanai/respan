import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_burr._adapter import _ACTIVE_SPANS


@pytest.fixture
def telemetry(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    _ACTIVE_SPANS.set(())
    yield provider, exporter
    provider.shutdown()
    assert _ACTIVE_SPANS.get() == ()
