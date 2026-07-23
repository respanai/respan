from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

import respan_instrumentation_mirascope._instrumentation as instrumentation
from respan_instrumentation_mirascope import MirascopeInstrumentor


def _exporter(monkeypatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.mirascope.status"),
    )
    return exporter


def test_backend_error_attributes_preserve_upstream_status(monkeypatch) -> None:
    exporter = _exporter(monkeypatch)
    model = SimpleNamespace(model_id="openai/gpt-4.1-mini")

    class ProviderError(RuntimeError):
        status_code = 503

    def call(self, content):
        raise ProviderError("provider unavailable")

    with pytest.raises(ProviderError):
        instrumentation._call_wrapper(call)(model, "hello")

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["status_code"] == 503
    assert attrs["error.message"] == "provider unavailable"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": "ProviderError",
        "message": "provider unavailable",
        "status": "error",
    }


def test_lifecycle_is_reference_counted_and_idempotent(monkeypatch) -> None:
    installed = 0
    removed = 0

    def install() -> None:
        nonlocal installed
        installed += 1

    def remove() -> None:
        nonlocal removed
        removed += 1

    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(instrumentation, "_install_patches", install)
    monkeypatch.setattr(instrumentation, "_remove_patches", remove)
    monkeypatch.setattr(
        instrumentation.importlib, "import_module", lambda name: object()
    )

    first = MirascopeInstrumentor()
    second = MirascopeInstrumentor()
    first.activate()
    first.activate()
    second.activate()
    assert installed == 1

    first.deactivate()
    first.deactivate()
    assert removed == 0
    second.deactivate()
    assert removed == 1
