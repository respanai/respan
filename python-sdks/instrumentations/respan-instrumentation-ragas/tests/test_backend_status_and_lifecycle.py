from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

import respan_instrumentation_ragas._instrumentation as instrumentation
from respan_instrumentation_ragas import RagasInstrumentor


def _exporter(monkeypatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.ragas.status"),
    )
    return exporter


def test_backend_error_attributes_preserve_upstream_status(monkeypatch) -> None:
    exporter = _exporter(monkeypatch)

    class RateLimitError(RuntimeError):
        status_code = 429

    def evaluate(dataset):
        raise RateLimitError("ragas rate limited")

    wrapped = instrumentation._sync_wrapper(evaluate, kind="evaluation")
    with pytest.raises(RateLimitError):
        wrapped([{"question": "q"}])

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["status_code"] == 429
    assert attrs["error.message"] == "ragas rate limited"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == {
        "error": "RateLimitError",
        "message": "ragas rate limited",
        "status": "error",
    }


def test_success_sets_backend_status(monkeypatch) -> None:
    exporter = _exporter(monkeypatch)

    class Result:
        status_code = 201

    wrapped = instrumentation._sync_wrapper(lambda dataset: Result(), kind="evaluation")
    wrapped([])
    assert exporter.get_finished_spans()[0].attributes["status_code"] == 201


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

    first = RagasInstrumentor()
    second = RagasInstrumentor()
    first.activate()
    first.activate()
    second.activate()
    assert installed == 1

    first.deactivate()
    first.deactivate()
    assert removed == 0
    second.deactivate()
    assert removed == 1
