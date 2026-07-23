from __future__ import annotations

import asyncio
import json
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from ragas.metrics.collections import ExactMatch, StringPresence

import respan_instrumentation_ragas._instrumentation as instrumentation
from respan_instrumentation_ragas import RagasInstrumentor


@pytest.fixture
def instrumented(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.ragas.collections"),
    )
    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    instrumentor = RagasInstrumentor()
    instrumentor.activate()
    yield exporter
    instrumentor.deactivate()


def test_modern_sync_async_and_batch_surfaces_emit_one_span_each(
    instrumented,
) -> None:
    metric = ExactMatch()
    assert metric.score(reference="Paris", response="Paris").value == 1.0
    assert asyncio.run(metric.ascore(reference="Paris", response="Lyon")).value == 0.0
    batch = metric.batch_score(
        [
            {"reference": "a", "response": "a"},
            {"reference": "b", "response": "c"},
        ]
    )
    assert [result.value for result in batch] == [1.0, 0.0]

    spans = instrumented.get_finished_spans()
    assert len(spans) == 3
    assert spans[-1].attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "exact_match.batch"
    )
    assert all(span.attributes["status_code"] == 200 for span in spans)


def test_modern_metric_failure_has_backend_visible_error(instrumented) -> None:
    with pytest.raises(AssertionError):
        StringPresence().score(reference=1, response="text")
    attrs = instrumented.get_finished_spans()[0].attributes
    assert attrs["status_code"] == 500
    assert "valid reference string" in attrs["error.message"]
    output = json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert output["status"] == "error"
    assert output["error"] == "AssertionError"
    assert "valid reference string" in output["message"]
