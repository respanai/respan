from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

import respan_instrumentation_ragas._instrumentation as instrumentation
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        instrumentation.trace,
        "get_tracer",
        lambda *args, **kwargs: provider.get_tracer("test.ragas"),
    )
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", True)
    return exporter


def test_sync_metric_success_has_canonical_task_contract(spans) -> None:
    metric = SimpleNamespace(name="exact_match")

    def score(self, sample):
        return 1.0

    wrapped = instrumentation._sync_wrapper(score, kind="metric")
    assert wrapped(metric, {"response": "Paris"}) == 1.0

    span = spans.get_finished_spans()[0]
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert attrs[RESPAN_LOG_METHOD] == "tracing_integration"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "exact_match"
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == 1.0
    assert "traceloop.span.kind" not in attrs


@pytest.mark.asyncio
async def test_async_metric_error_is_recorded_and_reraised(spans) -> None:
    metric = SimpleNamespace(name="failing_metric")

    async def score(self, sample):
        raise RuntimeError("deterministic metric failure")

    wrapped = instrumentation._async_wrapper(score, kind="metric")
    with pytest.raises(RuntimeError, match="deterministic metric failure"):
        await wrapped(metric, {"response": "bad"})

    span = spans.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_capture_content_false_omits_inputs_and_outputs(spans, monkeypatch) -> None:
    monkeypatch.setattr(instrumentation, "_CAPTURE_CONTENT", False)

    def evaluate(dataset, **kwargs):
        return {"secret_result": 1}

    wrapped = instrumentation._sync_wrapper(evaluate, kind="evaluation")
    wrapped([{"secret": "value"}], experiment_name="privacy")

    attrs = spans.get_finished_spans()[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] == "privacy"


def test_sync_to_async_evaluation_creates_one_root_span(spans) -> None:
    async def aevaluate(dataset):
        return {"score": 1}

    wrapped_async = instrumentation._async_wrapper(aevaluate, kind="evaluation")

    def evaluate(dataset):
        return asyncio.run(wrapped_async(dataset))

    wrapped_sync = instrumentation._sync_wrapper(evaluate, kind="evaluation")
    assert wrapped_sync([{"question": "q"}]) == {"score": 1}
    assert len(spans.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_experiment_run_contains_child_row_spans(spans) -> None:
    wrapper = SimpleNamespace(__name__="answer_question")

    async def row(self, item):
        return {"answer": item["answer"]}

    wrapped_row = instrumentation._async_wrapper(row, kind="experiment_row")

    async def run(self, dataset, name=None):
        return [await wrapped_row(self, item) for item in dataset]

    wrapped_run = instrumentation._async_wrapper(run, kind="experiment_run")
    result = await wrapped_run(wrapper, [{"answer": "Paris"}], name="demo")
    assert result == [{"answer": "Paris"}]

    finished = spans.get_finished_spans()
    assert {span.attributes[RESPAN_LOG_TYPE] for span in finished} == {"task"}
    row_span = next(
        span
        for span in finished
        if span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME].startswith(
            "ragas.experiment.row"
        )
    )
    run_span = next(
        span
        for span in finished
        if span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "ragas.experiment.demo"
    )
    assert row_span.parent.span_id == run_span.context.span_id
