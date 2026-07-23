from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode
from temporalio.contrib.opentelemetry import TracingInterceptor

from respan_instrumentation_temporal import TemporalInstrumentor
from respan_instrumentation_temporal import _instrumentation
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


class _FakeSpan:
    def __init__(self, name: str, attributes=None):
        self.name = name
        self.attributes = dict(attributes or {})
        self.status = None
        self.exceptions = []
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exc, *args, **kwargs):
        self.exceptions.append(exc)

    def end(self, *args, **kwargs):
        self.ended = True

    def get_span_context(self):
        return _instrumentation.trace.INVALID_SPAN_CONTEXT

    def is_recording(self):
        return True


class _FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, *args, **kwargs):
        span = _FakeSpan(name, kwargs.get("attributes"))
        self.spans.append(span)
        yield span

    def start_span(self, name, *args, **kwargs):
        span = _FakeSpan(name, kwargs.get("attributes"))
        self.spans.append(span)
        return span


@pytest.fixture(autouse=True)
def reset_instrumentor():
    TemporalInstrumentor._patches_applied = False
    TemporalInstrumentor._activation_count = 0
    TemporalInstrumentor._patched_targets.clear()
    yield
    TemporalInstrumentor._patches_applied = False
    TemporalInstrumentor._activation_count = 0
    TemporalInstrumentor._patched_targets.clear()


def _make_interceptor(*, capture_content=True):
    tracer = _FakeTracer()
    interceptor = _instrumentation._build_interceptor(
        TracingInterceptor,
        tracer=tracer,
        capture_content=capture_content,
        max_attribute_chars=16_000,
        always_create_workflow_spans=False,
    )
    return tracer, interceptor


def _assert_contract(attrs, log_type):
    assert attrs[RESPAN_LOG_TYPE] == log_type
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME]
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH]
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in attrs
    for banned_alias in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "span_tools",
        "has_tool_calls",
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
    ):
        assert banned_alias not in attrs


def test_workflow_start_maps_to_canonical_workflow_span():
    tracer, interceptor = _make_interceptor()
    temporal_input = SimpleNamespace(
        args=("Ada",),
        id="greeting-workflow-1",
        workflow="GreetingWorkflow",
        headers={},
    )

    with interceptor._start_as_current_span(
        "StartWorkflow:GreetingWorkflow",
        attributes={"temporalWorkflowID": "greeting-workflow-1"},
        input_with_headers=temporal_input,
        kind=SpanKind.CLIENT,
    ):
        pass

    span = tracer.spans[-1]
    _assert_contract(span.attributes, "workflow")
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == (
        "temporal.start_workflow.GreetingWorkflow"
    )
    assert (
        "greeting-workflow-1" in span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    assert "temporalWorkflowID" not in span.attributes
    assert span.attributes["status_code"] == 200


def test_activity_error_records_status_and_backend_error_fields():
    tracer, interceptor = _make_interceptor()

    with pytest.raises(RuntimeError, match="activity exploded"):
        with interceptor._start_as_current_span(
            "RunActivity:compose_greeting",
            attributes={"temporalActivityID": "activity-1"},
            kind=SpanKind.SERVER,
        ):
            raise RuntimeError("activity exploded")

    span = tracer.spans[-1]
    _assert_contract(span.attributes, "task")
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "activity exploded"
    assert (
        "activity exploded" in span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )


def test_completed_workflow_span_proxy_handles_success_and_error():
    tracer, interceptor = _make_interceptor()
    successful = interceptor.tracer.start_span(
        "CompleteWorkflow:GreetingWorkflow",
        attributes={"temporalRunID": "run-1"},
    )
    successful.end()
    assert successful.attributes[RESPAN_LOG_TYPE] == "workflow"
    assert successful.attributes["status_code"] == 200

    failed = interceptor.tracer.start_span(
        "CompleteWorkflow:BrokenWorkflow",
        attributes={"temporalRunID": "run-2"},
    )
    failed.record_exception(ValueError("workflow failed"))
    failed.end()
    assert failed.attributes["status_code"] == 500
    assert failed.attributes["error.message"] == "workflow failed"
    assert failed.ended


def test_capture_content_false_omits_args_ids_and_error_text():
    tracer, interceptor = _make_interceptor(capture_content=False)
    temporal_input = SimpleNamespace(
        args=("top-secret",),
        id="customer-workflow-id",
        workflow="SecretWorkflow",
        headers={},
    )
    with interceptor._start_as_current_span(
        "StartWorkflow:SecretWorkflow",
        attributes={"temporalWorkflowID": "customer-workflow-id"},
        input_with_headers=temporal_input,
        kind=SpanKind.CLIENT,
    ):
        pass

    attrs = tracer.spans[-1].attributes
    assert "top-secret" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert "customer-workflow-id" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert '"content_captured": false' in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]


@pytest.mark.asyncio
async def test_connect_injects_once_and_respects_existing_tracing_interceptor():
    instrumentor = TemporalInstrumentor()
    instrumentor._base_interceptor_class = TracingInterceptor
    instrumentor._interceptor = object()

    async def connect(*args, **kwargs):
        return kwargs

    result = await instrumentor._connect(connect, ("localhost:7233",), {})
    assert result["interceptors"] == [instrumentor._interceptor]

    existing = TracingInterceptor()
    result = await instrumentor._connect(
        connect,
        ("localhost:7233",),
        {"interceptors": [existing]},
    )
    assert result["interceptors"] == [existing]


def test_activate_and_deactivate_are_idempotent(monkeypatch):
    wrapped = []
    unwrapped = []
    monkeypatch.setattr(
        _instrumentation,
        "wrap_function_wrapper",
        lambda module, target, wrapper: wrapped.append((module, target)),
    )
    monkeypatch.setattr(
        _instrumentation,
        "unwrap",
        lambda module, target: unwrapped.append((module, target)),
    )

    instrumentor = TemporalInstrumentor()
    instrumentor.activate()
    instrumentor.activate()
    assert wrapped == [("temporalio.client", "Client.connect")]
    assert instrumentor._is_instrumented

    observer = TemporalInstrumentor()
    observer.activate()
    assert observer._is_instrumented
    assert TemporalInstrumentor._activation_count == 2

    instrumentor.deactivate()
    assert TemporalInstrumentor._patches_applied
    assert not unwrapped
    assert not instrumentor._is_instrumented

    observer.deactivate()
    observer.deactivate()
    assert unwrapped == [("temporalio.client", "Client.connect")]
    assert not observer._is_instrumented
    assert TemporalInstrumentor._activation_count == 0
