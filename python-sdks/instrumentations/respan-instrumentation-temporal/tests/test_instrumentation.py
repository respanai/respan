"""Contract tests against released Temporal interceptors and real OTel spans."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_temporal import TemporalInstrumentor, _instrumentation
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.utils.span_factory import propagate_attributes
from temporalio.client import Client, OutboundInterceptor, QueryWorkflowInput
from temporalio.contrib.opentelemetry import TracingInterceptor

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


@pytest.fixture
def pipeline():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def interceptor(provider, *, capture_content=True):
    return _instrumentation._build_interceptor(
        TracingInterceptor,
        tracer=provider.get_tracer("temporal-contract"),
        capture_content=capture_content,
        max_attribute_chars=None,
        always_create_workflow_spans=True,
    )


def query(args=()):
    return QueryWorkflowInput(
        id="actual-workflow-id",
        run_id="actual-run-id",
        query="status",
        args=args,
        reject_condition=None,
        headers={},
        ret_type=None,
        rpc_metadata={},
        rpc_timeout=None,
    )


class Transport(OutboundInterceptor):
    """Controlled transport boundary, retaining the real vendor interceptor."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def query_workflow(self, value):
        self.calls.append(value)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_native_query_returns_identical_result_and_complete_payload(pipeline):
    provider, exporter = pipeline
    payload = {
        "history": [
            {"arguments": list(range(70)), "vector": [i / 100 for i in range(90)]}
            for _ in range(45)
        ],
        "text": "x" * 20_000,
    }
    transport = Transport(result=payload)
    value = query((payload,))
    result = (
        await interceptor(provider).intercept_client(transport).query_workflow(value)
    )
    assert result is payload
    assert transport.calls == [value]
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert json.loads(attrs[INPUT])["input"]["args"] == [payload]
    assert json.loads(attrs[OUTPUT]) == payload
    assert "status_code" not in attrs
    assert not any(
        key.startswith("temporal")
        or key
        in ("model", "tools", "tool_calls", "prompt_tokens", "traceloop.span.kind")
        for key in attrs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("api_key=private failure"),
        asyncio.CancelledError("private cancellation"),
    ],
)
async def test_native_error_and_cancellation_identity_preserved(pipeline, error):
    provider, exporter = pipeline
    transport = Transport(error=error)
    with pytest.raises(type(error)) as caught:
        await interceptor(provider).intercept_client(transport).query_workflow(query())
    assert caught.value is error
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert "private failure" not in (span.status.description or "")
    assert not span.events
    assert "status_code" not in span.attributes


class UnobservedPayload(list):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("payload serialization must be skipped")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "sampler",
        "suppression",
        "disabled",
        "ambient",
        "supplied",
        "environment",
        "unknown-parent",
    ],
)
async def test_sampling_and_privacy_precede_payload_access(pipeline, monkeypatch, mode):
    provider, exporter = pipeline
    if mode == "sampler":
        provider = TracerProvider(sampler=ALWAYS_OFF)
    native = interceptor(provider, capture_content=mode != "disabled")
    token = None
    if mode == "suppression":
        token = context.attach(
            context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
    elif mode == "ambient":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    elif mode == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif mode == "unknown-parent":
        parent = provider.get_tracer("application-owned").start_span("parent")
        token = context.attach(trace.set_span_in_context(parent))
    elif mode == "supplied":
        denied = baggage.set_baggage(
            ENABLE_CONTENT_TRACING_KEY, "false", context=context.Context()
        )
        native._context_from_headers = lambda _headers: denied
        # Direct interceptor entry uses the same explicit context as the real
        # activity interceptor after native Temporal header extraction.
    UnobservedPayload.calls = 0
    payload = UnobservedPayload(["secret"])
    try:
        if mode == "supplied":
            with native._start_as_current_span(
                "RunActivity:actual",
                attributes={},
                input_with_headers=query((payload,)),
                kind=trace.SpanKind.SERVER,
                context=denied,
            ):
                pass
        else:
            result = object()
            assert (
                await native.intercept_client(Transport(result=result)).query_workflow(
                    query((payload,))
                )
                is result
            )
    finally:
        if token is not None:
            context.detach(token)
        if mode == "unknown-parent":
            parent.end()
    assert UnobservedPayload.calls == 0
    for span in exporter.get_finished_spans():
        assert "secret" not in span.to_json()
    if mode == "suppression":
        assert exporter.get_finished_spans() == ()
    if mode == "sampler":
        provider.shutdown()


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("initial", [False, True])
def test_observed_ancestor_decision_is_irreversible(pipeline, finished, initial):
    provider, exporter = pipeline
    canonical = interceptor(provider).tracer
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, initial))
    parent = canonical.start_span("RunWorkflow:Parent")
    context.detach(token)
    if not initial:
        assert not parent.allowed()
    if finished:
        parent.end()
    parent_context = trace.set_span_in_context(parent)
    child = canonical.start_span(
        "RunActivity:Child",
        context=parent_context,
        attributes={
            _instrumentation.TEMPORAL_CAPTURED_INPUT: {"args": ["private child"]}
        },
    )
    child.end()
    parent.end()
    child_span = next(
        s for s in exporter.get_finished_spans() if s.name == "RunActivity:Child"
    )
    assert ("private child" in child_span.to_json()) is initial


def test_ambient_and_supplied_privacy_are_combined_and_late_denial_scrubs(pipeline):
    provider, exporter = pipeline
    canonical = interceptor(provider).tracer
    with canonical.start_as_current_span(
        "RunActivity:actual",
        context=context.Context(),
        attributes={
            _instrumentation.TEMPORAL_CAPTURED_INPUT: {"args": ["private input"]}
        },
    ) as span:
        span.capture_output("private output")
        span.record_exception(RuntimeError("private diagnostic"))
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        span.end()
        context.detach(token)
        span.capture_output("later private output")
    finished = exporter.get_finished_spans()[0]
    assert "private" not in finished.to_json()
    assert finished.status.description is None
    assert "error.message" not in finished.attributes
    assert not finished.events


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["start", "attributes", "propagation", "end"])
async def test_telemetry_faults_keep_native_result_and_clean_spans(
    pipeline, monkeypatch, fault
):
    provider, exporter = pipeline
    native = interceptor(provider)
    tracer = native.tracer._tracer
    if fault == "start":
        monkeypatch.setattr(
            tracer,
            "start_span",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("trace start failed")),
        )
    elif fault in ("attributes", "end"):
        original = tracer.start_span

        def start(*args, **kwargs):
            span = original(*args, **kwargs)
            if fault == "attributes":
                monkeypatch.setattr(
                    span,
                    "set_attribute",
                    lambda *_a, **_k: (_ for _ in ()).throw(
                        RuntimeError("trace attribute failed")
                    ),
                )
            else:
                end = span.end

                def failed_end(*args, **kwargs):
                    end(*args, **kwargs)
                    raise RuntimeError("trace end failed")

                monkeypatch.setattr(span, "end", failed_end)
            return span

        monkeypatch.setattr(tracer, "start_span", start)
    else:
        monkeypatch.setattr(
            native,
            "_context_to_headers",
            lambda *_a: (_ for _ in ()).throw(RuntimeError("propagator failed")),
        )
    result = object()
    assert (
        await native.intercept_client(Transport(result=result)).query_workflow(query())
        is result
    )
    assert trace.get_current_span() is trace.INVALID_SPAN
    assert len(exporter.get_finished_spans()) == (0 if fault == "start" else 1)


@pytest.mark.asyncio
async def test_real_client_shared_configuration_and_foreign_wrapper(monkeypatch):
    original = Client.__dict__["connect"]
    first, second = TemporalInstrumentor(), TemporalInstrumentor()
    first.activate()
    second.activate()
    try:
        client = await Client.connect("localhost:7233", lazy=True)
        assert len(client.config()["interceptors"]) == 1
        with pytest.raises(ValueError, match="different settings"):
            TemporalInstrumentor(capture_content=False).activate()
        first.deactivate()
        assert TemporalInstrumentor._activation_count == 1
        later = Client.__dict__["connect"]

        async def foreign(cls, *args, **kwargs):
            return await later.__func__(cls, *args, **kwargs)

        descriptor = classmethod(foreign)
        monkeypatch.setattr(Client, "connect", descriptor)
        second.deactivate()
        assert Client.__dict__["connect"] is descriptor
        client = await Client.connect("localhost:7233", lazy=True)
        assert not client.config()["interceptors"]
    finally:
        first.deactivate()
        second.deactivate()
        Client.connect = original


@pytest.mark.asyncio
async def test_failed_activation_and_connect_setup_roll_back(monkeypatch):
    original = Client.__dict__["connect"]
    instrumentor = TemporalInstrumentor()
    monkeypatch.setattr(
        instrumentor,
        "_ensure_interceptor",
        lambda: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )
    instrumentor.activate()
    assert not instrumentor._is_instrumented
    assert Client.__dict__["connect"] is original
    calls = []

    async def connect(*args, **kwargs):
        calls.append((args, kwargs))
        return "native"

    assert await instrumentor._connect(connect, ("localhost",), {}) == "native"
    assert calls == [(("localhost",), {})]


def test_redaction_dataclass_and_metadata_propagation(pipeline):
    provider, exporter = pipeline

    @dataclass
    class Payload:
        values: list[int]
        api_key: str

    native = interceptor(provider)
    with (
        propagate_attributes(metadata={"run_id": "exact-marker", "api_key": "secret"}),
        native._start_as_current_span(
            "StartWorkflow:Actual",
            attributes={"temporalWorkflowID": "actual-id"},
            input_with_headers=query((Payload(list(range(70)), "secret"),)),
            kind=trace.SpanKind.CLIENT,
        ),
    ):
        pass
    span = exporter.get_finished_spans()[0]
    assert span.attributes["respan.metadata.run_id"] == "exact-marker"
    assert "secret" not in span.to_json()
    assert json.loads(span.attributes[INPUT])["input"]["args"][0]["values"] == list(
        range(70)
    )


def test_late_environment_veto_scrubs_metadata_and_diagnostics(pipeline, monkeypatch):
    provider, exporter = pipeline
    native = interceptor(provider)
    with (
        propagate_attributes(metadata={"private_field": "private metadata"}),
        native._start_as_current_span(
            "RunActivity:actual", attributes={}, kind=trace.SpanKind.SERVER
        ),
    ):
        span = trace.get_current_span()
        span.capture_input("private input")
        span.capture_output("private output")
        span.record_exception(RuntimeError("private diagnostics"))
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    finished = exporter.get_finished_spans()[0]
    assert "private" not in finished.to_json()
    assert finished.status.description is None
    assert "error.message" not in finished.attributes
    assert not finished.events


def test_initial_child_veto_scrubs_active_parent_after_ambient_restored(pipeline):
    provider, exporter = pipeline
    native = interceptor(provider)
    with native.tracer.start_as_current_span(
        "RunWorkflow:Parent",
        attributes={
            _instrumentation.TEMPORAL_CAPTURED_INPUT: {"args": ["private ancestor"]}
        },
    ) as parent:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        child = native.tracer.start_span("RunActivity:Child")
        child.end()
        context.detach(token)
        assert not parent.allowed()
    assert "private ancestor" not in "\n".join(
        s.to_json() for s in exporter.get_finished_spans()
    )


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("initial", [False, True])
def test_observer_captures_application_parent_privacy(
    pipeline, monkeypatch, finished, initial
):
    provider, exporter = pipeline
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    native = interceptor(provider)
    _instrumentation._observe_provider()
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, initial))
    parent = provider.get_tracer("application").start_span("application-parent")
    context.detach(token)
    if finished:
        parent.end()
    child = native.tracer.start_span(
        "RunActivity:Child",
        context=trace.set_span_in_context(parent),
        attributes={
            _instrumentation.TEMPORAL_CAPTURED_INPUT: {"args": ["private descendant"]}
        },
    )
    child.end()
    parent.end()
    span = next(
        s for s in exporter.get_finished_spans() if s.name == "RunActivity:Child"
    )
    assert ("private descendant" in span.to_json()) is initial


def test_late_proxy_provider_unobserved_parent_fails_closed(pipeline, monkeypatch):
    provider, exporter = pipeline
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    proxy_provider = trace.ProxyTracerProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: proxy_provider)
    instrumentor = TemporalInstrumentor()
    native = instrumentor.interceptor
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    # This parent was created before the observer could see the late provider.
    parent = provider.get_tracer("application").start_span("unobserved-parent")
    with native.tracer.start_as_current_span(
        "RunActivity:Child",
        context=trace.set_span_in_context(parent),
        attributes={
            _instrumentation.TEMPORAL_CAPTURED_INPUT: {
                "args": ["private late-provider payload"]
            }
        },
    ):
        pass
    parent.end()
    span = next(
        s for s in exporter.get_finished_spans() if s.name == "RunActivity:Child"
    )
    assert "private late-provider payload" not in span.to_json()


def test_released_native_payload_types_are_complete(pipeline):
    from datetime import datetime, timezone
    from uuid import UUID

    from pydantic import BaseModel
    from respan_instrumentation_temporal._serialization import to_jsonable
    from temporalio.api.common.v1 import Payload

    class Model(BaseModel):
        values: list[int]

    value = {
        "model": Model(values=list(range(75))),
        "protobuf": Payload(
            metadata={"encoding": b"json/plain"}, data=b"complete data"
        ),
        "binary": b"complete binary",
        "timestamp": datetime(2026, 10, 5, tzinfo=timezone.utc),
        "uuid": UUID("11111111-1111-1111-1111-111111111111"),
        "set": {1, 2, 3},
    }
    actual = to_jsonable(value)
    assert actual["model"]["values"] == list(range(75))
    assert actual["protobuf"]["data"] == "Y29tcGxldGUgZGF0YQ=="
    assert actual["binary"]["base64"] == "Y29tcGxldGUgYmluYXJ5"
    assert actual["timestamp"] == "2026-10-05T00:00:00+00:00"
    assert actual["uuid"] == "11111111-1111-1111-1111-111111111111"
    assert set(actual["set"]) == {1, 2, 3}


def test_unknown_serialization_hooks_are_not_invoked():
    from respan_instrumentation_temporal._serialization import to_jsonable

    class CustomerObject:
        def __getattribute__(self, name):
            raise AssertionError("unknown customer hook invoked")

    assert to_jsonable(CustomerObject()) == {"type": "CustomerObject"}


def test_quoted_secret_redaction_is_valid_idempotent_and_preserves_schema():
    from respan_instrumentation_temporal._serialization import (
        json_dumps,
        safe_error_message,
    )

    payload = {
        "schema": {
            "type": "object",
            "properties": {
                "api_key": {"type": "string", "default": "private default"},
                "enabled": {"type": "boolean", "default": False},
            },
        },
        "encoded": '{"client_secret":"private \\"quoted\\" value", "count":0}',
        "authorization": "Basic cHJpdmF0ZQ==",
        "message": "Bearer privatebearer",
    }
    text = json_dumps(payload)
    actual = json.loads(text)
    assert actual["schema"]["properties"]["api_key"]["type"] == "string"
    assert actual["schema"]["properties"]["api_key"]["default"] == "[REDACTED]"
    assert actual["schema"]["properties"]["enabled"]["default"] is False
    assert json.loads(actual["encoded"])["count"] == 0
    assert "private" not in text
    assert json_dumps(actual) == text
    assert "private" not in safe_error_message(
        RuntimeError('secret="private multiword value"'), capture_content=True
    )


def test_baggage_does_not_invoke_numeric_subclass_hooks():
    from respan_instrumentation_temporal._serialization import safe_baggage_value

    class Number(int):
        def __str__(self):
            raise AssertionError("customer hook invoked")

    assert safe_baggage_value("respan.metadata.value", Number(1)) == '{"type":"Number"}'


@pytest.mark.asyncio
async def test_released_native_failure_has_diagnostics_without_result(pipeline):
    from opentelemetry.semconv._incubating.attributes.error_attributes import (
        ERROR_MESSAGE,
    )
    from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

    provider, exporter = pipeline
    error = RuntimeError("native failure")
    with pytest.raises(RuntimeError) as caught:
        await (
            interceptor(provider)
            .intercept_client(Transport(error=error))
            .query_workflow(query())
        )
    assert caught.value is error
    span = exporter.get_finished_spans()[0]
    assert OUTPUT not in span.attributes
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == "RuntimeError"
    assert span.attributes[ERROR_MESSAGE] == "native failure"


def test_error_keeps_only_previously_captured_native_result(pipeline):
    provider, exporter = pipeline
    span = interceptor(provider).tracer.start_span("RunActivity:actual")
    span.capture_output({"native": [False, 0]})
    span.record_exception(RuntimeError("native diagnostic"))
    span.end()
    assert json.loads(exporter.get_finished_spans()[0].attributes[OUTPUT]) == {
        "native": [False, 0]
    }


@pytest.mark.asyncio
async def test_native_private_error_has_type_without_message_or_result(pipeline):
    from opentelemetry.semconv._incubating.attributes.error_attributes import (
        ERROR_MESSAGE,
    )
    from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

    provider, exporter = pipeline
    error = RuntimeError("private error message")
    native = interceptor(provider, capture_content=False)
    active = native.tracer.start_span("RunActivity:private")
    active.record_exception(error)
    assert active.status.description is None
    assert ERROR_MESSAGE not in active.attributes
    assert OUTPUT not in active.attributes
    assert active.attributes[ERROR_TYPE] == "RuntimeError"
    active.end()
    with pytest.raises(RuntimeError) as caught:
        await native.intercept_client(Transport(error=error)).query_workflow(query())
    assert caught.value is error
    span = exporter.get_finished_spans()[-1]
    assert OUTPUT not in span.attributes
    assert ERROR_MESSAGE not in span.attributes
    assert span.status.description is None
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == "RuntimeError"


def test_direct_owned_error_message_is_scrubbed_on_late_veto(pipeline):
    from opentelemetry.semconv._incubating.attributes.error_attributes import (
        ERROR_MESSAGE,
    )

    provider, exporter = pipeline
    span = interceptor(provider).tracer.start_span("RunActivity:actual")
    span.set_attribute(ERROR_MESSAGE, "private direct diagnostic")
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        span.end()
    finally:
        context.detach(token)
    assert ERROR_MESSAGE not in exporter.get_finished_spans()[0].attributes


def test_bare_native_error_status_has_no_invented_message_type_or_output(pipeline):
    from opentelemetry.semconv._incubating.attributes.error_attributes import (
        ERROR_MESSAGE,
    )
    from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

    provider, exporter = pipeline
    span = interceptor(provider).tracer.start_span("RunActivity:actual")
    span.set_status(trace.Status(trace.StatusCode.ERROR))
    assert span.status.description is None
    assert ERROR_MESSAGE not in span.attributes
    assert ERROR_TYPE not in span.attributes
    assert OUTPUT not in span.attributes
    span.end()
    finished = exporter.get_finished_spans()[0]
    assert finished.status.status_code is trace.StatusCode.ERROR
    assert finished.status.description is None
    assert ERROR_MESSAGE not in finished.attributes
    assert ERROR_TYPE not in finished.attributes
    assert OUTPUT not in finished.attributes
