"""Actual released SDK values, HTTP/SSE parsers and native provider behavior."""

import asyncio
import json

import pytest
from _native import Native
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from replicate.exceptions import ModelError, ReplicateError
from replicate.helpers import FileOutput
from respan_instrumentation_replicate import ReplicateInstrumentor
from respan_instrumentation_replicate import _instrumentation as adapter
from respan_instrumentation_replicate._serialization import json_string, native_value
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

INPUT = "traceloop.entity.input"
OUTPUT = "traceloop.entity.output"


@pytest.fixture
def pipeline():
    p = TracerProvider()
    m = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(m))
    owners = []

    def owner(**kwargs):
        i = ReplicateInstrumentor(tracer_provider=p, **kwargs)
        i.activate()
        owners.append(i)
        return i

    yield p, m, owner
    for i in reversed(owners):
        i.deactivate()
    p.shutdown()


def run(native, **kwargs):
    return native.client.run("owner/model", input={"prompt": "native"}, **kwargs)


def test_full_native_output_zero_usage_source_http_and_parent(pipeline):
    p, m, owner = pipeline
    owner()
    n = Native(
        {"object": "embedding", "embedding": [0.0] * 5001, "zero": 0, "flag": False}
    )
    with p.get_tracer("app").start_as_current_span("parent") as parent:
        result = run(n)
        assert len(result["embedding"]) == 5001
    s = m.get_finished_spans()[0]
    assert s.parent.span_id == parent.context.span_id
    assert (
        len(json.loads(s.attributes[OUTPUT])) == 5001
        and s.attributes["gen_ai.usage.input_tokens"] == 0
    )
    assert (
        "llm.usage.total_tokens" not in s.attributes
        and s.attributes["http.response.status_code"] == 201
    )
    assert s.attributes["respan.entity.log_type"] == "embedding"
    n.close()


@pytest.mark.parametrize("value", [False, 0, [], {}, "", {"items": [0] * 5001}])
def test_actual_false_zero_empty_and_complete_native_payloads(pipeline, value):
    _, m, owner = pipeline
    owner()
    n = Native(value)
    result = run(n)
    assert result == value
    assert json.loads(m.get_finished_spans()[0].attributes[OUTPUT]) == value
    n.close()


def test_sse_all_native_events_and_lazy_close_before_and_after_read(pipeline):
    _, m, owner = pipeline
    owner()
    n = Native(chunks=250)
    iterator = n.client.stream("owner/model", input={"prompt": "native"})
    assert not n.calls and not m.get_finished_spans()
    events = list(iterator)
    assert len(events) == 251
    s = m.get_finished_spans()[0]
    frames = json.loads(s.attributes[OUTPUT])
    assert len(frames) == 251 and frames[-2]["data"] == "chunk-249"
    assert s.attributes["gen_ai.completion.0.content"] == "".join(
        event.data for event in events if event.event.value == "output"
    )
    assert s.attributes["gen_ai.completion.0.role"] == "assistant"
    n = Native()
    iterator = n.client.stream("owner/model", input={"prompt": "native"})
    source = iterator.source
    iterator.close()
    assert source.gi_frame is None
    assert OUTPUT not in m.get_finished_spans()[-1].attributes
    n = Native()
    iterator = n.client.stream("owner/model", input={"prompt": "native"})
    event = next(iterator)
    assert event.data == "chunk-0"
    source = iterator.source
    iterator.close()
    assert source.gi_frame is None
    assert len(json.loads(m.get_finished_spans()[-1].attributes[OUTPUT])) == 1


@pytest.mark.asyncio
async def test_native_async_run_client_stream_prediction_stream_and_aclose(pipeline):
    _, m, owner = pipeline
    owner()
    n = Native("native")
    assert (
        await n.client.async_run("owner/model", input={"prompt": "native"}) == "native"
    )
    iterator = await n.client.async_stream("owner/model", input={"prompt": "native"})
    events = [e async for e in iterator]
    assert len(events) == 251
    prediction = await n.client.predictions.async_create(
        model="owner/model", input={"prompt": "native"}, stream=True
    )
    iterator = prediction.async_stream()
    source = iterator.source
    await iterator.aclose()
    assert source.ag_frame is None
    iterator = prediction.async_stream()
    events = [e async for e in iterator]
    assert len(events) == 251
    assert len(json.loads(m.get_finished_spans()[-1].attributes[OUTPUT])) == 251
    await n.client._async_client.aclose()
    n.close()


def test_native_prediction_and_file_return_identity_resources_and_management(pipeline):
    _, m, owner = pipeline
    owner()
    n = Native("https://replicate.delivery/controlled/output/file")
    output = run(n, use_file_output=True)
    assert type(output) is FileOutput and output.read() == b"native"
    s = m.get_finished_spans()[0]
    assert json.loads(s.attributes[OUTPUT]) == {"url": output.url}
    prediction = n.client.models.predictions.create(
        model="owner/model", input={"prompt": "native"}
    )
    assert prediction.wait() is None
    assert prediction.reload() is None
    assert prediction.cancel() is None
    page = n.client.predictions.list()
    assert page.results[0].id == "controlled"
    for s in m.get_finished_spans()[1:]:
        if s.name.endswith((".wait", ".reload", ".cancel")):
            assert OUTPUT not in s.attributes
    assert "__orig_class__" not in json.loads(
        m.get_finished_spans()[-1].attributes[OUTPUT]
    )
    n.close()


@pytest.mark.parametrize("error_kind", ["api", "model"])
def test_actual_errors_no_synthesized_result_or_guessed_http_status(
    pipeline, error_kind
):
    _, m, owner = pipeline
    owner()
    n = Native(status="failed") if error_kind == "model" else Native(http_status=429)
    with pytest.raises(ModelError if error_kind == "model" else ReplicateError):
        run(n)
    s = m.get_finished_spans()[0]
    assert s.status.status_code.name == "ERROR" and OUTPUT not in s.attributes
    assert s.attributes["error.type"] == (
        "ModelError" if error_kind == "model" else "ReplicateError"
    )
    assert s.attributes["http.response.status_code"] == (
        201 if error_kind == "model" else 429
    )
    n.close()


@pytest.mark.parametrize(
    "gate",
    [
        "constructor",
        "canonical",
        "legacy",
        "env",
        "active_parent",
        "preimported",
        "late_stream",
    ],
)
def test_irreversible_privacy_without_owned_payloads_or_diagnostics(
    pipeline, monkeypatch, gate
):
    p, m, owner = pipeline
    owner(capture_content=gate != "constructor")
    n = Native("PRIVATE result")
    token = None
    if gate == "env":
        monkeypatch.setenv("RESPAN_TRACE_CONTENT", "false")
    if gate in ("canonical", "legacy"):
        token = context.attach(
            context.set_value(
                ENABLE_CONTENT_TRACING_KEY if gate == "canonical" else "trace_content",
                False,
            )
        )
    with p.get_tracer("app").start_as_current_span("parent") as parent:
        if gate == "active_parent":
            parent.set_attribute("trace_content", False)
        if gate in ("preimported", "late_stream"):
            iterator = n.client.stream("owner/model", input={"prompt": "PRIVATE input"})
            if gate == "late_stream":
                next(iterator)
            from opentelemetry.context import detach as alias

            t = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            alias(t)
            list(iterator)
        else:
            assert (
                n.client.run("owner/model", input={"prompt": "PRIVATE input"})
                == "PRIVATE result"
            )
    if token:
        context.detach(token)
    leaves = [s for s in m.get_finished_spans() if s.name.startswith("replicate")]
    assert len(leaves) == 1
    for s in leaves:
        assert (
            INPUT not in s.attributes
            and OUTPUT not in s.attributes
            and "PRIVATE" not in str(s.attributes)
        )
        assert not s.events and not s.status.description
    n.close()


@pytest.mark.parametrize(
    "suppression",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_real_suppression_omits_leaf(pipeline, suppression):
    _, m, owner = pipeline
    owner()
    t = context.attach(context.set_value(suppression, True))
    n = Native("native")
    try:
        assert run(n) == "native"
    finally:
        context.detach(t)
    assert not m.get_finished_spans()
    n.close()


def test_sampling_before_telemetry_conversion(monkeypatch):
    p = TracerProvider(sampler=ALWAYS_OFF)
    m = InMemorySpanExporter()
    p.add_span_processor(SimpleSpanProcessor(m))
    i = ReplicateInstrumentor(tracer_provider=p)
    i.activate()
    monkeypatch.setattr(
        adapter,
        "native_value",
        lambda *a: (_ for _ in ()).throw(AssertionError("telemetry extraction")),
    )
    n = Native("native")
    assert run(n) == "native"
    i.deactivate()
    assert not m.get_finished_spans()
    n.close()
    p.shutdown()


@pytest.mark.parametrize("hook", ["observe_result", "observe_prediction", "finish"])
def test_observer_faults_preserve_native_outcome_and_restore_context(
    pipeline, monkeypatch, hook
):
    _, _m, owner = pipeline
    owner()
    before = context.get_current()
    monkeypatch.setattr(
        adapter._Call,
        hook,
        lambda *a: (_ for _ in ()).throw(RuntimeError("observer fault")),
    )
    n = Native("native")
    assert run(n) == "native"
    assert context.get_current() is before
    n.close()


def test_shared_lifecycle_conflict_and_foreign_owned_restore(pipeline):
    _, _m, owner = pipeline
    import replicate

    original = replicate.Client.run
    first = owner()
    second = owner()
    first.activate()
    first.deactivate()
    n = Native("native")
    assert run(n) == "native"
    with pytest.raises(ValueError):
        owner(capture_content=False)
    ours = replicate.Client.run

    def foreign(*a, **k):
        return ours(*a, **k)

    replicate.Client.run = foreign
    second.deactivate()
    assert replicate.Client.run is foreign and run(n) == "native"
    replicate.Client.run = original
    n.close()


def test_unknown_conversion_hooks_and_quoted_schema_redaction():
    class Hostile:
        def model_dump(self):
            raise AssertionError("unknown model_dump")

        def __str__(self):
            raise AssertionError("unknown str")

    assert native_value(Hostile()) == {"type": "Hostile"}
    value = {
        "type": "object",
        "properties": {"api_key": {"type": "string", "default": "PRIVATE"}},
        "arguments": '{"api_key":"PRIVATE "quoted" secret"}',
        "vector": [0] * 5001,
        "flag": False,
    }
    encoded = json_string(value)
    assert "PRIVATE" not in encoded and len(json.loads(encoded)["vector"]) == 5001
    assert "api_key" in json.loads(encoded)["properties"]


def test_native_polling_wait_updates_same_prediction_without_invented_return(pipeline):
    _, m, owner = pipeline
    owner()
    n = Native("native")
    n.statuses = ["starting", "processing", "succeeded"]
    prediction = n.client.predictions.create(
        model="owner/model", input={"prompt": "native"}, wait=False
    )
    identity = id(prediction)
    assert prediction.status == "starting"
    assert prediction.wait() is None
    assert (
        id(prediction) == identity
        and prediction.status == "succeeded"
        and prediction.output == "native"
    )
    assert (
        len(m.get_finished_spans()) == 2
        and OUTPUT not in m.get_finished_spans()[-1].attributes
    )
    n.close()


def test_native_sse_close_releases_http_response_and_foreign_binding(pipeline):
    _, m, owner = pipeline
    owner()
    n = Native()
    iterator = n.client.stream("owner/model", input={"prompt": "native"})
    next(iterator)
    response = n.responses[-1]
    assert not response.is_closed
    iterator.close()
    assert response.is_closed
    assert m.get_finished_spans()[-1].attributes["http.response.status_code"] == 200
    n.close()


def test_native_namespace_deployment_and_module_bound_methods(pipeline, monkeypatch):
    _, m, owner = pipeline
    owner()
    n = Native("native")
    prediction = n.client.deployments.predictions.create(
        deployment="owner/deployment", input={"prompt": "native"}
    )
    assert prediction.id == "controlled"
    import replicate

    monkeypatch.setattr(replicate.default_client, "_Client__client", n.client._client)
    assert replicate.run("owner/model", input={"prompt": "native"}) == "native"
    assert [s.name for s in m.get_finished_spans()] == [
        "replicate.deployments.predictions.create",
        "replicate.run",
    ]
    n.close()


def test_native_throw_identity_and_consumed_partial_data(pipeline):
    _, m, owner = pipeline
    owner()
    n = Native()
    iterator = n.client.stream("owner/model", input={"prompt": "native"})
    next(iterator)
    error = ValueError("native caller throw")
    try:
        iterator.throw(error)
    except ValueError as received:
        assert received is error
    else:
        raise AssertionError("native exception absent")
    assert len(json.loads(m.get_finished_spans()[-1].attributes[OUTPUT])) == 1
    assert n.responses[-1].is_closed
    n.close()


def test_two_pending_native_siblings_survive_exporter_suppression(pipeline):
    p, m, owner = pipeline
    owner()
    n = Native()
    with p.get_tracer("app").start_as_current_span("parent"):
        first = n.client.stream("owner/model", input={"prompt": "first"})
        second = n.client.stream("owner/model", input={"prompt": "second"})
        assert len(list(first)) == 251 and len(list(second)) == 251
    leaves = [s for s in m.get_finished_spans() if s.name.startswith("replicate")]
    assert len(leaves) == 2 and all(
        INPUT in s.attributes and OUTPUT in s.attributes for s in leaves
    )
    n.close()


def test_unknown_local_and_finished_veto_ancestors(pipeline):
    p, m, owner = pipeline
    unknown = p.get_tracer("app").start_span("unknown")
    owner()
    t = context.attach(trace.set_span_in_context(unknown))
    n = Native("native")
    try:
        assert run(n) == "native"
    finally:
        context.detach(t)
        unknown.end()
    assert INPUT not in m.get_finished_spans()[0].attributes
    known = p.get_tracer("app").start_span("known")
    known.set_attribute("trace_content", False)
    known.end()
    t = context.attach(trace.set_span_in_context(known))
    try:
        assert run(n) == "native"
    finally:
        context.detach(t)
    assert INPUT not in m.get_finished_spans()[-1].attributes
    n.close()


def test_true_remote_parent_preserved_and_unknown_carriers_bounded(pipeline):
    _p, m, owner = pipeline
    owner()
    remote = trace.NonRecordingSpan(
        trace.SpanContext(
            trace_id=123, span_id=456, is_remote=True, trace_flags=trace.TraceFlags(1)
        )
    )
    token = context.attach(trace.set_span_in_context(remote))
    n = Native("native")
    try:
        assert run(n) == "native"
    finally:
        context.detach(token)
    leaf = m.get_finished_spans()[0]
    assert leaf.parent.is_remote and INPUT in leaf.attributes
    policy = adapter._MANAGER.policy
    for i in range(5000):
        policy.enroll(
            trace.NonRecordingSpan(
                trace.SpanContext(
                    trace_id=123,
                    span_id=1000 + i,
                    is_remote=False,
                    trace_flags=trace.TraceFlags(1),
                )
            )
        )
    assert len(policy.closed) <= 4096 and not any(k[1] >= 1000 for k in policy.active)
    n.close()


def test_abandoned_generator_span_is_bodyless_without_draining(pipeline):
    import gc

    _, m, owner = pipeline
    owner()
    n = Native()
    iterator = n.client.stream("owner/model", input={"prompt": "native"})
    assert not n.calls
    del iterator
    gc.collect()
    assert (
        len(m.get_finished_spans()) == 1
        and OUTPUT not in m.get_finished_spans()[0].attributes
        and m.get_finished_spans()[0].attributes["respan.entity.log_type"] == "text"
        and INPUT not in m.get_finished_spans()[0].attributes
    )


@pytest.mark.asyncio
async def test_abandoned_async_generator_is_bodyless(pipeline):
    import gc

    _, m, owner = pipeline
    owner()
    n = Native()
    iterator = await n.client.async_stream("owner/model", input={"prompt": "native"})
    assert not n.calls
    del iterator
    gc.collect()
    await asyncio.sleep(0)
    assert (
        len(m.get_finished_spans()) == 1
        and OUTPUT not in m.get_finished_spans()[0].attributes
        and m.get_finished_spans()[0].attributes["respan.entity.log_type"] == "text"
    )


def test_scrub_and_finish_fault_cannot_replace_native_outcome(pipeline, monkeypatch):
    _p, m, owner = pipeline
    owner()
    n = Native("native")
    before = context.get_current()

    def fail(*args):
        raise RuntimeError("observer fault")

    monkeypatch.setattr(adapter._Call, "scrub", fail)
    monkeypatch.setattr(adapter._Call, "observe_result", fail)
    assert run(n) == "native"
    assert context.get_current() is before
    assert (
        len(m.get_finished_spans()) == 1
        and OUTPUT not in m.get_finished_spans()[0].attributes
    )
    n.close()


def test_complete_embedding_result_extras_preserved_separately(pipeline):
    _, m, owner = pipeline
    owner()
    value = {
        "object": "embedding",
        "embedding": [0] * 5001,
        "extra": {"false": False, "zero": 0, "empty": []},
    }
    n = Native(value)
    assert run(n) == value
    s = m.get_finished_spans()[0]
    assert len(json.loads(s.attributes[OUTPUT])) == 5001
    assert json.loads(s.attributes["respan.metadata.replicate.result"]) == value
    n.close()


@pytest.mark.parametrize(
    "fragments",
    [
        ['Authorization: Bearer "controlled-private"'],
        ["Authorization: Be", 'arer "controlled-private"'],
        ["api_key=", "controlled-private"],
    ],
)
def test_native_sse_quoted_and_fragmented_credentials_preserve_events(
    pipeline, fragments
):
    _, m, owner = pipeline
    owner()
    n = Native(chunks=len(fragments), output_chunks=fragments)
    events = list(n.client.stream("owner/model", input={"prompt": "controlled"}))
    assert [event.data for event in events[:-1]] == fragments
    attrs = dict(m.get_finished_spans()[0].attributes)
    assert "controlled-private" not in json.dumps(attrs)
    assert attrs["gen_ai.completion.0.role"] == "assistant"
    assert len(json.loads(attrs[OUTPUT])) == len(fragments) + 1
    n.close()
