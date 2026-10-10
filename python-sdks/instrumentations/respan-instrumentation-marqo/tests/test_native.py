import inspect
import json

import marqo
import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags
from respan_instrumentation_marqo import MarqoInstrumentor
from respan_instrumentation_marqo import _native_instrumentation as impl
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

I = "traceloop.entity.input"
O = "traceloop.entity.output"


def body(span, field=O):
    return json.loads(span.attributes[field])


def empty(spans):
    for s in spans:
        assert (
            I not in s.attributes
            and O not in s.attributes
            and "error.message" not in s.attributes
        )
        assert not s.events and s.status.description is None


def test_native_complete_search_and_request(runtime):
    _c, index, _p, m, _o, requests = runtime
    result = index.search(q={"native": 0.0}, limit=75, offset=0, show_highlights=False)
    assert len(result["hits"]) == 75 and result["hits"][0]["flag"] is False
    span = m.get_finished_spans()[0]
    assert span.attributes["respan.entity.log_type"] == "task" and body(span) == result
    data = body(span, I)
    assert data["kwargs"]["offset"] == 0 and data["kwargs"]["show_highlights"] is False
    assert (
        len(data["native_requests"]) == 1
        and data["native_requests"][0]["body"] == requests[0]["body"]
    )
    assert (
        span.attributes["http.response.status_code"] == 200
        and "gen_ai.request.model" not in span.attributes
    )


def test_native_embedding_full_vectors_and_envelope(runtime):
    if not hasattr(marqo.index.Index, "embed"):
        pytest.skip("native minimum has no embed API")
    result = runtime[1].embed("native", content_type=None)
    span = runtime[3].get_finished_spans()[0]
    assert (
        body(span) == [result["embeddings"][0]["embedding"]]
        and len(body(span)[0]) == 5001
    )
    assert (
        span.attributes["gen_ai.request.model"] == result["model"]
        and span.attributes["llm.request.type"] == "embedding"
    )
    extra = json.loads(span.attributes["respan.metadata.marqo.result"])
    assert extra["records"][0]["flag"] is False and extra["empty"] == ""
    assert "gen_ai.usage.input_tokens" not in span.attributes


def test_handle_factory_does_not_invent_work(runtime):
    c, _index, _p, m, _o, requests = runtime
    assert type(c.index("docs")) is marqo.index.Index
    assert not requests and not m.get_finished_spans()


@pytest.mark.parametrize("method", ["create_index", "delete_index"])
def test_native_alias_lifecycle_one_span(runtime, method):
    c, _index, _p, m, _o, _requests = runtime
    result = getattr(c, method)("docs")
    assert result["acknowledged"] and len(m.get_finished_spans()) == 1
    assert body(m.get_finished_spans()[0]) == result


@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("get_stats", (), {}),
        ("get_settings", (), {}),
        ("get_document", ("doc",), {"expose_facets": True}),
        ("get_documents", (["doc"],), {"expose_facets": True}),
        ("delete_documents", (["doc"],), {}),
        ("eject_model", ("actual-model",), {}),
        ("recommend", (["doc"],), {"limit": 0}),
    ],
)
def test_released_native_resource_result_preserved(runtime, name, args, kwargs):
    if not hasattr(marqo.index.Index, name):
        pytest.skip("native minimum API unavailable")
    if (
        name == "eject_model"
        and "model_device"
        in inspect.signature(marqo.index.Index.eject_model).parameters
    ):
        kwargs = {**kwargs, "model_device": "cpu"}
    result = getattr(runtime[1], name)(*args, **kwargs)
    span = runtime[3].get_finished_spans()[0]
    assert body(span) == result
    assert span.attributes["respan.entity.log_type"] == "task"


@pytest.mark.parametrize("name", ["add_documents", "update_documents"])
def test_native_full_write_batches(runtime, name):
    docs = [
        {"_id": str(i), "text": "native", "vector": [float(i)] * 5001, "flag": False}
        for i in range(75)
    ]
    kwargs = {"documents": docs}
    if name == "add_documents":
        kwargs["tensor_fields"] = ["text"]
    result = getattr(runtime[1], name)(**kwargs)
    span = runtime[3].get_finished_spans()[0]
    assert body(span) == result
    assert (
        len(body(span, I)["kwargs"]["documents"]) == 75
        and len(body(span, I)["kwargs"]["documents"][0]["vector"]) == 5001
    )


@pytest.mark.parametrize(
    "flag,value",
    [
        (ENABLE_CONTENT_TRACING_KEY, False),
        ("trace_content", False),
        ("override_enable_content_tracing", False),
        (context._SUPPRESS_INSTRUMENTATION_KEY, True),
        (SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True),
    ],
)
def test_native_privacy_suppression(runtime, flag, value):
    token = context.attach(context.set_value(flag, value))
    try:
        assert len(runtime[1].search(q="PRIVATE")["hits"]) == 75
    finally:
        context.detach(token)
    spans = runtime[3].get_finished_spans()
    if value is True:
        assert not spans
    else:
        assert len(spans) == 1
        empty(spans)


@pytest.mark.parametrize("env", ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
def test_environment_capture(runtime, monkeypatch, env):
    monkeypatch.setenv(env, "false")
    runtime[1].search(q="PRIVATE")
    empty(runtime[3].get_finished_spans())


def test_native_error_preserved_no_fake_output(runtime):
    with pytest.raises(marqo.errors.MarqoWebError) as found:
        runtime[1].health()
    span = runtime[3].get_finished_spans()[0]
    assert found.value.status_code == 503
    assert (
        span.status.status_code is StatusCode.ERROR
        and span.attributes["http.response.status_code"] == 503
    )
    assert span.attributes["error.type"] == "MarqoWebError" and O not in span.attributes
    assert "PRIVATE" not in json.dumps(dict(span.attributes))


@pytest.mark.parametrize("kind", ["active", "finished", "unknown", "remote"])
def test_native_ancestor_bounds(runtime, kind):
    p = runtime[2]
    if kind in ["active", "finished"]:
        parent = p.get_tracer("app").start_span("parent")
        parent.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
        if kind == "finished":
            parent.end()
    else:
        parent = NonRecordingSpan(
            SpanContext(123, 456, kind == "remote", TraceFlags(1))
        )
    with trace.use_span(parent):
        runtime[1].search(q="PRIVATE")
    span = runtime[3].get_finished_spans()[-1]
    if kind == "remote":
        assert I in span.attributes
    else:
        empty([span])
    if kind == "active":
        parent.end()


def test_late_end_readable_veto(runtime, monkeypatch):
    end = Span.end

    def veto(s, *a, **kw):
        if s.name.startswith("marqo."):
            s.add_event("private", {"body": "PRIVATE"})
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            try:
                return end(s, *a, **kw)
            finally:
                context.detach(token)
        return end(s, *a, **kw)

    monkeypatch.setattr(Span, "end", veto)
    with pytest.raises(marqo.errors.MarqoWebError):
        runtime[1].health()
    empty(runtime[3].get_finished_spans())


@pytest.mark.parametrize("stage", ["request", "observe", "capture", "error"])
def test_observer_faults_preserve_native_result_error(runtime, monkeypatch, stage):
    def fault(self, *a, **k):
        if self.span and self.span.is_recording():
            self.keys.add(I)
            self.span.set_attribute(I, "PRIVATE")
            self.span.add_event("private", {"body": "PRIVATE"})
        raise RuntimeError("observer")

    monkeypatch.setattr(impl._Call, stage, fault)
    if stage == "error":
        with pytest.raises(marqo.errors.MarqoWebError):
            runtime[1].health()
    else:
        assert len(runtime[1].search(q="native")["hits"]) == 75
    empty(runtime[3].get_finished_spans())
    assert not impl._MANAGER.states


@pytest.mark.parametrize("mode", ["attach", "detach", "set"])
def test_mutate_faults_restore_outcomes_and_context(runtime, monkeypatch, mode):
    before = context.get_current()
    tokens = []
    if mode == "attach":
        attach = context.attach

        def fault(c):
            t = attach(c)
            tokens.append(t)
            if len(tokens) == 1:
                raise RuntimeError("attach mutated")
            return t

        monkeypatch.setattr(context, "attach", fault)
    elif mode == "detach":

        def fault(t):
            raise RuntimeError("detach failed")

        monkeypatch.setattr(context, "detach", fault)
    else:
        set_attribute = Span.set_attribute

        def fault(s, k, v):
            set_attribute(s, k, v)
            if k == I:
                raise RuntimeError("set mutated")

        monkeypatch.setattr(Span, "set_attribute", fault)
    assert len(runtime[1].search(q="native")["hits"]) == 75
    assert context.get_current() is before
    if mode in ["attach", "set"]:
        empty(runtime[3].get_finished_spans())


def test_native_shared_foreign_and_conflict(runtime):
    p = runtime[2]
    owner = runtime[4]
    other = MarqoInstrumentor(tracer_provider=p)
    other.activate()
    owner.deactivate()
    with pytest.raises(ValueError):
        MarqoInstrumentor(tracer_provider=p, capture_content=False).activate()
    runtime[1].search(q="native")
    assert len(runtime[3].get_finished_spans()) == 1
    original = marqo.index.Index.search

    def foreign(*a, **k):
        return original(*a, **k)

    marqo.index.Index.search = foreign
    other.deactivate()
    assert marqo.index.Index.search is foreign
    marqo.index.Index.search = original.__wrapped__


def test_sampler_and_constructor_gate_before_conversion(runtime, monkeypatch):
    owner = runtime[4]
    owner.deactivate()

    def forbidden(v):
        raise AssertionError("observer conversion")

    monkeypatch.setattr(impl, "json_string", forbidden)
    for capture, sampler in [(False, None), (True, ALWAYS_OFF)]:
        provider = TracerProvider(sampler=sampler) if sampler else TracerProvider()
        o = MarqoInstrumentor(tracer_provider=provider, capture_content=capture)
        o.activate()
        try:
            assert len(runtime[1].search(q="native")["hits"]) == 75
        finally:
            o.deactivate()
            provider.shutdown()


def test_structural_serializer_no_unknown_hooks_and_complete_schema():
    from respan_instrumentation_marqo._serialization import json_string

    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("hash")
            raise AssertionError

        def __eq__(cls, other):
            calls.append("eq")
            raise AssertionError

    class Unknown(metaclass=Meta):
        def __str__(self):
            calls.append("str")
            raise AssertionError

        def __iter__(self):
            calls.append("iter")
            raise AssertionError

        def model_dump(self):
            calls.append("dump")
            raise AssertionError

    d = {
        "unknown": Unknown(),
        "schema": {
            "type": "object",
            "properties": {"password": {"type": "string", "default": "PRIVATE"}},
            "required": ["password"],
        },
        "text": r"Bearer \"PRIVATE SPACE\" Basic \'PRIVATE TWO\'",
        "history": list(range(75)),
    }
    out = json.loads(json_string(d))
    assert (
        not calls and out["unknown"] == "<Unknown>" and "PRIVATE" not in json.dumps(out)
    )
    assert out["schema"]["required"] == ["password"] and len(out["history"]) == 75
    assert json_string(out) == json_string(json.loads(json_string(out)))


def test_native_client_listing_and_bulk(runtime):
    c, _index, _p, m, _o, _requests = runtime
    assert c.get_indexes()["results"] == []
    queries = [{"index": "docs", "q": "native", "limit": 0}]
    result = c.bulk_search(queries)
    spans = m.get_finished_spans()
    assert len(spans) == 2 and body(spans[1]) == result
    assert body(spans[1], I)["args"][0] == queries


def test_native_direct_static_create_delete(runtime):
    c, index, _p, m, _o, _requests = runtime
    result = marqo.index.Index.create(
        c.config, "docs", model="actual", normalize_embeddings=False
    )
    assert result["acknowledged"] and len(m.get_finished_spans()) == 1
    assert index.delete()["acknowledged"] and len(m.get_finished_spans()) == 2


def test_native_local_cloud_status_error_no_fake_http(runtime):
    if not hasattr(marqo.index.Index, "get_status"):
        pytest.skip("native minimum no cloud status API")
    with pytest.raises(marqo.errors.UnsupportedOperationError):
        runtime[1].get_status()
    span = runtime[3].get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR and O not in span.attributes
    assert "http.response.status_code" not in span.attributes


def test_partial_activation_restores_descriptors(runtime, monkeypatch):
    owner = runtime[4]
    owner.deactivate()
    original = marqo.index.Index.search
    patch = impl._Manager.patch
    calls = []

    def fault(self, *args):
        patch(self, *args)
        calls.append(1)
        if len(calls) == 6:
            raise RuntimeError("partial install")

    monkeypatch.setattr(impl._Manager, "patch", fault)
    with pytest.raises(RuntimeError):
        owner.activate()
    assert impl._MANAGER is None and marqo.index.Index.search is original
    assert len(runtime[1].search(q="native")["hits"]) == 75


def test_native_end_mutate_then_raise_preserves_result(runtime, monkeypatch):
    end = Span.end

    def fault(s, *a, **kw):
        end(s, *a, **kw)
        raise RuntimeError("end mutation")

    monkeypatch.setattr(Span, "end", fault)
    assert len(runtime[1].search(q="native")["hits"]) == 75
    assert len(runtime[3].get_finished_spans()) == 1 and not impl._MANAGER.states


def test_native_bare_error_status_retained_no_diagnostic_invention(runtime):
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import Status

    class Bare(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_status(Status(StatusCode.ERROR))

        def on_end(self, span):
            pass

    runtime[2].add_span_processor(Bare())
    assert len(runtime[1].search(q="native")["hits"]) == 75
    span = runtime[3].get_finished_spans()[0]
    assert (
        span.status.status_code is StatusCode.ERROR and span.status.description is None
    )
    assert (
        "error.type" not in span.attributes and "error.message" not in span.attributes
    )
