import gc
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pymilvus
import pytest
from conftest import populated
from opentelemetry import context, trace
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from respan_instrumentation_milvus import MilvusInstrumentor
from respan_instrumentation_milvus import _native_instrumentation as impl
from respan_instrumentation_milvus._serialization import (
    json_value,
    native_storage,
    redact_text,
)

I = "traceloop.entity.input"
O = "traceloop.entity.output"


def spans(memory, operation):
    return [
        s
        for s in memory.get_finished_spans()
        if s.attributes.get("db.operation") == operation
    ]


def one(memory, operation):
    rows = spans(memory, operation)
    assert len(rows) == 1
    return rows[0]


def bodyless(rows):
    for span in rows:
        assert (
            I not in span.attributes
            and O not in span.attributes
            and "error.message" not in span.attributes
        )
        assert not span.events and span.status.description is None
        assert "PRIVATE" not in json.dumps(dict(span.attributes))


@pytest.mark.parametrize(
    "method",
    [
        "query",
        "get",
        "has_collection",
        "describe_collection",
        "list_collections",
        "get_collection_stats",
        "upsert",
        "delete",
        "flush",
        "drop_collection",
    ],
)
def test_actual_native_operations(engine, runtime, method):
    if not hasattr(MilvusClient, method):
        pytest.skip("Selected native SDK does not expose this client API")
    name, data = populated(engine)
    memory = runtime[1]
    memory.clear()
    options = {"collection_name": name}
    if method == "query":
        options.update(filter="id>=0", output_fields=["id", "vector", "text", "flag"])
    elif method == "get":
        options.update(ids=[0], output_fields=["id", "vector", "text", "flag"])
    elif method == "list_collections":
        options = {}
    elif method == "upsert":
        options.update(data=data)
    elif method == "delete":
        options.update(ids=[1])
    result = getattr(engine, method)(**options)
    span = one(memory, method)
    assert (
        span.attributes["respan.entity.log_type"] == "task"
        and span.attributes["db.system"] == "milvus"
    )
    assert span.attributes["traceloop.entity.path"] == ""
    assert json.loads(span.attributes[O]) == json_value(result)
    assert not any(
        k.startswith("gen_ai.")
        or k in ("status_code", "traceloop.span.kind", "http.response.status_code")
        for k in span.attributes
    )


def test_full_lazy_native75x5001_without_mutating_return(engine, runtime, monkeypatch):
    name, _data = populated(engine, count=75, dimension=5001)
    runtime[1].clear()
    observed = []
    original = impl._Call.result

    def result(state, value):
        observed.append(value)
        return original(state, value)

    monkeypatch.setattr(impl._Call, "result", result)
    result = engine.query(
        name, filter="id>=0", limit=75, output_fields=["id", "vector", "text", "flag"]
    )
    assert result is observed[-1]
    hybrid = getattr(pymilvus.client.types, "HybridExtraList", None)
    if hybrid is not None:
        raw = native_storage(result, hybrid)
        assert not any(raw["_materialized_bitmap"])
    out = json.loads(one(runtime[1], "query").attributes[O])
    assert len(out) == 75 and len(out[0]["vector"]) == 5001
    assert out == json_value(result) and out[0]["flag"] is False


@pytest.mark.parametrize(
    "flag,value",
    [
        ("respan_enable_content_tracing", False),
        ("trace_content", False),
        ("override_enable_content_tracing", False),
        (context._SUPPRESS_INSTRUMENTATION_KEY, True),
        (SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True),
    ],
)
def test_actual_capture_suppression(engine, runtime, flag, value):
    name, _ = populated(engine)
    runtime[1].clear()
    token = context.attach(context.set_value(flag, value))
    try:
        assert engine.query(name, filter="id>=0")
    finally:
        context.detach(token)
    rows = runtime[1].get_finished_spans()
    if value is True:
        assert not rows
    else:
        bodyless(rows)


@pytest.mark.parametrize("env", ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
@pytest.mark.parametrize("value", ["false", "0", "off", "no"])
def test_environment_capture(engine, runtime, monkeypatch, env, value):
    name, _ = populated(engine)
    runtime[1].clear()
    monkeypatch.setenv(env, value)
    engine.query(name, filter="id>=0")
    bodyless(runtime[1].get_finished_spans())


def test_sampling_constructor_and_supplied_bounds_before_extraction(
    engine, runtime, monkeypatch
):
    name, _ = populated(engine)
    runtime[2].deactivate()

    def forbidden(*a, **kw):
        raise AssertionError("telemetry conversion")

    monkeypatch.setattr(impl, "json_value", forbidden)
    for config in (
        {"capture_content": False},
        {"context": context.set_value("respan_enable_content_tracing", False)},
        {"tracer_provider": TracerProvider(sampler=ALWAYS_OFF)},
    ):
        p = config.get("tracer_provider", runtime[0])
        owner = MilvusInstrumentor(**{**config, "tracer_provider": p})
        owner.activate()
        try:
            assert engine.query(name, filter="id>=0")
        finally:
            owner.deactivate()


@pytest.mark.parametrize("kind", ["active", "finished", "unknown", "remote"])
def test_ancestor_bounds(engine, runtime, kind):
    name, _ = populated(engine)
    runtime[1].clear()
    if kind in ("active", "finished"):
        parent = runtime[0].get_tracer("app").start_span("parent")
        parent.set_attribute("respan_enable_content_tracing", False)
        if kind == "finished":
            parent.end()
            runtime[1].clear()
    else:
        parent = NonRecordingSpan(SpanContext(1, 2, kind == "remote", TraceFlags(1)))
    with trace.use_span(parent, end_on_exit=False):
        engine.query(name, filter="id>=0")
    span = one(runtime[1], "query")
    if kind == "remote":
        assert I in span.attributes
    else:
        bodyless([span])
    if kind == "active":
        parent.end()


def test_actual_error_no_fake_output(engine, runtime):
    runtime[1].clear()
    with pytest.raises(pymilvus.exceptions.MilvusException):
        engine.query("absent", filter="id>=0")
    span = one(runtime[1], "query")
    assert (
        span.status.status_code == StatusCode.ERROR
        and O not in span.attributes
        and "error.type" in span.attributes
    )


@pytest.mark.parametrize(
    "stage", ["start", "set", "attach", "detach", "end", "combined"]
)
def test_native_outcome_context_and_bareerror_faults(
    engine, runtime, monkeypatch, stage
):
    name, _ = populated(engine)
    runtime[1].clear()
    before = context.get_current()

    class Bare(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_status(Status(StatusCode.ERROR))
            span.add_event("private", {"body": "PRIVATE"})

        def on_end(self, span):
            pass

    runtime[0].add_span_processor(Bare())

    def fault(*a, **kw):
        raise RuntimeError("observer")

    if stage == "start":
        monkeypatch.setattr(runtime[0], "get_tracer", fault)
    elif stage == "set":
        original = Span.set_attribute

        def setter(span, key, value):
            original(span, key, value)
            if key == I:
                raise RuntimeError("after mutation")

        monkeypatch.setattr(Span, "set_attribute", setter)
    elif stage == "attach":
        original = context.attach

        def attach(carrier):
            original(carrier)
            raise RuntimeError("after mutation")

        monkeypatch.setattr(context, "attach", attach)
    elif stage == "detach":
        monkeypatch.setattr(context, "detach", fault)
    elif stage == "end":
        original = Span.end

        def end(span, *a, **kw):
            original(span, *a, **kw)
            raise RuntimeError("after mutation")

        monkeypatch.setattr(Span, "end", end)
    else:
        monkeypatch.setattr(impl._Call, "result", fault)
        monkeypatch.setattr(impl._Call, "scrub", fault)
    assert engine.query(name, filter="id>=0") and context.get_current() is before
    assert impl._MANAGER is None or not list(impl._MANAGER.observer.states)
    if stage in ("set", "attach", "combined"):
        rows = runtime[1].get_finished_spans()
        bodyless(rows)
        assert all(
            s.status.status_code == StatusCode.ERROR
            and s.attributes["traceloop.entity.path"] == ""
            and "error.type" not in s.attributes
            for s in rows
        )


def test_late_readable_end_veto(engine, runtime, monkeypatch):
    name, _ = populated(engine)
    runtime[1].clear()
    original = Span.end

    def end(span, *a, **kw):
        span.add_event("private", {"body": "PRIVATE"})
        span.set_status(Status(StatusCode.ERROR, "PRIVATE"))
        span.set_attribute("error.message", "PRIVATE")
        token = context.attach(
            context.set_value("respan_enable_content_tracing", False)
        )
        try:
            return original(span, *a, **kw)
        finally:
            context.detach(token)

    monkeypatch.setattr(Span, "end", end)
    engine.query(name, filter="id>=0")
    bodyless(runtime[1].get_finished_spans())


@pytest.mark.parametrize("mode", ["exhaust", "close", "veto", "deactivate", "gc"])
@pytest.mark.skipif(
    not hasattr(MilvusClient, "query_iterator"),
    reason="Native2.4.1 has no client iterator API",
)
def test_actual_iterator_lifecycle_identity(engine, runtime, mode):
    name, _ = populated(engine, count=5)
    runtime[1].clear()
    iterator = engine.query_iterator(
        name, batch_size=2, filter="id>=0", output_fields=["id", "vector"]
    )
    assert not runtime[1].get_finished_spans()
    token = (
        context.attach(context.set_value("respan_enable_content_tracing", False))
        if mode == "veto"
        else None
    )
    try:
        if mode in ("exhaust", "veto"):
            batches = []
            while True:
                value = iterator.next()
                batches.append(json_value(value))
                if not len(value):
                    break
            iterator.close()
        elif mode == "close":
            assert iterator.close() is None
        elif mode == "deactivate":
            runtime[2].deactivate()
            assert iterator.next()
            iterator.close()
        else:
            del iterator
            gc.collect()
    finally:
        if token:
            context.detach(token)
    rows = spans(runtime[1], "query_iterator")
    assert len(rows) == 1
    if mode == "exhaust":
        assert (
            rows[0].status.status_code == StatusCode.OK
            and json.loads(rows[0].attributes[O]) == batches
        )
    elif mode == "veto":
        bodyless(rows)
    else:
        assert rows[0].status.status_code == StatusCode.UNSET
    assert impl._MANAGER is None or not list(impl._MANAGER.observer.states)


def test_native_pending_siblings(engine, runtime, monkeypatch):
    name, _ = populated(engine)
    runtime[1].clear()
    barrier = threading.Barrier(2)
    original = impl._Call.result

    def result(state, value):
        if state.operation == "client.query":
            barrier.wait(timeout=10)
        return original(state, value)

    monkeypatch.setattr(impl._Call, "result", result)
    with runtime[0].get_tracer("app").start_as_current_span("parent"):
        carrier = context.get_current()

        def call():
            token = context.attach(carrier)
            try:
                return engine.query(name, filter="id>=0")
            finally:
                context.detach(token)

        with ThreadPoolExecutor(2) as pool:
            values = list(pool.map(lambda _: call(), range(2)))
    rows = spans(runtime[1], "query")
    assert (
        len(values) == len(rows) == 2
        and rows[0].parent == rows[1].parent
        and all(O in s.attributes for s in rows)
    )


def test_native_shared_foreign_partial_rollback(runtime, monkeypatch):
    p, _m, owner = runtime
    other = MilvusInstrumentor(tracer_provider=p)
    other.activate()
    owner.deactivate()
    with pytest.raises(RuntimeError):
        MilvusInstrumentor(tracer_provider=p, capture_content=False).activate()
    other.deactivate()
    restored = inspect.getattr_static(MilvusClient, "query")
    patch = impl._Manager.patch
    count = []

    def failing(manager, *args):
        patch(manager, *args)
        count.append(1)
        if len(count) == 4:
            raise RuntimeError("partial")

    monkeypatch.setattr(impl._Manager, "patch", failing)
    with pytest.raises(RuntimeError):
        owner.activate()
    assert (
        inspect.getattr_static(MilvusClient, "query") is restored
        and impl._MANAGER is None
    )


@pytest.mark.parametrize(
    "value",
    [
        r"Bearer \"PRIVATE SPACE\" Basic \'PRIVATE TWO\'",
        'Authorization: Bearer "PRIVATE SPACE"',
        "token=PRIVATE",
        "https://name:PRIVATE@host/path?api_key=PRIVATE",
    ],
)
def test_redaction_schema_unknown_hooks(value):
    assert "PRIVATE" not in redact_text(value) and redact_text(
        redact_text(value)
    ) == redact_text(value)
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append("eq")
            raise AssertionError

        def __hash__(cls):
            calls.append("hash")
            raise AssertionError

    class Unknown(metaclass=Meta):
        def __iter__(self):
            calls.append("iter")
            raise AssertionError

        def __str__(self):
            calls.append("str")
            raise AssertionError

    assert json_value(Unknown()) is None and not calls
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "default": "PRIVATE", "example": "PRIVATE"}
        },
        "required": ["api_key"],
    }
    assert "api_key" in json_value(schema)[
        "properties"
    ] and "PRIVATE" not in json.dumps(json_value(schema))


@pytest.mark.skipif(
    not hasattr(pymilvus, "AsyncMilvusClient"), reason="Native2.4.1 has no async client"
)
@pytest.mark.asyncio
async def test_actual_async_client(native_server, engine, runtime):
    name, _ = populated(engine)
    runtime[1].clear()
    client = pymilvus.AsyncMilvusClient(uri=native_server)
    result = await client.query(name, filter="id>=0", output_fields=["id", "vector"])
    assert result and json.loads(one(runtime[1], "query").attributes[O]) == json_value(
        result
    )
    await client.close()


@pytest.mark.skipif(
    not hasattr(MilvusClient, "hybrid_search"),
    reason="Native2.4.1 has no hybrid client API",
)
def test_native_hybrid_typed_requests(engine, runtime):
    name, _ = populated(engine)
    indexes = engine.prepare_index_params()
    indexes.add_index(field_name="vector", index_type="FLAT", metric_type="L2")
    engine.create_index(name, indexes)
    engine.load_collection(name)
    runtime[1].clear()
    reqs = [
        AnnSearchRequest([[0.0] * 4], "vector", {"metric_type": "L2", "params": {}}, 2),
        AnnSearchRequest([[1.0] * 4], "vector", {"metric_type": "L2", "params": {}}, 2),
    ]
    result = engine.hybrid_search(
        name, reqs, RRFRanker(), limit=2, output_fields=["id", "vector"]
    )
    assert result and json.loads(
        one(runtime[1], "hybrid_search").attributes[O]
    ) == json_value(result)
    inp = json.loads(one(runtime[1], "hybrid_search").attributes[I])
    assert len(inp["arguments"]["reqs"]) == 2


@pytest.mark.skipif(
    not hasattr(MilvusClient, "optimize"), reason="Minimum has no OptimizeTask API"
)
def test_actual_optimize_task_native_error_identity(engine, runtime):
    name, _ = populated(engine)
    runtime[1].clear()
    with runtime[0].get_tracer("app").start_as_current_span("parent"):
        task = engine.optimize(name, target_size="invalid", wait=False)
        assert type(task).__name__ == "OptimizeTask"
        with pytest.raises(pymilvus.exceptions.ParamError) as found:
            task.result(timeout=10)
        raw = native_storage(task, type(task))
        assert found.value is raw["_exception"]
    span = one(runtime[1], "optimize")
    assert span.status.status_code == StatusCode.ERROR and span.parent is not None
    assert O not in span.attributes
    assert not list(impl._MANAGER.observer.states)


@pytest.mark.skipif(
    not hasattr(MilvusClient, "optimize"), reason="Minimum has no OptimizeTask API"
)
@pytest.mark.parametrize("mode", ["parent", "suppressed"])
def test_actual_native_cpu_worker_bounds(engine, runtime, mode):
    name, _ = populated(engine)
    runtime[1].clear()
    with runtime[0].get_tracer("app").start_as_current_span("parent"):
        token = (
            context.attach(
                context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
            )
            if mode == "suppressed"
            else None
        )
        try:
            task = engine.optimize(name, target_size="1MB", wait=False)
            task.result(timeout=0.05)
            task.cancel()
            task.join(timeout=3)
        finally:
            if token is not None:
                context.detach(token)
            engine.close()
    rows = runtime[1].get_finished_spans()
    native = [s for s in rows if s.attributes.get("db.system") == "milvus"]
    if mode == "suppressed":
        assert not native
    else:
        optimize = one(runtime[1], "optimize")
        children = [s for s in native if s.attributes["db.operation"] != "optimize"]
        assert children and all(
            s.parent.span_id == optimize.context.span_id for s in children
        )


def test_known_request_and_client_subclass_getters_are_not_observed(engine, runtime):
    calls = []

    class Request(AnnSearchRequest):
        @property
        def param(self):
            calls.append("getter")
            raise AssertionError("telemetry getter")

    request = Request([[0.0] * 5001], "vector", {"flag": False}, 0)
    converted = json_value(request)
    assert converted["_param"]["flag"] is False and len(converted["_data"][0]) == 5001
    assert not calls


def test_runtime_descriptor_restore_and_held_setter_veto(engine, runtime, monkeypatch):
    name, _ = populated(engine)
    runtime[1].clear()
    original = Span.set_attribute

    def setter(span, key, value):
        if key == I:
            token = context.attach(
                context.set_value("respan_enable_content_tracing", False)
            )
            context.detach(token)
        return original(span, key, value)

    monkeypatch.setattr(Span, "set_attribute", setter)
    assert engine.query(name, filter="id>=0")
    bodyless(runtime[1].get_finished_spans())
    owner = runtime[2]
    owner.deactivate()
    rt = context._RUNTIME_CONTEXT
    before = dict(vars(rt))
    owner.activate()
    owner.deactivate()
    assert vars(rt) == before


def test_actual_upstream_metric_fault_preserves_native_result(
    engine, runtime, monkeypatch
):
    import opentelemetry.instrumentation.milvus as upstream
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics._internal.instrument import _Histogram

    name, _ = populated(engine)
    runtime[2].deactivate()
    metric_provider = MeterProvider()
    monkeypatch.setattr(
        upstream, "get_meter", lambda *a, **kw: metric_provider.get_meter("native-test")
    )

    def fault(*a, **kw):
        raise RuntimeError("metric observer")

    monkeypatch.setattr(_Histogram, "record", fault)
    runtime[2].activate()
    assert runtime[2]._is_instrumented and not impl._MANAGER.suppressed()
    runtime[1].clear()
    actual = engine.query(name, filter="id>=0")
    span = one(runtime[1], "query")
    assert actual and json.loads(span.attributes[O]) == json_value(actual)
    assert (
        "error.type" not in span.attributes and span.status.status_code == StatusCode.OK
    )
    metric_provider.shutdown()


@pytest.mark.skipif(
    not hasattr(MilvusClient, "query_iterator"), reason="Minimum has no client iterator"
)
def test_actual_iterator_detach_mutate_then_raise(engine, runtime, monkeypatch):
    name, _ = populated(engine, count=3)
    iterator = engine.query_iterator(
        name, batch_size=2, filter="id>=0", output_fields=["id", "vector"]
    )
    ambient = context.get_current()
    raw = impl._MANAGER.original_detach

    def detach(token):
        raw(token)
        raise RuntimeError("observer detach after native reset")

    monkeypatch.setattr(context, "detach", detach)
    try:
        result = iterator.next()
    finally:
        monkeypatch.undo()
        iterator.close()
    assert len(result) == 2
    assert context.get_current() is ambient


def test_actual_sdk_nonsecret_url_fields(engine, runtime):
    name, _ = populated(engine, count=1)
    url = "https://user:PRIVATE@host/path?api_key=PRIVATE&zero=0"
    engine.insert(name, [{"id": 55, "vector": [1.0, 0.0, 0.0, 0.0], "url": url}])
    span = runtime[1].get_finished_spans()[-1]
    data = json.loads(span.attributes["traceloop.entity.input"])
    value = data["arguments"]["data"][0]["url"]
    assert "PRIVATE" not in value and "zero=0" in value


def test_actual_canonical_priority_under_native_attribute_limit(engine, runtime):
    class Fill(SpanProcessor):
        def on_start(self, span, parent_context=None):
            for index in range(150):
                span.set_attribute("respan.metadata.extra_" + str(index), index)

        def on_end(self, span):
            pass

    name, _ = populated(engine, count=1)
    runtime[0].add_span_processor(Fill())
    runtime[1].clear()
    result = engine.query(name, filter="id>=0")
    assert result
    span = runtime[1].get_finished_spans()[-1]
    assert "traceloop.entity.input" in span.attributes
    assert "traceloop.entity.output" in span.attributes
