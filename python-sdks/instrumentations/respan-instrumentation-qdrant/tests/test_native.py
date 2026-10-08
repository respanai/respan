import inspect
import json
from datetime import date, datetime, timezone
from typing import ClassVar

import numpy as np
import pytest
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import StatusCode
from qdrant_client import AsyncQdrantClient, QdrantClient, grpc, models
from respan_instrumentation_qdrant import QdrantInstrumentor
from respan_instrumentation_qdrant import _native_instrumentation as impl
from respan_instrumentation_qdrant._serialization import (
    json_string,
    json_value,
    redact_text,
)


def bodyless(spans):
    assert spans
    for span in spans:
        assert "traceloop.entity.input" not in span.attributes
        assert "traceloop.entity.output" not in span.attributes
        assert "error.message" not in span.attributes
        assert not span.events and span.status.description is None


@pytest.mark.parametrize(
    "method",
    [
        "get_collections",
        "get_collection",
        "collection_exists",
        "count",
        "retrieve",
        "scroll",
        "query_points",
        "query_batch_points",
        "search",
    ],
)
def test_native_reads(engine, runtime, method):
    if not hasattr(engine, method):
        pytest.skip("Selected native SDK does not expose this API")
    args = {
        "get_collections": {},
        "get_collection": {"collection_name": "native"},
        "collection_exists": {"collection_name": "native"},
        "count": {"collection_name": "native"},
        "retrieve": {"collection_name": "native", "ids": [0, 1], "with_vectors": True},
        "scroll": {"collection_name": "native", "with_vectors": True, "limit": 75},
        "query_points": {
            "collection_name": "native",
            "query": [1.0, 0.0, 0.0],
            "with_vectors": True,
        },
        "query_batch_points": {
            "collection_name": "native",
            "requests": [models.QueryRequest(query=[1.0, 0.0, 0.0], with_vector=True)],
        }
        if hasattr(models, "QueryRequest")
        else {},
        "search": {
            "collection_name": "native",
            "query_vector": [1.0, 0.0, 0.0],
            "with_vectors": True,
        },
    }[method]
    result = getattr(engine, method)(**args)
    span = runtime[1].get_finished_spans()[-1]
    assert json.loads(span.attributes["traceloop.entity.output"]) == json_value(result)
    assert span.attributes["respan.entity.log_type"] == "task"
    assert span.attributes["db.system"] == "qdrant"
    assert "traceloop.span.kind" not in span.attributes
    assert (
        "status_code" not in span.attributes
        and "http.response.status_code" not in span.attributes
    )
    assert not any("model" in key or "usage" in key for key in span.attributes)


@pytest.mark.parametrize(
    "method",
    [
        "set_payload",
        "overwrite_payload",
        "delete_payload",
        "clear_payload",
        "update_vectors",
        "delete",
        "upload_points",
        "upload_collection",
    ],
)
def test_native_writes(engine, runtime, method):
    if method == "set_payload":
        result = engine.set_payload(
            "native",
            {
                "nested": [{"value": 0, "flag": False}],
                "authorization": 'Bearer "PRIVATE SPACE"',
            },
            points=[0],
        )
    elif method == "overwrite_payload":
        result = engine.overwrite_payload(
            "native", {"flag": False, "empty": "", "zero": 0}, points=[0]
        )
    elif method == "delete_payload":
        result = engine.delete_payload("native", ["body"], points=[0])
    elif method == "clear_payload":
        result = engine.clear_payload("native", points_selector=[0])
    elif method == "update_vectors":
        result = engine.update_vectors(
            "native", [models.PointVectors(id=0, vector=[0.0, 1.0, 0.0])]
        )
    elif method == "delete":
        result = engine.delete("native", points_selector=[0])
    elif method == "upload_points":
        result = engine.upload_points(
            "native",
            (models.PointStruct(id=i + 10, vector=[1.0, 0.0, 0.0]) for i in range(75)),
        )
        assert engine.count("native").count == 77
    else:
        result = engine.upload_collection(
            "native",
            vectors=np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            ids=[10, 11],
            payload=[{"flag": False}, {}],
        )
    spans = [s for s in runtime[1].get_finished_spans() if s.name == "qdrant." + method]
    assert len(spans) == 1
    assert json.loads(spans[0].attributes["traceloop.entity.output"]) == json_value(
        result
    )
    assert "PRIVATE" not in str(spans[0].attributes)


def test_full_native_75x5001(engine, runtime):
    engine.create_collection(
        "full",
        vectors_config=models.VectorParams(size=5001, distance=models.Distance.DOT),
    )
    points = [
        models.PointStruct(
            id=i,
            vector=[float(j % 7) for j in range(5001)],
            payload={
                "flag": False,
                "zero": 0,
                "empty": "",
                "history": list(range(75)),
                "api_key": "PRIVATE",
            },
        )
        for i in range(75)
    ]
    engine.upsert("full", points=points)
    write = runtime[1].get_finished_spans()[-1]
    assert (
        len(json.loads(write.attributes["traceloop.entity.input"])["kwargs"]["points"])
        == 75
    )
    result = engine.retrieve("full", ids=list(range(75)), with_vectors=True)
    output = json.loads(
        runtime[1].get_finished_spans()[-1].attributes["traceloop.entity.output"]
    )
    assert len(output) == len(result) == 75 and len(output[-1]["vector"]) == 5001
    assert output[-1]["vector"] == result[-1].vector
    assert output[-1]["payload"] == {
        "flag": False,
        "zero": 0,
        "empty": "",
        "history": list(range(75)),
        "api_key": "[REDACTED]",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("respan_enable_content_tracing", False),
        ("trace_content", False),
        ("override_enable_content_tracing", False),
        (_SUPPRESS_INSTRUMENTATION_KEY, True),
        (SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True),
    ],
)
def test_capture_suppression(engine, runtime, key, value):
    token = context.attach(context.set_value(key, value))
    try:
        result = engine.retrieve("native", [0], with_vectors=True)
        assert result[0].payload["body"] == "PRIVATE"
    finally:
        context.detach(token)
    if value is True:
        assert not runtime[1].get_finished_spans()
    else:
        bodyless(runtime[1].get_finished_spans())


@pytest.mark.parametrize(
    "variable,value",
    [
        ("RESPAN_TRACE_CONTENT", "false"),
        ("TRACELOOP_TRACE_CONTENT", "0"),
        ("RESPAN_TRACE_CONTENT", "OFF"),
        ("TRACELOOP_TRACE_CONTENT", "no"),
    ],
)
def test_environment_capture(engine, runtime, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    assert engine.retrieve("native", [0])[0].id == 0
    bodyless(runtime[1].get_finished_spans())


@pytest.mark.parametrize("disabled", ["constructor", "sampling", "supplied", "ambient"])
def test_gate_before_unknown_conversion(engine, disabled):
    class Point(models.PointStruct):
        calls: ClassVar[int] = 0

        def model_dump(self, *args, **kwargs):
            Point.calls += 1
            return super().model_dump(*args, **kwargs)

    point = Point(id=22, vector=[1.0, 0.0, 0.0])
    provider = (
        TracerProvider(sampler=ALWAYS_OFF)
        if disabled == "sampling"
        else TracerProvider()
    )
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    owner = QdrantInstrumentor(
        tracer_provider=provider,
        capture_content=disabled != "constructor",
        context=context.set_value("respan_enable_content_tracing", False)
        if disabled == "supplied"
        else context.Context(),
    )
    owner.activate()
    token = context.attach(
        context.set_value("respan_enable_content_tracing", disabled != "ambient")
    )
    try:
        engine.upsert("native", [point])
        assert Point.calls == 0
    finally:
        context.detach(token)
        owner.deactivate()
        provider.shutdown()
    if disabled == "sampling":
        assert not memory.get_finished_spans()
    else:
        bodyless(memory.get_finished_spans())


def test_native_error_no_fake_output(engine, runtime):
    with pytest.raises(ValueError) as raised:
        engine.retrieve("missing", [0])
    span = runtime[1].get_finished_spans()[-1]
    assert raised.value.args and span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == type(raised.value).__name__
    assert "traceloop.entity.output" not in span.attributes
    assert (
        "status_code" not in span.attributes
        and "http.response.status_code" not in span.attributes
    )


@pytest.mark.parametrize("kind", ["active", "finished", "unknownlocal", "remote"])
def test_native_parent_bounds(engine, runtime, kind):
    provider, memory, _ = runtime
    if kind in ("active", "finished"):
        parent = provider.get_tracer("app").start_span(
            "parent", attributes={"respan_enable_content_tracing": False}
        )
        parent.set_attribute("respan_enable_content_tracing", True)
        if kind == "finished":
            parent.end()
    else:
        parent = trace.NonRecordingSpan(
            trace.SpanContext(
                trace_id=123,
                span_id=456,
                is_remote=kind == "remote",
                trace_flags=trace.TraceFlags(1),
            )
        )
    token = context.attach(trace.set_span_in_context(parent))
    try:
        assert engine.retrieve("native", [0])[0].id == 0
    finally:
        context.detach(token)
        if kind == "active":
            parent.end()
    own = [s for s in memory.get_finished_spans() if s.name.startswith("qdrant.")]
    if kind == "remote":
        assert "traceloop.entity.output" in own[-1].attributes
    else:
        bodyless(own)


def test_actual_readable_end_veto(engine, runtime, monkeypatch):
    original = Span.end

    def end(span, *args, **kwargs):
        if span.name.startswith("qdrant."):
            token = context.attach(
                context.set_value("respan_enable_content_tracing", False)
            )
            try:
                return original(span, *args, **kwargs)
            finally:
                context.detach(token)
        return original(span, *args, **kwargs)

    monkeypatch.setattr(Span, "end", end)
    assert engine.retrieve("native", [0])[0].payload["body"] == "PRIVATE"
    bodyless(runtime[1].get_finished_spans())


@pytest.mark.parametrize(
    "stage", ["input", "result", "setter", "attach", "detach", "end"]
)
def test_telemetry_fault_native_outcome_and_context(
    engine, runtime, monkeypatch, stage
):
    ambient = context.get_current()

    def fault(*args, **kwargs):
        raise RuntimeError("observer fault")

    if stage == "input":
        monkeypatch.setattr(impl, "json_string", fault)
    elif stage == "result":
        monkeypatch.setattr(impl._Call, "observe", fault)
    elif stage == "setter":
        monkeypatch.setattr(Span, "set_attribute", fault)
    elif stage == "attach":
        monkeypatch.setattr(context, "attach", fault)
    elif stage == "detach":
        monkeypatch.setattr(context, "detach", fault)
    else:
        monkeypatch.setattr(Span, "end", fault)
    assert engine.retrieve("native", [0])[0].id == 0
    assert context.get_current() is ambient
    assert not list(impl._MANAGER.states)


def test_compatible_owners_foreign_and_partial_rollback(engine, runtime, monkeypatch):
    provider, _, owner = runtime
    native = inspect.getattr_static(QdrantClient, "get_collections")
    second = QdrantInstrumentor(tracer_provider=provider)
    second.activate()
    owner.deactivate()
    assert inspect.getattr_static(QdrantClient, "get_collections") is native
    with pytest.raises(ValueError):
        QdrantInstrumentor(capture_content=False, tracer_provider=provider).activate()

    def foreign(*args, **kwargs):
        return native(*args, **kwargs)

    monkeypatch.setattr(QdrantClient, "get_collections", foreign)
    second.deactivate()
    assert inspect.getattr_static(QdrantClient, "get_collections") is foreign
    assert engine.get_collections().collections


def test_native_proto_input(engine, runtime):
    point = grpc.PointStruct(
        id=grpc.PointId(num=10),
        vectors=grpc.Vectors(vector=grpc.Vector(data=[1.0, 0.0, 0.0])),
    )
    runtime[2].deactivate()
    with pytest.raises(AttributeError) as bare:
        engine.upsert("native", [point])
    runtime[2].activate()
    with pytest.raises(AttributeError) as observed:
        engine.upsert("native", [point])
    assert observed.value.args == bare.value.args
    span = runtime[1].get_finished_spans()[-1]
    assert (
        json.loads(span.attributes["traceloop.entity.input"])["args"][1][0]["id"]["num"]
        == "10"
    )
    assert "traceloop.entity.output" not in span.attributes


def test_safe_full_schema_unknown_hooks_and_dates():
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("hash")
            return 0

        def __eq__(cls, other):
            calls.append("eq")
            return False

    class Unknown(metaclass=Meta):
        def __getattribute__(self, key):
            calls.append(key)
            return super().__getattribute__(key)

        def __str__(self):
            calls.append("str")
            return "PRIVATE"

        def __iter__(self):
            calls.append("iter")
            return iter([])

    value = Unknown()
    assert json_value(value) is None and not calls
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "default": "PRIVATE"},
            "flag": {"type": "boolean", "default": False},
        },
    }
    out = json_value(schema)
    assert out["properties"]["api_key"] == {"type": "string", "default": "[REDACTED]"}
    assert out["properties"]["flag"]["default"] is False
    assert json_value(date(2026, 10, 9)) == "2026-10-09"
    assert (
        json_value(datetime(2026, 10, 9, tzinfo=timezone.utc))
        == "2026-10-09T00:00:00+00:00"
    )
    assert json.loads(json_string(list(range(5001)))) == list(range(5001))


@pytest.mark.parametrize(
    "text",
    [
        'Bearer "PRIVATE SPACE"',
        r"Bearer \"PRIVATE SPACE\"",
        'Basic "PRIVATE SPACE"',
        '{"authorization":"Bearer PRIVATE","false":false,"zero":0}',
        "https://user:PRIVATE@host/path?api_key=PRIVATE&zero=0",
    ],
)
def test_valid_idempotent_credentials(text):
    clean = redact_text(text)
    assert "PRIVATE" not in clean and redact_text(clean) == clean
    if text.startswith("{"):
        assert json.loads(clean)["false"] is False
    if text.startswith("https:"):
        assert "zero=0" in clean


async def test_native_async_local_outcomes(runtime):
    client = AsyncQdrantClient(":memory:")
    try:
        assert await client.create_collection(
            "async",
            vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT),
        )
        assert await client.upsert(
            "async",
            [models.PointStruct(id=0, vector=[1.0, 0.0, 0.0], payload={"flag": False})],
        )
        result = await client.retrieve("async", [0], with_vectors=True)
        assert result[0].vector == [1.0, 0.0, 0.0]
        assert json.loads(
            runtime[1].get_finished_spans()[-1].attributes["traceloop.entity.output"]
        ) == json_value(result)
    finally:
        await client.close()


def test_private_child_does_not_deny_public_sibling(engine, runtime):
    provider, memory, _ = runtime
    with provider.get_tracer("app").start_as_current_span("parent"):
        with provider.get_tracer("app").start_as_current_span(
            "private-child", attributes={"respan_enable_content_tracing": False}
        ):
            assert engine.retrieve("native", [0])[0].id == 0
        assert engine.retrieve("native", [1])[0].id == 1
    spans = [s for s in memory.get_finished_spans() if s.name == "qdrant.retrieve"]
    bodyless(spans[:1])
    assert json.loads(spans[1].attributes["traceloop.entity.output"])[0]["id"] == 1


def test_native_attribute_capacity_keeps_canonical_io(engine, runtime):
    from opentelemetry.sdk.trace import SpanProcessor

    class Metadata(SpanProcessor):
        def on_start(self, span, parent_context=None):
            for i in range(150):
                span.set_attribute(f"respan.metadata.extra_{i}", i)

    runtime[0].add_span_processor(Metadata())
    result = engine.retrieve("native", [0])
    attrs = runtime[1].get_finished_spans()[-1].attributes
    assert json.loads(attrs["traceloop.entity.input"])["args"][1] == [0]
    assert json.loads(attrs["traceloop.entity.output"]) == json_value(result)
    assert attrs["db.system"] == "qdrant"
    assert attrs["db.operation"] == "retrieve"
    assert attrs["db.name"] == "native"
    assert attrs["traceloop.entity.name"] == "qdrant.retrieve"


def test_end_mutate_privacy_then_raise_scrubs_held_span(engine, runtime, monkeypatch):
    held = []

    def end(span, *args, **kwargs):
        held.append(span)
        span.set_attribute("respan_enable_content_tracing", False)
        raise RuntimeError("observer end fault")

    monkeypatch.setattr(Span, "end", end)
    assert engine.retrieve("native", [0])[0].id == 0
    assert held and not list(impl._MANAGER.states)
    bodyless(held)


def test_bare_error_private_remains_error(engine, runtime):
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import Status

    class BareError(SpanProcessor):
        def on_start(self, span, parent_context=None):
            if span.name.startswith("qdrant."):
                span.set_status(Status(StatusCode.ERROR, "PRIVATE"))
                span.add_event("PRIVATE", {"body": "PRIVATE"})

    runtime[0].add_span_processor(BareError())
    token = context.attach(context.set_value("respan_enable_content_tracing", False))
    try:
        assert engine.retrieve("native", [0])[0].id == 0
    finally:
        context.detach(token)
    spans = runtime[1].get_finished_spans()
    bodyless(spans)
    assert spans[-1].status.status_code == StatusCode.ERROR
    assert "error.type" not in spans[-1].attributes


def test_partial_activation_failure_restores_exact_native_methods(engine, monkeypatch):
    original = impl._Manager.patch
    methods = {
        name: inspect.getattr_static(QdrantClient, name)
        for name in ("retrieve", "get_collection", "get_collections")
    }
    count = 0

    def patch(manager, obj, name, replacement):
        nonlocal count
        original(manager, obj, name, replacement)
        count += 1
        if count == 3:
            raise RuntimeError("partial activation fault")

    monkeypatch.setattr(impl._Manager, "patch", patch)
    provider = TracerProvider()
    try:
        with pytest.raises(RuntimeError, match="partial activation"):
            QdrantInstrumentor(tracer_provider=provider).activate()
        assert impl._MANAGER is None and not impl._OWNERS
        for name, method in methods.items():
            assert inspect.getattr_static(QdrantClient, name) is method
        assert engine.retrieve("native", [0])[0].id == 0
    finally:
        provider.shutdown()


@pytest.mark.parametrize("stage", ["attach", "detach"])
def test_mutate_then_raise_context_fault(engine, runtime, monkeypatch, stage):
    ambient = context.get_current()
    native = getattr(context, stage)

    def fault(*args, **kwargs):
        native(*args, **kwargs)
        raise RuntimeError("observer mutated context fault")

    monkeypatch.setattr(context, stage, fault)
    assert engine.retrieve("native", [0])[0].id == 0
    assert context.get_current() is ambient
    assert impl._CURRENT.get() is None and not list(impl._MANAGER.states)


def test_native_sparse_prefetch_fusion_and_multivectors(engine, runtime):
    if not hasattr(models, "Prefetch"):
        pytest.skip("Qdrant 1.9 does not expose universal query and prefetch models")
    engine.create_collection(
        "typed",
        vectors_config={
            "dense": models.VectorParams(size=3, distance=models.Distance.DOT),
            "multi": models.VectorParams(
                size=3,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    engine.upsert(
        "typed",
        [
            models.PointStruct(
                id=i,
                vector={
                    "dense": [1.0, float(i), 0.0],
                    "multi": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "sparse": models.SparseVector(
                        indices=[0, 3], values=[1.0, float(i + 1)]
                    ),
                },
                payload={"category": "even" if i % 2 == 0 else "odd", "flag": False},
            )
            for i in range(4)
        ],
    )
    result = engine.query_points(
        "typed",
        prefetch=[
            models.Prefetch(query=[1.0, 0.0, 0.0], using="dense", limit=4),
            models.Prefetch(
                query=models.SparseVector(indices=[0], values=[1.0]),
                using="sparse",
                limit=4,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=4,
        with_vectors=True,
    )
    assert len(result.points) == 4
    assert json.loads(
        runtime[1].get_finished_spans()[-1].attributes["traceloop.entity.output"]
    ) == json_value(result)
    multi = engine.query_points(
        "typed", query=[[1.0, 0.0, 0.0]], using="multi", with_vectors=True
    )
    assert multi.points and len(multi.points[0].vector["multi"]) == 2
    facet = engine.facet("typed", "category")
    assert sum(hit.count for hit in facet.hits) == 4


def test_native_http_typed_result_and_sourced_status(native_http, runtime):
    from qdrant_client.http.exceptions import UnexpectedResponse

    client = QdrantClient(url=native_http[0])
    try:
        result = client.retrieve("native", [1], with_vectors=True)
        assert type(result[0]) is models.Record
        assert json.loads(
            runtime[1].get_finished_spans()[-1].attributes["traceloop.entity.output"]
        ) == json_value(result)
        assert (
            "http.response.status_code"
            not in runtime[1].get_finished_spans()[-1].attributes
        )
        with pytest.raises(UnexpectedResponse) as raised:
            client.retrieve("missing", [1])
        assert raised.value.status_code == 400
        span = runtime[1].get_finished_spans()[-1]
        assert span.attributes["http.response.status_code"] == 400
        assert span.attributes["error.type"] == "UnexpectedResponse"
        assert span.status.status_code == StatusCode.ERROR
        assert "traceloop.entity.output" not in span.attributes
        assert "PRIVATE" not in str(span.attributes)
    finally:
        client.close()


async def test_pending_native_async_late_privacy_and_sibling(native_http, runtime):
    import asyncio

    from opentelemetry.sdk.trace import SpanProcessor

    held = []

    class Observe(SpanProcessor):
        def on_start(self, span, parent_context=None):
            if span.name == "qdrant.retrieve":
                held.append(span)

    runtime[0].add_span_processor(Observe())
    client = AsyncQdrantClient(url=native_http[0])
    controls = native_http[1]
    controls["delay"] = True
    try:
        with runtime[0].get_tracer("app").start_as_current_span("parent"):
            pending = asyncio.create_task(
                client.retrieve("native", [0], with_vectors=True)
            )
            assert await asyncio.to_thread(controls["pending"].wait, 5)
            held[0].set_attribute("respan_enable_content_tracing", False)
            sibling = await client.retrieve("native", [1], with_vectors=True)
            assert sibling[0].id == 1
            controls["release"].set()
            result = await pending
            assert result[0].id == 0
        spans = [
            s for s in runtime[1].get_finished_spans() if s.name == "qdrant.retrieve"
        ]
        public = next(s for s in spans if s.context.span_id == held[1].context.span_id)
        private = next(s for s in spans if s.context.span_id == held[0].context.span_id)
        assert json.loads(public.attributes["traceloop.entity.output"])[0]["id"] == 1
        bodyless([private])
        assert not list(impl._MANAGER.states)
    finally:
        controls["release"].set()
        await client.close()


def test_actual_native_discard_late_readable_veto(engine, runtime, monkeypatch):
    from opentelemetry.trace import Status

    end = Span.end

    def fail(self, *args):
        raise RuntimeError("observation failure")

    def veto(span, *args, **kwargs):
        if span.name.startswith("qdrant."):
            span.add_event("private", {"body": "PRIVATE"})
            span.set_status(Status(StatusCode.ERROR, "PRIVATE"))
            span.set_attribute("error.message", "PRIVATE")
            token = context.attach(
                context.set_value("respan_enable_content_tracing", False)
            )
            try:
                return end(span, *args, **kwargs)
            finally:
                context.detach(token)
        return end(span, *args, **kwargs)

    monkeypatch.setattr(impl._Call, "observe", fail)
    monkeypatch.setattr(Span, "end", veto)
    value = engine.retrieve("native", [0], with_vectors=True)
    assert len(value) == 1
    rows = runtime[1].get_finished_spans()
    assert len(rows) == 1
    row = rows[0]
    assert (
        not row.events
        and row.status.description is None
        and "error.message" not in row.attributes
    )

    bodyless(rows)
    assert row.status.status_code == StatusCode.ERROR
    assert "error.type" not in row.attributes
    assert not list(impl._MANAGER.states)


def test_foreign_native_wrapper_reactivation(engine, runtime, monkeypatch):
    wrapped = QdrantClient.retrieve

    def foreign(*args, **kwargs):
        return wrapped(*args, **kwargs)

    monkeypatch.setattr(QdrantClient, "retrieve", foreign)
    runtime[2].deactivate()
    runtime[2].activate()
    before = len(runtime[1].get_finished_spans())
    assert engine.retrieve("native", [0])[0].id == 0
    spans = runtime[1].get_finished_spans()[before:]
    assert len(spans) == 1
