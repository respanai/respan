"""Actual released Chroma persistent/Rust/HTTP operations and native models."""

import asyncio
import importlib.metadata
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import chromadb
import pytest
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
)
from respan_instrumentation_chroma import ChromaInstrumentor
from respan_instrumentation_chroma import _instrumentation as impl
from respan_instrumentation_chroma._serialization import json_value, redact_text

INPUT = "traceloop.entity.input"
OUTPUT = "traceloop.entity.output"


def collection(engine, name="native_collection", n=2, dimension=4):
    col = engine.create_collection(name, embedding_function=None)
    col.add(
        ids=[f"id-{i}" for i in range(n)],
        embeddings=[[float(j % 3) for j in range(dimension)] for _ in range(n)],
        documents=[f"native document {i}" for i in range(n)],
        metadatas=[{"rank": i, "flag": False} for i in range(n)],
    )
    return col


def one(memory, operation):
    spans = [
        s
        for s in memory.get_finished_spans()
        if s.attributes.get("db.operation") == operation
    ]
    assert len(spans) == 1
    return spans[0]


def bodyless(spans):
    for span in spans:
        assert INPUT not in span.attributes and OUTPUT not in span.attributes
        assert "error.message" not in span.attributes
        assert span.status.description is None
        assert not span.events
        assert "PRIVATE" not in json.dumps(dict(span.attributes))


@pytest.mark.parametrize(
    "method", ["count", "get", "peek", "query", "update", "upsert", "delete", "modify"]
)
def test_native_collection_operations(engine, runtime, method):
    col = collection(engine)
    memory = runtime[1]
    memory.clear()
    if method == "query":
        result = col.query(
            query_embeddings=[[0.0, 1.0, 2.0, 0.0]],
            n_results=1,
            where={"flag": False},
            include=["embeddings", "documents", "metadatas", "distances"],
        )
    elif method in {"update", "upsert"}:
        result = getattr(col, method)(
            ids=["id-0"],
            embeddings=[[0.0, 1.0, 2.0, 0.0]],
            documents=["changed"],
            metadatas=[{"rank": 0, "flag": False}],
        )
    elif method == "delete":
        result = col.delete(ids=["id-0"])
    elif method == "modify":
        result = col.modify(metadata={"flag": False, "zero": 0})
    elif method == "get":
        result = col.get(include=["embeddings", "documents", "metadatas"])
    else:
        result = getattr(col, method)()
    span = one(memory, method)
    assert span.attributes["respan.entity.log_type"] == "task"
    assert span.attributes["db.system"] == "chroma"
    assert span.status.status_code == StatusCode.OK
    assert json.loads(span.attributes[OUTPUT]) == json_value(result)
    assert not any(
        key.startswith("gen_ai.") or key in ("status_code", "traceloop.span.kind")
        for key in span.attributes
    )


@pytest.mark.parametrize(
    "method",
    [
        "count_collections",
        "list_collections",
        "get_collection",
        "get_or_create_collection",
        "get_version",
        "heartbeat",
        "delete_collection",
    ],
)
def test_native_client_operations(engine, runtime, method):
    collection(engine)
    memory = runtime[1]
    memory.clear()
    if method in {"get_collection", "get_or_create_collection", "delete_collection"}:
        result = getattr(engine, method)(
            "native_collection",
            **({"embedding_function": None} if method != "delete_collection" else {}),
        )
    else:
        result = getattr(engine, method)()
    span = one(memory, method)
    assert json.loads(span.attributes[OUTPUT]) == json_value(result)
    assert span.attributes["respan.entity.log_type"] == "task"


def test_native_full_vectors_and_records(engine, runtime):
    col = collection(engine, n=75, dimension=5001)
    memory = runtime[1]
    memory.clear()
    result = col.get(include=["embeddings", "documents", "metadatas"])
    output = json.loads(one(memory, "get").attributes[OUTPUT])
    assert len(output["ids"]) == 75 and len(output["embeddings"][0]) == 5001
    assert output == json_value(result)
    memory.clear()
    result = col.query(
        query_embeddings=[[float(i % 3) for i in range(5001)]],
        n_results=75,
        include=["embeddings", "documents", "distances"],
    )
    output = json.loads(one(memory, "query").attributes[OUTPUT])
    assert len(output["ids"][0]) == 75 and len(output["embeddings"][0][0]) == 5001
    assert output == json_value(result)


def test_native_none_and_error_identity(engine, runtime):
    col = engine.create_collection("native_none", embedding_function=None)
    memory = runtime[1]
    memory.clear()
    assert col.add(ids=["id-0"], embeddings=[[0.0, 1.0]]) is None
    assert one(memory, "add").attributes[OUTPUT] == "null"
    memory.clear()
    with pytest.raises(chromadb.errors.DuplicateIDError) as native:
        col.add(ids=["duplicate", "duplicate"], embeddings=[[0.0, 1.0], [0.0, 1.0]])
    span = one(memory, "add")
    assert OUTPUT not in span.attributes
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == type(native.value).__name__
    assert "status_code" not in span.attributes


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
def test_capture_and_suppression(engine, runtime, key, value):
    col = collection(engine)
    memory = runtime[1]
    memory.clear()
    token = context.attach(context.set_value(key, value))
    try:
        result = col.get(include=["documents", "embeddings"])
    finally:
        context.detach(token)
    assert result["ids"]
    if value is True:
        assert not memory.get_finished_spans()
    else:
        bodyless(memory.get_finished_spans())


@pytest.mark.parametrize("env", ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
@pytest.mark.parametrize("value", ["false", "0", "off", "no"])
def test_environment_capture(engine, runtime, monkeypatch, env, value):
    col = collection(engine)
    memory = runtime[1]
    memory.clear()
    monkeypatch.setenv(env, value)
    col.get()
    bodyless(memory.get_finished_spans())


def test_constructor_capture_and_native_sampling(engine):
    col = collection(engine)
    for enabled, sampler in [(False, None), (True, ALWAYS_OFF)]:
        provider = TracerProvider(sampler=sampler) if sampler else TracerProvider()
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        owner = ChromaInstrumentor(capture_content=enabled, tracer_provider=provider)
        owner.activate()
        try:
            assert col.get()["ids"]
            bodyless(memory.get_finished_spans())
        finally:
            owner.deactivate()
            provider.shutdown()
        if sampler:
            assert not memory.get_finished_spans()


def test_known_collection_storage_does_not_call_name_getter(engine, runtime):
    col = collection(engine)
    memory = runtime[1]
    memory.clear()

    class Native(Collection):
        @property
        def name(self):
            raise AssertionError("telemetry-only getter")

    if importlib.metadata.version("chromadb").startswith("0.5."):
        pytest.skip(
            "Native0.5 Collection is a pydantic model rather than CollectionCommon"
        )
    shadow = Native(client=col._client, model=col._model, embedding_function=None)
    result = shadow.get()
    assert result["ids"]
    assert INPUT in one(memory, "get").attributes


def test_unknown_metaclass_and_schema_storage():
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append("eq")
            raise AssertionError

        __hash__ = type.__hash__

        @property
        def __mro__(cls):
            calls.append("mro")
            raise AssertionError

    class Unknown(metaclass=Meta):
        def __iter__(self):
            calls.append("iter")
            raise AssertionError

        def __str__(self):
            calls.append("str")
            raise AssertionError

    assert json_value({"unknown": Unknown()}) == {"unknown": None}
    assert not calls
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "default": "PRIVATE", "example": "PRIVATE"}
        },
        "required": ["api_key"],
    }
    output = json_value(schema)
    assert (
        "api_key" in output["properties"]
        and output["properties"]["api_key"]["default"] == "[REDACTED]"
    )
    assert json_value({"properties": {"api_key": "PRIVATE"}}) == {
        "properties": {"api_key": "[REDACTED]"}
    }


@pytest.mark.parametrize(
    "value",
    [
        r"Bearer \"PRIVATE SPACE\" Basic \'PRIVATE TWO\'",
        'Authorization: Bearer "PRIVATE"',
        "token=PRIVATE",
        "https://name:PRIVATE@host/path?api_key=PRIVATE",
    ],
)
def test_credentials_valid_idempotent(value):
    result = redact_text(value)
    assert "PRIVATE" not in result and redact_text(result) == result


@pytest.mark.parametrize("kind", ["active", "finished", "unknown", "remote"])
def test_native_parent_bounds(engine, runtime, kind):
    col = collection(engine)
    provider, memory, _ = runtime
    memory.clear()
    if kind in {"active", "finished"}:
        parent = provider.get_tracer("native-test").start_span("parent")
        parent.set_attribute("respan_enable_content_tracing", False)
        if kind == "finished":
            parent.end()
            memory.clear()
    else:
        parent = NonRecordingSpan(SpanContext(1, 2, kind == "remote", TraceFlags(1)))
    with trace.use_span(parent, end_on_exit=False):
        col.get()
    span = one(memory, "get")
    if kind == "remote":
        assert INPUT in span.attributes
    else:
        bodyless([span])
    if kind == "active":
        parent.end()


def test_actual_readable_end_veto(engine, runtime, monkeypatch):
    col = collection(engine)
    memory = runtime[1]
    memory.clear()
    original = Span.end

    def end(span, *args, **kwargs):
        span.add_event("private", {"body": "PRIVATE"})
        span.set_status(Status(StatusCode.ERROR, "PRIVATE"))
        span.set_attribute("error.message", "PRIVATE")
        token = context.attach(
            context.set_value("respan_enable_content_tracing", False)
        )
        try:
            return original(span, *args, **kwargs)
        finally:
            context.detach(token)

    monkeypatch.setattr(Span, "end", end)
    assert col.get()["ids"]
    bodyless(memory.get_finished_spans())


@pytest.mark.parametrize("fault", ["start", "set", "attach", "detach", "end"])
def test_native_telemetry_faults_preserve_result_and_ambient(
    engine, runtime, monkeypatch, fault
):
    col = collection(engine)
    provider, memory, _ = runtime
    memory.clear()
    before = context.get_current()
    if fault == "start":
        monkeypatch.setattr(
            provider,
            "get_tracer",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("observer")),
        )
    elif fault == "set":
        original = Span.set_attribute

        def setter(span, key, value):
            original(span, key, value)
            if key == INPUT:
                raise RuntimeError("mutate then raise")

        monkeypatch.setattr(Span, "set_attribute", setter)
    elif fault == "attach":
        original = context.attach

        def attach(carrier):
            original(carrier)
            raise RuntimeError("mutate then raise")

        monkeypatch.setattr(context, "attach", attach)
    elif fault == "detach":
        monkeypatch.setattr(
            context,
            "detach",
            lambda token: (_ for _ in ()).throw(RuntimeError("detach")),
        )
    else:
        original = Span.end

        def end(span, *args, **kwargs):
            original(span, *args, **kwargs)
            raise RuntimeError("end mutation")

        monkeypatch.setattr(Span, "end", end)
    result = col.get()
    assert result["ids"]
    assert context.get_current() is before
    assert not list(impl._MANAGER.observer.states)
    if fault in {"set", "attach"}:
        bodyless(memory.get_finished_spans())


def test_owner_lifecycle_preserves_foreign_wrapper(runtime):
    provider, _, owner = runtime
    wrapped = inspect.getattr_static(Collection, "get")
    second = ChromaInstrumentor(tracer_provider=provider)
    second.activate()
    owner.deactivate()
    assert inspect.getattr_static(Collection, "get") is wrapped
    with pytest.raises(RuntimeError):
        ChromaInstrumentor(capture_content=False, tracer_provider=provider).activate()

    def foreign(*a, **kw):
        return wrapped(*a, **kw)

    Collection.get = foreign
    second.deactivate()
    assert Collection.get is foreign
    Collection.get = wrapped.__wrapped__


def test_native_runtime_descriptor_presence(runtime):
    owner = runtime[2]
    owner.deactivate()
    rt = context._RUNTIME_CONTEXT
    before = dict(vars(rt))
    owner.activate()
    owner.deactivate()
    assert vars(rt) == before


@pytest.mark.skipif(
    not hasattr(chromadb, "AsyncHttpClient"),
    reason="Native Chroma0.5 exposes no async client API",
)
@pytest.mark.asyncio
async def test_actual_async_native_server(native_server, runtime):
    client = await chromadb.AsyncHttpClient(
        host="127.0.0.1",
        port=native_server,
        settings=Settings(anonymized_telemetry=False),
    )
    col = await client.create_collection(
        "native_async_" + uuid4().hex[:8], embedding_function=None
    )
    await col.add(ids=["a"], embeddings=[[0.0, 1.0]], documents=["native async"])
    result = await col.query(
        query_embeddings=[[0.0, 1.0]], n_results=1, include=["documents", "embeddings"]
    )
    assert result["ids"] == [["a"]]
    span = one(runtime[1], "query")
    assert json.loads(span.attributes[OUTPUT]) == json_value(result)


@pytest.mark.skipif(
    not hasattr(chromadb, "AsyncHttpClient"),
    reason="Native Chroma0.5 exposes no async client API",
)
@pytest.mark.asyncio
async def test_two_pending_native_bodies_under_same_parent(native_server, runtime):
    provider, memory, _ = runtime
    client = await chromadb.AsyncHttpClient(
        host="127.0.0.1",
        port=native_server,
        settings=Settings(anonymized_telemetry=False),
    )
    col = await client.create_collection(
        "native_siblings_" + uuid4().hex[:8], embedding_function=None
    )
    await col.add(ids=["a"], embeddings=[[0.0, 1.0]])
    memory.clear()
    with provider.get_tracer("native-test").start_as_current_span("parent"):
        results = await asyncio.gather(
            col.get(include=["embeddings"]), col.get(include=["embeddings"])
        )
    assert all(result["ids"] == ["a"] for result in results)
    spans = [
        s
        for s in memory.get_finished_spans()
        if s.attributes.get("db.operation") == "get"
    ]
    assert len(spans) == 2 and spans[0].parent == spans[1].parent
    assert all(OUTPUT in s.attributes for s in spans)


def test_pending_native_persistent_siblings(engine, runtime, monkeypatch):
    col = collection(engine)
    provider, memory, _ = runtime
    memory.clear()
    barrier = threading.Barrier(2)
    original = impl._Call.result

    def result(state, value):
        if state.operation == "collection.get":
            barrier.wait(timeout=10)
        return original(state, value)

    monkeypatch.setattr(impl._Call, "result", result)
    with provider.get_tracer("app").start_as_current_span("parent"):
        carrier = context.get_current()

        def call():
            token = context.attach(carrier)
            try:
                return col.get(include=["embeddings"])
            finally:
                context.detach(token)

        with ThreadPoolExecutor(2) as pool:
            values = list(pool.map(lambda _: call(), range(2)))
    spans = [
        span
        for span in memory.get_finished_spans()
        if span.attributes.get("db.operation") == "get"
    ]
    assert len(values) == len(spans) == 2 and spans[0].parent == spans[1].parent
    assert all(OUTPUT in span.attributes for span in spans)


def test_native_embedding_callback_once_and_result_identity(
    engine, runtime, monkeypatch
):
    calls = []

    class Embedding:
        @staticmethod
        def name():
            return "controlled-native"

        def __call__(self, input):
            calls.append(input)
            return [[0.0, 1.0, 2.0, 0.0] for _ in input]

    col = engine.create_collection("native_callback", embedding_function=Embedding())
    col.add(ids=["native"], documents=["native text"])
    assert len(calls) == 1 and calls[0] == ["native text"]
    actual = []
    original = impl._Call.result

    def result(state, value):
        actual.append(value)
        return original(state, value)

    monkeypatch.setattr(impl._Call, "result", result)
    native = col.get(include=["embeddings", "documents"])
    assert native is actual[-1]
    assert json.loads(one(runtime[1], "get").attributes[OUTPUT]) == json_value(native)


@pytest.mark.parametrize("field", ["constructor", "supplied", "ambient"])
def test_initial_supplied_ambient_bounds_cannot_widen(engine, runtime, field):
    col = collection(engine)
    provider, memory, owner = runtime
    owner.deactivate()
    options = {"tracer_provider": provider}
    if field == "constructor":
        options["capture_content"] = False
    elif field == "supplied":
        options["context"] = context.set_value("respan_enable_content_tracing", False)
    else:
        options["context"] = context.Context()
    private = ChromaInstrumentor(**options)
    private.activate()
    memory.clear()
    token = context.attach(
        context.set_value("respan_enable_content_tracing", field != "ambient")
    )
    try:
        assert col.get()["ids"]
    finally:
        context.detach(token)
        private.deactivate()
    bodyless(memory.get_finished_spans())


def test_private_on_start_events_are_scrubbed(engine, runtime):
    from opentelemetry.sdk.trace import SpanProcessor

    class Event(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.add_event("private", {"body": "PRIVATE"})
            span.set_status(Status(StatusCode.ERROR, "PRIVATE"))

        def on_end(self, span):
            pass

    col = collection(engine)
    runtime[0].add_span_processor(Event())
    runtime[1].clear()
    token = context.attach(context.set_value("respan_enable_content_tracing", False))
    try:
        assert col.get()["ids"]
    finally:
        context.detach(token)
    bodyless(runtime[1].get_finished_spans())


def test_partial_delegate_activation_rollback(engine, runtime, monkeypatch):
    owner = runtime[2]
    owner.deactivate()
    original = inspect.getattr_static(Collection, "get")
    patch = impl._Manager.patch
    calls = []

    def fault(manager, *args):
        patch(manager, *args)
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("partial activation")

    monkeypatch.setattr(impl._Manager, "patch", fault)
    with pytest.raises(RuntimeError):
        owner.activate()
    assert impl._MANAGER is None
    assert inspect.getattr_static(Collection, "get") is original
    assert collection(engine).get()["ids"]


def test_combined_observer_fault_preserves_bare_error_and_common_path(
    engine, runtime, monkeypatch
):
    from opentelemetry.sdk.trace import SpanProcessor

    class BareError(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_status(Status(StatusCode.ERROR))
            span.add_event("private", {"body": "PRIVATE"})

        def on_end(self, span):
            pass

    col = collection(engine)
    runtime[0].add_span_processor(BareError())
    runtime[1].clear()

    def fault(state, *args):
        raise RuntimeError("observer failure")

    monkeypatch.setattr(impl._Call, "result", fault)
    monkeypatch.setattr(impl._Call, "scrub", fault)
    native = col.get()
    assert native["ids"]
    span = one(runtime[1], "get")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["traceloop.entity.path"] == ""
    assert "error.type" not in span.attributes
    bodyless([span])


@pytest.mark.parametrize("operation", ["get_indexing_status", "fork", "search"])
def test_latest_native_capability_outcomes(engine, runtime, operation):
    if not hasattr(Collection, operation):
        pytest.skip("Native minimum does not expose this API")
    col = collection(engine)
    runtime[1].clear()
    options = {"name": "native_fork"} if operation == "fork" else {}
    if operation == "search":
        from chromadb import Search

        options = {"searches": Search().limit(1)}
    try:
        value = getattr(col, operation)(**options)
    except Exception as error:  # noqa: BLE001 - native capability errors are source outcomes.
        span = one(runtime[1], operation)
        assert span.attributes["error.type"] == type(error).__name__
        assert OUTPUT not in span.attributes
    else:
        span = one(runtime[1], operation)
        assert json.loads(span.attributes[OUTPUT]) == json_value(value)
