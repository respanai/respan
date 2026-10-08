import json
import uuid

import grpc
import pytest
from _native import Protocol
from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, _Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_instrumentation_weaviate import WeaviateInstrumentor
from respan_instrumentation_weaviate import _instrumentation as adapter
from respan_instrumentation_weaviate._serialization import safe_text, to_jsonable
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from weaviate.classes.config import DataType, Property
from weaviate.classes.tenants import Tenant
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateQueryError

INPUT = "traceloop.entity.input"
OUTPUT = "traceloop.entity.output"


@pytest.fixture
def runtime():
    with Protocol() as native, native.client() as client:
        collection = client.collections.create("Docs")
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        inst = WeaviateInstrumentor()
        inst.activate(tracer_provider=provider)
        yield native, client, collection, provider, exporter, inst
        inst.deactivate()
        provider.shutdown()


def last(runtime):
    return runtime[4].get_finished_spans()[-1]


def bodyless(span):
    assert INPUT not in span.attributes and OUTPUT not in span.attributes
    assert not span.events and not span.status.description
    assert "error.message" not in span.attributes


@pytest.mark.parametrize(
    "method",
    ["fetch_objects", "near_vector", "near_text", "bm25", "hybrid", "near_object"],
)
def test_native_queries(runtime, method):
    native, _, col, _, exporter, _ = runtime
    params = {"include_vector": True, "limit": 3}
    params.update(
        {
            "near_vector": {"near_vector": [0.25] * 4},
            "near_text": {"query": "native"},
            "bm25": {"query": "native"},
            "hybrid": {"query": "native", "vector": [0.25] * 4},
            "near_object": {"near_object": uuid.UUID(int=1)},
        }.get(method, {})
    )
    result = getattr(col.query, method)(**params)
    assert (
        type(result).__name__ in {"QueryReturn", "GenerativeReturn"}
        and len(result.objects) == 3
    )
    out = json.loads(last(runtime).attributes[OUTPUT])
    assert len(out["objects"]) == 3
    assert (
        out["objects"][0]["properties"]["flag"] is False
        and out["objects"][0]["properties"]["zero"] == 0
    )
    assert last(runtime).attributes["db.collection.name"] == "Docs"
    assert last(runtime).attributes[RESPAN_LOG_TYPE] == "task"
    assert (
        len(exporter.get_finished_spans()) == 1
        and native.grpc_requests[-1][0] == "Search"
    )


def test_full_vectors_and_history_native_result(runtime):
    native, _, col, _, _, _ = runtime
    native.count = 75
    native.dim = 5001
    result = col.query.fetch_objects(limit=75, include_vector=True)
    output = json.loads(last(runtime).attributes[OUTPUT])
    assert len(result.objects) == len(output["objects"]) == 75
    assert (
        len(result.objects[0].vector["default"])
        == len(output["objects"][0]["vector"]["default"])
        == 5001
    )
    assert output["objects"][0]["properties"]["flag"] is False
    assert (
        output["objects"][0]["metadata"]["distance"] is None
    )  # native options did not request it


def test_actual_native_query_uuid_value(runtime):
    from weaviate.util import _WeaviateUUIDInt

    result = runtime[2].query.fetch_objects(include_vector=True, limit=3)
    assert type(result.objects[0].uuid) is _WeaviateUUIDInt
    expected = uuid.UUID.__str__(
        uuid.UUID(int=uuid.UUID.int.__get__(result.objects[0].uuid, uuid.UUID))
    )
    output = json.loads(last(runtime).attributes[OUTPUT])
    assert output["objects"][0]["uuid"] == expected
    assert expected == "00000000-0000-0000-0000-000000000000"


def test_native_uuid_and_opaque_subclass_metadata_no_customer_hooks(
    runtime, monkeypatch
):
    from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES
    from weaviate.util import _WeaviateUUIDInt

    hooks = []

    def getter(self, name):
        hooks.append("native-getter")
        return object.__getattribute__(self, name)

    def stringify(self):
        hooks.append("native-str")
        return "unexpected customer string"

    class OpaqueUUID(uuid.UUID):
        def __getattribute__(self, name):
            hooks.append("opaque-getter")
            raise AssertionError(name)

        def __str__(self):
            hooks.append("opaque-str")
            raise AssertionError("str")

    native_value = _WeaviateUUIDInt(0)
    opaque_value = OpaqueUUID(int=1)
    monkeypatch.setattr(_WeaviateUUIDInt, "__getattribute__", getter)
    monkeypatch.setattr(_WeaviateUUIDInt, "__str__", stringify)
    runtime[5].deactivate()
    bare = runtime[2].query.fetch_objects(include_vector=True, limit=3)
    assert type(bare.objects[0].uuid) is _WeaviateUUIDInt and not hooks
    runtime[5].activate(tracer_provider=runtime[3])
    token = _PROPAGATED_ATTRIBUTES.set(
        {"metadata": {"native_uuid": native_value, "opaque_uuid": opaque_value}}
    )
    try:
        actual = runtime[2].query.fetch_objects(include_vector=True, limit=3)
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)
    assert type(actual.objects[0].uuid) is _WeaviateUUIDInt and not hooks
    output = json.loads(last(runtime).attributes[OUTPUT])
    assert output["objects"][0]["uuid"] == "00000000-0000-0000-0000-000000000000"
    assert (
        last(runtime).attributes["respan.metadata.native_uuid"]
        == "00000000-0000-0000-0000-000000000000"
    )
    assert json.loads(last(runtime).attributes["respan.metadata.opaque_uuid"]) == {
        "type": "OpaqueUUID"
    }
    assert to_jsonable(opaque_value) == {"type": "OpaqueUUID"} and not hooks


@pytest.mark.parametrize(
    "method",
    ["insert", "insert_many", "update", "replace", "delete_by_id", "exists", "ingest"],
)
def test_native_data_and_ingest_protocol(runtime, method):
    native, _, col, _, exp, _ = runtime
    if method == "insert":
        result = col.data.insert(
            {"text": "native", "flag": False, "zero": 0},
            uuid=uuid.UUID(int=1),
            vector=[0.25] * 5001,
        )
    elif method == "insert_many":
        result = col.data.insert_many([{"text": "one"}, {"text": "two"}])
    elif method == "ingest":
        result = col.data.ingest([{"text": "one"}, {"text": "two"}])
    elif method == "exists":
        result = col.data.exists(uuid.UUID(int=1))
    elif method == "delete_by_id":
        result = col.data.delete_by_id(uuid.UUID(int=1))
    else:
        result = getattr(col.data, method)(uuid.UUID(int=1), {"text": "changed"})
    spans = exp.get_finished_spans()
    assert len(spans) == 1
    out = json.loads(spans[0].attributes[OUTPUT])
    if method in {"ingest", "insert_many"}:
        assert len(result.uuids) == len(out["uuids"]) == 2
        assert len(out["_all_responses"]) == 2 and out["has_errors"] is False
    elif method == "insert":
        assert (
            out == str(result)
            and len(json.loads(spans[0].attributes[INPUT])["kwargs"]["vector"]) == 5001
        )
    elif result is None:
        assert out is None
    else:
        assert out is result
    if method == "ingest":
        assert any(name == "BatchStream" for name, _ in native.grpc_requests)


@pytest.mark.parametrize(
    "method",
    ["create", "create_from_dict", "exists", "list_all", "export_config", "delete"],
)
def test_native_collections(runtime, method):
    _, client, _, _, exp, _ = runtime
    if method == "create":
        result = client.collections.create("New")
    elif method == "create_from_dict":
        result = client.collections.create_from_dict(
            {"class": "New", "vectorizer": "none"}
        )
    elif method == "export_config":
        result = client.collections.export_config("Docs")
    elif method == "list_all":
        result = getattr(client.collections, method)()
    else:
        result = getattr(client.collections, method)("Docs")
    assert len(exp.get_finished_spans()) == 1 and OUTPUT in last(runtime).attributes
    if method.startswith("create"):
        assert result.name == "New"


@pytest.mark.parametrize("method", ["get", "get_shards", "add_property"])
def test_native_config(runtime, method):
    _, _, col, _, _, _ = runtime
    result = (
        col.config.add_property(Property(name="new", data_type=DataType.TEXT))
        if method == "add_property"
        else getattr(col.config, method)()
    )
    out = json.loads(last(runtime).attributes[OUTPUT])
    if method == "get":
        assert (
            out["name"] == result.name == "Docs"
            and out["inverted_index_config"]["index_null_state"] is False
        )
    if result is None:
        assert out is None


@pytest.mark.parametrize("method", ["over_all", "near_vector", "hybrid"])
def test_native_aggregate(runtime, method):
    _, _, col, _, _, _ = runtime
    kwargs = {"total_count": True}
    if method == "near_vector":
        kwargs.update(near_vector=[0.25] * 4, distance=0.5)
    if method == "hybrid":
        kwargs.update(query="native", object_limit=3)
    result = getattr(col.aggregate, method)(**kwargs)
    assert (
        result.total_count
        == json.loads(last(runtime).attributes[OUTPUT])["total_count"]
        == 3
    )


@pytest.mark.parametrize("method", ["create", "remove", "update", "get"])
def test_native_tenants(runtime, method):
    _, _, col, _, _, _ = runtime
    result = (
        col.tenants.get()
        if method == "get"
        else getattr(col.tenants, method)([Tenant(name="tenant")])
    )
    assert OUTPUT in last(runtime).attributes
    if method == "get":
        assert "tenant" in result


@pytest.mark.asyncio
async def test_native_async_protocols_and_context(runtime):
    native, _, _, provider, exp, _ = runtime
    async with native.async_client() as c:
        col = c.collections.use("Docs")
        ambient = context.get_current()
        with trace.use_span(
            provider.get_tracer("app").start_span("parent"), end_on_exit=True
        ):
            result = await col.query.fetch_objects(include_vector=True)
            await col.data.insert({"text": "async"})
            aggregate = await col.aggregate.over_all(total_count=True)
            ingest = await col.data.ingest([{"text": "one"}, {"text": "two"}])
        assert context.get_current() is ambient
        assert (
            len(result.objects) == 3
            and aggregate.total_count == 3
            and len(ingest.uuids) == 2
        )
        children = [
            s for s in exp.get_finished_spans() if s.name.startswith("weaviate.")
        ]
        assert len(children) == 4 and all(OUTPUT in s.attributes for s in children)
        assert len({s.parent.span_id for s in children}) == 1


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_before_inspection(runtime, key, monkeypatch):
    monkeypatch.setattr(adapter, "_identity", lambda _: pytest.fail("inspected"))
    token = context.attach(context.set_value(key, True))
    try:
        assert len(runtime[2].query.fetch_objects().objects) == 3
    finally:
        context.detach(token)
    assert not runtime[4].get_finished_spans()


def test_sampler_before_inspection(runtime, monkeypatch):
    runtime[5].deactivate()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    inst = WeaviateInstrumentor()
    inst.activate(tracer_provider=provider)
    monkeypatch.setattr(adapter, "_identity", lambda _: pytest.fail("inspected"))
    try:
        assert len(runtime[2].query.fetch_objects().objects) == 3
    finally:
        inst.deactivate()
        provider.shutdown()


@pytest.mark.parametrize(
    "flag", [ENABLE_CONTENT_TRACING_KEY, "override_enable_content_tracing"]
)
def test_initial_context_privacy(runtime, flag):
    token = context.attach(context.set_value(flag, False))
    try:
        assert len(runtime[2].query.fetch_objects().objects) == 3
    finally:
        context.detach(token)
    bodyless(last(runtime))


@pytest.mark.parametrize("name", ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
def test_env_privacy(runtime, name, monkeypatch):
    monkeypatch.setenv(name, "off")
    runtime[2].query.fetch_objects()
    bodyless(last(runtime))


@pytest.mark.parametrize("finished", [False, True])
def test_application_ancestor_attributes(runtime, finished):
    parent = runtime[3].get_tracer("app").start_span("parent")
    parent.set_attribute("traceloop.enable_content_tracing", False)
    if finished:
        parent.end()
    with trace.use_span(parent):
        runtime[2].query.fetch_objects()
    bodyless(last(runtime))
    parent.end()


@pytest.mark.parametrize("remote", [False, True])
def test_unknown_local_remote_carriers(runtime, remote):
    carrier = trace.NonRecordingSpan(
        trace.SpanContext(123, 456, remote, trace.TraceFlags(1))
    )
    with trace.use_span(carrier):
        runtime[2].query.fetch_objects()
    if remote:
        assert OUTPUT in last(runtime).attributes
    else:
        bodyless(last(runtime))


def test_late_readable_veto(runtime, monkeypatch):
    original = _Span.end
    originals = []

    def end(span, *a, **k):
        if span.name.startswith("weaviate."):
            originals.append(span)
            span.set_attribute(ENABLE_CONTENT_TRACING_KEY, False)
            span.record_exception(ValueError('Bearer "controlled secret"'))
        return original(span, *a, **k)

    monkeypatch.setattr(_Span, "end", end)
    runtime[2].query.fetch_objects()
    bodyless(last(runtime))


@pytest.mark.parametrize(
    "fault", ["attributes", "attach", "detach", "end", "serialization"]
)
def test_telemetry_fault_native_result_and_exact_context(runtime, fault, monkeypatch):
    ambient = context.get_current()
    if fault == "attributes":
        monkeypatch.setattr(
            _Span,
            "set_attribute",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("telemetry")),
        )
    elif fault == "serialization":
        monkeypatch.setattr(
            adapter,
            "json_dumps",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("telemetry")),
        )
    elif fault in {"attach", "detach"}:
        original = getattr(context, fault)

        def broken(value):
            original(value)
            raise RuntimeError("telemetry")

        monkeypatch.setattr(context, fault, broken)
    else:
        original = _Span.end

        def broken(span, *a, **k):
            original(span, *a, **k)
            raise RuntimeError("telemetry")

        monkeypatch.setattr(_Span, "end", broken)
    assert len(runtime[2].query.fetch_objects().objects) == 3
    assert context.get_current() is ambient


@pytest.mark.parametrize("transport", ["grpc", "http"])
def test_native_errors_no_synthetic_output_status(runtime, transport):
    if transport == "grpc":
        runtime[0].grpc_error = grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(WeaviateQueryError) as raised:
            runtime[2].query.fetch_objects()
    else:
        with pytest.raises(UnexpectedStatusCodeError) as raised:
            runtime[1].collections.use("Missing").config.get()
    span = last(runtime)
    assert (
        span.attributes["error.type"] == type(raised.value).__name__
        and OUTPUT not in span.attributes
    )
    assert span.status.status_code is trace.StatusCode.ERROR
    if transport == "grpc":
        assert "http.response.status_code" not in span.attributes
    else:
        assert span.attributes["http.response.status_code"] == 404


@pytest.mark.parametrize(
    "text",
    [
        'Bearer "controlled secret"',
        'Basic "controlled secret"',
        r"Bearer \"controlled secret\"",
        r"Basic \"controlled secret\"",
        'Authorization: Bearer "controlled secret"',
        "https://user:password@host/path?api%5Fkey=controlled&answer=0",
    ],
)
def test_native_input_redaction(runtime, text):
    runtime[2].data.insert(
        {
            "text": text,
            "schema": {
                "properties": {
                    "private_key": {"type": "string", "default": "controlled"}
                },
                "flag": False,
                "zero": 0,
            },
            "encoded": '{"api_key":"controlled secret","flag":false,"zero":0}',
        }
    )
    value = last(runtime).attributes[INPUT]
    assert "controlled" not in value
    assert (
        "private_key" in value
        and json.loads(value)["args"][0]["schema"]["flag"] is False
    )
    assert safe_text(safe_text(text)) == safe_text(text)
    assert runtime[0].requests[-1][2]["properties"]["text"] == text


def test_unknown_hooks_metadata_sampler_and_native(runtime):
    hooks = []

    class Meta(type):
        def __hash__(self):
            hooks.append("hash")
            return type.__hash__(self)

    class Opaque(metaclass=Meta):
        def __getattribute__(self, name):
            hooks.append("get")
            raise AssertionError(name)

    assert to_jsonable({"opaque": Opaque()}) == {"opaque": {"type": "Opaque"}}
    from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES

    token = _PROPAGATED_ATTRIBUTES.set({"metadata": {"opaque": Opaque()}})
    try:
        runtime[2].query.fetch_objects()
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)
    assert not hooks


def test_shared_config_and_foreign_restore(runtime, monkeypatch):
    peer = WeaviateInstrumentor()
    peer.activate(tracer_provider=runtime[3])
    with pytest.raises(ValueError):
        WeaviateInstrumentor().activate(tracer_provider=TracerProvider())
    runtime[5].deactivate()
    runtime[2].query.fetch_objects()
    assert len(runtime[4].get_finished_spans()) == 1
    peer.deactivate()
    runtime[4].clear()
    runtime[2].query.fetch_objects()
    assert not runtime[4].get_finished_spans()
    assert not adapter._PATCHES and not adapter._POLICIES


def test_foreign_startup_fault_native_context_scrub(runtime):
    seen = []
    ambient = context.get_current()

    class Broken(SpanProcessor):
        def on_start(self, span, parent_context=None):
            seen.append(span)
            span.record_exception(ValueError('Bearer "controlled secret"'))
            context.attach(context.set_value("poison", True))
            raise RuntimeError("processor")

        def on_end(self, span):
            pass

    runtime[3].add_span_processor(Broken())
    assert (
        len(runtime[2].query.fetch_objects().objects) == 3
        and context.get_current() is ambient
    )
    assert len(seen) == 1 and not seen[0].is_recording() and not seen[0].events


def test_mutating_setter_rollback(monkeypatch):
    from weaviate.collections.collections.sync import _Collections

    original = _Collections.create
    provider = TracerProvider()
    processors = provider._active_span_processor._span_processors
    failed = False

    def setter(owner, name, value):
        nonlocal failed
        setattr(owner, name, value)
        if not failed and isinstance(owner, type):
            failed = True
            raise RuntimeError("mutated")

    monkeypatch.setattr(adapter, "setattr", setter, raising=False)
    with pytest.raises(RuntimeError):
        WeaviateInstrumentor().activate(tracer_provider=provider)
    assert (
        _Collections.create is original
        and provider._active_span_processor._span_processors == processors
    )
    assert not adapter._PATCHES and not adapter._POLICIES


def test_native_collection_iterator_keeps_lazy_identity(runtime):
    from weaviate.collections.iterator import _ObjectIterator

    iterator = runtime[2].iterator(include_vector=True)
    assert type(iterator) is _ObjectIterator and not runtime[4].get_finished_spans()
    values = list(iterator)
    assert len(values) == 3 and len(runtime[4].get_finished_spans()) == 2
    assert json.loads(last(runtime).attributes[OUTPUT])["objects"] == []


@pytest.mark.asyncio
async def test_native_async_iterator_and_preawait_deactivate(runtime):
    from weaviate.collections.iterator import _ObjectAIterator

    async with runtime[0].async_client() as client:
        col = client.collections.use("Docs")
        iterator = col.iterator(include_vector=True)
        assert (
            type(iterator) is _ObjectAIterator and not runtime[4].get_finished_spans()
        )
        values = [value async for value in iterator]
        assert len(values) == 3 and len(runtime[4].get_finished_spans()) == 2
        pending = col.query.fetch_objects()
        runtime[5].deactivate()
        result = await pending
        assert len(result.objects) == 3 and len(runtime[4].get_finished_spans()) == 2


def test_supplied_parent_and_ambient_privacy_are_combined(runtime):
    ambient = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    supplied = context.set_value(ENABLE_CONTENT_TRACING_KEY, True, context.Context())
    parent = runtime[3].get_tracer("app").start_span("supplied", context=supplied)
    context.detach(ambient)
    with trace.use_span(parent):
        runtime[2].query.fetch_objects()
    bodyless(last(runtime))
    parent.end()


def test_foreign_bare_error_no_invented_diagnostic(runtime, monkeypatch):
    original = _Span.end

    def end(span, *a, **k):
        if span.name.startswith("weaviate."):
            span._status = trace.Status(trace.StatusCode.ERROR)
        return original(span, *a, **k)

    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    monkeypatch.setattr(_Span, "end", end)
    try:
        runtime[2].query.fetch_objects()
    finally:
        context.detach(token)
    span = last(runtime)
    bodyless(span)
    assert (
        span.status.status_code is trace.StatusCode.ERROR
        and "error.type" not in span.attributes
    )


def test_owned_observer_add_then_raise_rollback(monkeypatch):
    provider = TracerProvider()
    original = provider.add_span_processor

    def add(processor):
        original(processor)
        raise RuntimeError("registered")

    monkeypatch.setattr(provider, "add_span_processor", add)
    with pytest.raises(RuntimeError):
        WeaviateInstrumentor().activate(tracer_provider=provider)
    assert (
        not adapter._POLICIES and not provider._active_span_processor._span_processors
    )


def test_foreign_wrapper_retained_after_deactivation(runtime, monkeypatch):
    owner = type(runtime[2].query)
    installed = owner.fetch_objects

    def foreign(instance, *args, **kwargs):
        return installed(instance, *args, **kwargs)

    monkeypatch.setattr(owner, "fetch_objects", foreign)
    runtime[5].deactivate()
    assert owner.fetch_objects is foreign
    assert len(runtime[2].query.fetch_objects().objects) == 3
    assert not runtime[4].get_finished_spans()


def test_real_httpx_callback_is_native_once(runtime):
    calls = []
    runtime[1]._connection._client.event_hooks["response"].append(
        lambda response: calls.append(response.status_code)
    )
    identifier = runtime[2].data.insert({"text": "callback"})
    assert type(identifier) is uuid.UUID and calls == [200]
    assert len(runtime[4].get_finished_spans()) == 1


def test_late_actual_provider_unknown_existing_parent_closed(runtime, monkeypatch):
    runtime[5].deactivate()
    late = TracerProvider()
    exporter = InMemorySpanExporter()
    late.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    instrumentor = WeaviateInstrumentor()
    instrumentor.activate()
    existing = late.get_tracer("app").start_span("unobserved-before-provider")
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", late)
    try:
        with trace.use_span(existing):
            runtime[2].query.fetch_objects()
        bodyless(exporter.get_finished_spans()[-1])
        runtime[2].query.fetch_objects()
        assert OUTPUT in exporter.get_finished_spans()[-1].attributes
    finally:
        existing.end()
        instrumentor.deactivate()
        late.shutdown()


def test_actual_native_registry_no_customer_metaclass_descriptor(runtime, monkeypatch):
    from weaviate.collections.classes import config

    hooks = []

    class Meta(type):
        @property
        def __module__(self):
            hooks.append("module")
            return "weaviate.collections.classes.config"

    class Extension(metaclass=Meta):
        pass

    monkeypatch.setattr(config, "ControlledExtension", Extension, raising=False)
    runtime[5].deactivate()
    inst = WeaviateInstrumentor()
    inst.activate(tracer_provider=runtime[3])
    try:
        assert len(runtime[2].query.fetch_objects().objects) == 3
        assert not hooks
    finally:
        inst.deactivate()
