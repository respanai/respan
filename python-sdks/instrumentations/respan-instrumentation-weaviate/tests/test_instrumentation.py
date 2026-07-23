import json
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

from respan_instrumentation_weaviate import WeaviateInstrumentor
from respan_instrumentation_weaviate import _instrumentation
from respan_instrumentation_weaviate._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
    PatchSpec,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


class _FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.status = None
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status


class _FakeTracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, **_kwargs):
        span = _FakeSpan(name)
        self.spans.append(span)
        yield span


def _module(monkeypatch, name, class_name, target_class):
    module = ModuleType(name)
    setattr(module, class_name, target_class)
    monkeypatch.setitem(sys.modules, name, module)


def _install_fake_weaviate(monkeypatch):
    class _Collections:
        def create(self, name, vector_config=None):
            return {"name": name, "created": True}

        def exists(self, name):
            return name != "missing"

        def delete(self, name):
            if name == "missing":
                raise RuntimeError("collection missing")
            return None

    class _DataCollection:
        name = "Docs"
        _tenant = None

        def insert(self, properties, uuid=None, vector=None):
            return uuid or "generated-id"

        def update(self, uuid, properties=None, vector=None):
            return None

    class _DataCollectionAsync:
        name = "AsyncDocs"
        _tenant = "tenant-a"

        async def insert(self, properties, uuid=None, vector=None):
            return uuid or "async-id"

    class _QueryCollection:
        name = "Docs"

        def near_vector(self, near_vector, limit=10, **kwargs):
            return {
                "objects": [{"uuid": "one", "distance": 0.01, "vector": near_vector}]
            }

    modules = [
        (
            "weaviate.collections.collections.sync",
            "_Collections",
            _Collections,
            "collections",
            ("create", "exists", "delete"),
            False,
        ),
        (
            "weaviate.collections.data.sync",
            "_DataCollection",
            _DataCollection,
            "data",
            ("insert", "update"),
            False,
        ),
        (
            "weaviate.collections.data.async_",
            "_DataCollectionAsync",
            _DataCollectionAsync,
            "data",
            ("insert",),
            True,
        ),
        (
            "weaviate.collections.query",
            "_QueryCollection",
            _QueryCollection,
            "query",
            ("near_vector",),
            False,
        ),
    ]
    specs = []
    for module_name, class_name, target, label, methods, is_async in modules:
        _module(monkeypatch, module_name, class_name, target)
        specs.append(PatchSpec(module_name, class_name, label, methods, is_async))
    monkeypatch.setattr(_instrumentation, "WEAVIATE_PATCH_SPECS", tuple(specs))
    return _Collections, _DataCollection, _DataCollectionAsync, _QueryCollection


@pytest.fixture(autouse=True)
def reset_instrumentor():
    WeaviateInstrumentor._patches_applied = False
    WeaviateInstrumentor._activation_count = 0
    WeaviateInstrumentor._patched_targets = []
    yield
    for target, method in reversed(WeaviateInstrumentor._patched_targets):
        _instrumentation.unwrap(target, method)
    WeaviateInstrumentor._patches_applied = False
    WeaviateInstrumentor._activation_count = 0
    WeaviateInstrumentor._patched_targets = []


def _assert_contract(span):
    attrs = span.attributes
    assert attrs[RESPAN_LOG_TYPE] == "task"
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_NAME]
    assert attrs[SpanAttributes.TRACELOOP_ENTITY_PATH]
    for banned_alias in (
        "tools",
        "tool_calls",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "span_tools",
        "has_tool_calls",
        RESPAN_SPAN_TOOLS,
        RESPAN_SPAN_TOOL_CALLS,
        RESPAN_SPAN_HANDOFFS,
    ):
        assert banned_alias not in attrs


def test_package_exports_weaviate_instrumentor():
    assert WeaviateInstrumentor is _instrumentation.WeaviateInstrumentor
    assert WeaviateInstrumentor.name == "weaviate"


def test_collection_data_and_query_operations_emit_spans(monkeypatch):
    Collections, Data, _, Query = _install_fake_weaviate(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = WeaviateInstrumentor()
    instrumentor.activate()

    Collections().create("Docs", vector_config={"vectorizer": "none"})
    object_id = Data().insert(
        {"text": "Respan traces Weaviate"},
        vector=[0.1, 0.2, 0.3],
    )
    result = Query().near_vector([0.1, 0.2, 0.3], limit=1)

    assert object_id == "generated-id"
    assert result["objects"][0]["uuid"] == "one"
    assert [span.name for span in tracer.spans] == [
        "weaviate.collections.create",
        "weaviate.data.insert",
        "weaviate.query.near_vector",
    ]
    for span in tracer.spans:
        _assert_contract(span)
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
        assert span.status.status_code is StatusCode.OK
    assert (
        '"collection_name": "Docs"'
        in tracer.spans[1].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    instrumentor.deactivate()


@pytest.mark.asyncio
async def test_async_data_operation_is_awaited(monkeypatch):
    _, _, AsyncData, _ = _install_fake_weaviate(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = WeaviateInstrumentor()
    instrumentor.instrument()

    result = await AsyncData().insert(
        {"text": "async"},
        vector=[0.3, 0.2, 0.1],
    )

    assert result == "async-id"
    assert [span.name for span in tracer.spans] == ["weaviate.data.insert"]
    _assert_contract(tracer.spans[0])
    instrumentor.uninstrument()


def test_error_status_and_content_control(monkeypatch):
    Collections, _, _, _ = _install_fake_weaviate(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = WeaviateInstrumentor(capture_content=False)
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="collection missing"):
        Collections().delete("missing")

    span = tracer.spans[-1]
    assert span.status.status_code is StatusCode.ERROR
    assert span.exceptions
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "collection missing"
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    _assert_contract(span)
    instrumentor.deactivate()


def test_full_vectors_survive_canonical_input_and_output(monkeypatch):
    _, _, _, Query = _install_fake_weaviate(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = WeaviateInstrumentor()
    instrumentor.activate()
    vector = [0.125] * (MAX_ATTRIBUTE_CHARS + 17)
    tags = [f"tag-{index}" for index in range(MAX_PREVIEW_ITEMS + 17)]

    Query().near_vector(vector, limit=1, filters={"tags": tags})

    span = tracer.spans[-1]
    entity_input = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    entity_output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert entity_input["near_vector"] == vector
    assert entity_output["objects"][0]["vector"] == vector
    assert len(entity_output["objects"][0]["vector"]) == len(vector)
    assert (
        len(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        > MAX_ATTRIBUTE_CHARS
    )
    assert (
        len(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        > MAX_ATTRIBUTE_CHARS
    )
    assert entity_input["kwargs"]["filters"]["tags"]["truncated"] is True
    instrumentor.deactivate()


def test_lifecycle_is_idempotent_and_reference_counted(monkeypatch):
    Collections, _, _, _ = _install_fake_weaviate(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    first = WeaviateInstrumentor()
    second = WeaviateInstrumentor()
    first.activate()
    first.activate()
    second.activate()

    Collections().exists("Docs")
    assert len(tracer.spans) == 1
    first.deactivate()
    Collections().exists("Docs")
    assert len(tracer.spans) == 2
    second.deactivate()
    Collections().exists("Docs")
    assert len(tracer.spans) == 2


def test_real_dynamic_batch_runtime_target_is_patched_and_unwrapped(monkeypatch):
    from weaviate.collections.batch.collection import _BatchCollection

    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    monkeypatch.setattr(
        _instrumentation,
        "WEAVIATE_PATCH_SPECS",
        (
            PatchSpec(
                "weaviate.collections.batch.collection",
                "_BatchCollection",
                "batch",
                ("add_object", "flush"),
            ),
        ),
    )
    original_add_object = _BatchCollection.add_object
    original_flush = _BatchCollection.flush
    instrumentor = WeaviateInstrumentor()
    instrumentor.activate()

    batch = object.__new__(_BatchCollection)
    batch._BatchCollection__name = "Docs"
    batch._BatchCollection__tenant = None
    batch._BatchBase__active_requests = 0
    batch._BatchBase__batch_objects = []
    batch._BatchBase__batch_references = []
    object_ids = iter(("object-1", "object-2", "object-3"))
    batch._add_object = lambda **_kwargs: next(object_ids)

    assert (
        batch.add_object(
            {"text": "first"},
            vector=[0.1, 0.2, 0.3],
        )
        == "object-1"
    )
    batch.add_object({"text": "second"}, vector=[0.4, 0.5, 0.6])
    batch.add_object({"text": "third"}, vector=[0.7, 0.8, 0.9])
    batch.flush()

    assert [span.name for span in tracer.spans] == [
        "weaviate.batch.add_object",
        "weaviate.batch.add_object",
        "weaviate.batch.add_object",
        "weaviate.batch.flush",
    ]
    for span in tracer.spans:
        _assert_contract(span)
        assert span.status.status_code is StatusCode.OK
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
    first_input = json.loads(
        tracer.spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    assert first_input["operation"] == "batch.add_object"
    assert first_input["properties"] == {"text": "first"}
    assert first_input["vector"] == [0.1, 0.2, 0.3]

    instrumentor.deactivate()
    assert _BatchCollection.add_object is original_add_object
    assert _BatchCollection.flush is original_flush
    batch.flush()
    assert len(tracer.spans) == 4
