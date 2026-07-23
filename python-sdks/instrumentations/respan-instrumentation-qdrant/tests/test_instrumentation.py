import json
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

from respan_instrumentation_qdrant import QdrantInstrumentor
from respan_instrumentation_qdrant import _instrumentation
from respan_instrumentation_qdrant._constants import (
    MAX_ATTRIBUTE_CHARS,
    MAX_PREVIEW_ITEMS,
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


def _install_fake_qdrant(monkeypatch):
    module = ModuleType("qdrant_client")

    class QdrantClient:
        def create_collection(self, collection_name, vectors_config=None):
            return {"collection_name": collection_name, "created": True}

        def upsert(self, collection_name, points, api_key=None):
            return {"collection_name": collection_name, "points": points}

        def query_points(self, collection_name, query, limit=10):
            if collection_name == "missing":
                raise RuntimeError("collection does not exist")
            return {"points": [{"id": 1, "score": 0.99}], "query": query}

    class AsyncQdrantClient:
        async def upsert(self, collection_name, points):
            return {"collection_name": collection_name, "count": len(points)}

        async def query_points(self, collection_name, query, limit=10):
            return {"points": [{"id": 2, "score": 0.9}], "query": query}

    module.QdrantClient = QdrantClient
    module.AsyncQdrantClient = AsyncQdrantClient
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    return QdrantClient, AsyncQdrantClient


@pytest.fixture(autouse=True)
def reset_instrumentor():
    QdrantInstrumentor._patches_applied = False
    QdrantInstrumentor._activation_count = 0
    QdrantInstrumentor._patched_targets = []
    yield
    for target, method in reversed(QdrantInstrumentor._patched_targets):
        _instrumentation.unwrap(target, method)
    QdrantInstrumentor._patches_applied = False
    QdrantInstrumentor._activation_count = 0
    QdrantInstrumentor._patched_targets = []


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


def test_package_exports_qdrant_instrumentor():
    assert QdrantInstrumentor is _instrumentation.QdrantInstrumentor
    assert QdrantInstrumentor.name == "qdrant"


def test_sync_operations_emit_canonical_task_spans(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.instrument()

    client = QdrantClient()
    client.create_collection("docs", vectors_config={"size": 3})
    client.upsert(
        "docs",
        [{"id": 1, "vector": [0.1, 0.2, 0.3]}],
        api_key="do-not-export",
    )
    result = client.query_points("docs", [0.1, 0.2, 0.3], limit=1)

    assert result["points"][0]["id"] == 1
    assert [span.name for span in tracer.spans] == [
        "qdrant.create_collection",
        "qdrant.upsert",
        "qdrant.query_points",
    ]
    for span in tracer.spans:
        _assert_contract(span)
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
        assert span.status.status_code is StatusCode.OK
    assert (
        "<redacted>"
        in tracer.spans[1].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_async_operations_are_awaited_and_traced(monkeypatch):
    _, AsyncQdrantClient = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.activate()

    result = await AsyncQdrantClient().upsert(
        "docs",
        [{"id": 2, "vector": [0.3, 0.2, 0.1]}],
    )

    assert result["count"] == 1
    assert [span.name for span in tracer.spans] == ["qdrant.upsert"]
    _assert_contract(tracer.spans[0])
    instrumentor.deactivate()


def test_errors_are_recorded_and_reraised(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="collection does not exist"):
        QdrantClient().query_points("missing", [0.0, 0.0, 0.0])

    span = tracer.spans[-1]
    assert span.status.status_code is StatusCode.ERROR
    assert span.exceptions
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "collection does not exist"
    assert (
        "collection does not exist"
        in span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    instrumentor.deactivate()


def test_capture_content_false_omits_inputs_and_outputs(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = QdrantInstrumentor(capture_content=False)
    instrumentor.activate()

    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])

    attrs = tracer.spans[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    _assert_contract(tracer.spans[0])
    instrumentor.deactivate()


def test_full_vectors_survive_canonical_input_and_output(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = QdrantInstrumentor()
    instrumentor.activate()
    vector = [0.125] * (MAX_ATTRIBUTE_CHARS + 17)
    tags = [f"tag-{index}" for index in range(MAX_PREVIEW_ITEMS + 17)]

    QdrantClient().upsert(
        "docs",
        [{"id": 1, "vector": vector, "payload": {"tags": tags}}],
    )

    span = tracer.spans[-1]
    entity_input = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    entity_output = json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert entity_input["points"][0]["vector"] == vector
    assert entity_output["points"][0]["vector"] == vector
    assert len(entity_input["points"][0]["vector"]) == len(vector)
    assert (
        len(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        > MAX_ATTRIBUTE_CHARS
    )
    assert (
        len(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        > MAX_ATTRIBUTE_CHARS
    )
    assert entity_input["points"][0]["payload"]["tags"]["truncated"] is True
    instrumentor.deactivate()


def test_lifecycle_is_idempotent_and_reference_counted(monkeypatch):
    QdrantClient, _ = _install_fake_qdrant(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    first = QdrantInstrumentor()
    second = QdrantInstrumentor()

    first.activate()
    first.activate()
    second.activate()
    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])
    assert len(tracer.spans) == 1

    first.deactivate()
    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])
    assert len(tracer.spans) == 2

    second.deactivate()
    QdrantClient().query_points("docs", [0.1, 0.2, 0.3])
    assert len(tracer.spans) == 2
