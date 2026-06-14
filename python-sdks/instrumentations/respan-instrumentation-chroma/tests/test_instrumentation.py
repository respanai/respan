import sys
from contextlib import contextmanager
from types import ModuleType

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_chroma import ChromaInstrumentor
from respan_instrumentation_chroma import _instrumentation
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
    def start_as_current_span(self, name):
        span = _FakeSpan(name)
        self.spans.append(span)
        yield span


def _install_fake_chroma_modules(monkeypatch):
    chromadb_module = ModuleType("chromadb")
    api_module = ModuleType("chromadb.api")
    client_module = ModuleType("chromadb.api.client")
    models_module = ModuleType("chromadb.api.models")
    collection_module = ModuleType("chromadb.api.models.Collection")

    class Collection:
        name = "docs"
        id = "collection-id"
        tenant = "default_tenant"
        database = "default_database"

        def add(self, ids, embeddings=None, metadatas=None, documents=None, **kwargs):
            return None

        def query(self, query_embeddings=None, n_results=10, include=None, **kwargs):
            return {
                "ids": [["doc-1"]],
                "documents": [["Respan traces Chroma operations."]],
                "embeddings": [[[0.1, 0.2, 0.3]]],
                "distances": [[0.0]],
            }

        def count(self):
            return 1

    class Client:
        def create_collection(self, name, **kwargs):
            collection = Collection()
            collection.name = name
            return collection

        def list_collections(self):
            return [Collection()]

    client_module.Client = Client
    collection_module.Collection = Collection
    chromadb_module.api = api_module
    api_module.client = client_module
    api_module.models = models_module
    models_module.Collection = collection_module

    for module in (
        chromadb_module,
        api_module,
        client_module,
        models_module,
        collection_module,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    return Client, Collection


@pytest.fixture(autouse=True)
def reset_chroma_instrumentor():
    ChromaInstrumentor._patches_applied = False
    yield
    ChromaInstrumentor._patches_applied = False


def test_package_exports_chroma_instrumentor():
    assert ChromaInstrumentor is _instrumentation.ChromaInstrumentor
    assert ChromaInstrumentor.name == "chroma"


def test_chroma_methods_emit_task_spans(monkeypatch):
    Client, _ = _install_fake_chroma_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)

    instrumentor = ChromaInstrumentor()
    instrumentor.activate()

    client = Client()
    collection = client.create_collection("docs")
    collection.add(
        ids=["doc-1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Respan traces Chroma operations."],
    )
    result = collection.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=1)

    assert result["ids"] == [["doc-1"]]
    assert [span.name for span in tracer.spans] == [
        "chroma.client.create_collection",
        "chroma.collection.add",
        "chroma.collection.query",
    ]

    for span in tracer.spans:
        attrs = span.attributes
        assert attrs[RESPAN_LOG_TYPE] == "task"
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
            "total_request_tokens",
            "span_tools",
            "has_tool_calls",
            RESPAN_SPAN_TOOLS,
            RESPAN_SPAN_TOOL_CALLS,
            RESPAN_SPAN_HANDOFFS,
        ):
            assert banned_alias not in attrs

    add_input = tracer.spans[1].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    query_output = tracer.spans[2].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert '"embeddings": {"count": 1, "dimensions": 3}' in add_input
    assert '"embeddings": {"count": 1, "dimensions": 3}' in query_output
    assert "Respan traces Chroma operations." in query_output
    assert "['Respan traces Chroma operations.']" not in query_output

    instrumentor.deactivate()


def test_chroma_method_errors_are_recorded(monkeypatch):
    _, Collection = _install_fake_chroma_modules(monkeypatch)
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)

    def broken_count(self):
        raise RuntimeError("count failed")

    monkeypatch.setattr(Collection, "count", broken_count)

    instrumentor = ChromaInstrumentor()
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="count failed"):
        Collection().count()

    span = tracer.spans[-1]
    assert span.name == "chroma.collection.count"
    assert span.exceptions
    assert "count failed" in span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
