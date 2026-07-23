import asyncio
import sys
from contextlib import contextmanager
from types import ModuleType

from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_milvus import MilvusInstrumentor
from respan_instrumentation_milvus import (
    _native_instrumentation as native_instrumentation,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class _Span:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status


class _Tracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, **_kwargs):
        span = _Span(name)
        self.spans.append(span)
        yield span


def test_sync_and_async_milvus_operations(monkeypatch):
    pymilvus = ModuleType("pymilvus")

    class MilvusClient:
        def insert(self, collection_name, data):
            return {"insert_count": len(data), "collection": collection_name}

        def search(self, collection_name, data, limit=10):
            return [[{"id": 1, "distance": 0.01}]]

        def close(self):
            return None

    class AsyncMilvusClient:
        async def query(self, collection_name, filter=""):
            return [{"id": 1, "collection": collection_name, "filter": filter}]

        async def close(self):
            return None

    pymilvus.MilvusClient = MilvusClient
    pymilvus.AsyncMilvusClient = AsyncMilvusClient
    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)

    tracer = _Tracer()
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", lambda _: tracer)
    MilvusInstrumentor._patches_applied = False
    instrumentor = MilvusInstrumentor()
    instrumentor.activate()

    assert MilvusClient().insert("docs", [{"id": 1}])["insert_count"] == 1
    assert MilvusClient().search("docs", [[0.1, 0.2]])[0]
    assert asyncio.run(AsyncMilvusClient().query("docs", "id == 1"))[0]["id"] == 1
    assert [span.name for span in tracer.spans] == [
        "milvus.client.insert",
        "milvus.client.search",
        "milvus.client.query",
    ]
    for span in tracer.spans:
        assert span.attributes[RESPAN_LOG_TYPE] == "task"
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
    assert not hasattr(MilvusClient.close, "__wrapped__")

    instrumentor.deactivate()
