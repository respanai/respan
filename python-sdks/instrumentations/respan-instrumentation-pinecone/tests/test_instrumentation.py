import asyncio
import sys
from contextlib import contextmanager
from types import ModuleType

from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_pinecone import PineconeInstrumentor
from respan_instrumentation_pinecone import (
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


def _module(monkeypatch, name):
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_sync_async_and_embedding_operations(monkeypatch):
    pinecone = _module(monkeypatch, "pinecone")
    pinecone.__path__ = []
    index_module = _module(monkeypatch, "pinecone.index")
    async_client = _module(monkeypatch, "pinecone.async_client")
    async_client.__path__ = []
    async_index_module = _module(monkeypatch, "pinecone.async_client.async_index")
    client = _module(monkeypatch, "pinecone.client")
    client.__path__ = []
    inference_module = _module(monkeypatch, "pinecone.client.inference")

    class Index:
        def upsert(self, vectors, namespace=""):
            return {"upserted_count": len(vectors), "namespace": namespace}

        def query(self, vector, top_k):
            return {"matches": [{"id": "doc-1", "score": 0.9}], "top_k": top_k}

    class AsyncIndex:
        async def fetch(self, ids, namespace=""):
            return {"vectors": {ids[0]: {"values": [0.1, 0.2]}}}

    class Inference:
        def embed(self, model, inputs):
            return {"model": model, "data": [{"values": [0.1, 0.2]}]}

    index_module.Index = Index
    async_index_module.AsyncIndex = AsyncIndex
    inference_module.Inference = Inference

    tracer = _Tracer()
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", lambda _: tracer)
    PineconeInstrumentor._patches_applied = False
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()

    assert Index().upsert([("doc-1", [0.1, 0.2])])["upserted_count"] == 1
    assert Index().query([0.1, 0.2], 1)["matches"]
    assert asyncio.run(AsyncIndex().fetch(["doc-1"]))["vectors"]
    assert Inference().embed("multilingual-e5-large", ["hello"])["data"]
    assert [span.name for span in tracer.spans] == [
        "pinecone.index.upsert",
        "pinecone.index.query",
        "pinecone.index.fetch",
        "pinecone.inference.embed",
    ]
    assert tracer.spans[-1].attributes[RESPAN_LOG_TYPE] == "embedding"
    assert tracer.spans[-1].attributes[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    assert (
        tracer.spans[-1].attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "multilingual-e5-large"
    )
    for span in tracer.spans:
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes

    instrumentor.deactivate()
