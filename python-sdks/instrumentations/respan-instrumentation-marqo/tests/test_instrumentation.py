import sys
from contextlib import contextmanager
from types import ModuleType

import pytest
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_marqo import MarqoInstrumentor
from respan_instrumentation_marqo import (
    _native_instrumentation as native_instrumentation,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class _Span:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exception):
        self.exceptions.append(exception)


class _Tracer:
    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, name, **_kwargs):
        span = _Span(name)
        self.spans.append(span)
        yield span


def test_search_and_embed_emit_canonical_spans(monkeypatch):
    marqo = ModuleType("marqo")
    marqo.__path__ = []
    client_module = ModuleType("marqo.client")
    index_module = ModuleType("marqo.index")

    class Client:
        def index(self, index_name):
            return Index(index_name)

    class Index:
        def __init__(self, index_name="docs"):
            self.index_name = index_name

        def search(self, q, limit=10):
            return {"hits": [{"_id": "doc-1", "_score": 0.9}], "query": q}

        def embed(self, content):
            return {"embeddings": [[0.1, 0.2]], "content": content}

        def health(self):
            raise RuntimeError("marqo unavailable")

    client_module.Client = Client
    index_module.Index = Index
    marqo.client = client_module
    marqo.index = index_module
    for module in (marqo, client_module, index_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    tracer = _Tracer()
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", lambda _: tracer)
    MarqoInstrumentor._patches_applied = False
    instrumentor = MarqoInstrumentor()
    instrumentor.activate()
    second_instrumentor = MarqoInstrumentor()
    second_instrumentor.activate()

    index = Client().index("docs")
    assert index.search("observability")["hits"]
    assert index.embed(["hello"])["embeddings"]
    assert [span.name for span in tracer.spans] == [
        "marqo.client.index",
        "marqo.index.search",
        "marqo.index.embed",
    ]
    assert tracer.spans[1].attributes[RESPAN_LOG_TYPE] == "task"
    assert tracer.spans[2].attributes[RESPAN_LOG_TYPE] == "embedding"
    assert tracer.spans[2].attributes[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    for span in tracer.spans:
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes

    with pytest.raises(RuntimeError, match="marqo unavailable"):
        index.health()
    failed_span = tracer.spans[-1]
    assert failed_span.name == "marqo.index.health"
    assert failed_span.status.status_code == StatusCode.ERROR
    assert [str(exc) for exc in failed_span.exceptions] == ["marqo unavailable"]

    instrumentor.deactivate()
    traced_count = len(tracer.spans)
    assert index.search("still active")["hits"]
    assert len(tracer.spans) == traced_count + 1

    second_instrumentor.deactivate()
    traced_count = len(tracer.spans)
    assert index.search("deactivated")["hits"]
    assert len(tracer.spans) == traced_count
