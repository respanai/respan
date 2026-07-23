from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

from respan_instrumentation_elasticsearch import ElasticsearchInstrumentor
from respan_instrumentation_elasticsearch import _instrumentation
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_SPAN_HANDOFFS,
    RESPAN_SPAN_TOOL_CALLS,
    RESPAN_SPAN_TOOLS,
)


class _FakeSpan:
    def __init__(self, name: str):
        self.name = name
        self.attributes: dict[str, object] = {}
        self.status = None
        self.exceptions: list[BaseException] = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def record_exception(self, exc):
        self.exceptions.append(exc)


class _FakeTracer:
    def __init__(self):
        self.spans: list[_FakeSpan] = []

    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        span = _FakeSpan(name)
        self.spans.append(span)
        yield span


class _Response:
    def __init__(self, body, status=200):
        self.body = body
        self.meta = SimpleNamespace(status=status)


@pytest.fixture(autouse=True)
def reset_instrumentor():
    ElasticsearchInstrumentor._patches_applied = False
    ElasticsearchInstrumentor._activation_count = 0
    ElasticsearchInstrumentor._patched_targets.clear()
    yield
    ElasticsearchInstrumentor._patches_applied = False
    ElasticsearchInstrumentor._activation_count = 0
    ElasticsearchInstrumentor._patched_targets.clear()


def _assert_contract(attrs):
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


def test_sync_search_emits_canonical_task_span(monkeypatch):
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = ElasticsearchInstrumentor()

    response = instrumentor._trace_sync(
        lambda *args, **kwargs: _Response(
            {"took": 2, "hits": {"hits": [{"_id": "doc-1"}]}}
        ),
        ("POST", "/articles/_search"),
        {"body": {"query": {"match": {"title": "tracing"}}}},
    )

    assert response.body["took"] == 2
    span = tracer.spans[-1]
    assert span.name == "elasticsearch.search"
    _assert_contract(span.attributes)
    assert (
        '"title": "tracing"' in span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )
    assert '"doc-1"' in span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert span.attributes["status_code"] == 200


def test_reuses_official_elasticsearch_span_without_duplicate(monkeypatch):
    native_span = _FakeSpan("search")
    tracer = _FakeTracer()
    monkeypatch.setattr(
        _instrumentation, "_active_elasticsearch_span", lambda: native_span
    )
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = ElasticsearchInstrumentor()

    instrumentor._trace_sync(
        lambda *args, **kwargs: _Response({"hits": {"hits": []}}),
        ("POST", "/articles/_search"),
        {"body": {"query": {"match_all": {}}}},
    )

    assert tracer.spans == []
    _assert_contract(native_span.attributes)
    assert native_span.attributes["status_code"] == 200


@pytest.mark.asyncio
async def test_async_index_emits_canonical_task_span(monkeypatch):
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = ElasticsearchInstrumentor()

    async def perform(*args, **kwargs):
        return _Response({"result": "created", "_id": "doc-1"}, status=201)

    await instrumentor._trace_async(
        perform,
        ("PUT", "/articles/_doc/doc-1"),
        {"body": {"title": "Async tracing"}},
    )

    span = tracer.spans[-1]
    assert span.name == "elasticsearch.index"
    _assert_contract(span.attributes)
    assert span.attributes["elasticsearch.target"] == "/articles/_doc/:id"
    assert span.attributes["status_code"] == 201


def test_transport_exception_records_backend_visible_error(monkeypatch):
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = ElasticsearchInstrumentor()

    def perform(*args, **kwargs):
        raise ConnectionError("cluster unavailable")

    with pytest.raises(ConnectionError, match="cluster unavailable"):
        instrumentor._trace_sync(perform, ("GET", "/_cluster/health"), {})

    span = tracer.spans[-1]
    _assert_contract(span.attributes)
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["status_code"] == 500
    assert span.attributes["error.message"] == "cluster unavailable"
    assert span.exceptions


def test_http_error_response_is_marked_error(monkeypatch):
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = ElasticsearchInstrumentor()

    instrumentor._trace_sync(
        lambda *args, **kwargs: _Response(
            {"error": {"type": "index_not_found_exception", "reason": "missing"}},
            status=404,
        ),
        ("GET", "/missing/_doc/id-1"),
        {},
    )

    span = tracer.spans[-1]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["status_code"] == 404
    assert span.attributes["error.message"] == "missing"


def test_capture_content_false_omits_bodies_and_raw_document_id(monkeypatch):
    tracer = _FakeTracer()
    monkeypatch.setattr(_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = ElasticsearchInstrumentor(capture_content=False)

    instrumentor._trace_sync(
        lambda *args, **kwargs: _Response({"secret": "response-secret"}),
        ("PUT", "/articles/_doc/customer-123"),
        {"body": {"secret": "request-secret"}},
    )

    attrs = tracer.spans[-1].attributes
    assert "request-secret" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert "response-secret" not in attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert "customer-123" not in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    assert '"content_captured": false' in attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]

    def fail(*args, **kwargs):
        raise ConnectionError("secret cluster hostname")

    with pytest.raises(ConnectionError, match="secret cluster hostname"):
        instrumentor._trace_sync(fail, ("GET", "/_cluster/health"), {})
    error_span = tracer.spans[-1]
    assert error_span.attributes["error.message"] == "ConnectionError"
    assert (
        "secret cluster hostname"
        not in error_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    assert not error_span.exceptions

    instrumentor._trace_sync(
        lambda *args, **kwargs: _Response(
            {"error": {"reason": "secret backend detail"}}, status=404
        ),
        ("GET", "/missing/_doc/customer-123"),
        {},
    )
    http_error_span = tracer.spans[-1]
    assert http_error_span.attributes["error.message"] == (
        "Elasticsearch returned HTTP 404"
    )
    assert (
        "secret backend detail"
        not in http_error_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )


def test_activate_and_deactivate_are_idempotent(monkeypatch):
    wrapped: list[tuple[str, str]] = []
    unwrapped: list[tuple[str, str]] = []
    fake_module = SimpleNamespace(
        Transport=type("Transport", (), {"perform_request": lambda self: None}),
        AsyncTransport=type(
            "AsyncTransport", (), {"perform_request": lambda self: None}
        ),
    )
    monkeypatch.setattr(
        _instrumentation.importlib, "import_module", lambda _: fake_module
    )
    monkeypatch.setattr(
        _instrumentation,
        "wrap_function_wrapper",
        lambda module, target, wrapper: wrapped.append((module, target)),
    )
    monkeypatch.setattr(
        _instrumentation,
        "unwrap",
        lambda module, target: unwrapped.append((module, target)),
    )

    instrumentor = ElasticsearchInstrumentor()
    instrumentor.activate()
    instrumentor.activate()
    assert len(wrapped) == 2
    assert instrumentor._is_instrumented

    observer = ElasticsearchInstrumentor()
    observer.activate()
    assert observer._is_instrumented
    assert ElasticsearchInstrumentor._activation_count == 2

    instrumentor.deactivate()
    assert ElasticsearchInstrumentor._patches_applied
    assert not unwrapped
    assert not instrumentor._is_instrumented

    observer.deactivate()
    observer.deactivate()
    assert len(unwrapped) == 2
    assert not observer._is_instrumented
    assert ElasticsearchInstrumentor._activation_count == 0
