from __future__ import annotations

import asyncio
import json
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv.trace import SpanAttributes as OTelSpanAttributes
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from pinecone import Index

try:
    from pinecone.async_client.async_index import AsyncIndex
except ImportError:
    AsyncIndex = None
from pinecone.exceptions import PineconeApiException
from respan_instrumentation_pinecone import PineconeInstrumentor
from respan_instrumentation_pinecone import (
    _native_instrumentation as native_instrumentation,
)
from respan_instrumentation_pinecone._serialization import json_dumps
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


class _Handler(BaseHTTPRequestHandler):
    server_version = "PineconeContractFixture/1.0"

    def log_message(self, *_args) -> None:
        return

    def _json(self, status: int, value: object) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path.startswith("/indexes"):
            self._json(200, {"indexes": []})
            return
        if self.path.startswith("/vectors/list"):
            if "error" in self.path:
                self._json(400, {"message": "expected list error"})
            else:
                result = {
                    "vectors": [
                        {"id": "doc-2" if "paginationToken" in self.path else "doc-1"}
                    ],
                    "namespace": "demo",
                }
                if "paginationToken" not in self.path:
                    result["pagination"] = {"next": "next-page"}
                self._json(200, result)
            return
        if self.path.startswith("/namespaces"):
            self._json(200, {"namespaces": [{"name": "demo", "recordCount": 1}]})
            return
        if self.path.startswith("/bulk/imports"):
            self._json(
                200,
                {
                    "data": [
                        {
                            "id": "import-1",
                            "uri": "s3://fixture",
                            "status": "Completed",
                            "createdAt": "2026-10-03T00:00:00Z",
                        }
                    ]
                },
            )
            return
        if self.path.startswith("/vectors/fetch"):
            self._json(
                200,
                {
                    "namespace": "demo",
                    "vectors": {
                        "doc-1": {
                            "id": "doc-1",
                            "values": [0.1, 0.2],
                            "metadata": {"topic": "tracing"},
                        }
                    },
                },
            )
            return
        self._json(404, {"message": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path.endswith("/documents/upsert"):
            self._json(200, {"upserted_count": 1})
        elif self.path.endswith("/documents/search"):
            self._json(
                200,
                {
                    "namespace": "demo",
                    "matches": [{"_id": "doc-1", "_score": 0.99, "title": "tracing"}],
                    "usage": {"read_units": 1},
                },
            )
        elif self.path.endswith("/documents/fetch"):
            self._json(
                200,
                {
                    "namespace": "demo",
                    "documents": {"doc-1": {"_id": "doc-1", "title": "tracing"}},
                    "usage": {"read_units": 1},
                },
            )
        elif self.path.endswith("/documents/update"):
            self._json(200, {"matched_records": 1})
        elif self.path.endswith("/documents/delete"):
            if "/error/" in self.path:
                self._json(503, {"message": "deterministic Pinecone outage"})
            else:
                self._json(200, {"matched_records": 1})
        elif self.path.endswith("/documents/list"):
            if "/error/" in self.path:
                self._json(503, {"message": "deterministic page failure"})
            elif body.get("pagination_token"):
                self._json(
                    200,
                    {
                        "documents": [{"_id": "doc-2"}],
                        "namespace": "demo",
                        "usage": {"read_units": 1},
                    },
                )
            else:
                self._json(
                    200,
                    {
                        "documents": [{"_id": "doc-1"}],
                        "namespace": "demo",
                        "usage": {"read_units": 1},
                        "pagination": {"next": "second-page"},
                    },
                )
        elif self.path == "/describe_index_stats":
            self._json(
                200,
                {
                    "dimension": 2,
                    "indexFullness": 0,
                    "namespaces": {},
                    "totalVectorCount": 1,
                },
            )
        elif self.path == "/vectors/upsert":
            self._json(200, {"upsertedCount": 1})
        elif self.path == "/query":
            self._json(
                200,
                {
                    "namespace": "demo",
                    "matches": [
                        {
                            "id": "doc-1",
                            "score": 0.99,
                            "values": [0.1, 0.2],
                            "metadata": {"topic": "tracing"},
                        }
                    ],
                },
            )
        elif self.path == "/vectors/delete":
            self._json(503, {"message": "deterministic Pinecone outage"})
        else:
            self._json(404, {"message": "not found"})


@contextmanager
def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def exporter(monkeypatch):
    provider = TracerProvider()
    span_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", provider.get_tracer)
    yield provider, span_exporter
    provider.shutdown()


def test_real_current_sdk_exports_sync_async_and_failure(exporter):
    provider, span_exporter = exporter
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    try:
        with _server() as host:
            index = Index(
                host=host, api_key="pinecone-contract-secret", ssl_verify=False
            )

            async def async_fetch() -> object:
                async_index = AsyncIndex(
                    host=host,
                    api_key="pinecone-contract-secret",
                    ssl_verify=False,
                )
                try:
                    return await async_index.fetch(ids=["doc-1"], namespace="demo")
                finally:
                    await async_index.close()

            tracer = provider.get_tracer("pinecone.contract")
            with tracer.start_as_current_span("root") as root:
                index.describe_index_stats()
                index.upsert(
                    vectors=[{"id": "doc-1", "values": [0.1, 0.2]}],
                    namespace="demo",
                )
                index.query(
                    vector=[0.1, 0.2],
                    top_k=1,
                    namespace="demo",
                    include_metadata=True,
                    include_values=True,
                )
                if AsyncIndex is not None:
                    asyncio.run(async_fetch())
                else:
                    index.fetch(ids=["doc-1"], namespace="demo")
                with pytest.raises(PineconeApiException):
                    index.delete(ids=["doc-1"], namespace="demo")

            spans = span_exporter.get_finished_spans()
            names = Counter(span.name for span in spans)
            assert names == Counter(
                {
                    "root": 1,
                    "pinecone.index.describe_index_stats": 1,
                    "pinecone.index.upsert": 1,
                    "pinecone.index.query": 1,
                    "pinecone.index.fetch": 1,
                    "pinecone.index.delete": 1,
                }
            )
            assert len({span.context.span_id for span in spans}) == len(spans)
            client_spans = [span for span in spans if span.name != "root"]
            assert all(
                span.parent.span_id == root.context.span_id for span in client_spans
            )
            assert all(
                span.attributes[RESPAN_LOG_TYPE] == "task" for span in client_spans
            )
            assert all(
                SpanAttributes.TRACELOOP_SPAN_KIND not in span.attributes
                for span in client_spans
            )
            assert all(
                span.attributes[OTelSpanAttributes.DB_SYSTEM] == "pinecone"
                for span in client_spans
            )
            for span in client_spans:
                assert json.loads(
                    span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
                )
                if SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes:
                    json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])

            failed = next(span for span in client_spans if span.name.endswith("delete"))
            assert failed.status.status_code is StatusCode.ERROR
            assert failed.attributes[HTTP_RESPONSE_STATUS_CODE] == 503
            assert (
                "deterministic Pinecone outage" in failed.attributes[ERROR_MESSAGE_ATTR]
            )
            assert failed.events[0].name == "exception"
            exported = json.dumps([dict(span.attributes) for span in spans])
            assert "pinecone-contract-secret" not in exported
            assert "0x" not in exported
    finally:
        instrumentor.deactivate()


def test_serializer_is_bounded_redacted_and_never_calls_repr_or_str():
    calls = {"repr": 0, "str": 0}

    class Hostile:
        def __repr__(self):
            calls["repr"] += 1
            raise AssertionError("repr must not be called")

        def __str__(self):
            calls["str"] += 1
            raise AssertionError("str must not be called")

    encoded = json_dumps(
        {
            "api_key": "plain-secret",
            "client_secret": "another-secret",
            "nested": {"auth_token": "token-value", "value": Hostile()},
            "unicode": "😀" * 10_000,
        }
    )
    assert len(encoded.encode("utf-8")) <= 16_000
    parsed = json.loads(encoded)
    assert parsed
    assert "plain-secret" not in encoded
    assert "another-secret" not in encoded
    assert "token-value" not in encoded
    assert calls == {"repr": 0, "str": 0}


def test_multiple_instances_share_lifecycle_and_reject_config_mismatch(monkeypatch):
    class IndexFixture:
        def query(self, vector, top_k):
            return {"matches": [], "vector": vector, "top_k": top_k}

    module = SimpleNamespace(Index=IndexFixture)
    monkeypatch.setattr(
        native_instrumentation.importlib,
        "import_module",
        lambda name: (
            module
            if name == "pinecone.index"
            else (_ for _ in ()).throw(ImportError(name))
        ),
    )
    first = PineconeInstrumentor()
    second = PineconeInstrumentor()
    first.activate()
    second.activate()
    try:
        assert PineconeInstrumentor._activation_count == 2
        first.deactivate()
        assert PineconeInstrumentor._activation_count == 1
        assert PineconeInstrumentor._patches_applied is True
        with pytest.raises(ValueError):
            PineconeInstrumentor(capture_content=False).activate()
    finally:
        second.deactivate()
    assert PineconeInstrumentor._activation_count == 0
    assert PineconeInstrumentor._patches_applied is False


@pytest.mark.skipif(
    not hasattr(Index, "documents"), reason="Document API requires Pinecone 10"
)
def test_documents_sync_async_and_error_spans(exporter):
    from pinecone import TextQuery

    provider, span_exporter = exporter
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    try:
        with _server() as host:
            with (
                Index(host=host, api_key="fixture-secret") as index,
                provider.get_tracer("test").start_as_current_span(
                    "documents-parent"
                ) as parent,
            ):
                result = index.documents.upsert(
                    namespace="demo", documents=[{"_id": "doc-1", "title": "tracing"}]
                )
                assert result.upserted_count == 1
                index.documents.search(
                    namespace="demo",
                    score_by=[TextQuery(query="tracing", fields=["title"])],
                    top_k=1,
                )

                async def fetch():
                    async with AsyncIndex(
                        host=host, api_key="fixture-secret"
                    ) as client:
                        return await client.documents.fetch(
                            namespace="demo", ids=["doc-1"]
                        )

                assert asyncio.run(fetch()).documents["doc-1"].title == "tracing"
                with pytest.raises(PineconeApiException):
                    index.documents.delete(namespace="error", ids=["doc-1"])
            spans = [
                s
                for s in span_exporter.get_finished_spans()
                if s.name.startswith("pinecone.")
            ]
            assert len(spans) == 4
            assert all(s.parent.span_id == parent.context.span_id for s in spans)
            by_name = {s.name: s for s in spans}
            output = json.loads(
                by_name["pinecone.documents.upsert"].attributes[
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT
                ]
            )
            assert output["upserted_count"] == 1
            output = json.loads(
                by_name["pinecone.documents.search"].attributes[
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT
                ]
            )
            assert output["matches"][0]["title"] == "tracing"
            assert (
                by_name["pinecone.documents.delete"].status.status_code
                is StatusCode.ERROR
            )
            assert all(s.attributes[RESPAN_LOG_TYPE] == "task" for s in spans)
    finally:
        instrumentor.deactivate()


@pytest.mark.skipif(
    not hasattr(Index, "documents"), reason="Document API requires Pinecone 10"
)
@pytest.mark.parametrize("use_async", [False, True])
def test_all_document_methods_and_lazy_pages(exporter, use_async):
    from pinecone.models.pagination import AsyncPaginator, Paginator

    provider, span_exporter = exporter
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    try:
        with _server() as host:

            async def run_async():
                async with AsyncIndex(host=host, api_key="fixture-secret") as index:
                    await index.documents.batch_upsert(
                        namespace="demo",
                        documents=[{"_id": "doc-1", "title": "tracing"}],
                        show_progress=False,
                    )
                    await index.documents.upsert(
                        namespace="demo", documents=[{"_id": "doc-1"}]
                    )
                    await index.documents.update(
                        namespace="demo", documents=[{"_id": "doc-1", "title": "new"}]
                    )
                    await index.documents.delete(namespace="demo", ids=["doc-1"])
                    await index.documents.search(
                        namespace="demo",
                        score_by=[
                            {"type": "text", "query": "tracing", "fields": ["title"]}
                        ],
                        top_k=1,
                    )
                    await index.documents.fetch(namespace="demo", ids=["doc-1"])
                    before = len(span_exporter.get_finished_spans())
                    paginator = index.documents.list(namespace="demo")
                    assert isinstance(paginator, AsyncPaginator)
                    assert len(span_exporter.get_finished_spans()) == before
                    assert [item.id for item in await paginator.to_list()] == [
                        "doc-1",
                        "doc-2",
                    ]
                    assert paginator.pagination_token is None
                    with pytest.raises(PineconeApiException):
                        await index.documents.list(namespace="error").to_list()

            with provider.get_tracer("test").start_as_current_span("parent") as parent:
                if use_async:
                    asyncio.run(run_async())
                else:
                    with Index(host=host, api_key="fixture-secret") as index:
                        index.documents.batch_upsert(
                            namespace="demo",
                            documents=[{"_id": "doc-1", "title": "tracing"}],
                            show_progress=False,
                        )
                        index.documents.upsert(
                            namespace="demo", documents=[{"_id": "doc-1"}]
                        )
                        index.documents.update(
                            namespace="demo",
                            documents=[{"_id": "doc-1", "title": "new"}],
                        )
                        index.documents.delete(namespace="demo", ids=["doc-1"])
                        index.documents.search(
                            namespace="demo",
                            score_by=[
                                {
                                    "type": "text",
                                    "query": "tracing",
                                    "fields": ["title"],
                                }
                            ],
                            top_k=1,
                        )
                        index.documents.fetch(namespace="demo", ids=["doc-1"])
                        before = len(span_exporter.get_finished_spans())
                        paginator = index.documents.list(namespace="demo")
                        assert isinstance(paginator, Paginator)
                        assert len(span_exporter.get_finished_spans()) == before
                        assert [
                            item.id for page in paginator.pages() for item in page.items
                        ] == ["doc-1", "doc-2"]
                        with pytest.raises(PineconeApiException):
                            index.documents.list(namespace="error").to_list()
            spans = [
                span
                for span in span_exporter.get_finished_spans()
                if span.name.startswith("pinecone.")
            ]
            assert len(spans) == 9
            assert all(span.parent.span_id == parent.context.span_id for span in spans)
            pages = [span for span in spans if span.name == "pinecone.documents.list"]
            assert len(pages) == 3
            assert (
                json.loads(pages[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
                    "items"
                ][0]["id"]
                == "doc-1"
            )
            assert pages[2].status.status_code is StatusCode.ERROR
            assert len({span.context.span_id for span in spans}) == 9
    finally:
        instrumentor.deactivate()


@pytest.mark.skipif(
    not hasattr(Index, "documents"), reason="Document API requires Pinecone 10"
)
def test_paginator_validation_capture_opt_out_and_deactivation(exporter):
    _, span_exporter = exporter
    instrumentor = PineconeInstrumentor(capture_content=False)
    instrumentor.activate()
    try:
        with _server() as host, Index(host=host, api_key="fixture-secret") as index:
            with pytest.raises(ValueError):
                index.documents.list(namespace="")
            assert (
                span_exporter.get_finished_spans()[0].status.status_code
                is StatusCode.ERROR
            )
            span_exporter.clear()
            first = next(index.documents.list(namespace="demo").pages())
            assert first.items[0].id == "doc-1"
            spans = span_exporter.get_finished_spans()
            assert len(spans) == 1
            assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in spans[0].attributes
            assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in spans[0].attributes
            paginator = index.documents.list(namespace="demo")
            instrumentor.deactivate()
            assert len(paginator.to_list()) == 2
            assert len(span_exporter.get_finished_spans()) == 1
    finally:
        instrumentor.deactivate()


@pytest.mark.skipif(AsyncIndex is None, reason="Requires current async client")
def test_released_async_iterators_and_control_paginator(exporter):
    from opentelemetry import trace
    from pinecone import AsyncPinecone
    from pinecone.models.pagination import AsyncPaginator

    provider, span_exporter = exporter
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    try:
        with _server() as host:

            async def run():
                async with AsyncIndex(host=host, api_key="fixture-key") as index:
                    stream = index.list(namespace="demo")
                    assert hasattr(stream, "__anext__")
                    assert not span_exporter.get_finished_spans()
                    values = []
                    async for page in stream:
                        assert (
                            trace.get_current_span().get_span_context().span_id
                            == parent.context.span_id
                        )
                        values.extend(v.id for v in page.vectors)
                    assert values == ["doc-1", "doc-2"]
                    assert [
                        p.namespaces[0].name async for p in index.list_namespaces()
                    ] == ["demo"]
                    assert [item.id async for item in index.list_imports()] == [
                        "import-1"
                    ]
                    early = index.list(namespace="demo")
                    await anext(early)
                    await early.aclose()
                    with pytest.raises(
                        PineconeApiException, match="expected list error"
                    ):
                        await anext(index.list(namespace="error"))
                async with AsyncPinecone(host=host, api_key="fixture-key") as client:
                    paginator = client.indexes.list()
                    assert isinstance(paginator, AsyncPaginator)
                    assert await paginator.to_list() == []

            with provider.get_tracer("test").start_as_current_span("parent") as parent:
                asyncio.run(run())
            spans = [
                s for s in span_exporter.get_finished_spans() if s.name != "parent"
            ]
            assert len(spans) == 6
            assert all(s.parent.span_id == parent.context.span_id for s in spans)
            assert spans[4].status.status_code is StatusCode.ERROR
            assert (
                json.loads(spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
                    "count"
                ]
                == 2
            )
            assert (
                json.loads(spans[3].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
                    "count"
                ]
                == 1
            )
    finally:
        instrumentor.deactivate()


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_async_generator_cancellation_preserves_exception_and_closes_source(
    exporter, cleanup_error
):
    _, span_exporter = exporter
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    closed = []

    async def source():
        try:
            yield "first"
            await asyncio.Event().wait()
        finally:
            closed.append(True)
            if cleanup_error:
                raise RuntimeError("source cleanup error")

    async def run():
        stream = instrumentor._trace_async_generator("index.list", source, None, (), {})
        assert await anext(stream) == "first"
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        task.cancel("cancel fixture")
        # Native generator cleanup itself replaces cancellation before the
        # wrapper sees it; preserve the actual source exception unchanged.
        expected_type = RuntimeError if cleanup_error else asyncio.CancelledError
        expected_message = "source cleanup error" if cleanup_error else "cancel fixture"
        with pytest.raises(expected_type, match=expected_message):
            await task

    try:
        asyncio.run(run())
        assert closed == [True]
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code is StatusCode.ERROR
    finally:
        instrumentor.deactivate()


@pytest.mark.parametrize("nested", [False, True])
def test_bypassed_async_generator_early_close_closes_native_source(exporter, nested):
    _, span_exporter = exporter
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    closed = []

    async def source():
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    stream = instrumentor._trace_async_generator("index.list", source, None, (), {})
    if not nested:
        instrumentor.deactivate()

    async def run():
        token = PineconeInstrumentor._active_call.set(nested)
        try:
            assert await anext(stream) == "first"
            await stream.aclose()
        finally:
            PineconeInstrumentor._active_call.reset(token)

    try:
        asyncio.run(run())
        assert closed == [True]
        assert not span_exporter.get_finished_spans()
    finally:
        instrumentor.deactivate()


def test_missing_callable_signature_keeps_existing_wrapper(exporter, monkeypatch):
    _, span_exporter = exporter
    original_signature = native_instrumentation.inspect.signature

    def signature(target):
        if getattr(target, "__name__", None) == "query":
            raise ValueError("native callable has no signature")
        return original_signature(target)

    monkeypatch.setattr(native_instrumentation.inspect, "signature", signature)
    instrumentor = PineconeInstrumentor()
    instrumentor.activate()
    try:
        with _server() as host, Index(host=host, api_key="fixture-key") as index:
            index.query(vector=[0.1, 0.2], top_k=1)
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "pinecone.index.query"
    finally:
        instrumentor.deactivate()
