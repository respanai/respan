import asyncio
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_lancedb import LanceDBInstrumentor
from respan_instrumentation_lancedb import (
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


def _install_modules(monkeypatch):
    lancedb = ModuleType("lancedb")
    lancedb.__path__ = []
    db = ModuleType("lancedb.db")
    table = ModuleType("lancedb.table")
    query = ModuleType("lancedb.query")

    class LanceDBConnection:
        def create_table(self, name, data=None):
            return {"name": name, "rows": len(data or [])}

    class LanceTable:
        name = "docs"

        def add(self, data):
            return {"rows_added": len(data)}

    class LanceQueryBuilder:
        def to_list(self):
            return [{"id": "doc-1", "vector": [0.1, 0.2]}]

    class AsyncConnection:
        async def table_names(self):
            return ["docs"]

    class AsyncTable:
        async def add(self, data):
            return {"rows_added": len(data)}

    class AsyncQueryBase:
        async def to_list(self):
            return [{"id": "doc-1"}]

    db.LanceDBConnection = LanceDBConnection
    db.AsyncConnection = AsyncConnection
    table.LanceTable = LanceTable
    table.AsyncTable = AsyncTable
    query.LanceQueryBuilder = LanceQueryBuilder
    query.AsyncQueryBase = AsyncQueryBase
    lancedb.db = db
    lancedb.table = table
    lancedb.query = query
    for module in (lancedb, db, table, query):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return LanceDBConnection, LanceTable, LanceQueryBuilder, AsyncConnection


@pytest.fixture(autouse=True)
def reset():
    LanceDBInstrumentor._patches_applied = False
    yield
    LanceDBInstrumentor._patches_applied = False


def test_sync_operations_emit_contract_clean_task_spans(monkeypatch):
    Connection, Table, Query, _ = _install_modules(monkeypatch)
    tracer = _Tracer()
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = LanceDBInstrumentor()
    instrumentor.activate()

    assert Connection().create_table("docs", [{"vector": [0.1, 0.2]}])["rows"] == 1
    assert Table().add([{"vector": [0.1, 0.2]}])["rows_added"] == 1
    assert Query().to_list()[0]["id"] == "doc-1"

    assert [span.name for span in tracer.spans] == [
        "lancedb.connection.create_table",
        "lancedb.table.add",
        "lancedb.query.to_list",
    ]
    for span in tracer.spans:
        assert span.attributes[RESPAN_LOG_TYPE] == "task"
        assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT in span.attributes
        for alias in ("tools", "tool_calls", "model", "prompt_tokens", "span_tools"):
            assert alias not in span.attributes

    instrumentor.deactivate()


def test_async_operations_are_awaited_inside_span(monkeypatch):
    *_, AsyncConnection = _install_modules(monkeypatch)
    tracer = _Tracer()
    monkeypatch.setattr(native_instrumentation.trace, "get_tracer", lambda _: tracer)
    instrumentor = LanceDBInstrumentor()
    instrumentor.activate()

    assert asyncio.run(AsyncConnection().table_names()) == ["docs"]
    assert tracer.spans[-1].attributes[RESPAN_LOG_TYPE] == "task"
