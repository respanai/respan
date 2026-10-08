"""Genuine bundled PostgreSQL engine, native pgvector codecs and libpq clients."""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Mapping

import pgvector.psycopg as vector_adapter
import psycopg
import pytest

try:
    from pgvector import Bit, HalfVector, SparseVector, Vector
except ImportError:
    from pgvector.utils import Bit, HalfVector, SparseVector, Vector
from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, _Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode
from respan_instrumentation_pgvector import PGVectorInstrumentor
from respan_instrumentation_pgvector import _instrumentation as adapter
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from .native_engine import postgres

INPUT = SpanAttributes.TRACELOOP_ENTITY_INPUT
OUTPUT = SpanAttributes.TRACELOOP_ENTITY_OUTPUT


@pytest.fixture(scope="module")
def dsn():
    with postgres() as value:
        with psycopg.connect(value, autocommit=True) as connection:
            connection.execute("CREATE EXTENSION vector")
            assert connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname='vector'"
            ).fetchone() == ("0.8.7",)
            vector_adapter.register_vector(connection)
            connection.execute(
                "CREATE TABLE docs(id integer,embedding vector(5001),payload jsonb,flag boolean,zero integer,empty text)"
            )
            payload = {
                "history": [{"position": i} for i in range(75)],
                "password": "controlled-json-secret",
                "properties": {
                    "api_key": {"type": "string", "default": "controlled-schema-secret"}
                },
            }
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO docs VALUES(%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            i,
                            Vector([0.0] * 5001),
                            psycopg.types.json.Jsonb(payload),
                            False,
                            0,
                            "",
                        )
                        for i in range(75)
                    ],
                )
        yield value


@pytest.fixture
def runtime(dsn):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    owners = []

    def activate(**kwargs):
        owner = PGVectorInstrumentor(tracer_provider=provider, **kwargs)
        owner.activate()
        owners.append(owner)
        return owner

    yield dsn, provider, exporter, activate
    for owner in reversed(owners):
        owner.deactivate()
    provider.shutdown()


def bodyless(span):
    assert INPUT not in span.attributes and OUTPUT not in span.attributes
    assert not span.events and not span.status.description
    assert "error.message" not in span.attributes


def fetch(dsn, asynchronous=False, count=1):
    if asynchronous:

        async def run():
            async with await psycopg.AsyncConnection.connect(
                dsn, autocommit=True
            ) as connection:
                await vector_adapter.register_vector_async(connection)
                cursor = await connection.execute(
                    "SELECT embedding,payload,flag,zero,empty FROM docs ORDER BY id LIMIT %s",
                    (count,),
                )
                rows = await cursor.fetchall()
                await cursor.close()
                return rows

        return asyncio.run(run())
    with psycopg.connect(dsn, autocommit=True) as connection:
        vector_adapter.register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT embedding,payload,flag,zero,empty FROM docs ORDER BY id LIMIT %s",
                (count,),
            )
            return cursor.fetchall()


def vector_values(value):
    return (
        value.to_list()
        if type(value) is Vector or type(value) is HalfVector
        else value.tolist()
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_full75_rows5001_vectors(runtime, asynchronous):
    dsn, _, exporter, activate = runtime
    activate()
    rows = fetch(dsn, asynchronous, count=75)
    assert (
        type(rows) is list
        and len(rows) == 75
        and len(vector_values(rows[0][0])) == 5001
    )
    span = next(
        s for s in exporter.get_finished_spans() if s.name.endswith(".fetchall")
    )
    output = json.loads(span.attributes[OUTPUT])
    assert len(output) == 75 and all(len(row[0]) == 5001 for row in output)
    assert (
        len(output[0][1]["history"]) == 75 and output[0][1]["password"] == "[REDACTED]"
    )
    assert output[0][1]["properties"]["api_key"] == {
        "type": "string",
        "default": "[REDACTED]",
    }
    assert (
        output[0][2:] == [False, 0, ""]
        and rows[0][1]["password"] == "controlled-json-secret"
    )
    assert (
        span.attributes["respan.entity.log_type"] == "task"
        and span.attributes["db.system.name"] == "postgresql"
    )
    assert span.attributes["db.namespace"] == "postgres"
    assert not any(k.startswith(("gen_ai.", "llm.")) for k in span.attributes)
    assert not any(
        k in span.attributes for k in ("status_code", "model", "traceloop.span.kind")
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["vector", "halfvec", "sparsevec", "bit"])
def test_native_all_registered_types(runtime, asynchronous, kind):
    dsn, _, exporter, activate = runtime
    activate()
    native = {
        "vector": Vector([0.25] * 5001),
        "halfvec": HalfVector([0.25] * 5001),
        "sparsevec": SparseVector({0: 1.0, 5000: 2.0}, 5001),
        "bit": Bit("1" + "0" * 5000),
    }[kind]
    query = f"SELECT %s::{kind if kind != 'bit' else 'bit(5001)'}"
    if asynchronous:

        async def run():
            async with await psycopg.AsyncConnection.connect(
                dsn, autocommit=True
            ) as connection:
                await vector_adapter.register_vector_async(connection)
                cursor = await connection.execute(query, (native,), binary=True)
                row = await cursor.fetchone()
                await cursor.close()
                return row

        row = asyncio.run(run())
    else:
        with psycopg.connect(dsn, autocommit=True) as connection:
            vector_adapter.register_vector(connection)
            cursor = connection.execute(query, (native,), binary=True)
            row = cursor.fetchone()
            cursor.close()
    output = json.loads(
        next(
            s for s in exporter.get_finished_spans() if s.name.endswith(".fetchone")
        ).attributes[OUTPUT]
    )[0]
    if kind in ("vector", "halfvec"):
        assert len(output) == 5001 and output[0] == 0.25
    elif kind == "sparsevec":
        assert output == {
            "dimensions": 5001,
            "indices": [0, 5000],
            "values": [1.0, 2.0],
        }
    else:
        import base64

        assert base64.b64decode(output["base64"]) == bytes(row[0])
        assert len(bytes(row[0])) == 4 + (5001 + 7) // 8
        request = json.loads(
            next(
                s for s in exporter.get_finished_spans() if s.name.endswith(".execute")
            ).attributes[INPUT]
        )
        assert request["arguments"][1][0] == {"length": 5001, "bits": "1" + "0" * 5000}
    assert row[0] is not None


@pytest.mark.parametrize(
    "gate", ["capture", "canonical", "traceloop", "respan_env", "traceloop_env"]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_private_bodyless(runtime, monkeypatch, gate, asynchronous):
    dsn, _, exporter, activate = runtime
    activate(capture_content=gate != "capture")
    token = None
    if gate in ("canonical", "traceloop"):
        token = context.attach(
            context.set_value(
                ENABLE_CONTENT_TRACING_KEY
                if gate == "canonical"
                else "override_enable_content_tracing",
                False,
            )
        )
    if gate.endswith("env"):
        monkeypatch.setenv(
            "RESPAN_TRACE_CONTENT"
            if gate == "respan_env"
            else "TRACELOOP_TRACE_CONTENT",
            "false",
        )
    try:
        rows = fetch(dsn, asynchronous)
    finally:
        if token is not None:
            context.detach(token)
    assert len(vector_values(rows[0][0])) == 5001
    for span in exporter.get_finished_spans():
        bodyless(span)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_native_suppression_before_start(runtime, key, asynchronous):
    dsn, _, exporter, activate = runtime
    activate()
    token = context.attach(context.set_value(key, True))
    try:
        rows = fetch(dsn, asynchronous)
    finally:
        context.detach(token)
    assert len(rows) == 1 and not exporter.get_finished_spans()


@pytest.mark.parametrize("sampler", [False, True])
def test_unknown_mapping_row_hooks_match_bare(dsn, sampler):
    calls = []

    class Row(Mapping):
        def __init__(self, values):
            self.values = values

        def __iter__(self):
            calls.append("iter")
            return iter(self.values)

        def __len__(self):
            calls.append("len")
            return len(self.values)

        def __getitem__(self, key):
            calls.append("get")
            return self.values[key]

        def items(self):
            calls.append("items")
            return [("unsafe", 1)]

        def __str__(self):
            calls.append("str")
            return "opaque"

        def __repr__(self):
            calls.append("repr")
            return "opaque"

    def factory(cursor):
        return lambda values: Row(values)

    provider = TracerProvider(sampler=ALWAYS_OFF) if sampler else TracerProvider()
    owner = PGVectorInstrumentor(tracer_provider=provider)
    owner.activate()
    try:
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor(row_factory=factory) as cursor:
                cursor.execute("SELECT 1")
                rows = cursor.fetchall()
            assert type(rows[0]) is Row
            observed = list(calls)
            calls.clear()
            owner.deactivate()
            with connection.cursor(row_factory=factory) as cursor:
                cursor.execute("SELECT 1")
                rows = cursor.fetchall()
            assert calls == observed == []
    finally:
        owner.deactivate()
        provider.shutdown()


@pytest.mark.parametrize("method", ["fetchone", "fetchmany", "fetchall"])
def test_real_tuple_password_column_redaction(runtime, method):
    dsn, _, exporter, activate = runtime
    activate()
    with (
        psycopg.connect(dsn, autocommit=True) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT %(password)s::text AS password",
            {"password": "controlled-column-secret"},
        )
        result = getattr(cursor, method)()
    expected = ("[REDACTED]",) if method == "fetchone" else [("[REDACTED]",)]
    span = next(
        s for s in exporter.get_finished_spans() if s.name.endswith("." + method)
    )
    assert json.loads(span.attributes[OUTPUT]) == json.loads(json.dumps(expected))
    assert (
        ("controlled-column-secret",) in result
        if method != "fetchone"
        else result == ("controlled-column-secret",)
    )
    assert "controlled-column-secret" not in json.dumps(
        [dict(s.attributes) for s in exporter.get_finished_spans()]
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_server_cursor_protocol(runtime, asynchronous):
    dsn, _, exporter, activate = runtime
    activate()
    if asynchronous:

        async def run():
            async with await psycopg.AsyncConnection.connect(dsn) as connection:
                async with connection.cursor(name="controlled_portal") as cursor:
                    await cursor.execute("SELECT id FROM docs ORDER BY id")
                    first = await cursor.fetchmany(2)
                    rest = await cursor.fetchall()
                    assert type(cursor) is psycopg.AsyncServerCursor
                assert cursor.closed
                return first, rest

        first, rest = asyncio.run(run())
    else:
        with psycopg.connect(dsn) as connection:
            with connection.cursor(name="controlled_portal") as cursor:
                cursor.execute("SELECT id FROM docs ORDER BY id")
                first = cursor.fetchmany(2)
                rest = cursor.fetchall()
                assert type(cursor) is psycopg.ServerCursor
            assert cursor.closed
    assert len(first) == 2 and len(rest) == 73
    spans = exporter.get_finished_spans()
    assert any(
        "server_cursor.fetchmany" in s.name
        and json.loads(s.attributes[OUTPUT]) == [[0], [1]]
        for s in spans
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_execute_return_identity_and_cursor_iteration(
    runtime, monkeypatch, asynchronous
):
    dsn, _, _, activate = runtime
    observed = []
    cls = psycopg.AsyncCursor if asynchronous else psycopg.Cursor
    original = cls.execute
    if asynchronous:

        async def foreign(self, *args, **kwargs):
            value = await original(self, *args, **kwargs)
            observed.append(value)
            return value
    else:

        def foreign(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            observed.append(value)
            return value

    monkeypatch.setattr(cls, "execute", foreign)
    activate()
    if asynchronous:

        async def run():
            async with await psycopg.AsyncConnection.connect(
                dsn, autocommit=True
            ) as connection:
                cursor = await connection.execute("SELECT 1 UNION ALL SELECT 2")
                assert cursor is observed[-1] and type(cursor) is psycopg.AsyncCursor
                rows = [row async for row in cursor]
                await cursor.close()
                return rows

        rows = asyncio.run(run())
    else:
        with psycopg.connect(dsn, autocommit=True) as connection:
            cursor = connection.execute("SELECT 1 UNION ALL SELECT 2")
            assert cursor is observed[-1] and type(cursor) is psycopg.Cursor
            rows = list(cursor)
            cursor.close()
    assert rows == [(1,), (2,)]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_executemany_returns_none_and_returning_sets(runtime, asynchronous):
    dsn, _, exporter, activate = runtime
    activate()
    query = "INSERT INTO docs(id) VALUES (%s) RETURNING id"
    if asynchronous:

        async def run():
            async with await psycopg.AsyncConnection.connect(dsn) as connection:
                async with connection.cursor() as cursor:
                    value = await cursor.executemany(
                        query, [(1000,), (1001,)], returning=True
                    )
                    first = await cursor.fetchone()
                    cursor.nextset()
                    second = await cursor.fetchone()
                    assert value is None
                await connection.rollback()
                return first, second

        result = asyncio.run(run())
    else:
        with psycopg.connect(dsn) as connection:
            with connection.cursor() as cursor:
                value = cursor.executemany(query, [(1000,), (1001,)], returning=True)
                first = cursor.fetchone()
                cursor.nextset()
                second = cursor.fetchone()
                assert value is None
            connection.rollback()
            result = first, second
    assert result == ((1000,), (1001,))
    span = next(
        s for s in exporter.get_finished_spans() if s.name.endswith(".executemany")
    )
    assert json.loads(span.attributes[OUTPUT]) is None


def test_native_generator_params_consumed_once(runtime):
    dsn, _, _, activate = runtime
    activate()
    seen = []

    def values():
        for i in range(75):
            seen.append(i)
            yield (2000 + i,)

    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            assert (
                cursor.executemany("INSERT INTO docs(id) VALUES (%s)", values()) is None
            )
        connection.rollback()
    assert seen == list(range(75))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_error_identity_sqlstate_no_fake_result(
    runtime, monkeypatch, asynchronous
):
    dsn, _, exporter, activate = runtime
    seen = []
    cls = psycopg.AsyncCursor if asynchronous else psycopg.Cursor
    original = cls.execute
    if asynchronous:

        async def foreign(self, *args, **kwargs):
            try:
                return await original(self, *args, **kwargs)
            except psycopg.errors.UndefinedTable as error:
                seen.append(error)
                raise
    else:

        def foreign(self, *args, **kwargs):
            try:
                return original(self, *args, **kwargs)
            except psycopg.errors.UndefinedTable as error:
                seen.append(error)
                raise

    monkeypatch.setattr(cls, "execute", foreign)
    activate()
    if asynchronous:

        async def run():
            async with await psycopg.AsyncConnection.connect(
                dsn, autocommit=True
            ) as connection:
                with pytest.raises(psycopg.errors.UndefinedTable) as caught:
                    await connection.execute(
                        "SELECT * FROM controlled_missing_relation"
                    )
                assert caught.value is seen[-1]

        asyncio.run(run())
    else:
        with psycopg.connect(dsn, autocommit=True) as connection:
            with pytest.raises(psycopg.errors.UndefinedTable) as caught:
                connection.execute("SELECT * FROM controlled_missing_relation")
            assert caught.value is seen[-1]
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert (
        span.attributes["error.type"] == "UndefinedTable"
        and span.attributes["db.response.status_code"] == "42P01"
    )
    assert (
        OUTPUT not in span.attributes
        and "status_code" not in span.attributes
        and "http.response.status_code" not in span.attributes
    )


@pytest.mark.parametrize(
    "attribute", [ENABLE_CONTENT_TRACING_KEY, "traceloop.enable_content_tracing"]
)
def test_initial_and_active_parent_veto(runtime, attribute):
    dsn, provider, exporter, activate = runtime
    activate()
    with provider.get_tracer("application").start_as_current_span(
        "parent", attributes={attribute: False}
    ) as parent:
        parent.set_attribute(attribute, True)
        fetch(dsn)
    for span in exporter.get_finished_spans():
        if span.name.startswith("pgvector."):
            bodyless(span)
    exporter.clear()
    with provider.get_tracer("application").start_as_current_span("active") as parent:
        parent.set_attribute(attribute, False)
        with provider.get_tracer("application").start_as_current_span("generic"):
            pass
        parent.set_attribute(attribute, True)
        fetch(dsn)
    for span in exporter.get_finished_spans():
        if span.name.startswith("pgvector."):
            bodyless(span)


def test_remote_unknown_finished_carriers(runtime):
    dsn, provider, exporter, activate = runtime
    activate()
    remote = trace.NonRecordingSpan(
        trace.SpanContext(123, 456, True, trace.TraceFlags(1))
    )
    token = context.attach(trace.set_span_in_context(remote))
    try:
        fetch(dsn)
    finally:
        context.detach(token)
    assert all(INPUT in s.attributes for s in exporter.get_finished_spans())
    exporter.clear()
    parent = provider.get_tracer("application").start_span("observed")
    key = parent.get_span_context()
    parent.end()
    unknown = trace.NonRecordingSpan(
        trace.SpanContext(key.trace_id + 1, key.span_id, False, trace.TraceFlags(1))
    )
    token = context.attach(trace.set_span_in_context(unknown))
    try:
        fetch(dsn)
    finally:
        context.detach(token)
    for span in exporter.get_finished_spans():
        if span.name.startswith("pgvector."):
            bodyless(span)


def test_native_sampler_has_no_conversion(runtime, monkeypatch):
    dsn, _, _, _ = runtime
    provider = TracerProvider(sampler=ALWAYS_OFF)
    owner = PGVectorInstrumentor(tracer_provider=provider)
    owner.activate()
    calls = []
    original = adapter.native_json

    def spy(value, *args, **kwargs):
        calls.append(True)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(adapter, "native_json", spy)
    try:
        assert len(fetch(dsn)) == 1 and not calls
    finally:
        owner.deactivate()
        provider.shutdown()


@pytest.mark.parametrize(
    "fault", ["startup", "attribute", "attach", "detach", "end", "serde"]
)
def test_native_telemetry_faults_preserve_outcome(runtime, monkeypatch, fault):
    dsn, provider, _, activate = runtime
    ambient = context.get_current()
    if fault == "startup":

        class Foreign(SpanProcessor):
            def on_start(self, span, parent_context=None):
                raise RuntimeError("controlled startup")

        provider.add_span_processor(Foreign())
    activate()
    if fault == "attribute":
        original = _Span.set_attribute
        seen = []

        def setter(self, key, value):
            original(self, key, value)
            if key == INPUT and not seen:
                seen.append(True)
                raise RuntimeError("controlled attr")

        monkeypatch.setattr(_Span, "set_attribute", setter)
    if fault == "attach":
        original = context.attach
        seen = []

        def attach(ctx):
            value = original(ctx)
            if not seen:
                seen.append(True)
                raise RuntimeError("controlled attach")
            return value

        monkeypatch.setattr(context, "attach", attach)
    if fault == "detach":

        def detach(*args):
            raise RuntimeError("controlled detach")

        monkeypatch.setattr(context, "detach", detach)
    if fault == "end":

        def end(*args, **kwargs):
            raise RuntimeError("controlled end")

        monkeypatch.setattr(_Span, "end", end)
    if fault == "serde":

        def serializer(*args, **kwargs):
            raise RuntimeError("controlled serde")

        monkeypatch.setattr(adapter, "native_json", serializer)
    assert len(fetch(dsn)) == 1 and context.get_current() is ambient
    assert not adapter._PENDING


def test_late_readable_and_retained_immutable_scrub(runtime, monkeypatch):
    dsn, _, exporter, activate = runtime
    activate()
    original = _Span.end
    retained = []

    def end(self, *args, **kwargs):
        retained.append(self)
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            return original(self, *args, **kwargs)
        finally:
            context.detach(token)

    monkeypatch.setattr(_Span, "end", end)
    fetch(dsn)
    for span in exporter.get_finished_spans():
        bodyless(span)
    for span in retained:
        assert (
            INPUT not in (span.attributes or {})
            and OUTPUT not in (span.attributes or {})
            and not span.events
        )


def test_private_foreign_events_and_bare_error(runtime):
    dsn, provider, exporter, activate = runtime

    class Foreign(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.add_event("controlled diagnostic", {"text": "controlled detail"})
            span.set_status(Status(StatusCode.ERROR, "controlled detail"))

    provider.add_span_processor(Foreign())
    activate(capture_content=False)
    fetch(dsn)
    for span in exporter.get_finished_spans():
        bodyless(span)
        assert span.status.status_code is StatusCode.ERROR
        assert "error.type" not in span.attributes


def test_bare_foreign_error_survives_success(runtime):
    dsn, provider, exporter, activate = runtime

    class Foreign(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_status(Status(StatusCode.ERROR))

    provider.add_span_processor(Foreign())
    activate()
    fetch(dsn)
    for span in exporter.get_finished_spans():
        assert span.status.status_code is StatusCode.ERROR
        assert (
            "error.type" not in span.attributes
            and "error.message" not in span.attributes
        )
        assert span.status.description is None
        if OUTPUT in span.attributes:
            assert (
                "error" not in json.loads(span.attributes[OUTPUT])
                if type(json.loads(span.attributes[OUTPUT])) is dict
                else True
            )


def test_context_false_native_row_callback_latches(runtime):
    dsn, _, exporter, activate = runtime
    activate()
    ambient = context.get_current()
    tokens = []

    def factory(cursor):
        def make(values):
            tokens.append(
                context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            )
            return tuple(values)

        return make

    with (
        psycopg.connect(dsn, autocommit=True) as connection,
        connection.cursor(row_factory=factory) as cursor,
    ):
        cursor.execute("SELECT 1")
        assert cursor.fetchall() == [(1,)]
    assert context.get_current() is ambient
    span = next(s for s in exporter.get_finished_spans() if s.name.endswith("fetchall"))
    bodyless(span)


def test_shared_lifecycle_descriptor_foreign_and_mutate_rollback(runtime, monkeypatch):
    _dsn, provider, _, activate = runtime
    first = activate()
    second = activate()
    first.deactivate()
    with pytest.raises(ValueError):
        PGVectorInstrumentor(tracer_provider=provider, capture_content=False).activate()
    installed = psycopg.Cursor.execute

    @functools.wraps(installed)
    def foreign(*args, **kwargs):
        return installed(*args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, "execute", foreign)
    second.deactivate()
    assert psycopg.Cursor.execute is foreign
    assert all(
        type(p).__name__ != "AncestorPolicy"
        for p in provider._active_span_processor._span_processors
    )
    monkeypatch.setattr(psycopg.Cursor, "execute", installed.__wrapped__)
    prior = psycopg.Cursor.execute
    seen = []

    def setter(owner, name, value):
        setattr(owner, name, value)
        if owner is psycopg.Cursor and name == "execute" and not seen:
            seen.append(True)
            raise RuntimeError("controlled activation")

    monkeypatch.setattr(adapter, "setattr", setter, raising=False)
    owner = PGVectorInstrumentor(tracer_provider=provider)
    with pytest.raises(RuntimeError):
        owner.activate()
    assert (
        psycopg.Cursor.execute is prior and not adapter._PATCHES and not adapter._OWNERS
    )


def test_native_deactivate_pending_callback(runtime):
    dsn, _, exporter, activate = runtime
    owner = activate()

    def factory(cursor):
        def make(values):
            owner.deactivate()
            return tuple(values)

        return make

    with (
        psycopg.connect(dsn, autocommit=True) as connection,
        connection.cursor(row_factory=factory) as cursor,
    ):
        cursor.execute("SELECT 1")
        assert cursor.fetchall() == [(1,)]
    span = next(s for s in exporter.get_finished_spans() if s.name.endswith("fetchall"))
    bodyless(span)
    assert not adapter._PATCHES and not adapter._OWNERS


def test_source_context_siblings_ignore_internal_export_suppression(runtime):
    dsn, provider, exporter, activate = runtime
    activate()
    with provider.get_tracer("application").start_as_current_span("parent"):
        fetch(dsn)
        fetch(dsn)
    spans = [s for s in exporter.get_finished_spans() if s.name.endswith("fetchall")]
    assert len(spans) == 2 and all(
        INPUT in s.attributes and OUTPUT in s.attributes for s in spans
    )


def test_native_psycopg2_registration(runtime):
    import pgvector.psycopg2 as pg2
    import psycopg2

    dsn, _, exporter, activate = runtime
    activate()
    connection = psycopg2.connect(dsn)
    try:
        assert pg2.register_vector(connection) is None
        with connection.cursor() as cursor:
            cursor.execute("SELECT embedding FROM docs LIMIT 1")
            row = cursor.fetchone()
        assert len(vector_values(row[0])) == 5001
    finally:
        connection.close()
    assert (
        len(exporter.get_finished_spans()) == 1
        and json.loads(exporter.get_finished_spans()[0].attributes[OUTPUT]) is None
    )


def test_credential_text_bytes_schema_redaction_native(runtime):
    dsn, _, exporter, activate = runtime
    activate()
    value = {
        "quoted": 'Bearer "controlled bearer secret"',
        "url": "https://example.invalid/?api%5Fkey=controlled-url-secret",
        "properties": {
            "api_key": {"type": "string", "example": "controlled-schema-secret"}
        },
        "flag": False,
        "zero": 0,
        "empty": "",
    }
    with psycopg.connect(dsn, autocommit=True) as connection:
        cursor = connection.execute(
            "SELECT %s::jsonb,%s::bytea",
            (psycopg.types.json.Jsonb(value), b"password=controlled-byte-secret"),
        )
        row = cursor.fetchone()
        cursor.close()
    assert row[0] == value and bytes(row[1]) == b"password=controlled-byte-secret"
    encoded = json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert not any(
        s in encoded
        for s in [
            "controlled bearer secret",
            "controlled-url-secret",
            "controlled-schema-secret",
        ]
    )
    import base64

    output = json.loads(
        next(
            s for s in exporter.get_finished_spans() if s.name.endswith("fetchone")
        ).attributes[OUTPUT]
    )
    assert b"controlled-byte-secret" not in base64.b64decode(output[1]["base64"])


def test_native_empty_null_false_zero_and_composed_sql(runtime):
    dsn, _, exporter, activate = runtime
    activate()
    with psycopg.connect(dsn, autocommit=True) as connection:
        cursor = connection.execute(
            psycopg.sql.SQL("SELECT {},{},{}").format(
                psycopg.sql.Literal(False),
                psycopg.sql.Literal(0),
                psycopg.sql.Literal(""),
            )
        )
        assert cursor.fetchone() == (False, 0, "")
        cursor = connection.execute("SELECT id FROM docs WHERE false")
        assert cursor.fetchall() == [] and cursor.fetchone() is None
    outputs = [
        json.loads(s.attributes[OUTPUT])
        for s in exporter.get_finished_spans()
        if s.name.endswith(("fetchall", "fetchone"))
    ]
    assert [False, 0, ""] in outputs and [] in outputs and None in outputs


def test_released_respan_disabled(runtime, monkeypatch):
    from respan_tracing.core.tracer import RespanTracer

    dsn, _, exporter, activate = runtime
    monkeypatch.setattr(RespanTracer, "_instance", None)
    RespanTracer(is_enabled=False)
    activate()
    fetch(dsn)
    assert not exporter.get_finished_spans()


def test_mutable_numpy_registry_does_not_inspect_opaque_rows(runtime):
    import numpy as np

    dsn, _, _, activate = runtime
    hooks = []

    class Opaque:
        @property
        def dtype(self):
            hooks.append(True)
            raise RuntimeError("controlled customer getter")

    np.sctypeDict["controlled_opaque"] = Opaque

    def factory(cursor):
        return lambda values: Opaque()

    try:
        with psycopg.connect(dsn, autocommit=True, row_factory=factory) as connection:
            rows = connection.execute("SELECT 1").fetchall()
            assert type(rows[0]) is Opaque and not hooks
            activate()
            rows = connection.execute("SELECT 1").fetchall()
            assert type(rows[0]) is Opaque and not hooks
    finally:
        np.sctypeDict.pop("controlled_opaque")


@pytest.mark.asyncio
async def test_deactivate_pending_async_query_finalizes_once(runtime):
    dsn, _, exporter, activate = runtime
    async with await psycopg.AsyncConnection.connect(
        dsn, autocommit=True
    ) as connection:
        cursor = connection.cursor()
        owner = activate()
        task = asyncio.create_task(cursor.execute("SELECT pg_sleep(0.1),1"))
        for _ in range(1000):
            if adapter._PENDING:
                break
            await asyncio.sleep(0)
        assert adapter._PENDING
        owner.deactivate()
        assert len(exporter.get_finished_spans()) == 1
        response = await task
        assert response is cursor and not cursor.closed
        assert len(exporter.get_finished_spans()) == 1
        bodyless(exporter.get_finished_spans()[0])
        await cursor.close()
