import gc
import inspect
import json
from tempfile import TemporaryDirectory

import lancedb
import pyarrow as pa
import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, _Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from respan_instrumentation_lancedb import LanceDBInstrumentor
from respan_instrumentation_lancedb import _instrumentation as adapter
from respan_instrumentation_lancedb._serialization import safe_text
from respan_instrumentation_lancedb._translator import native_json
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

INPUT = "traceloop.entity.input"
OUTPUT = "traceloop.entity.output"


def rows(n=20, dimension=4):
    return [
        {
            "id": i,
            "text": "native searchable row",
            "vector": [float(i)] * dimension,
            "flag": False,
            "zero": 0,
        }
        for i in range(n)
    ]


@pytest.fixture
def runtime():
    with TemporaryDirectory() as path:
        db = lancedb.connect(path)
        table = db.create_table("docs", rows())
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        inst = LanceDBInstrumentor()
        inst.activate(tracer_provider=provider)
        yield path, db, table, provider, exporter, inst
        inst.deactivate()
        provider.shutdown()


def query_span(exporter):
    return next(
        s for s in reversed(exporter.get_finished_spans()) if ".query." in s.name
    )


def bodyless(span):
    assert INPUT not in span.attributes and OUTPUT not in span.attributes
    assert not span.events and not span.status.description
    assert "error.message" not in span.attributes


@pytest.mark.parametrize("method", ["to_list", "to_arrow", "to_pandas", "explain_plan"])
def test_sync_query_actual_result_and_config(runtime, method):
    _, _, table, _, exporter, _ = runtime
    q = (
        table.search([0.0] * 4)
        .where("id >= 0")
        .select(["id", "vector", "flag", "zero"])
        .limit(20)
    )
    result = getattr(q, method)()
    span = query_span(exporter)
    assert span.attributes[RESPAN_LOG_TYPE] == "task"
    assert span.attributes["db.collection.name"] == "docs"
    assert "traceloop.span.kind" not in span.attributes
    payload = json.loads(span.attributes[INPUT])
    assert payload["native_configuration"]["limit"] == 20
    assert payload["native_configuration"]["where"] == "id >= 0"
    if method == "explain_plan":
        assert json.loads(span.attributes[OUTPUT]) == result
    else:
        assert len(json.loads(span.attributes[OUTPUT])) == 20
    assert len(exporter.get_finished_spans()) == 1


def test_full_native_arrow_vectors_schema_and_false_zero(runtime):
    _, db, _, _, exporter, _ = runtime
    data = pa.Table.from_pylist(rows(75, 5001))
    schema = data.schema.with_metadata({b"api_key": b"controlled", b"zero": b"0"})
    table = db.create_table("full", data.cast(schema))
    result = table.search().limit(75).to_arrow()
    span = query_span(exporter)
    out = json.loads(span.attributes[OUTPUT])
    assert type(result) is pa.Table and result.num_rows == 75
    assert len(out) == 75 and len(out[0]["vector"]) == 5001
    assert out[0]["flag"] is False and out[0]["zero"] == 0
    envelope = json.loads(span.attributes["respan.metadata.lancedb.result"])
    assert envelope["schemas"][0]["fields"][0]["name"] == "id"
    assert "controlled" not in json.dumps(envelope)


@pytest.mark.parametrize(
    "method",
    [
        "create_table",
        "open_table",
        "table_names",
        "drop_table",
        "add",
        "update",
        "delete",
        "optimize",
    ],
)
def test_connection_and_table_native_operations(runtime, method):
    _, db, table, _, exporter, _ = runtime
    if method == "create_table":
        result = db.create_table("new", rows(1))
    elif method == "open_table":
        result = db.open_table("docs")
    elif method == "table_names":
        result = db.table_names()
    elif method == "drop_table":
        result = db.drop_table("docs")
    elif method == "add":
        result = table.add(rows(1))
    elif method == "update":
        result = table.update(where="id=0", values={"text": "updated"})
    elif method == "delete":
        result = table.delete("id=0")
    else:
        result = table.optimize()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].attributes["db.operation.name"] == method
    assert OUTPUT in spans[0].attributes
    if result is None:
        assert json.loads(spans[0].attributes[OUTPUT]) is None


def test_sync_merge_builder_and_execution(runtime):
    _, _, table, _, exporter, _ = runtime
    merge = table.merge_insert("id")
    assert type(merge).__name__ == "LanceMergeInsertBuilder"
    assert not exporter.get_finished_spans()
    result = (
        merge.when_matched_update_all().when_not_matched_insert_all().execute(rows(1))
    )
    assert len(exporter.get_finished_spans()) == 1
    span = exporter.get_finished_spans()[0]
    assert span.name == "lancedb.merge.execute"
    assert (
        json.loads(span.attributes[INPUT])["native_configuration"][
            "when_matched_update_all"
        ]
        is True
    )
    assert json.loads(span.attributes[OUTPUT]) == native_json(result)


@pytest.mark.parametrize(
    "consumer", ["read_all", "read_next_batch", "iterator", "context_close"]
)
def test_sync_record_batch_reader_keeps_native_identity_and_no_drain(runtime, consumer):
    _, _, table, _, exporter, _ = runtime
    reader = table.search().limit(20).to_batches(batch_size=3)
    assert type(reader) is pa.RecordBatchReader
    assert json.loads(query_span(exporter).attributes[OUTPUT]) == {
        "type": "RecordBatchReader"
    }
    if consumer == "read_all":
        assert reader.read_all().num_rows == 20
    elif consumer == "read_next_batch":
        assert reader.read_next_batch().num_rows <= 20
    elif consumer == "iterator":
        assert sum(b.num_rows for b in reader) == 20
    else:
        with reader as same:
            assert same is reader
    reader.close()
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["to_list", "to_arrow", "to_pandas", "explain_plan"])
async def test_async_query_native_result(runtime, method):
    path, _, _, _, exporter, _ = runtime
    db = await lancedb.connect_async(path)
    table = await db.open_table("docs")
    exporter.clear()
    query = (await table.search([0.0] * 4)).limit(20).where("id >= 0")
    result = await getattr(query, method)()
    span = query_span(exporter)
    payload = json.loads(span.attributes[INPUT])
    assert any(s["method"] == "table.search" for s in payload["builder"])
    assert OUTPUT in span.attributes
    if method == "to_arrow":
        assert type(result) is pa.Table
    db.close()


@pytest.mark.asyncio
async def test_async_merge_is_original_sync_builder_and_real_awaited_result(runtime):
    path, _, _, _, exporter, _ = runtime
    db = await lancedb.connect_async(path)
    table = await db.open_table("docs")
    exporter.clear()
    builder = table.merge_insert("id")
    assert not inspect.isawaitable(builder)
    assert not exporter.get_finished_spans()
    result = await builder.when_not_matched_insert_all().execute(rows(1))
    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].name == "lancedb.merge.execute"
    assert json.loads(spans[0].attributes[OUTPUT]) == native_json(result)
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", ["read_all", "iterator", "deactivate", "gc"])
async def test_async_reader_native_identity_and_lazy_completion(runtime, consumer):
    path, _, _, _, exporter, inst = runtime
    from lancedb.arrow import AsyncRecordBatchReader

    db = await lancedb.connect_async(path)
    table = await db.open_table("docs")
    exporter.clear()
    reader = await table.query().limit(20).to_batches(max_batch_length=3)
    assert type(reader) is AsyncRecordBatchReader and reader.__aiter__() is reader
    assert not exporter.get_finished_spans()
    if consumer == "read_all":
        batches = await reader.read_all()
        assert sum(b.num_rows for b in batches) == 20
    elif consumer == "iterator":
        assert sum([b.num_rows async for b in reader]) == 20
    elif consumer == "deactivate":
        inst.deactivate()
        assert sum([b.num_rows async for b in reader]) == 20
    else:
        del reader
        gc.collect()
    assert len(exporter.get_finished_spans()) == 1
    span = exporter.get_finished_spans()[0]
    if consumer in ("read_all", "iterator"):
        assert sum(len(b) for b in json.loads(span.attributes[OUTPUT])) == 20
    else:
        assert OUTPUT not in span.attributes
    db.close()


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_before_native_payload_inspection(runtime, key, monkeypatch):
    _, _, table, _, exporter, _ = runtime
    monkeypatch.setattr(
        adapter, "dumps", lambda _: pytest.fail("inspected suppressed data")
    )
    token = context.attach(context.set_value(key, True))
    try:
        assert len(table.search().limit(20).to_list()) == 20
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()


def test_sampler_before_payload_inspection(monkeypatch):
    p = TracerProvider(sampler=ALWAYS_OFF)
    i = LanceDBInstrumentor()
    i.activate(tracer_provider=p)
    monkeypatch.setattr(
        adapter, "dumps", lambda _: pytest.fail("inspected unsampled data")
    )
    try:
        with TemporaryDirectory() as path:
            d = lancedb.connect(path)
            t = d.create_table("docs", rows(1))
            assert len(t.search().to_list()) == 1
    finally:
        i.deactivate()
        p.shutdown()


@pytest.mark.parametrize(
    "flag", [ENABLE_CONTENT_TRACING_KEY, "override_enable_content_tracing"]
)
def test_initial_privacy_cannot_widen(runtime, flag):
    _, _, table, _, exporter, _ = runtime
    token = context.attach(context.set_value(flag, False))
    query = table.search()
    context.detach(token)
    query.to_list()
    bodyless(query_span(exporter))


@pytest.mark.parametrize("finished", [False, True])
def test_observed_application_ancestor_attribute_veto(runtime, finished):
    _, _, table, provider, exporter, _ = runtime
    parent = provider.get_tracer("app").start_span(
        "parent", attributes={ENABLE_CONTENT_TRACING_KEY: False}
    )
    parent.set_attribute(ENABLE_CONTENT_TRACING_KEY, True)
    if finished:
        parent.end()
    token = context.attach(trace.set_span_in_context(parent))
    try:
        table.search().limit(20).to_list()
    finally:
        context.detach(token)
    bodyless(query_span(exporter))
    if not finished:
        parent.end()


@pytest.mark.parametrize("remote", [False, True])
def test_unknown_local_carrier_closed_remote_carrier_allowed(runtime, remote):
    _, _, table, _, exporter, _ = runtime
    carrier = trace.NonRecordingSpan(
        trace.SpanContext(123, 456, remote, trace.TraceFlags(1))
    )
    token = context.attach(trace.set_span_in_context(carrier))
    try:
        table.search().limit(20).to_list()
    finally:
        context.detach(token)
    span = query_span(exporter)
    if remote:
        assert OUTPUT in span.attributes
    else:
        bodyless(span)


def test_late_readable_snapshot_veto_before_export(runtime, monkeypatch):
    _, _, table, _, exporter, _ = runtime
    ambient = context.get_current()
    original = _Span.end

    def end(span, *args, **kwargs):
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            return original(span, *args, **kwargs)
        finally:
            context.detach(token)

    monkeypatch.setattr(_Span, "end", end)
    table.search().limit(20).to_list()
    bodyless(query_span(exporter))
    assert context.get_current() is ambient


@pytest.mark.parametrize("fault", ["dumps", "set_attribute", "attach", "detach", "end"])
def test_telemetry_fault_native_result_and_exact_context(runtime, fault, monkeypatch):
    _, _, table, _, _, _ = runtime
    ambient = context.get_current()
    if fault == "dumps":
        monkeypatch.setattr(
            adapter, "dumps", lambda _: (_ for _ in ()).throw(RuntimeError("telemetry"))
        )
    elif fault == "set_attribute":
        monkeypatch.setattr(
            _Span,
            "set_attribute",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("telemetry")),
        )
    elif fault in ("attach", "detach"):
        original = getattr(context, fault)

        def broken(value):
            original(value)
            raise RuntimeError("telemetry")

        monkeypatch.setattr(context, fault, broken)
    else:
        original = _Span.end

        def broken(span, *a, **k):
            original(span, *a, **k)
            raise RuntimeError("telemetry")

        monkeypatch.setattr(_Span, "end", broken)
    assert len(table.search().limit(20).to_list()) == 20
    assert context.get_current() is ambient


def test_actual_native_error_identity_no_synthetic_output(runtime, monkeypatch):
    _, _, table, _, exporter, _ = runtime
    with pytest.raises((ValueError, RuntimeError)) as raised:
        table.search().where("invalid ! SQL").to_list()
    span = query_span(exporter)
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes["error.type"] == type(raised.value).__name__
    assert (
        OUTPUT not in span.attributes
        and "http.response.status_code" not in span.attributes
    )


@pytest.mark.parametrize(
    "text",
    [
        'Bearer "controlled secret"',
        'Basic "controlled secret"',
        r"Bearer \"controlled secret\"",
        r"Basic \"controlled secret\"",
        'Authorization: Bearer "controlled secret"',
        "https://user:password@host/path?api%5Fkey=controlled&answer=0",
    ],
)
def test_redaction_native_rows_and_idempotence(runtime, text):
    _, db, _, _, exporter, _ = runtime
    t = db.create_table(
        "secret",
        [
            {
                "id": 0,
                "text": text,
                "vector": [0.0],
                "schema": {
                    "properties": {
                        "private_key": {"type": "string", "default": "controlled"}
                    },
                    "zero": 0,
                    "flag": False,
                },
            }
        ],
    )
    result = t.search().to_list()
    out = query_span(exporter).attributes[OUTPUT]
    assert "controlled" not in out
    assert "private_key" in out and json.loads(out)[0]["schema"]["zero"] == 0
    assert result[0]["text"] == text
    assert safe_text(safe_text(text)) == safe_text(text)


def test_unknown_hooks_never_used_for_telemetry(runtime):
    class Hostile:
        def __repr__(self):
            raise AssertionError("repr")

        def __iter__(self):
            raise AssertionError("iter")

        def to_pylist(self):
            raise AssertionError("conversion")

    assert native_json({"unknown": Hostile()}) == {"unknown": {"type": "Hostile"}}


def test_shared_reference_config_conflict_foreign_restore(runtime):
    _, _, table, provider, exporter, inst = runtime
    peer = LanceDBInstrumentor()
    peer.activate(tracer_provider=provider)
    with pytest.raises(ValueError):
        LanceDBInstrumentor().activate(tracer_provider=TracerProvider())
    inst.deactivate()
    table.search().limit(20).to_list()
    assert len(exporter.get_finished_spans()) == 1
    peer.deactivate()
    exporter.clear()
    table.search().limit(20).to_list()
    assert not exporter.get_finished_spans()
    assert not adapter._PATCHES and not adapter._POLICIES


@pytest.mark.asyncio
async def test_two_pending_native_readers_and_late_veto(runtime):
    path, _, _, provider, exporter, _ = runtime
    db = await lancedb.connect_async(path)
    table = await db.open_table("docs")
    parent = provider.get_tracer("app").start_span("parent")
    token = context.attach(trace.set_span_in_context(parent))
    try:
        first = await table.query().limit(2).to_batches()
        second = await table.query().limit(2).to_batches()
        await first.read_all()
        await second.read_all()
        captured = [s for s in exporter.get_finished_spans() if ".query." in s.name]
        assert len(captured) == 2 and all(OUTPUT in s.attributes for s in captured)
        private = await table.query().limit(2).to_batches()
        veto = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            await private.__anext__()
        finally:
            context.detach(veto)
        await private.read_all()
        bodyless(query_span(exporter))
    finally:
        context.detach(token)
        parent.end()
        db.close()


@pytest.mark.parametrize("kind", ["scalar", "fts"])
def test_native_indexes_and_full_text_query(runtime, kind):
    _, _, table, _, exporter, _ = runtime
    if kind == "scalar":
        table.create_scalar_index("id")
    else:
        table.create_fts_index("text", use_tantivy=False)
        matches = table.search("native", query_type="fts").limit(20).to_list()
        assert len(matches) == 20
    assert any(
        s.attributes["db.operation.name"]
        == ("create_scalar_index" if kind == "scalar" else "create_fts_index")
        for s in exporter.get_finished_spans()
    )


def test_native_generator_data_is_consumed_only_by_sdk(runtime):
    _, db, _, _, exporter, _ = runtime
    count = 0

    def data():
        nonlocal count
        for row in rows(75):
            count += 1
            yield pa.RecordBatch.from_pylist([row])

    schema = pa.Table.from_pylist(rows(1)).schema
    table = db.create_table("iterable", data(), schema=schema)
    assert count == 75 and table.count_rows() == 75
    span = exporter.get_finished_spans()[-1]
    assert json.loads(span.attributes[INPUT])["arguments"]["data"] == {
        "type": "generator"
    }


def test_foreign_processor_startup_failure_scrubs_and_closes_actual_span(runtime):
    _, _, table, provider, _, _ = runtime
    ambient = context.get_current()
    seen = []

    class Broken(SpanProcessor):
        def on_start(self, span, parent_context=None):
            seen.append(span)
            span.record_exception(ValueError('Bearer "controlled secret"'))
            context.attach(context.set_value("poison", True))
            raise RuntimeError("processor")

        def on_end(self, span):
            pass

    provider.add_span_processor(Broken())
    assert len(table.search().limit(20).to_list()) == 20
    assert context.get_current() is ambient
    assert len(seen) == 1 and not seen[0].is_recording()
    assert not seen[0].events and INPUT not in seen[0].attributes


def test_activation_mutate_then_raise_rolls_back_native_class_and_processor(
    monkeypatch,
):
    provider = TracerProvider()
    original = lancedb.db.LanceDBConnection.create_table
    processors = provider._active_span_processor._span_processors
    failed = False

    def setter(owner, name, value):
        nonlocal failed
        setattr(owner, name, value)
        if not failed and isinstance(owner, type):
            failed = True
            raise RuntimeError("setter mutated")

    monkeypatch.setattr(adapter, "setattr", setter, raising=False)
    with pytest.raises(RuntimeError):
        LanceDBInstrumentor().activate(tracer_provider=provider)
    assert lancedb.db.LanceDBConnection.create_table is original
    assert provider._active_span_processor._span_processors == processors
    assert not adapter._PATCHES and not adapter._POLICIES


def test_foreign_wrapper_identity_survives_deactivation(runtime, monkeypatch):
    _, _, table, _, _, inst = runtime
    owner = lancedb.query.LanceQueryBuilder
    installed = owner.to_list

    def foreign(instance, *args, **kwargs):
        return installed(instance, *args, **kwargs)

    monkeypatch.setattr(owner, "to_list", foreign)
    inst.deactivate()
    assert owner.to_list is foreign
    assert len(table.search().limit(20).to_list()) == 20


def test_known_schema_secret_defaults_and_encoded_json_arguments():
    schema = {
        "properties": {
            "api_key": {"type": "string", "default": "controlled"},
            "flag": {"type": "boolean", "default": False},
            "zero": {"type": "integer", "default": 0},
        }
    }
    payload = native_json(schema)
    from respan_instrumentation_lancedb._translator import dumps

    result = json.loads(dumps(payload))
    assert result["properties"]["api_key"]["type"] == "string"
    assert result["properties"]["api_key"]["default"] == "[REDACTED]"
    assert result["properties"]["flag"]["default"] is False
    assert result["properties"]["zero"]["default"] == 0
    value = '{"api_key":"controlled secret","zero":0,"flag":false}'
    clean = safe_text(value)
    assert json.loads(clean)["zero"] == 0 and "controlled" not in clean
    assert safe_text(clean) == clean


def test_released_respan_disabled_tracer_does_not_capture(runtime, monkeypatch):
    from respan_tracing.core.tracer import RespanTracer

    _, _, table, _, exporter, _ = runtime
    monkeypatch.setattr(RespanTracer, "_instance", None)
    RespanTracer(is_enabled=False)
    assert len(table.search().limit(20).to_list()) == 20
    assert not exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_actual_empty_async_reader_exhaustion_is_captured(runtime):
    path, _, _, _, exporter, _ = runtime
    db = await lancedb.connect_async(path)
    table = await db.open_table("docs")
    reader = await table.query().where("id<0").to_batches()
    assert await reader.read_all() == []
    assert json.loads(query_span(exporter).attributes[OUTPUT]) == []
    db.close()


def test_actual_arrow_extension_conversion_hooks_are_not_called():
    calls = []

    class Scalar(pa.ExtensionScalar):
        def as_py(self, **kwargs):
            calls.append("as_py")
            raise AssertionError("customer conversion")

    class Extension(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.int64(), "respan.controlled")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

        def __arrow_ext_scalar_class__(self):
            return Scalar

        def __str__(self):
            calls.append("str")
            raise AssertionError("customer formatting")

    extension = Extension()
    array = pa.ExtensionArray.from_storage(extension, pa.array([0, 1]))
    value = pa.Table.from_arrays([array], names=["extension"])
    result = native_json(value)
    assert result["type"] == "Table" and calls == []


def test_actual_native_dataframe_unknown_label_adds_no_telemetry_hooks(runtime):
    import pandas as pd

    _, db, _, provider, _, inst = runtime
    calls = []

    class Label:
        def __hash__(self):
            calls.append("hash")
            return 17

        def __str__(self):
            calls.append("str")
            return "controlled"

    observed = []
    inst.deactivate()
    for name in ("bare", "instrumented"):
        if name == "instrumented":
            inst.activate(tracer_provider=provider)
        frame = pd.DataFrame([[0]], columns=[Label()])
        calls.clear()
        table = db.create_table(name, data=frame)
        observed.append((tuple(calls), table.count_rows()))
    assert observed[0] == observed[1]


def test_native_pandas_string_blocks_remain_complete(runtime):
    _, _, table, _, exporter, _ = runtime
    frame = table.search().limit(20).to_pandas()
    result = json.loads(query_span(exporter).attributes[OUTPUT])
    assert len(frame) == len(result) == 20
    assert result[0]["text"] == "native searchable row"
    assert len(result[0]["vector"]) == 4 and result[0]["flag"] is False


@pytest.mark.parametrize("suppression", [False, True])
def test_actual_hybrid_worker_query_legs_do_not_emit_detached_spans(
    runtime, suppression
):
    _, _, table, provider, exporter, _ = runtime
    table.create_fts_index("text", use_tantivy=False)
    exporter.clear()
    parent = provider.get_tracer("app").start_span("parent")
    token = context.attach(trace.set_span_in_context(parent))
    suppressed_token = (
        context.attach(context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True))
        if suppression
        else None
    )
    try:
        builder = (
            table.search(query_type="hybrid").vector([0.0] * 4).text("native").limit(2)
        )
        result = builder.to_arrow()
        assert type(result) is pa.Table and result.num_rows == 2
    finally:
        if suppressed_token is not None:
            context.detach(suppressed_token)
        context.detach(token)
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("lancedb.")]
    if suppression:
        assert spans == []
    else:
        assert (
            len(spans) == 1
            and spans[0].parent.span_id == parent.get_span_context().span_id
        )
        assert json.loads(spans[0].attributes[OUTPUT]) == native_json(result)
    parent.end()
