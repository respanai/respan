"""Native SDK registration and invocation fixtures; no fake vendor modules."""

import asyncio
import inspect
import json
from contextlib import AsyncExitStack, asynccontextmanager

import pytest
import restate
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_restate import RestateInstrumentor
from respan_instrumentation_restate import _instrumentation as adapter
from respan_instrumentation_restate._policy import content_allowed
from respan_instrumentation_restate._serialization import json_string
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
    RESPAN_METADATA,
    RESPAN_THREADS_ID,
    RESPAN_TRACE_GROUP_ID,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from restate.handler import invoke_handler
from restate.server_context import (
    ServerInvocationContext,
    _restate_context_var,
    restate_context_is_replaying,
)
from restate.server_types import ReceiveChannel
from restate.vm import Invocation, VMWrapper


@pytest.fixture
def runtime(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(adapter.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(adapter.trace, "get_tracer", provider.get_tracer)
    instrumentor = RestateInstrumentor()
    instrumentor.activate()
    yield provider, exporter, instrumentor
    instrumentor.deactivate()
    provider.shutdown()


async def native_invoke(component, name, value, *, replaying=False, codec=None):
    handler = component.handlers[name]
    encoded = handler.handler_io.input_serde.serialize(value)
    invocation = Invocation(
        "inv-native", 7, [], encoded, "order-native", "tenant", "limit", "idem"
    )
    queue = asyncio.Queue()
    receive = ReceiveChannel(queue.get)

    async def send(event):
        pass

    native = ServerInvocationContext(
        VMWrapper([("content-type", "application/vnd.restate.invocation.v5")]),
        handler,
        invocation,
        {},
        send,
        receive,
        **(
            {"journal_codec": codec}
            if "journal_codec" in inspect.signature(ServerInvocationContext).parameters
            else {}
        ),
    )
    token = _restate_context_var.set(native)
    replay_token = restate_context_is_replaying.set(replaying)
    try:
        async with AsyncExitStack() as stack:
            for manager in handler.context_managers or ():
                await stack.enter_async_context(manager())
            return await invoke_handler(
                handler,
                native,
                encoded,
                **(
                    {"journal_codec": codec}
                    if "journal_codec" in inspect.signature(invoke_handler).parameters
                    else {}
                ),
            )
    finally:
        restate_context_is_replaying.reset(replay_token)
        _restate_context_var.reset(token)
        await receive.close()


def component(kind="service", managers=None):
    constructors = {
        "service": restate.Service,
        "object": restate.VirtualObject,
        "workflow": restate.Workflow,
    }
    item = constructors[kind](
        "NativeComponent",
        metadata={"team": "audit", "api_key": "private"},
        invocation_context_managers=managers,
    )
    calls = []

    async def fn(ctx, request: dict):
        calls.append((ctx, request))
        return {"accepted": request}

    decorator = (
        item.main(name="run") if kind == "workflow" else item.handler(name="run")
    )
    decorator(fn)
    return item, calls


@pytest.mark.parametrize("kind", ["service", "object", "workflow"])
@pytest.mark.parametrize("replaying", [False, True])
def test_native_invocation_preserves_calls_results_context_and_contract(
    runtime, kind, replaying
):
    provider, exporter, _ = runtime
    item, calls = component(kind)
    value = {
        "messages": [{"role": "user", "content": "Ada"}],
        "vectors": list(range(300)),
    }
    with provider.get_tracer("test").start_as_current_span("outer"):
        result = asyncio.run(native_invoke(item, "run", value, replaying=replaying))
    assert json.loads(result) == {"accepted": value}
    assert len(calls) == 1 and isinstance(calls[0][0], ServerInvocationContext)
    child, parent = exporter.get_finished_spans()
    assert child.parent.span_id == parent.context.span_id
    assert child.status.status_code is StatusCode.OK
    attrs = child.attributes
    assert attrs[RESPAN_LOG_TYPE] == ("workflow" if kind == "workflow" else "task")
    assert attrs[RESPAN_TRACE_GROUP_ID] == "inv-native"
    assert attrs[RESPAN_THREADS_ID] == "order-native"
    assert json.loads(attrs[RESPAN_METADATA])["restate"]["replaying"] is replaying
    assert "private" not in attrs[RESPAN_METADATA]
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT])["input"] == value
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert "status_code" not in attrs
    assert "traceloop.span.kind" not in attrs


def test_custom_serde_is_called_once_and_opaque_input_is_omitted(runtime):
    _, exporter, _ = runtime

    class Serde(restate.serde.Serde):
        calls = 0

        def deserialize(self, buf):
            self.calls += 1
            return json.loads(buf)

        def serialize(self, value):
            return json.dumps(value).encode()

    serde = Serde()
    item = restate.Service("CustomSerde")

    @item.handler(name="run", input_serde=serde)
    async def fn(ctx, request):
        return request

    assert json.loads(asyncio.run(native_invoke(item, "run", {"ok": 1}))) == {"ok": 1}
    assert serde.calls == 1
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT
        not in exporter.get_finished_spans()[0].attributes
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("secret=credential"),
        restate.TerminalError("secret=credential", status_code=409),
        asyncio.CancelledError("secret=credential"),
    ],
)
def test_original_errors_are_preserved_and_only_real_status_is_used(runtime, error):
    _, exporter, _ = runtime
    item = restate.Service("Fail")

    @item.handler(name="fail")
    async def fn(ctx, request):
        raise error

    with pytest.raises(type(error)) as caught:
        asyncio.run(native_invoke(item, "fail", {"x": 1}))
    assert caught.value is error
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert "credential" not in json.dumps(dict(span.attributes))
    assert "credential" not in (span.status.description or "")
    assert span.attributes.get("status_code") == (
        409 if isinstance(error, restate.TerminalError) else None
    )
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression_precedes_payload_inspection(runtime, monkeypatch, key):
    _, exporter, _ = runtime
    item, calls = component()
    monkeypatch.setattr(
        adapter, "_invocation_details", lambda ctx: pytest.fail("payload inspected")
    )
    token = context.attach(context.set_value(key, True))
    try:
        asyncio.run(native_invoke(item, "run", {"x": 1}))
    finally:
        context.detach(token)
    assert len(calls) == 1 and not exporter.get_finished_spans()


def test_sampling_precedes_payload_inspection(runtime, monkeypatch):
    provider, exporter, _ = runtime
    provider.sampler = ALWAYS_OFF
    monkeypatch.setattr(
        adapter, "_invocation_details", lambda ctx: pytest.fail("payload inspected")
    )
    item, calls = component()
    asyncio.run(native_invoke(item, "run", {"x": 1}))
    assert len(calls) == 1 and not exporter.get_finished_spans()


@pytest.mark.parametrize("veto", ["ambient", "environment", "capture"])
def test_initial_privacy_avoids_payload_and_diagnostics(runtime, monkeypatch, veto):
    _, exporter, instrumentor = runtime
    if veto == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif veto == "capture":
        instrumentor.deactivate()
        instrumentor._capture_content = False
        instrumentor.activate()
    monkeypatch.setattr(
        adapter, "_invocation_details", lambda ctx: pytest.fail("payload inspected")
    )
    item = restate.Service("Private")

    @item.handler(name="run")
    async def fn(ctx, request):
        raise RuntimeError("private-body")

    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if veto == "ambient"
        else None
    )
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(native_invoke(item, "run", {"private": "body"}))
    finally:
        if token is not None:
            context.detach(token)
    span = exporter.get_finished_spans()[0]
    assert "private-body" not in json.dumps(dict(span.attributes))
    assert span.status.description is None and not span.events
    assert RESPAN_METADATA not in span.attributes


def test_supplied_context_veto_and_ambient_veto_both_apply(runtime):
    provider, exporter, _ = runtime
    denied = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    allowed = context.set_value(ENABLE_CONTENT_TRACING_KEY, True)
    assert not content_allowed(True, denied)
    token = context.attach(denied)
    try:
        assert not content_allowed(True, allowed)
    finally:
        context.detach(token)
    parent = provider.get_tracer("test").start_span("parent", context=denied)
    with trace.use_span(parent):
        item, _ = component()
        asyncio.run(native_invoke(item, "run", {"private": "body"}))
    parent.end()
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT
        not in exporter.get_finished_spans()[0].attributes
    )


def test_finished_ancestor_veto_survives_context_restore(runtime):
    provider, exporter, _ = runtime
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    parent = provider.get_tracer("test").start_span("parent")
    parent.end()
    context.detach(token)
    with trace.use_span(parent):
        item, _ = component()
        asyncio.run(native_invoke(item, "run", {"private": "body"}))
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT
        not in exporter.get_finished_spans()[-1].attributes
    )


def test_late_provider_unobserved_local_parent_fails_closed(runtime, monkeypatch):
    _, exporter, _ = runtime
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    parent = provider.get_tracer("test").start_span("unobserved")
    monkeypatch.setattr(adapter.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(adapter.trace, "get_tracer", provider.get_tracer)
    with trace.use_span(parent):
        item, _ = component()
        asyncio.run(native_invoke(item, "run", {"private": "body"}))
    parent.end()
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT
        not in exporter.get_finished_spans()[0].attributes
    )
    provider.shutdown()


def test_active_veto_is_irreversible_for_ancestors_and_descendants(runtime):
    provider, exporter, _ = runtime
    item = restate.Service("Nested")

    @item.handler(name="child")
    async def child(ctx, request):
        return request

    @item.handler(name="run")
    async def fn(ctx, request):
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            await native_invoke(item, "child", {"private": "child"})
        finally:
            context.detach(token)
        await native_invoke(item, "child", {"private": "later"})
        return request

    with provider.get_tracer("test").start_as_current_span("parent"):
        asyncio.run(native_invoke(item, "run", {"private": "initial"}))
        asyncio.run(native_invoke(item, "child", {"private": "sibling"}))
    for span in exporter.get_finished_spans()[:-1]:
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert not span.events and span.status.description is None


def test_final_veto_clears_already_captured_input_before_detach(runtime):
    _, exporter, _ = runtime
    item = restate.Service("LateVeto")

    @item.handler(name="run")
    async def fn(ctx, request):
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        raise RuntimeError("private-diagnostic")

    with pytest.raises(RuntimeError):
        asyncio.run(native_invoke(item, "run", {"private": "initial"}))
    span = exporter.get_finished_spans()[0]
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert not span.events and span.status.description is None


@pytest.mark.parametrize(
    "fault",
    ["details", "start", "attribute", "status", "event", "end", "attach", "detach"],
)
def test_telemetry_faults_preserve_native_results_errors_and_cleanup(
    runtime, monkeypatch, fault
):
    provider, _, _ = runtime
    cleanup = []

    @asynccontextmanager
    async def manager():
        cleanup.append("enter")
        try:
            yield
        finally:
            cleanup.append("exit")

    item, calls = component(managers=[manager])

    def fail(*args, **kwargs):
        raise RuntimeError("telemetry-fault")

    tracer = provider.get_tracer("restate")
    if fault == "details":
        monkeypatch.setattr(adapter, "_invocation_details", fail)
    elif fault == "start":
        monkeypatch.setattr(adapter.trace, "get_tracer", lambda *a, **k: tracer)
        monkeypatch.setattr(tracer, "start_span", fail)
    elif fault in {"attach", "detach"}:
        monkeypatch.setattr(adapter.otel_context, fault, fail)
    else:
        from opentelemetry.sdk.trace import Span

        monkeypatch.setattr(
            Span,
            {
                "attribute": "set_attribute",
                "status": "set_status",
                "event": "add_event",
                "end": "end",
            }[fault],
            fail,
        )
    assert json.loads(asyncio.run(native_invoke(item, "run", {"ok": 1}))) == {
        "accepted": {"ok": 1}
    }
    assert len(calls) == 1 and cleanup == ["enter", "exit"]
    error = RuntimeError("native-failure")
    failing = restate.Service("Fail")

    @failing.handler(name="run")
    async def fn(ctx, request):
        raise error

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(native_invoke(failing, "run", {"ok": 1}))
    assert caught.value is error


def test_shared_lifecycle_foreign_wrapper_and_registration_cleanup(runtime):
    _, _, first = runtime
    second = RestateInstrumentor()
    second.activate()
    patch = adapter._PATCHED_TARGETS[0]
    foreign = lambda *args, **kwargs: patch.replacement(*args, **kwargs)
    setattr(patch.owner, patch.name, foreign)
    try:
        first.deactivate()
        assert adapter._ENABLED
        second.deactivate()
        assert getattr(patch.owner, patch.name) is foreign
        item, _ = component()
        assert adapter._invocation_context not in (
            item.handlers["run"].context_managers or ()
        )
    finally:
        second.deactivate()
        setattr(patch.owner, patch.name, patch.original)


def test_partial_install_rolls_back_without_removing_foreign_targets(
    runtime, monkeypatch
):
    _, _, first = runtime
    first.deactivate()
    original = inspect.getattr_static(restate.service.Service, "handler")
    real_import = adapter.importlib.import_module

    def import_module(name):
        if name == "restate.object":
            raise RuntimeError("partial-failure")
        return real_import(name)

    monkeypatch.setattr(adapter.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="partial-failure"):
        RestateInstrumentor().activate()
    assert inspect.getattr_static(restate.service.Service, "handler") is original
    assert not adapter._PATCHED_TARGETS and not adapter._ENABLED


def test_mismatched_shared_configuration_is_rejected(runtime):
    with pytest.raises(ValueError, match="capture_content"):
        RestateInstrumentor(capture_content=False).activate()


def test_json_preserves_complete_native_history_vectors_and_arguments():
    value = {
        "messages": [{"role": "user", "content": "x" * 6000}] * 80,
        "vector": list(range(300)),
        "arguments": '{"exact":1}',
        "token": "private",
    }
    parsed = json.loads(json_string(value))
    assert (
        len(parsed["messages"]) == 80 and len(parsed["messages"][0]["content"]) == 6000
    )
    assert (
        parsed["vector"] == value["vector"]
        and parsed["arguments"] == value["arguments"]
    )
    assert parsed["token"] == "[REDACTED]"


def test_generic_child_initial_veto_latches_recording_ancestor(runtime):
    provider, exporter, _ = runtime
    item = restate.Service("GenericChild")

    @item.handler(name="run")
    async def fn(ctx, request):
        denied = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
        child = provider.get_tracer("test").start_span("generic", context=denied)
        child.end()
        return request

    asyncio.run(native_invoke(item, "run", {"private": "initial"}))
    span = exporter.get_finished_spans()[-1]
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes


def test_descriptor_restore_is_exact_and_existing_native_managers_remain(runtime):
    _, _, instrumentor = runtime
    patches = list(adapter._PATCHED_TARGETS)
    instrumentor.deactivate()
    for patch in patches:
        assert inspect.getattr_static(patch.owner, patch.name) is patch.original
    for _ in range(3):
        instrumentor.activate()
        instrumentor.deactivate()
    for patch in patches:
        assert inspect.getattr_static(patch.owner, patch.name) is patch.original


def test_detach_fault_still_restores_native_parent_context(runtime, monkeypatch):
    provider, _, _ = runtime
    item, _ = component()

    def fail(token):
        raise RuntimeError("detach-fault")

    monkeypatch.setattr(adapter.otel_context, "detach", fail)

    async def invoke():
        parent = provider.get_tracer("test").start_span("outer")
        token = context._RUNTIME_CONTEXT.attach(trace.set_span_in_context(parent))
        try:
            await native_invoke(item, "run", {"ok": 1})
            assert trace.get_current_span() is parent
        finally:
            context._RUNTIME_CONTEXT.detach(token)
            parent.end()

    asyncio.run(invoke())


def test_event_time_veto_removes_all_diagnostics_before_end(runtime, monkeypatch):
    _, exporter, _ = runtime
    from opentelemetry.sdk.trace import Span

    original = Span.add_event

    def event(self, *args, **kwargs):
        original(self, *args, **kwargs)
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))

    monkeypatch.setattr(Span, "add_event", event)
    item = restate.Service("EventVeto")

    @item.handler(name="run")
    async def fn(ctx, request):
        raise RuntimeError("diagnostic-private")

    with pytest.raises(RuntimeError):
        asyncio.run(native_invoke(item, "run", {"private": "body"}))
    span = exporter.get_finished_spans()[0]
    assert not span.events and span.status.description is None
    assert "diagnostic-private" not in json.dumps(dict(span.attributes))


def test_native_journal_codec_callbacks_remain_single_and_input_is_opaque(runtime):
    if "journal_codec" not in inspect.signature(invoke_handler).parameters:
        pytest.skip("JournalValueCodec is absent from this native SDK version")
    from restate.entry_codec import JournalValueCodec

    class Codec(JournalValueCodec):
        decodes = 0

        def encode(self, value):
            return value

        async def decode(self, value):
            self.decodes += 1
            return value

    _, exporter, _ = runtime
    codec = Codec()
    item, calls = component()
    result = asyncio.run(native_invoke(item, "run", {"ok": 1}, codec=codec))
    assert json.loads(result) == {"accepted": {"ok": 1}}
    assert codec.decodes == 1 and len(calls) == 1
    assert (
        SpanAttributes.TRACELOOP_ENTITY_INPUT
        not in exporter.get_finished_spans()[0].attributes
    )


def test_serializer_does_not_call_hostile_container_or_number_methods():
    from collections.abc import Mapping, Sequence

    from respan_instrumentation_restate._serialization import safe_text

    class HostileMapping(Mapping):
        def __getitem__(self, key):
            pytest.fail("unknown mapping inspected")

        def __iter__(self):
            pytest.fail("unknown mapping iterated")

        def __len__(self):
            raise AssertionError("unknown mapping length inspected")

    class HostileSequence(Sequence):
        def __getitem__(self, key):
            pytest.fail("unknown sequence inspected")

        def __len__(self):
            raise AssertionError("unknown sequence length inspected")

    class HostileInt(int):
        def __int__(self):
            pytest.fail("unknown integer converted")

    class HostileFloat(float):
        def __float__(self):
            pytest.fail("unknown float converted")

    for value in (HostileMapping(), HostileSequence(), HostileInt(1), HostileFloat(1)):
        assert json.loads(json_string(value)) == {"type": type(value).__name__}
        assert safe_text(value) == f"<{type(value).__name__}>"


def test_quoted_json_arguments_authorization_and_schema_redaction_are_idempotent():
    from respan_instrumentation_restate._serialization import safe_text

    encoded = '{"arguments":"{\\"api_key\\":\\"two word credential\\",\\"false\\":false,\\"zero\\":0}","authorization":"Basic cHJpdmF0ZQ=="}'
    cleaned = safe_text(encoded, truncate=False)
    parsed = json.loads(cleaned)
    arguments = json.loads(parsed["arguments"])
    assert arguments == {"api_key": "[REDACTED]", "false": False, "zero": 0}
    assert parsed["authorization"] == "[REDACTED]"
    assert safe_text(cleaned, truncate=False) == cleaned
    diagnostic = safe_text(
        'failed secret="two word credential" Bearer private-value Basic cHJpdmF0ZQ=='
    )
    assert (
        "two word" not in diagnostic
        and "credential" not in diagnostic
        and "private-value" not in diagnostic
        and "cHJpdmF0ZQ" not in diagnostic
    )
    schema = {
        "properties": {
            "api_key": {
                "type": "string",
                "description": "credential parameter",
                "default": "two word secret",
            },
            "enabled": {"type": "boolean", "default": False},
            "count": {"type": "integer", "default": 0},
        }
    }
    parsed_schema = json.loads(json_string(schema))
    assert parsed_schema["properties"]["api_key"]["type"] == "string"
    assert parsed_schema["properties"]["api_key"]["default"] == "[REDACTED]"
    assert parsed_schema["properties"]["enabled"]["default"] is False
    assert parsed_schema["properties"]["count"]["default"] == 0
