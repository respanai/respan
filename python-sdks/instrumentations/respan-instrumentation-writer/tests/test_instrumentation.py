"""Released Writer SDK/native OTel contract and lifecycle regression tests."""

import gc
import json
import weakref

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as S
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from pydantic import BaseModel
from respan_instrumentation_writer import WriterInstrumentor
from respan_instrumentation_writer import _instrumentation as adapter
from respan_instrumentation_writer._serialization import json_dumps, safe_text
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.utils.span_factory import _PROPAGATED_ATTRIBUTES
from writerai import APIError, AsyncStream, RateLimitError, Stream

from tests._native import NativeRuntime, invoke

IN = S.TRACELOOP_ENTITY_INPUT
OUT = S.TRACELOOP_ENTITY_OUTPUT
OPS = (
    "chat",
    "completion",
    "graph",
    "application",
    "vision",
    "translation",
    "web_search",
    "parse_pdf",
)


@pytest.fixture
def setup():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = WriterInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    yield instrumentor, provider, exporter
    instrumentor.deactivate()
    provider.shutdown()


def owned(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.instrumentation_scope.name == "writer"
    ]


@pytest.mark.parametrize("operation", OPS)
def test_native_sync_resource_full_payload_and_callbacks(setup, operation):
    _, _, exporter = setup
    runtime = NativeRuntime()
    with runtime.client() as client:
        result = invoke(client, operation)
        assert type(result).__module__.startswith("writerai.types.")
    spans = owned(exporter)
    assert len(spans) == runtime.response_callbacks == len(runtime.requests) == 1
    attrs = spans[0].attributes
    assert (
        json.loads(attrs[OUT])[0][
            "native_extra" if operation != "chat" else "feedback"
        ]["zero"]
        == 0
    )
    assert not any(
        k in attrs
        for k in (
            "status_code",
            "tools",
            "model",
            "respan.span.tool_calls",
            "traceloop.span.kind",
        )
    )
    if operation in ("chat", "completion"):
        assert attrs[S.LLM_USAGE_PROMPT_TOKENS] == 0
        assert attrs[S.LLM_USAGE_COMPLETION_TOKENS] == 3
        assert attrs[S.LLM_REQUEST_TYPE] == (
            "chat" if operation == "chat" else "completion"
        )
        assert attrs["gen_ai.response.model"] != attrs[S.LLM_REQUEST_MODEL]
    if operation in ("application", "parse_pdf"):
        assert (
            "application_id" if operation == "application" else "file_id"
        ) in json.loads(attrs[IN])
    if operation in ("web_search", "parse_pdf"):
        assert attrs["respan.entity.log_type"] == "tool"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPS)
async def test_native_async_resource_outcomes(setup, operation):
    _, _, exporter = setup
    runtime = NativeRuntime()
    async with runtime.async_client() as client:
        result = await invoke(client, operation)
        assert type(result).__module__.startswith("writerai.types.")
    assert (
        len(owned(exporter)) == runtime.response_callbacks == len(runtime.requests) == 1
    )
    assert OUT in owned(exporter)[0].attributes


@pytest.mark.parametrize("operation", ("chat", "completion", "graph", "application"))
def test_native_300_stream_identity_type_original_chunks_and_close(setup, operation):
    _, _, exporter = setup
    runtime = NativeRuntime()
    with runtime.client() as client:
        stream = invoke(client, operation, stream=True)
        assert type(stream) is Stream and runtime.bodies[0].reads == 0
        native = stream
        chunks = list(stream)
        assert stream is native and len(chunks) == 300
        stream.close()
    attrs = owned(exporter)[0].attributes
    assert len(json.loads(attrs[OUT])) == 300
    assert attrs["gen_ai.completion.0.content"] == "".join(
        str(i) + "," for i in range(300)
    )
    assert runtime.bodies[0].closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("chat", "completion", "graph", "application"))
async def test_native_async_300_stream_and_original_close(setup, operation):
    _, _, exporter = setup
    runtime = NativeRuntime()
    async with runtime.async_client() as client:
        stream = await invoke(client, operation, stream=True)
        assert type(stream) is AsyncStream and runtime.bodies[0].reads == 0
        chunks = [chunk async for chunk in stream]
        await stream.close()
        assert len(chunks) == 300
    assert len(json.loads(owned(exporter)[0].attributes[OUT])) == 300
    assert runtime.bodies[0].closes == 1


@pytest.mark.parametrize(
    "action", ("unread_close", "context", "early_close", "deactivate", "gc")
)
def test_native_pending_stream_protocol_cleanup_once(setup, action):
    instrumentor, _, exporter = setup
    runtime = NativeRuntime()
    with runtime.client() as client:
        stream = invoke(client, "chat", stream=True)
        if action == "context":
            with stream as native:
                assert native is stream
                next(stream)
        elif action == "early_close":
            next(stream)
            stream.close()
        elif action == "deactivate":
            instrumentor.deactivate()
            assert runtime.bodies[0].closes == 0
            assert len(list(stream)) == 300
            stream.close()
        elif action == "gc":
            ref = weakref.ref(stream)
            del stream
            gc.collect()
            assert ref() is None
        else:
            stream.close()
    assert len(owned(exporter)) == 1
    assert (
        OUT not in owned(exporter)[0].attributes
        if action in ("unread_close", "deactivate", "gc")
        else OUT in owned(exporter)[0].attributes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("unread_close", "early_close", "deactivate"))
async def test_native_async_pending_close_cleanup(setup, action):
    instrumentor, _, exporter = setup
    runtime = NativeRuntime()
    async with runtime.async_client() as client:
        stream = await invoke(client, "chat", stream=True)
        if action == "early_close":
            await stream.__anext__()
        if action == "deactivate":
            instrumentor.deactivate()
            assert runtime.bodies[0].closes == 0
            assert len([x async for x in stream]) == 300
        await stream.close()
    assert len(owned(exporter)) == 1


@pytest.mark.parametrize(
    "key",
    (
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ),
)
def test_native_suppression_precedes_all_telemetry_inspection(setup, monkeypatch, key):
    _, _, exporter = setup
    calls = []
    original = adapter.request_attributes
    monkeypatch.setattr(
        adapter,
        "request_attributes",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    runtime = NativeRuntime()
    token = context.attach(context.set_value(key, True))
    try:
        with runtime.client() as client:
            invoke(client, "chat")
    finally:
        context.detach(token)
    assert calls == [] and owned(exporter) == [] and len(runtime.requests) == 1


def test_actual_sampler_before_body_and_model_inspection(monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    inst = WriterInstrumentor(tracer_provider=provider)
    calls = []
    monkeypatch.setattr(
        adapter, "request_attributes", lambda *a, **k: calls.append(1) or {}
    )
    inst.activate()
    try:
        with NativeRuntime().client() as client:
            invoke(client, "chat")
    finally:
        inst.deactivate()
        provider.shutdown()
    assert calls == [] and exporter.get_finished_spans() == ()


@pytest.mark.parametrize("kind", ("canonical", "traceloop", "env", "capture"))
def test_initial_content_veto_cannot_widen(setup, monkeypatch, kind):
    inst, provider, exporter = setup
    runtime = NativeRuntime()
    if kind == "capture":
        inst.deactivate()
        inst = WriterInstrumentor(tracer_provider=provider, capture_content=False)
        inst.activate()
    key = (
        ENABLE_CONTENT_TRACING_KEY
        if kind == "canonical"
        else "override_enable_content_tracing"
    )
    token = (
        context.attach(context.set_value(key, False))
        if kind in ("canonical", "traceloop")
        else None
    )
    if kind == "env":
        monkeypatch.setenv("RESPAN_TRACE_CONTENT", "off")
    try:
        with runtime.client() as client:
            stream = invoke(client, "chat", stream=True)
            if token is not None:
                context.detach(token)
                token = None
            if kind == "env":
                monkeypatch.delenv("RESPAN_TRACE_CONTENT")
            list(stream)
            stream.close()
    finally:
        if token is not None:
            context.detach(token)
        if kind == "capture":
            inst.deactivate()
    attrs = owned(exporter)[0].attributes
    assert IN not in attrs and OUT not in attrs and owned(exporter)[0].events == ()


@pytest.mark.parametrize(
    "key", (ENABLE_CONTENT_TRACING_KEY, "traceloop.enable_content_tracing")
)
@pytest.mark.parametrize("phase", ("initial", "active", "finished"))
def test_native_application_ancestor_attribute_denial_is_irreversible(
    setup, key, phase
):
    _, provider, exporter = setup
    tracer = provider.get_tracer("application")
    runtime = NativeRuntime()
    parent = tracer.start_span(
        "parent", attributes={key: False} if phase == "initial" else {}
    )
    with trace.use_span(parent, end_on_exit=False):
        if phase == "initial":
            parent.set_attribute(key, True)
        elif phase == "active":
            parent.set_attribute(key, False)
            with tracer.start_as_current_span("generic-child"):
                pass
            parent.set_attribute(key, True)
        else:
            parent.set_attribute(key, False)
            parent.end()
        with runtime.client() as client:
            invoke(client, "chat")
    parent.end()
    attrs = owned(exporter)[0].attributes
    assert IN not in attrs and OUT not in attrs


def test_late_explicit_detach_without_consumption_removes_retained_body(setup):
    _, _, exporter = setup
    runtime = NativeRuntime()
    with runtime.client() as client:
        stream = invoke(client, "chat", stream=True)
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        context.detach(token)
        list(stream)
        stream.close()
    assert (
        IN not in owned(exporter)[0].attributes
        and OUT not in owned(exporter)[0].attributes
    )


def test_actual_supplied_and_ambient_context_remote_and_unknown_tuple_carriers(setup):
    _, provider, exporter = setup
    runtime = NativeRuntime()
    tracer = provider.get_tracer("app")
    parent = tracer.start_span("observed")
    sc = parent.get_span_context()
    for remote in (False, True):
        carrier = NonRecordingSpan(
            SpanContext(
                sc.trace_id + 1, sc.span_id, is_remote=remote, trace_flags=TraceFlags(1)
            )
        )
        token = context.attach(trace.set_span_in_context(carrier))
        try:
            with runtime.client() as client:
                invoke(client, "chat")
        finally:
            context.detach(token)
    assert (
        IN not in owned(exporter)[0].attributes and IN in owned(exporter)[1].attributes
    )
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    supplied = context.set_value(ENABLE_CONTENT_TRACING_KEY, True, context.Context())
    parent2 = tracer.start_span("supplied", context=supplied)
    context.detach(token)
    with trace.use_span(parent2), runtime.client() as client:
        invoke(client, "chat")
    parent.end()
    assert IN not in owned(exporter)[2].attributes


def test_native_two_pending_siblings_survive_exporter_suppression(setup):
    _, provider, exporter = setup
    runtime = NativeRuntime()
    with (
        provider.get_tracer("app").start_as_current_span("parent"),
        runtime.client() as client,
    ):
        a = invoke(client, "chat", stream=True)
        b = invoke(client, "chat", stream=True)
        list(a)
        a.close()
        list(b)
        b.close()
    assert all(IN in s.attributes and OUT in s.attributes for s in owned(exporter))


def test_native_generator_full_history_schema_vector_false_zero_and_parse(setup):
    _, _, exporter = setup
    runtime = NativeRuntime(vector=True)
    schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "default": "controlled-secret"},
                    "zero": {"type": "integer", "default": 0},
                    "flag": {"type": "boolean", "default": False},
                },
            },
        },
    }
    with runtime.client() as client:
        response = client.chat.chat(
            model="native-writer",
            messages=(
                {"role": "user", "content": "x" * 10000 + str(i)} for i in range(75)
            ),
            tools=[schema],
            temperature=0,
            max_tokens=0,
            extra_query={"answer": 0},
        )
        assert (
            len(runtime.requests[-1]["messages"]) == 75
            and len(response.native_vector) == 5001
        )

        class Output(BaseModel):
            value: int
            summary: str
            sentiment: str

        parsed = client.chat.parse(
            model="native-writer",
            messages=[{"role": "user", "content": "parse"}],
            response_format=Output,
        )
        assert parsed.choices[0].message.parsed.value == 0
    attrs = owned(exporter)[0].attributes
    request = json.loads(attrs[IN])
    output = json.loads(attrs[OUT])[0]
    assert (
        len(request["messages"]) == 75
        and len(request["messages"][-1]["content"]) > 10000
    )
    assert len(output["native_vector"]) == 5001 and output["feedback"]["false"] is False
    assert (
        request["tools"][0]["function"]["parameters"]["properties"]["api_key"][
            "default"
        ]
        == "[REDACTED]"
    )
    assert (
        request["tools"][0]["function"]["parameters"]["properties"]["zero"]["default"]
        == 0
    )
    assert (
        json.loads(owned(exporter)[1].attributes[IN])["response_format"]["type"]
        == "json_schema"
    )
    assert len(owned(exporter)) == 2


def test_native_empty_feedback_not_gated_on_projected_messages(setup):
    _, _, exporter = setup
    runtime = NativeRuntime(empty=True)
    with runtime.client() as client:
        response = invoke(client, "chat")
        assert response.choices == []
    assert (
        json.loads(owned(exporter)[0].attributes[OUT])[0]["feedback"]["blocked"] is True
    )
    assert "gen_ai.completion.0.content" not in owned(exporter)[0].attributes


def test_native_fragmented_and_quoted_credentials_complete_arguments(setup):
    _, _, exporter = setup
    runtime = NativeRuntime(chunks=2, secret_fragments=True)
    with runtime.client() as client:
        native = list(invoke(client, "chat", stream=True))
        assert "controlled-" in native[0].choices[0].delta.content
    attrs = owned(exporter)[0].attributes
    assert "controlled-" not in attrs[OUT] and 'secret"' not in attrs[OUT]
    arguments = json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["function"][
        "arguments"
    ]
    assert json.loads(arguments) == {
        "private_key": "[REDACTED]",
        "zero": 0,
        "flag": False,
    }


@pytest.mark.parametrize("kind", ("http", "sse"))
def test_native_errors_identity_actual_http_partial_output_no_fabrication(setup, kind):
    _, _, exporter = setup
    runtime = NativeRuntime(error=kind == "http", stream_error=kind == "sse")
    with (
        runtime.client() as client,
        pytest.raises(RateLimitError if kind == "http" else APIError) as error,
    ):
        if kind == "http":
            invoke(client, "chat")
        else:
            list(invoke(client, "chat", stream=True))
    span = owned(exporter)[0]
    assert (
        span.status.status_code.name == "ERROR"
        and span.attributes["error.type"] == type(error.value).__name__
    )
    if kind == "http":
        assert (
            span.attributes["http.response.status_code"] == 429
            and OUT not in span.attributes
        )
    else:
        assert len(json.loads(span.attributes[OUT])) == 2
    assert "status_code" not in span.attributes


def test_native_retry_and_callbacks_are_sdk_owned(setup):
    _, _, exporter = setup
    runtime = NativeRuntime(retry=True)
    with runtime.client(max_retries=1) as client:
        invoke(client, "chat")
    assert (
        len(runtime.requests) == runtime.response_callbacks == 2
        and len(owned(exporter)) == 1
    )


@pytest.mark.parametrize(
    "fault", ("request", "response", "attributes", "end", "detach")
)
def test_telemetry_faults_preserve_native_results_cleanup_and_ambient(
    setup, monkeypatch, fault
):
    _, _, _exporter = setup
    ambient = context.get_current()
    runtime = NativeRuntime()

    def fail(*args, **kwargs):
        context.attach(context.set_value("telemetry-fault", True))
        raise RuntimeError("telemetry fault")

    if fault == "request":
        monkeypatch.setattr(adapter, "request_attributes", fail)
    elif fault == "response":
        monkeypatch.setattr(adapter, "response_attributes", fail)
    elif fault == "attributes":
        monkeypatch.setattr(adapter._Call, "set_attributes", fail)
    elif fault == "end":
        monkeypatch.setattr("opentelemetry.sdk.trace._Span.end", fail)
    else:
        monkeypatch.setattr(context, "detach", fail)
    with runtime.client() as client:
        response = invoke(client, "chat")
        assert response.choices[0].message.content == "native response"
    assert context.get_current() is ambient and len(runtime.requests) == 1


def test_foreign_start_fault_ends_scrubs_owned_partial_span(monkeypatch):
    provider = TracerProvider()
    seen = []
    ambient = context.get_current()

    class Foreign(SpanProcessor):
        def on_start(self, span, parent_context=None):
            seen.append(span)
            span.add_event("private", {"private": "controlled"})
            context.attach(context.set_value("fault", True))
            raise RuntimeError("foreign telemetry")

    foreign = Foreign()
    provider.add_span_processor(foreign)
    inst = WriterInstrumentor(tracer_provider=provider)
    inst.activate()
    try:
        with NativeRuntime().client() as client:
            invoke(client, "chat")
        assert (
            seen[0].end_time is not None
            and not seen[0].events
            and context.get_current() is ambient
        )
    finally:
        inst.deactivate()
        provider.shutdown()


def test_actual_unknown_payload_and_propagated_metaclass_hooks_not_executed(setup):
    _, _, exporter = setup
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("hash")
            return type.__hash__(cls)

        def __eq__(cls, other):
            calls.append("eq")
            return cls is other

    class Unknown(metaclass=Meta):
        def __getattribute__(self, key):
            calls.append("get")
            raise AssertionError(key)

        def __bool__(self):
            calls.append("bool")
            raise AssertionError

    value = Unknown()
    assert json.loads(json_dumps(value)) == {"type": "Unknown"}
    token = _PROPAGATED_ATTRIBUTES.set({"customer_identifier": value})
    try:
        with NativeRuntime().client() as client:
            invoke(client, "chat")
    finally:
        _PROPAGATED_ATTRIBUTES.reset(token)
    assert calls == [] and len(owned(exporter)) == 1


def test_native_shared_conflicts_foreign_and_mutate_then_raise_rollback(
    setup, monkeypatch
):
    inst, provider, _ = setup
    other = WriterInstrumentor(tracer_provider=provider)
    other.activate()
    with pytest.raises(ValueError):
        WriterInstrumentor(capture_content=False, tracer_provider=provider).activate()
    from writerai.resources.chat import ChatResource

    original = ChatResource.chat

    def foreign(*args, **kwargs):
        return original(*args, **kwargs)

    ChatResource.chat = foreign
    inst.deactivate()
    other.deactivate()
    assert ChatResource.chat is foreign
    ChatResource.chat = original.__wrapped__
    import builtins

    def fail(owner, name, value):
        builtins.setattr(owner, name, value)
        if owner is ChatResource and name == "chat":
            raise RuntimeError("mutated setter")

    monkeypatch.setattr(adapter, "setattr", fail, raising=False)
    fresh = WriterInstrumentor(tracer_provider=provider)
    with pytest.raises(RuntimeError):
        fresh.activate()
    assert ChatResource.chat is original.__wrapped__ and not adapter._PATCHES


def test_serializer_quoted_auth_json_schema_and_urls_idempotent():
    for raw in [
        'Authorization: Bearer "controlled-secret"',
        "Basic 'controlled secret'",
        'private_key="controlled secret"',
        "https://example.invalid/path?api%5Fkey=controlled-secret",
    ]:
        safe = safe_text(raw)
        assert (
            "controlled-secret" not in safe
            and "controlled secret" not in safe
            and safe_text(safe) == safe
        )
    raw = '{"private_key":"controlled secret","zero":0,"flag":false}'
    safe = safe_text(raw)
    assert (
        json.loads(safe) == {"private_key": "[REDACTED]", "zero": 0, "flag": False}
        and safe_text(safe) == safe
    )


def test_native_plural_cookie_credentials_are_redacted(setup):
    _, _, exporter = setup
    with NativeRuntime().client() as client:
        invoke(client, "chat", extra_body={"cookies": "controlled-cookie-secret"})
    assert json.loads(owned(exporter)[0].attributes[IN])["cookies"] == "[REDACTED]"
