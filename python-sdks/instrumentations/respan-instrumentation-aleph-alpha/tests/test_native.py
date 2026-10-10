"""Contract evidence uses native request/response structs and real transport."""

import gc
import inspect
import json
from enum import Enum

import pytest
from aleph_alpha_client import (
    AsyncClient,
    BatchSemanticEmbeddingRequest,
    ChatRequest,
    Client,
    CompletionRequest,
    EmbeddingRequest,
    EmbeddingV2Request,
    EvaluationRequest,
    ExplanationRequest,
    Message,
    Prompt,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
)
from aleph_alpha_client.chat import Role, StreamOptions
from aleph_alpha_client.embedding import InstructableEmbeddingRequest
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes as AI
from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode, TraceFlags
from respan_instrumentation_aleph_alpha import AlephAlphaInstrumentor
from respan_instrumentation_aleph_alpha._serialization import json_value, redact_text
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

MODEL = "controlled-aleph-model"


def req(method, text="native controlled input"):
    prompt = Prompt.from_text(text)
    if method.startswith("chat"):
        return ChatRequest(
            model=MODEL,
            messages=[Message(Role.User, text)],
            stream_options=StreamOptions(include_usage=True),
        )
    if method.startswith("complete"):
        return CompletionRequest(
            prompt, maximum_tokens=0, temperature=0.0, stop_sequences=[]
        )
    if method == "embed":
        return EmbeddingRequest(prompt, layers=[-1], pooling=["mean"])
    if method == "embeddings":
        return EmbeddingV2Request([text, "second"])
    if method == "semantic_embed":
        return SemanticEmbeddingRequest(
            prompt, SemanticRepresentation.Query, normalize=False
        )
    if method == "batch_semantic_embed":
        return BatchSemanticEmbeddingRequest(
            [prompt, prompt], SemanticRepresentation.Document
        )
    if method == "instructable_embed":
        return InstructableEmbeddingRequest(prompt, "native instruction")
    if method == "evaluate":
        return EvaluationRequest(prompt, completion_expected="native completion")
    return ExplanationRequest(prompt, target="native target")


def call(host, method="complete", request=None):
    client = Client(token="controlled-fixture-token", host=host, total_retries=0)
    try:
        return getattr(client, method)(
            request if request is not None else req(method), MODEL
        )
    finally:
        client.session.close()


def attrs(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0], spans[0].attributes


def no_body(exporter):
    span, data = attrs(exporter)
    assert AI.TRACELOOP_ENTITY_INPUT not in data
    assert AI.TRACELOOP_ENTITY_OUTPUT not in data
    assert not any(
        key.startswith(
            (
                AI.LLM_PROMPTS + ".",
                AI.LLM_COMPLETIONS + ".",
                "respan.metadata.aleph_alpha",
            )
        )
        or key == "error.message"
        for key in data
    )
    assert "PRIVATE" not in json.dumps(dict(data))
    assert span.status.description is None
    assert data[AI.LLM_SYSTEM] == "alephalpha"
    assert data[AI.LLM_REQUEST_TYPE] == "chat"


@pytest.mark.parametrize(
    "method",
    [
        "chat",
        "complete",
        "embed",
        "embeddings",
        "semantic_embed",
        "batch_semantic_embed",
        "instructable_embed",
        "evaluate",
        "explain",
    ],
)
def test_native_sync_contract(host, runtime, method):
    _, exporter, _ = runtime
    result = call(host, method)
    assert type(result).__module__.startswith("aleph_alpha_client.")
    span, data = attrs(exporter)
    assert span.status.status_code == StatusCode.OK
    assert data[RESPAN_LOG_TYPE] == (
        "chat"
        if method == "chat"
        else "text"
        if method == "complete"
        else "embedding"
        if "embed" in method
        else "task"
    )
    assert AI.TRACELOOP_ENTITY_INPUT in data and AI.TRACELOOP_ENTITY_OUTPUT in data
    output = json.loads(data[AI.TRACELOOP_ENTITY_OUTPUT])
    if method in ("semantic_embed", "instructable_embed"):
        assert output == result.embedding and len(output) == 5001
        assert data[AI.LLM_USAGE_PROMPT_TOKENS] == 0
        assert AI.LLM_USAGE_TOTAL_TOKENS not in data
    elif method == "batch_semantic_embed":
        assert output == result.embeddings and len(output[0]) == 5001
    elif method == "embed":
        assert len(next(iter(output.values()))) == 5001
    elif method == "embeddings":
        assert len(output) == 2 and len(output[0]) == 5001
    elif method == "complete":
        assert data[AI.LLM_REQUEST_MAX_TOKENS] == 0
        assert AI.LLM_USAGE_TOTAL_TOKENS not in data
    if method in ("chat", "evaluate", "explain"):
        assert AI.LLM_USAGE_COMPLETION_TOKENS not in data
    assert "status_code" not in data and "http.response.status_code" not in data
    assert not set(data) & {
        "model",
        "tools",
        "tool_calls",
        "prompt_tokens",
        "total_tokens",
        "respan.span.tools",
        AI.TRACELOOP_SPAN_KIND,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        "chat",
        "complete",
        "embed",
        "semantic_embed",
        "batch_semantic_embed",
        "instructable_embed",
        "evaluate",
        "explain",
    ],
)
async def test_native_async_contract(host, runtime, method):
    _, exporter, _ = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        result = await getattr(client, method)(req(method), MODEL)
        assert type(result).__module__.startswith("aleph_alpha_client.")
    span, data = attrs(exporter)
    assert span.status.status_code == StatusCode.OK
    assert AI.TRACELOOP_ENTITY_OUTPUT in data
    if method == "batch_semantic_embed":
        assert len(json.loads(data[AI.TRACELOOP_ENTITY_INPUT])) == 2


@pytest.mark.parametrize(
    "key",
    [
        "respan_enable_content_tracing",
        "trace_content",
        "override_enable_content_tracing",
    ],
)
def test_actual_context_flags(host, runtime, key):
    _, exporter, _ = runtime
    token = context.attach(context.set_value(key, False))
    try:
        call(host, request=req("complete", "PRIVATE"))
    finally:
        context.detach(token)
    no_body(exporter)


@pytest.mark.parametrize("env", ["RESPAN_TRACE_CONTENT", "TRACELOOP_TRACE_CONTENT"])
@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_environment_capture(host, runtime, monkeypatch, env, value):
    _, exporter, _ = runtime
    monkeypatch.setenv(env, value)
    call(host, request=req("complete", "PRIVATE"))
    no_body(exporter)


@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY]
)
def test_suppression_before_capture(host, runtime, key):
    _, exporter, _ = runtime
    token = context.attach(context.set_value(key, True))
    try:
        call(host)
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()


def test_actual_sampler_off(host):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AlephAlphaInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    try:
        call(host)
    finally:
        instrumentor.deactivate()
        provider.shutdown()
    assert not exporter.get_finished_spans()


def test_empty_candidate_payload(host, runtime):
    _, exporter, _ = runtime
    result = call(host, request=req("complete", "empty-candidates"))
    assert result.completions == []
    _, data = attrs(exporter)
    assert json.loads(data[AI.TRACELOOP_ENTITY_OUTPUT])["completions"] == []
    assert f"{AI.LLM_COMPLETIONS}.0.content" not in data


def test_native_error_identity_and_source_status(host, runtime):
    _, exporter, instrumentor = runtime
    instrumentor.deactivate()
    with pytest.raises(ValueError) as baseline:
        call(host, request=req("complete", "controlled-error"))
    instrumentor.activate()
    with pytest.raises(ValueError) as captured:
        call(host, request=req("complete", "controlled-error"))
    assert baseline.value.args == captured.value.args
    span, data = attrs(exporter)
    assert data["http.response.status_code"] == 400
    assert span.status.status_code == StatusCode.ERROR
    assert data["error.type"] == "ValueError"
    assert "PRIVATE-ERROR" not in data["error.message"]
    assert AI.TRACELOOP_ENTITY_OUTPUT not in data


def test_full_history_tools_schema(host, runtime):
    _, exporter, _ = runtime
    schema = {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "default": "PRIVATE",
                "example": "PRIVATE",
                "description": "name retained",
            },
            "count": {"type": "integer", "default": 0},
        },
        "required": ["api_key"],
    }
    request = ChatRequest(
        model=MODEL,
        messages=[Message(Role.User, f"message {i}") for i in range(75)],
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup_policy", "parameters": schema},
            }
        ],
        parallel_tool_calls=False,
    )
    result = call(host, "chat", request)
    _, data = attrs(exporter)
    assert len(json.loads(data[AI.TRACELOOP_ENTITY_INPUT])) == 75
    assert data[AI.LLM_SYSTEM] == "alephalpha"
    assert data[AI.LLM_REQUEST_TYPE] == "chat"
    functions = json.loads(data[AI.LLM_REQUEST_FUNCTIONS])
    assert (
        functions[0]["function"]["parameters"]["properties"]["api_key"]["default"]
        == "[REDACTED]"
    )
    assert (
        json.loads(data[f"{AI.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
        == result.message.tool_calls[0].id
    )
    metadata = json.loads(data["respan.metadata.aleph_alpha.request"])
    assert metadata["wire"][0]["parallel_tool_calls"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["complete_with_streaming", "chat_with_streaming"])
async def test_native_stream_exhaustion(host, runtime, method):
    _, exporter, _ = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        iterator = getattr(client, method)(req(method), MODEL)
        native = iterator.native
        assert inspect.isasyncgen(native)
        items = [item async for item in iterator]
        assert all(
            type(item).__module__.startswith("aleph_alpha_client.")
            or isinstance(item, Enum)
            for item in items
        )
    span, data = attrs(exporter)
    assert span.status.status_code == StatusCode.OK
    assert len(json.loads(data[AI.TRACELOOP_ENTITY_OUTPUT])) == len(items)
    assert data[f"{AI.LLM_COMPLETIONS}.0.content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("consume", [False, True])
async def test_native_stream_close(host, runtime, consume):
    _, exporter, _ = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        iterator = client.chat_with_streaming(req("chat"), MODEL)
        native = iterator.native
        if consume:
            await iterator.asend(None)
        await iterator.aclose()
        assert native.ag_frame is None
    span, data = attrs(exporter)
    assert span.status.status_code == StatusCode.UNSET
    assert (AI.TRACELOOP_ENTITY_OUTPUT in data) is consume


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat_with_streaming", "complete_with_streaming"])
async def test_split_quoted_stream_credentials(host, runtime, method):
    _, exporter, _ = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        items = [
            item
            async for item in getattr(client, method)(
                req(method, "split-secret"), MODEL
            )
        ]
    assert "PRIVATE-FRAGMENT" in str(items)
    _, data = attrs(exporter)
    assert "PRIVATE-FRAGMENT" not in json.dumps(dict(data))
    assert "[REDACTED]" in data[f"{AI.LLM_COMPLETIONS}.0.content"]


@pytest.mark.asyncio
async def test_pre_detach_veto_and_no_widening(host, runtime):
    _, exporter, _ = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        iterator = client.chat_with_streaming(req("chat", "PRIVATE"), MODEL)
        await anext(iterator)
        token = context.attach(
            context.set_value("respan_enable_content_tracing", False)
        )
        context.detach(token)
        assert iterator.state.policy.allowed is False
        await iterator.aclose()
    no_body(exporter)


@pytest.mark.asyncio
async def test_finished_parent_veto(host, runtime):
    provider, exporter, _ = runtime
    tracer = provider.get_tracer("native-test")
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        parent = tracer.start_span("parent")
        with trace.use_span(parent, end_on_exit=False):
            iterator = client.chat_with_streaming(req("chat", "PRIVATE"), MODEL)
            await anext(iterator)
        parent.set_attribute("respan_enable_content_tracing", False)
        parent.end()
        exporter.clear()
        await iterator.aclose()
    no_body(exporter)


@pytest.mark.parametrize("remote", [False, True])
def test_unknown_parent_carrier(host, runtime, remote):
    _, exporter, _ = runtime
    parent = NonRecordingSpan(
        SpanContext(123, 456, is_remote=remote, trace_flags=TraceFlags(1))
    )
    with trace.use_span(parent, end_on_exit=False):
        call(host)
    if remote:
        assert AI.TRACELOOP_ENTITY_INPUT in attrs(exporter)[1]
    else:
        no_body(exporter)


@pytest.mark.asyncio
async def test_two_pending_native_siblings(host, runtime):
    provider, exporter, _ = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        with provider.get_tracer("native-test").start_as_current_span("parent"):
            a = client.chat_with_streaming(req("chat"), MODEL)
            b = client.chat_with_streaming(req("chat"), MODEL)
            await anext(a)
            await anext(b)
            await a.aclose()
            await b.aclose()
    children = [
        span
        for span in exporter.get_finished_spans()
        if span.name.startswith("alephalpha.")
    ]
    assert len(children) == 2
    assert all(AI.TRACELOOP_ENTITY_OUTPUT in s.attributes for s in children)
    assert children[0].parent == children[1].parent


def test_shared_owner_conflict_foreign_restore(runtime):
    provider, _, first = runtime
    original = Client.complete
    second = AlephAlphaInstrumentor(tracer_provider=provider)
    second.activate()
    first.deactivate()
    assert Client.complete is original
    conflict = AlephAlphaInstrumentor(tracer_provider=provider, capture_content=False)
    with pytest.raises(RuntimeError):
        conflict.activate()

    def foreign(*args, **kwargs):
        return original(*args, **kwargs)

    Client.complete = foreign
    second.deactivate()
    assert Client.complete is foreign
    Client.complete = original.__wrapped__


def test_runtime_detach_descriptor_presence(runtime):
    _, _, instrumentor = runtime
    instrumentor.deactivate()
    before = dict(vars(context._RUNTIME_CONTEXT))
    instrumentor.activate()
    instrumentor.deactivate()
    assert vars(context._RUNTIME_CONTEXT) == before


@pytest.mark.asyncio
async def test_deactivate_unconsumed_and_gc(host, runtime):
    _, exporter, instrumentor = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        iterator = client.chat_with_streaming(req("chat", "PRIVATE"), MODEL)
        instrumentor.deactivate()
        no_body(exporter)
        result = await anext(iterator)
        assert result is not None
        await iterator.aclose()
        exporter.clear()
        instrumentor.activate()
        iterator = client.chat_with_streaming(req("chat"), MODEL)
        del iterator
        gc.collect()
        assert len(exporter.get_finished_spans()) == 1


@pytest.mark.parametrize(
    "fault", ["mutate_raise", "detach_false", "end_raise", "startup_raise"]
)
def test_observer_faults_preserve_native(host, runtime, monkeypatch, fault):
    provider, exporter, _ = runtime
    original_set = Span.set_attribute
    original_end = Span.end
    if fault == "startup_raise":
        monkeypatch.setattr(
            provider,
            "get_tracer",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("telemetry")),
        )
    elif fault == "end_raise":

        def end(self, *a, **kw):
            if self.name.startswith("alephalpha."):
                raise RuntimeError("telemetry")
            return original_end(self, *a, **kw)

        monkeypatch.setattr(Span, "end", end)
    else:

        def setter(self, key, value):
            if key == AI.TRACELOOP_ENTITY_INPUT:
                if fault == "detach_false":
                    token = context.attach(
                        context.set_value("respan_enable_content_tracing", False)
                    )
                    context.detach(token)
                original_set(self, key, value)
                if fault == "mutate_raise":
                    raise RuntimeError("telemetry")
                return None
            return original_set(self, key, value)

        monkeypatch.setattr(Span, "set_attribute", setter)
    result = call(host, request=req("complete", "PRIVATE"))
    assert result.completions[0].completion.endswith("PRIVATE")
    if fault in ("mutate_raise", "detach_false"):
        no_body(exporter)
    from respan_instrumentation_aleph_alpha import _instrumentation as module

    assert not list(module._MANAGER.observer.states)


def test_native_subclass_storage_unknown_hooks(host, runtime):
    _, exporter, _ = runtime

    class NativeRequest(CompletionRequest):
        @property
        def __dict__(self):
            raise AssertionError("unknown getter")

    result = call(
        host, request=NativeRequest(Prompt.from_text("native input"), maximum_tokens=2)
    )
    assert result.completions
    assert AI.TRACELOOP_ENTITY_INPUT in attrs(exporter)[1]

    class Unknown:
        def to_json(self):
            raise AssertionError("conversion hook")

        def __str__(self):
            raise AssertionError("string hook")

        def __iter__(self):
            raise AssertionError("iterator hook")

    assert json_value({"unknown": Unknown()}) == {"unknown": None}


@pytest.mark.parametrize(
    "secret",
    [
        'Authorization: Bearer "PRIVATE"',
        "authorization = 'Basic PRIVATE'",
        "token=PRIVATE",
        "https://name:PRIVATE@host/path?api_key=PRIVATE",
        '{"api_key":"PRIVATE\\"inside","zero":0,"empty":[],"false":false}',
    ],
)
def test_valid_idempotent_redaction(secret):
    value = redact_text(secret)
    assert "PRIVATE" not in value
    assert redact_text(value) == value
    if secret.startswith("{"):
        assert json.loads(value)["zero"] == 0


def test_schema_names_and_scalar_properties():
    assert (
        json_value({"properties": {"api_key": "PRIVATE"}})["properties"]["api_key"]
        == "[REDACTED]"
    )
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string", "example": "PRIVATE", "default": "PRIVATE"}
        },
        "required": ["api_key"],
    }
    converted = json_value(schema)
    assert converted["required"] == ["api_key"]
    assert converted["properties"]["api_key"]["example"] == "[REDACTED]"


@pytest.mark.parametrize(
    "value", [r"Authorization: Bearer \"PRIVATE SPACE\"", r"Basic \"PRIVATE TWO\""]
)
def test_literal_escaped_credentials(value):
    redacted = redact_text(value)
    assert "PRIVATE" not in redacted
    assert redact_text(redacted) == redacted


def test_native_numeric_configuration_keys(host, runtime):
    _, exporter, _ = runtime
    request = req("complete")
    from dataclasses import replace

    result = call(host, request=replace(request, logit_bias={0: 0.0, 7: -1.0}))
    assert result.completions
    data = json.loads(attrs(exporter)[1]["respan.metadata.aleph_alpha.request"])
    assert data["native"]["logit_bias"] == {"0": 0.0, "7": -1.0}


@pytest.mark.parametrize("ambient,supplied", [(False, True), (True, False)])
def test_ambient_and_supplied_content(host, ambient, supplied):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AlephAlphaInstrumentor(
        tracer_provider=provider,
        context=context.set_value("respan_enable_content_tracing", supplied),
    )
    instrumentor.activate()
    token = context.attach(context.set_value("respan_enable_content_tracing", ambient))
    try:
        call(host, request=req("complete", "PRIVATE"))
    finally:
        context.detach(token)
        instrumentor.deactivate()
        provider.shutdown()
    no_body(exporter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag,value",
    [
        ("respan_enable_content_tracing", False),
        (_SUPPRESS_INSTRUMENTATION_KEY, True),
        (SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True),
    ],
)
async def test_held_consumer_veto_before_scope_attach(host, runtime, flag, value):
    _, exporter, _ = runtime
    async with AsyncClient(
        token="controlled-fixture-token", host=host, total_retries=0
    ) as client:
        iterator = client.chat_with_streaming(req("chat", "PRIVATE-HELD"), MODEL)
        carrier = context.attach(context.set_value(flag, value))
        try:
            native = [item async for item in iterator]
            assert native
        finally:
            context.detach(carrier)
    no_body(exporter)


def test_owned_span_events_scrubbed(host):
    from opentelemetry.sdk.trace import SpanProcessor

    class PrivateEvent(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.record_exception(ValueError("PRIVATE-EVENT"))
            span.set_attribute("error.message", "PRIVATE-EVENT")

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(PrivateEvent())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AlephAlphaInstrumentor(
        tracer_provider=provider, capture_content=False
    )
    instrumentor.activate()
    try:
        result = call(host, request=req("complete", "PRIVATE-BODY"))
    finally:
        instrumentor.deactivate()
        provider.shutdown()
    assert result.completions
    no_body(exporter)
    assert not exporter.get_finished_spans()[0].events


def test_native_detach_fault_restores_ambient(host, runtime, monkeypatch):
    provider, exporter, _ = runtime
    parent = provider.get_tracer("native-test").start_span("parent")
    with trace.use_span(parent, end_on_exit=False):
        before = context.get_current()

        def failed(token):
            raise RuntimeError("telemetry detach")

        with monkeypatch.context() as patch:
            patch.setattr(context, "detach", failed)
            result = call(host)
        assert context.get_current() is before
        assert trace.get_current_span() is parent
        assert result.completions
    parent.end()
    assert len(exporter.get_finished_spans()) == 2


def test_native_structured_output_schema_is_observed_after_sdk_conversion(
    host, runtime
):
    from pydantic import BaseModel, Field

    class Structured(BaseModel):
        api_key: str = Field(default="PRIVATE-STRUCTURED")
        count: int = 0

    _, exporter, _ = runtime
    request = ChatRequest(
        model=MODEL,
        messages=[Message(Role.User, "native structured")],
        response_format=Structured,
    )
    result = call(host, "chat", request)
    assert result.message.content
    metadata = json.loads(attrs(exporter)[1]["respan.metadata.aleph_alpha.request"])
    schema = metadata["wire"][0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["api_key"]["default"] == "[REDACTED]"
    assert schema["properties"]["count"]["default"] == 0


def test_native_readable_end_snapshot_veto(host, monkeypatch):
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import Status

    class PrivateEvent(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.record_exception(ValueError("PRIVATE-END-EVENT"))
            span.set_attribute("error.message", "PRIVATE-END-EVENT")
            span.set_status(Status(StatusCode.ERROR, "PRIVATE-END-DESCRIPTION"))

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(PrivateEvent())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AlephAlphaInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    original = Span.end

    def ending(span, *args, **kwargs):
        token = context.attach(
            context.set_value("respan_enable_content_tracing", False)
        )
        try:
            return original(span, *args, **kwargs)
        finally:
            context.detach(token)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Span, "end", ending)
            result = call(host, request=req("complete", "PRIVATE-END-BODY"))
    finally:
        instrumentor.deactivate()
        provider.shutdown()
    assert result.completions
    no_body(exporter)
    span = exporter.get_finished_spans()[0]
    assert not span.events
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description is None
    assert span.attributes[AI.LLM_SYSTEM] == "alephalpha"
    assert span.attributes[AI.LLM_REQUEST_TYPE] == "chat"


def test_native_request_extra_unknown_metaclass_never_observed(host, runtime):
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append("eq")
            raise AssertionError("observer equality")

        __hash__ = type.__hash__

        @property
        def __mro__(cls):
            calls.append("mro")
            raise AssertionError("observer mro getter")

        @property
        def __dict__(cls):
            calls.append("dict")
            raise AssertionError("observer class getter")

    class Unknown(metaclass=Meta):
        pass

    request = req("complete")
    object.__setattr__(request, "observer_extra", Unknown())
    result = call(host, request=request)
    assert result.completions and not calls
    _, exporter, _ = runtime
    assert AI.TRACELOOP_ENTITY_INPUT in attrs(exporter)[1]


def test_native_attach_mutation_restores_caller(host, runtime, monkeypatch):
    original = context.attach
    before = context.get_current()

    def mutated(carrier):
        original(carrier)
        raise RuntimeError("telemetry attach mutation")

    with monkeypatch.context() as patch:
        patch.setattr(context, "attach", mutated)
        result = call(host)
    assert result.completions
    assert context.get_current() is before
    _, exporter, _ = runtime
    no_body(exporter)
