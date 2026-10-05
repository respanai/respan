"""Released Agent Framework regressions for privacy, streams and lifecycle."""

import asyncio
import json

import pytest
from agent_framework import (
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
    observability,
    tool,
)
from agent_framework.observability import ChatTelemetryLayer
from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_microsoft_agent_framework import (
    MicrosoftAgentFrameworkInstrumentor,
    _instrumentation,
    _policy,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer


class FixtureClient(ChatTelemetryLayer, BaseChatClient):
    OTEL_PROVIDER_NAME = "fixture"

    def __init__(self, mode="complete"):
        self.model = "fixture-maf-model"
        self.mode = mode
        self.closed = False
        super().__init__()

    def service_url(self):
        return "https://fixture.invalid/v1"

    def _inner_get_response(self, *, messages, options, stream=False, **kwargs):
        async def response():
            return ChatResponse(
                messages=Message("assistant", [Content.from_text("fixture answer")]),
                model=self.model,
                usage_details={
                    "input_token_count": 7,
                    "output_token_count": 3,
                    "cache_read_input_token_count": 2,
                    "reasoning_output_token_count": 1,
                },
            )

        async def updates():
            try:
                yield ChatResponseUpdate(
                    role="assistant",
                    contents=[Content.from_text("fixture ")],
                    model=self.model,
                )
                if self.mode == "error":
                    raise RuntimeError("controlled stream failure")
                if self.mode == "cancel":
                    await asyncio.Event().wait()
                yield ChatResponseUpdate(
                    role="assistant",
                    contents=[
                        Content.from_text("answer"),
                        Content.from_usage(
                            {"input_token_count": 7, "output_token_count": 3}
                        ),
                    ],
                    model=self.model,
                )
            finally:
                self.closed = True

        return (
            ResponseStream(updates(), finalizer=ChatResponse.from_updates)
            if stream
            else response()
        )


@pytest.fixture
def telemetry(monkeypatch):
    RespanTracer.reset_instance()
    settings = observability.OBSERVABILITY_SETTINGS
    monkeypatch.setattr(settings, "_user_disabled", False, raising=False)
    monkeypatch.setattr(settings, "enable_instrumentation", True)
    monkeypatch.setattr(settings, "enable_sensitive_data", False)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        observability,
        "get_tracer",
        lambda *a, **kw: provider.get_tracer("agent_framework"),
    )
    plugin = MicrosoftAgentFrameworkInstrumentor()
    plugin.activate()
    yield provider, exporter, plugin
    plugin.deactivate()
    while _policy._users:
        _policy.release()
    provider.shutdown()
    RespanTracer.reset_instance()


@pytest.mark.parametrize("mode", ["complete", "close", "error", "cancel"])
def test_stream_lifecycle_preserves_native_api_and_parent(telemetry, mode):
    if mode == "close" and not hasattr(ResponseStream, "close"):
        pytest.skip("Native 1.8.1 has no ResponseStream.close API")
    if mode == "cancel" and not hasattr(ResponseStream, "close"):
        pytest.xfail("Bare 1.8.1 does not finalize a cancelled native chat span")
    provider, exporter, _ = telemetry
    client = FixtureClient(mode)

    async def run():
        with provider.get_tracer("application").start_as_current_span("root") as root:
            stream = client.get_response(
                [Message("user", [Content.from_text("fixture prompt")])], stream=True
            )
            assert isinstance(stream, ResponseStream)
            await anext(stream)
            if mode == "cancel":
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            elif mode == "error":
                with pytest.raises(RuntimeError, match="controlled stream failure"):
                    await anext(stream)
            elif mode == "close":
                await stream.close()
            else:
                remaining = [item async for item in stream]
                assert remaining
                assert (await stream.get_final_response()).text == "fixture answer"
            assert trace.get_current_span() is root

    asyncio.run(run())
    assert client.closed
    chats = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get(RESPAN_LOG_TYPE) == "chat"
    ]
    assert len(chats) == 1
    root = next(s for s in exporter.get_finished_spans() if s.name == "root")
    assert chats[0].parent.span_id == root.context.span_id
    if mode in {"error", "cancel"}:
        assert chats[0].status.status_code is trace.StatusCode.ERROR


@pytest.mark.parametrize("policy", ["env", "runtime", "config"])
def test_content_opt_out_real_chat_and_tool(telemetry, monkeypatch, policy):
    _, exporter, plugin = telemetry
    if policy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if policy == "config":
        plugin.deactivate()
        plugin = MicrosoftAgentFrameworkInstrumentor(capture_content=False)
        plugin.activate()
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if policy == "runtime"
        else None
    )

    @tool
    def echo(value: str) -> str:
        return value

    async def run():
        await FixtureClient().get_response(
            [Message("user", [Content.from_text("private fixture sentinel")])]
        )
        await echo.invoke(arguments={"value": "private fixture sentinel"})

    try:
        asyncio.run(run())
    finally:
        if token is not None:
            context.detach(token)
        if policy == "config":
            plugin.deactivate()
    assert len(exporter.get_finished_spans()) == 2
    for span in exporter.get_finished_spans():
        assert "private fixture sentinel" not in str(span.attributes)
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


def test_stream_privacy_snapshot_survives_context_exit(telemetry):
    _, exporter, _ = telemetry

    async def run():
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            stream = FixtureClient().get_response(
                [Message("user", [Content.from_text("private fixture sentinel")])],
                stream=True,
            )
        finally:
            context.detach(token)
        async for _ in stream:
            pass

    asyncio.run(run())
    chat = exporter.get_finished_spans()[0]
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in chat.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in chat.attributes
    assert not any(k.startswith("gen_ai.completion.") for k in chat.attributes)


def test_native_suppression(telemetry):
    _, exporter, _ = telemetry

    @tool
    def echo(value: str) -> str:
        return value

    async def run():
        with suppress_instrumentation():
            await FixtureClient().get_response(
                [Message("user", [Content.from_text("suppressed fixture")])]
            )
            async for _ in FixtureClient().get_response(
                [Message("user", [Content.from_text("suppressed fixture")])],
                stream=True,
            ):
                pass
            await echo.invoke(arguments={"value": "suppressed fixture"})

    asyncio.run(run())
    assert exporter.get_finished_spans() == ()


def test_settings_shared_and_restored(telemetry):
    _, exporter, first = telemetry
    settings = observability.OBSERVABILITY_SETTINGS
    second = MicrosoftAgentFrameworkInstrumentor()
    second.activate()
    first.deactivate()
    assert settings.enable_sensitive_data is True
    asyncio.run(
        FixtureClient().get_response([Message("user", [Content.from_text("fixture")])])
    )
    assert len(exporter.get_finished_spans()) == 1
    second.deactivate()
    assert settings.enable_sensitive_data is False


def test_activation_failure_rolls_back_native_settings(telemetry, monkeypatch):
    _, _, first = telemetry
    first.deactivate()
    original = observability.ChatTelemetryLayer.get_response

    def fail(*args, **kwargs):
        raise RuntimeError("registration failure")

    monkeypatch.setattr(_instrumentation, "_acquire_shared_processor", fail)
    plugin = MicrosoftAgentFrameworkInstrumentor()
    with pytest.raises(RuntimeError, match="registration failure"):
        plugin.activate()
    assert not plugin._is_instrumented
    assert observability.ChatTelemetryLayer.get_response is original
    assert not observability.OBSERVABILITY_SETTINGS.enable_sensitive_data
    assert _policy._users == 0


def test_unrelated_workflow_attributes_are_untouched(telemetry):
    provider, exporter, _ = telemetry
    with provider.get_tracer("other-sdk").start_as_current_span("workflow.run") as span:
        span.set_attribute("workflow.name", "unrelated")
    assert dict(exporter.get_finished_spans()[0].attributes) == {
        "workflow.name": "unrelated"
    }


@pytest.mark.parametrize("value", [False, 0])
def test_falsy_native_tool_result(telemetry, value):
    _, exporter, _ = telemetry

    @tool
    def falsy() -> bool | int:
        return value

    asyncio.run(falsy.invoke(arguments={}))
    span = exporter.get_finished_spans()[0]
    assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == value

    assert (
        json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])["arguments"]
        is None
    )
    assert "gen_ai.tool.call.id" not in span.attributes


def test_sampled_out_provider_leaves_no_policy_cache(monkeypatch):
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    provider = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        observability,
        "get_tracer",
        lambda *a, **kw: provider.get_tracer("agent_framework"),
    )
    plugin = MicrosoftAgentFrameworkInstrumentor()
    plugin.activate()

    async def run():
        await FixtureClient().get_response(
            [Message("user", [Content.from_text("sampled-out fixture")])]
        )
        async for _ in FixtureClient().get_response(
            [Message("user", [Content.from_text("sampled-out fixture")])], stream=True
        ):
            pass

    try:
        asyncio.run(run())
        assert plugin._processor._content == {}
    finally:
        plugin.deactivate()
        provider.shutdown()


@pytest.mark.parametrize("usage", [None, {"input_token_count": 5}])
@pytest.mark.parametrize("private", [False, True])
def test_native_embedding_vectors_usage_identity(
    telemetry, monkeypatch, usage, private
):
    from agent_framework import BaseEmbeddingClient, Embedding, GeneratedEmbeddings
    from agent_framework.observability import EmbeddingTelemetryLayer

    _, exporter, _ = telemetry
    result = GeneratedEmbeddings(
        [Embedding([i / 128 for i in range(128)])], usage=usage
    )

    class Provider(BaseEmbeddingClient):
        async def get_embeddings(self, values, *, options=None):
            return result

    class Client(EmbeddingTelemetryLayer, Provider):
        model = "fixture-embedding"
        OTEL_PROVIDER_NAME = "fixture"

    if private:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    assert asyncio.run(Client().get_embeddings(["embedding fixture"])) is result
    (span,) = exporter.get_finished_spans()
    assert span.attributes[RESPAN_LOG_TYPE] == "embedding"
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
    if private:
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    else:
        assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
            next(iter(result)).vector
        ]
        assert json.loads(span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
            "embedding fixture"
        ]
    if usage:
        assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    else:
        assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in span.attributes


def test_embedding_failure_and_sampled_out(telemetry, monkeypatch):
    from agent_framework import BaseEmbeddingClient
    from agent_framework.observability import EmbeddingTelemetryLayer
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    _, exporter, plugin = telemetry

    class Provider(BaseEmbeddingClient):
        async def get_embeddings(self, values, *, options=None):
            raise ValueError("fixture embedding error")

    class Client(EmbeddingTelemetryLayer, Provider):
        model = "fixture-embedding"

    with pytest.raises(ValueError, match="fixture embedding error"):
        asyncio.run(Client().get_embeddings(["fixture"]))
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes[RESPAN_LOG_TYPE] == "embedding"
    plugin.deactivate()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(
        observability,
        "get_tracer",
        lambda *a, **kw: provider.get_tracer("agent_framework"),
    )
    plugin.activate()
    with pytest.raises(ValueError):
        asyncio.run(Client().get_embeddings(["fixture"]))
    assert plugin._processor._content == {}
    plugin.deactivate()
    provider.shutdown()


def test_native_enable_partial_failure_restores_policy(telemetry, monkeypatch):
    _, _, first = telemetry
    first.deactivate()
    settings = observability.OBSERVABILITY_SETTINGS

    def broken_enable(**kwargs):
        settings.enable_sensitive_data = True
        raise RuntimeError("native enable failure")

    monkeypatch.setattr(observability, "enable_instrumentation", broken_enable)
    plugin = MicrosoftAgentFrameworkInstrumentor(capture_content=False)
    with pytest.raises(RuntimeError, match="native enable failure"):
        plugin.activate()
    assert settings.enable_sensitive_data is False
    assert _policy._users == 0
    assert _policy.capture_content() is True


@pytest.mark.parametrize("tokens", [-1, 1.5, True])
def test_invalid_embedding_usage_is_not_exported(telemetry, tokens):
    from agent_framework import BaseEmbeddingClient, Embedding, GeneratedEmbeddings
    from agent_framework.observability import EmbeddingTelemetryLayer
    from opentelemetry.semconv._incubating.attributes import (
        gen_ai_attributes as GenAIAttributes,
    )

    _, exporter, _ = telemetry

    class Provider(BaseEmbeddingClient):
        async def get_embeddings(self, values, *, options=None):
            return GeneratedEmbeddings(
                [Embedding([0.1])], usage={"input_token_count": tokens}
            )

    class Client(EmbeddingTelemetryLayer, Provider):
        model = "fixture"

    asyncio.run(Client().get_embeddings(["fixture"]))
    attrs = exporter.get_finished_spans()[0].attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert SpanAttributes.LLM_USAGE_PROMPT_TOKENS not in attrs


def test_embedding_serialization_failure_keeps_response_identity(telemetry):
    from agent_framework import BaseEmbeddingClient, Embedding, GeneratedEmbeddings
    from agent_framework.observability import EmbeddingTelemetryLayer

    class Opaque:
        __slots__ = ()

        def __repr__(self):
            raise ValueError("cannot serialize fixture vector")

    result = GeneratedEmbeddings([Embedding(Opaque())])

    class Provider(BaseEmbeddingClient):
        async def get_embeddings(self, values, *, options=None):
            return result

    class Client(EmbeddingTelemetryLayer, Provider):
        model = "fixture"

    assert asyncio.run(Client().get_embeddings(["fixture"])) is result
    assert (
        SpanAttributes.TRACELOOP_ENTITY_OUTPUT
        not in telemetry[1].get_finished_spans()[0].attributes
    )
