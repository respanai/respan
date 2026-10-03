"""Native PydanticAI models and telemetry; no model provider is contacted."""

import asyncio
import json
from importlib.metadata import version

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import BaseModel
from pydantic_ai import Agent, Embedder
from pydantic_ai.embeddings.test import TestEmbeddingModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage
from respan_instrumentation_pydantic_ai import PydanticAIInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


@pytest.fixture
def runtime(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    plugins = []

    def activate(**kwargs):
        plugin = PydanticAIInstrumentor(**kwargs)
        plugin.activate()
        plugins.append(plugin)
        return plugin

    yield provider, exporter, activate
    for plugin in reversed(plugins):
        plugin.deactivate()
    provider.shutdown()


def test_released_embedder_sync_async_vectors_usage_and_parentage(runtime):
    provider, exporter, activate = runtime
    activate()
    embedder = Embedder(TestEmbeddingModel(dimensions=128))
    with provider.get_tracer("test").start_as_current_span("workflow") as parent:
        query = embedder.embed_query_sync("query text")
        documents = asyncio.run(
            embedder.embed_documents(["first document", "second document"])
        )
    spans = [s for s in exporter.get_finished_spans() if s.name != "workflow"]
    assert len(spans) == 2
    for span, response in zip(spans, [query, documents], strict=True):
        attrs = span.attributes
        assert span.parent.span_id == parent.context.span_id
        assert attrs[RESPAN_LOG_TYPE] == "embedding"
        assert attrs[SpanAttributes.LLM_SYSTEM] == "test"
        assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "test"
        assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
        assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == list(
            response.inputs
        )
        vectors = json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        assert vectors == response.embeddings
        assert len(vectors[0]) == 128
        assert (
            attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == response.usage.input_tokens
        )
        assert not {"inputs", "embeddings", "input_type", "inputs_count"}.intersection(
            attrs
        )


def test_released_content_opt_out_and_specific_embedder_ownership(runtime):
    _, exporter, activate = runtime
    embedder = Embedder(TestEmbeddingModel())
    previous = embedder.instrument
    first = activate(embedder=embedder, include_content=False)
    second = activate(embedder=embedder, include_content=False)
    first.deactivate()
    embedder.embed_query_sync("hidden input")
    assert len(exporter.get_finished_spans()) == 1
    attrs = exporter.get_finished_spans()[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert "hidden input" not in str(attrs)
    second.deactivate()
    assert embedder.instrument is previous
    embedder.embed_query_sync("no instrumentation")
    assert len(exporter.get_finished_spans()) == 1


def test_released_global_ownership_restores_agent_and_embedder(runtime):
    _, exporter, activate = runtime
    old_agent, old_embedder = Agent._instrument_default, Embedder._instrument_default
    first, second = activate(), activate()
    first.deactivate()
    Embedder(TestEmbeddingModel()).embed_query_sync("still active")
    assert len(exporter.get_finished_spans()) == 1
    second.deactivate()
    assert Agent._instrument_default is old_agent
    assert Embedder._instrument_default is old_embedder


@pytest.mark.parametrize("format_version", [5, 6])
def test_released_tool_history_usage_and_content_switch(runtime, format_version):
    if format_version == 6 and tuple(
        map(int, version("pydantic-ai-slim").split(".")[:2])
    ) < (2, 54):
        pytest.skip("Version 6 format is a current SDK opt-in")
    _, exporter, activate = runtime
    activate(version=format_version)
    calls = []

    def respond(messages, info):
        calls.append(messages)
        parts = (
            [ToolCallPart("add", {"a": 15, "b": 27}, tool_call_id="call-add")]
            if len(calls) == 1
            else [TextPart("42")]
        )
        return ModelResponse(
            parts,
            model_name="fixture-model",
            usage=RequestUsage(
                input_tokens=11,
                output_tokens=7,
                cache_read_tokens=3,
                cache_write_tokens=2,
                details={"reasoning_tokens": 4},
            ),
        )

    agent = Agent(FunctionModel(respond), name="calculator")

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    assert agent.run_sync("Add 15 and 27.").output == "42"
    spans = exporter.get_finished_spans()
    chats = sorted(
        [s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "chat"],
        key=lambda s: s.start_time,
    )
    assert len(chats) == 2
    first_calls = json.loads(
        chats[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
    )
    assert first_calls[0]["id"] == "call-add"
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" not in chats[1].attributes
    assert any(
        key.startswith(SpanAttributes.LLM_PROMPTS) and key.endswith(".tool_calls")
        for key in chats[1].attributes
    )
    for span in chats:
        assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11
        assert span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 7
        assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 18
        assert span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
        assert span.attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 4
    tool = next(s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "tool")
    assert tool.attributes["gen_ai.tool.call.id"] == "call-add"
    assert json.loads(tool.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])[
        "arguments"
    ] == {"a": 15, "b": 27}
    agent_span = next(s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "agent")
    assert not any(
        k.startswith(("gen_ai.usage.", "gen_ai.aggregated_usage.", "llm.usage."))
        for k in agent_span.attributes
    )


def test_released_agent_content_opt_out_and_structured_output(runtime):
    _, exporter, activate = runtime
    activate(include_content=False)

    class Answer(BaseModel):
        value: int

    agent = Agent(
        TestModel(custom_output_args={"value": 42}),
        output_type=Answer,
        name="structured",
    )
    assert agent.run_sync("private question").output.value == 42
    assert exporter.get_finished_spans()
    for span in exporter.get_finished_spans():
        attrs = span.attributes
        assert "private question" not in str(attrs)
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs


@pytest.mark.parametrize("fail", [False, True])
def test_released_async_stream_completion_and_failure(runtime, fail):
    _, exporter, activate = runtime
    activate()

    async def stream(messages, info):
        yield "stream "
        if fail:
            raise RuntimeError("synthetic stream failure")
        yield "answer"

    async def run():
        agent = Agent(FunctionModel(stream_function=stream), name="streaming")
        async with agent.run_stream("stream a response") as result:
            return [item async for item in result.stream_text()]

    if fail:
        with pytest.raises(RuntimeError, match="synthetic stream failure"):
            asyncio.run(run())
    else:
        assert asyncio.run(run())[-1] == "stream answer"
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(s.end_time is not None for s in spans)
    if fail:
        agent_span = next(
            s for s in spans if s.attributes.get(RESPAN_LOG_TYPE) == "agent"
        )
        assert agent_span.status.status_code == trace.StatusCode.ERROR
        # The released SDK leaves its child chat span UNSET for a stream
        # iteration error; the adapter preserves the native status.
        assert all(
            s.status.status_code in {trace.StatusCode.UNSET, trace.StatusCode.ERROR}
            for s in spans
        )


def test_unrelated_provider_spans_remain_untouched_and_nested_usage_is_not_doubled(
    runtime,
):
    provider, exporter, activate = runtime
    activate()
    attrs = {
        "gen_ai.system": "other",
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.input_tokens": 11,
        "gen_ai.usage.output_tokens": 7,
    }
    with provider.get_tracer("unrelated-sdk").start_as_current_span(
        "unrelated", attributes=attrs
    ):
        pass
    assert dict(exporter.get_finished_spans()[0].attributes) == attrs
    with (
        provider.get_tracer("pydantic-ai").start_as_current_span(
            "chat fixture", attributes=attrs
        ),
        provider.get_tracer("provider-sdk").start_as_current_span(
            "provider", attributes=attrs
        ),
    ):
        pass
    spans = exporter.get_finished_spans()
    wrapper = next(s for s in spans if s.name == "chat fixture")
    provider_span = next(s for s in spans if s.name == "provider")
    assert "gen_ai.usage.input_tokens" not in wrapper.attributes
    assert provider_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11


def test_embedding_activation_failure_restores_agent_and_capture(runtime, monkeypatch):
    from pydantic_ai.embeddings.instrumented import InstrumentedEmbeddingModel

    provider, _, activate = runtime
    previous = Agent._instrument_default
    original = InstrumentedEmbeddingModel._instrument

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic activation failure")

    monkeypatch.setattr(Embedder, "instrument_all", fail)
    with pytest.raises(RuntimeError, match="synthetic activation failure"):
        activate()
    assert Agent._instrument_default is previous
    assert InstrumentedEmbeddingModel._instrument is original
    assert len(provider._active_span_processor._span_processors) == 1


def test_released_embedding_failure_preserves_exception_and_content_switch(runtime):
    _, exporter, activate = runtime
    activate(include_content=False)

    class FailingModel(TestEmbeddingModel):
        async def embed(self, *args, **kwargs):
            raise RuntimeError("synthetic embedding failure")

    with pytest.raises(RuntimeError, match="synthetic embedding failure"):
        Embedder(FailingModel()).embed_query_sync("hidden failed input")
    span = exporter.get_finished_spans()[0]
    assert span.attributes[RESPAN_LOG_TYPE] == "embedding"
    assert span.status.status_code == trace.StatusCode.ERROR
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
