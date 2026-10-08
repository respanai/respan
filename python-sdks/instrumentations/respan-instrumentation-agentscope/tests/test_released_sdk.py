"""Exercise released AgentScope methods with synthetic HTTP and SDK fixtures."""

import asyncio
import functools
import json
from typing import ClassVar

import httpx2
import pytest
from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.embedding import EmbeddingResponse, EmbeddingUsage, OpenAIEmbeddingModel
from agentscope.message import TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatResponse, ChatUsage, OpenAIChatModel
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk, Toolkit
from openai import AsyncOpenAI
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_agentscope import AgentScopeInstrumentor
from respan_instrumentation_agentscope import _instrumentation as impl
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer


class Fixture:
    def __init__(self):
        self.requests = []
        self.clients = []

    async def handle(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if "fixture-failure" in json.dumps(body):
            return httpx2.Response(
                400,
                json={
                    "error": {
                        "message": "controlled fixture error",
                        "type": "invalid_request_error",
                    }
                },
            )
        if request.url.path.endswith("/embeddings"):
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "model": "fixture-embedding",
                    "data": [
                        {
                            "index": i,
                            "object": "embedding",
                            "embedding": [float(n) for n in range(256)],
                        }
                        for i, _ in enumerate(body["input"])
                    ],
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                },
            )
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        message = {"role": "assistant", "content": "fixture answer"}
        if body.get("tools") and not any(m["role"] == "tool" for m in body["messages"]):
            name = body["tools"][0]["function"]["name"]
            arguments = (
                {"city": "Tokyo"}
                if name != "generate_structured_output"
                else {"answer": "fixture value"}
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "fixture-tool-id",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        base = {"id": "fixture-response", "created": 0, "model": "fixture-chat"}
        if body.get("stream"):
            chunks = [
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "fixture "},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "stream"},
                            "finish_reason": "stop",
                        }
                    ],
                },
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": usage,
                },
            ]
            return httpx2.Response(
                200,
                text="".join("data: " + json.dumps(x) + "\n\n" for x in chunks)
                + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx2.Response(
            200,
            json={
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls"
                        if message.get("tool_calls")
                        else "stop",
                    }
                ],
                "usage": usage,
            },
        )

    def client(self):
        client = AsyncOpenAI(
            api_key="fixture-only",
            base_url="https://fixture.invalid/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
            max_retries=0,
        )
        self.clients.append(client)
        return client

    def model(self, stream=False):
        model = OpenAIChatModel(
            credential=OpenAICredential(
                api_key="fixture-only", base_url="https://fixture.invalid/v1"
            ),
            model="fixture-chat",
            stream=stream,
            max_retries=0,
            client_kwargs={
                "http_client": httpx2.AsyncClient(
                    transport=httpx2.MockTransport(self.handle)
                ),
                "max_retries": 0,
            },
        )
        return model

    def embedding(self):
        model = OpenAIEmbeddingModel(
            credential=OpenAICredential(
                api_key="fixture-only", base_url="https://fixture.invalid/v1"
            ),
            model="fixture-embedding",
            dimensions=256,
            max_retries=0,
        )
        model.client = self.client()
        model.batch_size = 2
        return model


@pytest.fixture
def recorded(monkeypatch):
    RespanTracer.reset_instance()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(impl, "_tracer", lambda: provider.get_tracer("released"))
    yield exporter, provider
    for patch in list(impl._PATCHES.values()):
        for owner in list(patch.owners):
            owner.deactivate()
    provider.shutdown()
    RespanTracer.reset_instance()


@pytest.fixture
def fixture():
    return Fixture()


def spans(recorded, kind=None):
    result = recorded[0].get_finished_spans()
    return [s for s in result if kind is None or s.attributes[RESPAN_LOG_TYPE] == kind]


@pytest.mark.asyncio
async def test_real_chat_stream_zero_usage_and_return_identity(recorded, fixture):
    plugin = AgentScopeInstrumentor()
    plugin.activate()
    model = fixture.model()
    result = await model([UserMsg(name="user", content="fixture prompt")])
    assert result.content[0].text == "fixture answer"
    model.stream = True
    source = await model([UserMsg(name="user", content="fixture stream")])
    assert hasattr(source, "aclose")
    chunks = [c async for c in source]
    assert chunks[-1].is_last
    assert chunks[-1].content[0].text == "fixture stream"
    captured = spans(recorded, "chat")
    assert len(captured) == 2
    for span in captured:
        assert span.attributes["gen_ai.usage.input_tokens"] == 0
        assert span.attributes["gen_ai.usage.output_tokens"] == 0
        assert span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 0
    assert captured[-1].attributes["gen_ai.completion.0.content"] == "fixture stream"
    assert captured[-1].attributes["gen_ai.is_streaming"] is True
    assert captured[0].attributes["gen_ai.is_streaming"] is False


@pytest.mark.asyncio
async def test_real_structured_generation(recorded, fixture):
    AgentScopeInstrumentor().activate()
    model = fixture.model()
    result = await model.generate_structured_output(
        [UserMsg(name="user", content="structure")],
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )
    assert result.content == {"answer": "fixture value"}
    captured = spans(recorded, "chat")
    assert len(captured) == 1
    assert (
        json.loads(captured[0].attributes["gen_ai.completion.0.content"])
        == result.content
    )
    assert captured[0].attributes["gen_ai.usage.input_tokens"] == 0


@pytest.mark.asyncio
async def test_real_embedding_batches_complete_vectors(recorded, fixture):
    AgentScopeInstrumentor().activate()
    model = fixture.embedding()
    result = await model(["first", "second", "third"])
    assert len(result.embeddings) == 3
    captured = spans(recorded, "embedding")
    assert len(captured) == 2
    assert sorted(
        len(json.loads(s.attributes["traceloop.entity.output"])) for s in captured
    ) == [1, 2]
    for span in captured:
        assert len(json.loads(span.attributes["traceloop.entity.output"])[0]) == 256
        assert json.loads(span.attributes["traceloop.entity.output"])[0][-1] == 255
        assert span.attributes["gen_ai.usage.input_tokens"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,usage",
    [("cache", EmbeddingUsage(tokens=0, time=0)), ("api", None)],
    ids=["cache", "missing"],
)
async def test_embedding_missing_or_cache_usage_not_invented(recorded, source, usage):
    class Embedding:
        model = "synthetic-embedding"

        async def _call_api(self, inputs):
            return EmbeddingResponse(
                embeddings=[[0.0, 1.0]], usage=usage, source=source
            )

    model = Embedding()
    AgentScopeInstrumentor(embedding_models=[model]).activate()
    result = await model._call_api(["text"])
    assert result.source == source
    assert "gen_ai.usage.input_tokens" not in spans(recorded, "embedding")[0].attributes


class Weather(ToolBase):
    name = "weather"
    is_concurrency_safe = True
    is_read_only = True
    description = "Synthetic weather"
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    async def check_permissions(self, tool_input, context):
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="fixture")

    async def call(self, city):
        return ToolChunk(content=[TextBlock(text=f"{city}: sunny")])


@pytest.mark.asyncio
async def test_real_agent_tool_parentage_and_correlation(recorded, fixture):
    AgentScopeInstrumentor().activate()
    agent = Agent(
        name="WeatherAgent",
        system_prompt="Use weather",
        model=fixture.model(),
        toolkit=Toolkit(tools=[Weather()]),
    )
    result = await agent.reply(UserMsg(name="user", content="weather please"))
    assert result.get_text_content() == "fixture answer"
    agent_span = spans(recorded, "agent")[0]
    tool = spans(recorded, "tool")[0]
    chats = spans(recorded, "chat")
    assert len(chats) == 2
    assert tool.parent.span_id == agent_span.context.span_id
    assert all(s.parent.span_id == agent_span.context.span_id for s in chats)
    call = json.loads(chats[0].attributes["gen_ai.completion.0.tool_calls"])[0]
    assert call["id"] == tool.attributes["gen_ai.tool.call.id"] == "fixture-tool-id"
    assert "gen_ai.completion.0.tool_calls" not in chats[1].attributes
    assert tool.start_time < tool.end_time <= agent_span.end_time


@pytest.mark.asyncio
async def test_real_agent_event_stream_no_caller_context_leak(recorded, fixture):
    AgentScopeInstrumentor().activate()
    agent = Agent(
        name="StreamAgent", system_prompt="reply", model=fixture.model(stream=True)
    )
    baseline = trace.get_current_span()
    events = []
    async for event in agent.reply_stream(UserMsg(name="user", content="stream")):
        assert trace.get_current_span() is baseline
        events.append(event)
    assert events
    assert (
        spans(recorded, "agent")[0].attributes["traceloop.entity.output"]
        == "fixture stream"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat", "embedding"])
async def test_provider_errors_no_fabricated_response_or_status(
    recorded, fixture, method
):
    AgentScopeInstrumentor().activate()
    model = fixture.model() if method == "chat" else fixture.embedding()
    with pytest.raises(Exception, match="controlled fixture error"):
        await model(
            [UserMsg(name="user", content="fixture-failure")]
            if method == "chat"
            else ["fixture-failure"]
        )
    span = spans(recorded, method)[0]
    assert span.status.status_code is StatusCode.ERROR
    assert "status_code" not in span.attributes
    assert "traceloop.entity.output" not in span.attributes
    assert not any(k.startswith("gen_ai.completion.") for k in span.attributes)
    assert "gen_ai.usage.input_tokens" not in span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["env", "context", "config"])
async def test_privacy_start_boundary_and_usage(recorded, fixture, monkeypatch, policy):
    AgentScopeInstrumentor(capture_content=policy != "config").activate()
    token = None
    if policy == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if policy == "context":
        token = context_api.attach(
            context_api.set_value(ENABLE_CONTENT_TRACING_KEY, False)
        )
    model = fixture.model()
    call = model([UserMsg(name="user", content="private-prompt")])
    if token is not None:
        context_api.detach(token)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    await call
    span = spans(recorded, "chat")[0]
    assert not any(
        k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in span.attributes
    )
    assert "traceloop.entity.input" not in span.attributes
    assert span.attributes["gen_ai.usage.input_tokens"] == 0


@pytest.mark.asyncio
async def test_suppressed_call_stays_suppressed_outside_scope(recorded, fixture):
    AgentScopeInstrumentor().activate()
    token = context_api.attach(
        context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    call = fixture.model()([UserMsg(name="user", content="hidden")])
    context_api.detach(token)
    await call
    assert not spans(recorded)


@pytest.mark.asyncio
async def test_sampler_is_respected_without_cached_tracer(
    recorded, fixture, monkeypatch
):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    provider.add_span_processor(SimpleSpanProcessor(recorded[0]))
    monkeypatch.setattr(impl, "_tracer", lambda: provider.get_tracer("sampled"))
    AgentScopeInstrumentor().activate()
    model = fixture.model()
    await model([UserMsg(name="user", content="dropped")])
    assert not spans(recorded)
    assert provider.sampler is ALWAYS_OFF
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(recorded[0]))
    await model([UserMsg(name="user", content="recorded")])
    assert len(spans(recorded)) == 1
    assert provider.sampler is ALWAYS_ON


@pytest.mark.asyncio
async def test_shared_owners_foreign_wrapper_and_inactive_inner(recorded, fixture):
    original = OpenAIChatModel.__call__
    first = AgentScopeInstrumentor()
    second = AgentScopeInstrumentor()
    first.activate()
    second.activate()
    first.deactivate()
    await fixture.model()([UserMsg(name="user", content="still traced")])
    assert len(spans(recorded, "chat")) == 1
    inner = OpenAIChatModel.__call__

    @functools.wraps(inner)
    def foreign(self, *a, **kw):
        return inner(self, *a, **kw)

    OpenAIChatModel.__call__ = foreign
    try:
        second.deactivate()
        assert OpenAIChatModel.__call__ is foreign
        await fixture.model()([UserMsg(name="user", content="inactive")])
        assert len(spans(recorded, "chat")) == 1
    finally:
        OpenAIChatModel.__call__ = original


def test_ownership_config_conflict_rollback(recorded):
    first = AgentScopeInstrumentor()
    first.activate()
    second = AgentScopeInstrumentor(capture_content=False)
    with pytest.raises(ValueError, match="same capture_content"):
        second.activate()
    assert not second._patches
    assert first._is_instrumented


def test_activation_rollback(recorded, monkeypatch):
    original = Agent.reply
    plugin = AgentScopeInstrumentor()
    patch = plugin._patch

    def failing(target, name, kind):
        if kind == "model_call":
            raise RuntimeError("injected activation failure")
        patch(target, name, kind)

    monkeypatch.setattr(plugin, "_patch", failing)
    with pytest.raises(RuntimeError, match="injected"):
        plugin.activate()
    assert Agent.reply is original
    assert not plugin._patches and not plugin._is_instrumented


@pytest.mark.asyncio
async def test_global_plus_explicit_agent_owner(recorded, fixture):
    agent = Agent(name="OwnedAgent", system_prompt="reply", model=fixture.model())
    global_owner = AgentScopeInstrumentor()
    global_owner.activate()
    local_owner = AgentScopeInstrumentor(
        agent=agent,
        instrument_models=False,
        instrument_tools=False,
        instrument_embeddings=False,
    )
    local_owner.activate()
    global_owner.deactivate()
    await agent.reply(UserMsg(name="user", content="still traced"))
    assert len(spans(recorded, "agent")) == 1
    local_owner.deactivate()
    assert "reply" not in vars(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "iteration", "close", "cancel"])
async def test_stream_close_error_and_parent_contract(recorded, failure):
    closed = []
    exception = RuntimeError("native stream failure")

    class Model:
        model = "fixture-stream"

        async def __call__(self, messages):
            async def source():
                try:
                    yield ChatResponse(content=[TextBlock(text="first")], is_last=False)
                    if failure == "iteration":
                        raise exception
                    if failure == "cancel":
                        raise asyncio.CancelledError("native cancel")
                    yield ChatResponse(
                        content=[TextBlock(text="last")],
                        is_last=True,
                        usage=ChatUsage(input_tokens=0, output_tokens=0, time=0),
                    )
                finally:
                    closed.append(True)
                    if failure == "close":
                        raise exception

            return source()

    model = Model()
    AgentScopeInstrumentor(models=[model]).activate()
    source = await model([UserMsg(name="user", content="prompt")])
    baseline = trace.get_current_span()
    assert (await anext(source)).content[0].text == "first"
    assert trace.get_current_span() is baseline
    if failure in ("iteration", "cancel"):
        with pytest.raises(BaseException) as caught:
            await anext(source)
        if failure == "iteration":
            assert caught.value is exception
        else:
            assert isinstance(caught.value, asyncio.CancelledError)
    elif failure == "close":
        with pytest.raises(RuntimeError) as caught:
            await source.aclose()
        assert caught.value is exception
    else:
        await source.aclose()
    assert closed == [True]
    assert trace.get_current_span() is baseline
    assert len(spans(recorded, "chat")) == 1
    assert "gen_ai.usage.input_tokens" not in spans(recorded, "chat")[0].attributes
    await source.aclose()
    assert len(spans(recorded, "chat")) == 1


@pytest.mark.asyncio
async def test_deactivated_lazy_stream_preserves_close(recorded):
    closed = []

    class CustomAgent:
        name = "lazy"

        async def reply_stream(self, inputs):
            try:
                yield UserMsg(name="user", content="synthetic")
                yield UserMsg(name="user", content="second")
            finally:
                closed.append(True)

    agent = CustomAgent()
    plugin = AgentScopeInstrumentor(
        agent=agent,
        instrument_models=False,
        instrument_embeddings=False,
        instrument_tools=False,
    )
    plugin.activate()
    stream = agent.reply_stream("input")
    plugin.deactivate()
    await anext(stream)
    await stream.aclose()
    assert closed == [True]
    assert not spans(recorded)


@pytest.mark.asyncio
async def test_lazy_stream_privacy_snapshot(recorded, fixture, monkeypatch):
    AgentScopeInstrumentor().activate()
    agent = Agent(
        name="PrivateAgent",
        system_prompt="private-system",
        model=fixture.model(stream=True),
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    stream = agent.reply_stream(UserMsg(name="user", content="private-input"))
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    async for _ in stream:
        pass
    assert len(spans(recorded)) == 2
    for span in spans(recorded):
        assert "traceloop.entity.input" not in span.attributes
        assert "traceloop.entity.output" not in span.attributes
        assert not any(
            k.startswith(("gen_ai.prompt.", "gen_ai.completion."))
            for k in span.attributes
        )


@pytest.mark.asyncio
async def test_no_usage_or_output_added_to_partial_or_empty_stream(recorded):
    class Model:
        model = "empty"

        async def __call__(self, messages):
            async def stream():
                if False:
                    yield None

            return stream()

    model = Model()
    AgentScopeInstrumentor(models=[model]).activate()
    source = await model([])
    assert [item async for item in source] == []
    span = spans(recorded, "chat")[0]
    assert "gen_ai.usage.input_tokens" not in span.attributes
    assert "gen_ai.completion.0.content" not in span.attributes
    assert "traceloop.entity.output" not in span.attributes


@pytest.mark.asyncio
async def test_mixed_content_and_tool_arguments_preserved(recorded):
    class Model:
        model = "mixed"

        async def __call__(self, messages):
            return ChatResponse(
                content=[
                    TextBlock(text="visible"),
                    ToolCallBlock(
                        id="call-mixed", name="weather", input='{"city":"Tokyo"}'
                    ),
                ],
                is_last=True,
            )

    model = Model()
    AgentScopeInstrumentor(models=[model]).activate()
    await model([UserMsg(name="user", content="request")])
    captured = spans(recorded, "chat")[0].attributes
    assert captured["gen_ai.completion.0.content"] == "visible"
    assert (
        json.loads(captured["gen_ai.completion.0.tool_calls"])[0]["id"] == "call-mixed"
    )


@pytest.mark.asyncio
async def test_serialization_failure_does_not_change_result(recorded):
    circular = {}
    circular["cycle"] = circular

    class Model:
        model = "opaque"

        async def __call__(self, messages):
            return circular

    model = Model()
    AgentScopeInstrumentor(models=[model]).activate()
    assert await model(circular) is circular
    assert len(spans(recorded, "chat")) == 1


@pytest.mark.asyncio
async def test_real_pipeline_workflow_with_agent_child(recorded, fixture):
    pipeline = pytest.importorskip(
        "agentscope.pipeline", reason="Pipelines added after minimum SDK"
    )
    model = fixture.model()
    original = fixture.handle

    async def without_tool(request):
        data = json.loads(request.content)
        data.pop("tools", None)
        return await original(httpx2.Request(request.method, request.url, json=data))

    # MockTransport retains the bound handler, so replace the client explicitly.
    model.client = AsyncOpenAI(
        api_key="fixture-only",
        base_url="https://fixture.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(without_tool)),
        max_retries=0,
    )
    AgentScopeInstrumentor().activate()
    leader = Agent(name="Leader", system_prompt="reply", model=model)
    team = pipeline.TeamPipeline(leader=leader, members=[])
    events = [
        event
        async for event in team.reply_stream(UserMsg(name="user", content="pipeline"))
    ]
    assert events
    workflows = spans(recorded, "workflow")
    agents = spans(recorded, "agent")
    assert len(workflows) == 1 and len(agents) == 1
    assert agents[0].parent.span_id == workflows[0].context.span_id
    await model.client.close()


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "4", None])
def test_invalid_or_missing_usage_does_not_manufacture_total(invalid):
    usage = impl.attrs._extract_usage(
        {"usage": {"input_tokens": invalid, "output_tokens": 2}}
    )
    assert usage[0] is None and usage[1] == 2 and usage[2] is None


def test_tool_only_content_empty_and_missing_call_id_not_invented():
    response = {
        "content": [
            {"type": "tool_call", "name": "weather", "input": {"city": "Tokyo"}}
        ]
    }
    result = impl.attrs._model_attributes(
        model={"model": "fixture"}, messages=[], tools=None, response=response
    )
    assert result["gen_ai.completion.0.content"] == ""
    assert result["traceloop.entity.output"] == ""
    calls = json.loads(result["gen_ai.completion.0.tool_calls"])
    assert "id" not in calls[0]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
