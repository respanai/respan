"""Released SDK tests; only the OpenAI HTTP transport is replaced."""

import asyncio
import inspect
import json

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import APIConnectionError, AsyncOpenAI
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from respan_instrumentation_autogen import (
    AutoGenInstrumentor,
    _instrumentation,
    _modern,
)
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY
from respan_tracing.core.tracer import RespanTracer
from wrapt import wrap_function_wrapper

try:
    import httpx2 as httpx
except ImportError:
    import httpx

MODEL = "gpt-4o-mini-2024-07-18"


def response(content="fixture answer", calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": ident,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for ident, name, arguments in calls
        ]
    return {
        "id": "fixture",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def chunk(text=None, terminal=False):
    value = {
        "id": "fixture",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": []
        if terminal
        else [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    if terminal:
        value["usage"] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    return ("data: " + json.dumps(value) + "\n\n").encode()


@pytest.fixture
def tracing(monkeypatch):
    RespanTracer.reset_instance()
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    plugin = AutoGenInstrumentor()
    plugin.activate()
    assert plugin._is_instrumented
    yield provider, exporter, plugin
    plugin.deactivate()
    assert _instrumentation._OWNERS == 0
    provider.shutdown()
    RespanTracer.reset_instance()


@pytest.fixture
def client_factory():
    clients = []

    async def factory(*responses, stream=None):
        queue = list(responses) or [response()]
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            if stream is not None:
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=stream
                )
            payload = queue.pop(0)
            if isinstance(payload, int):
                return httpx.Response(
                    payload,
                    json={
                        "error": {
                            "message": "controlled provider failure",
                            "type": "fixture",
                        }
                    },
                )
            return httpx.Response(200, json=payload)

        client = OpenAIChatCompletionClient(model=MODEL, api_key="fixture")
        await client._client.close()
        client._client = AsyncOpenAI(
            api_key="fixture",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client, requests

    yield factory
    for client in clients:
        asyncio.run(client.close())


def spans(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("respan.entity.log_type")
    ]


@pytest.mark.asyncio
async def test_completed_agent_is_ended_before_run_returns(tracing, client_factory):
    provider, exporter, _ = tracing
    client, requests = await client_factory()
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        result = await AssistantAgent("assistant", model_client=client).run(
            task="fixture prompt"
        )
        assert isinstance(result, TaskResult)
        assert result.messages[-1].content == "fixture answer"
        assert trace.get_current_span() is parent
        captured = spans(exporter)
        assert len(captured) == 2
        agent = next(
            s for s in captured if s.attributes["respan.entity.log_type"] == "agent"
        )
        chat = next(
            s for s in captured if s.attributes["respan.entity.log_type"] == "chat"
        )
        assert agent.parent.span_id == parent.context.span_id
        assert chat.parent.span_id == agent.context.span_id
        assert "fixture answer" in agent.attributes["traceloop.entity.output"]
        assert chat.attributes["gen_ai.usage.input_tokens"] == 3
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_tool_id_schema_error_and_history(tracing, client_factory):
    _, exporter, _ = tracing

    def fail_tool() -> int:
        """Produce a controlled failure."""
        raise ValueError("fixture tool failed")

    client, _ = await client_factory(
        response(calls=[("fixture-call", "fail_tool", "{}")]),
        response("handled failure"),
    )
    result = await AssistantAgent(
        "assistant", model_client=client, tools=[fail_tool], reflect_on_tool_use=True
    ).run(task="use tool")
    assert result.messages[-1].content == "handled failure"
    captured = spans(exporter)
    tool = next(s for s in captured if s.attributes["respan.entity.log_type"] == "tool")
    assert tool.attributes["gen_ai.tool.call.id"] == "fixture-call"
    assert tool.status.status_code is trace.StatusCode.ERROR
    assert (
        "status_code" not in tool.attributes
        and "http.response.status_code" not in tool.attributes
    )
    chats = [s for s in captured if s.attributes["respan.entity.log_type"] == "chat"]
    assert "fail_tool" in chats[0].attributes["llm.request.functions"]
    assert chats[1].attributes["gen_ai.prompt.3.role"] == "tool"
    assert chats[1].attributes["gen_ai.prompt.3.tool_call_id"] == "fixture-call"
    assert "fixture tool failed" in chats[1].attributes["gen_ai.prompt.3.content"]
    assert "gen_ai.completion.0.tool_calls" not in chats[1].attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["environment", "context"])
async def test_private_constructed_stream_and_tool_outputs(
    tracing, client_factory, monkeypatch, policy
):
    _, exporter, _ = tracing

    def private_tool() -> int:
        return 0

    client, _ = await client_factory(
        response(calls=[("private-call", "private_tool", "{}")])
    )
    token = None
    if policy == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    else:
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    stream = AssistantAgent(
        "private_agent", model_client=client, tools=[private_tool]
    ).run_stream(task="private prompt")
    if token:
        context.detach(token)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    result = [event async for event in stream][-1]
    assert result.messages[-1].content == "0"
    captured = spans(exporter)
    assert {s.attributes["respan.entity.log_type"] for s in captured} == {
        "agent",
        "chat",
        "tool",
    }
    for span in captured:
        assert "traceloop.entity.input" not in span.attributes
        assert "traceloop.entity.output" not in span.attributes
        assert not any(
            k.startswith(("gen_ai.prompt.", "gen_ai.completion."))
            for k in span.attributes
        )


@pytest.mark.asyncio
async def test_suppression_and_shared_owners(tracing, client_factory):
    _, exporter, first = tracing
    second = AutoGenInstrumentor()
    second.activate()
    first.deactivate()
    try:
        client, _ = await client_factory(response(), response())
        token = context.attach(
            context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            await AssistantAgent("hidden", model_client=client).run(task="suppressed")
        finally:
            context.detach(token)
        assert not spans(exporter)
        await AssistantAgent("visible", model_client=client).run(task="visible")
        assert len(spans(exporter)) == 2
    finally:
        second.deactivate()


class StreamFixture(httpx.AsyncByteStream):
    def __init__(self, mode):
        self.mode, self.closed = mode, False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        yield chunk("fixture ")
        if self.mode == "error":
            raise httpx.ReadError("controlled stream error")
        if self.mode == "cancel":
            self.waiting.set()
            await asyncio.Event().wait()
        yield chunk("answer")
        yield chunk(terminal=True)
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["complete", "close", "error", "cancel"])
async def test_stream_context_cleanup_and_real_return_api(
    tracing, client_factory, mode
):
    provider, exporter, _ = tracing
    fixture = StreamFixture(mode)
    client, _ = await client_factory(stream=fixture)
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        stream = client.create_stream([UserMessage(content="fixture", source="user")])
        assert all(
            hasattr(stream, api) for api in ("asend", "athrow", "aclose", "__anext__")
        )
        assert await anext(stream) == "fixture "
        assert trace.get_current_span() is parent
        if mode == "close":
            await stream.aclose()
        elif mode == "error":
            with pytest.raises((httpx.ReadError, APIConnectionError)):
                await anext(stream)
        elif mode == "cancel":
            task = asyncio.create_task(anext(stream))
            await fixture.waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            rest = [item async for item in stream]
            assert rest[-1].content == "fixture answer"
        assert trace.get_current_span() is parent
    (chat,) = spans(exporter)
    assert chat.parent.span_id == parent.context.span_id
    if mode in ("error", "cancel"):
        assert chat.status.status_code is trace.StatusCode.ERROR
        assert "error.message" in chat.attributes
    if mode != "complete":
        assert "gen_ai.completion.0.content" not in chat.attributes
        assert "gen_ai.usage.input_tokens" not in chat.attributes


@pytest.mark.asyncio
async def test_round_robin_connected_tree(tracing, client_factory):
    _, exporter, _ = tracing
    first, _ = await client_factory(response("first fixture"))
    second, _ = await client_factory(response("second fixture"))
    team = RoundRobinGroupChat(
        [
            AssistantAgent("first", model_client=first),
            AssistantAgent("second", model_client=second),
        ],
        max_turns=2,
    )
    result = await team.run(task="team fixture")
    assert result.messages[-1].content == "second fixture"
    captured = spans(exporter)
    assert len(captured) == 5
    roots = [s for s in captured if s.parent is None]
    assert len(roots) == 1
    identifiers = {s.context.span_id for s in captured}
    assert all(s.parent is None or s.parent.span_id in identifiers for s in captured)


@pytest.mark.asyncio
async def test_provider_error_has_no_fabricated_response_or_http_code(
    tracing, client_factory
):
    _, exporter, _ = tracing
    client, _ = await client_factory(401)
    with pytest.raises(Exception, match="controlled provider failure"):
        await AssistantAgent("assistant", model_client=client).run(task="error fixture")
    assert len(spans(exporter)) == 2
    for span in spans(exporter):
        assert span.status.status_code is trace.StatusCode.ERROR
        assert "traceloop.entity.output" not in span.attributes
        assert "status_code" not in span.attributes
        assert "gen_ai.usage.input_tokens" not in span.attributes
        assert span.attributes["http.response.status_code"] == 401


def test_foreign_wrapper_preserved_and_old_generation_inert(tracing):
    _, _, plugin = tracing
    owner = _modern.ModernAutoGenInstrumentor()
    original = next(x[2] for x in owner._owned if x[1] == "create")
    from autogen_ext.models.openai import BaseOpenAIChatCompletionClient

    wrapped = inspect.getattr_static(BaseOpenAIChatCompletionClient, "create")
    wrap_function_wrapper(
        BaseOpenAIChatCompletionClient,
        "create",
        lambda method, instance, args, kwargs: method(*args, **kwargs),
    )
    foreign = inspect.getattr_static(BaseOpenAIChatCompletionClient, "create")
    plugin.deactivate()
    assert inspect.getattr_static(BaseOpenAIChatCompletionClient, "create") is foreign
    assert not owner._generation["enabled"]
    BaseOpenAIChatCompletionClient.create = original
    assert wrapped is not original


def test_failed_activation_restores_partial_wrappers(monkeypatch):
    original = inspect.getattr_static(AssistantAgent, "on_messages_stream")
    real = _modern.wrap_function_wrapper
    count = 0

    def fail_after_one(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("controlled patch failure")
        return real(*args)

    monkeypatch.setattr(_modern, "wrap_function_wrapper", fail_after_one)
    with pytest.raises(RuntimeError, match="controlled patch failure"):
        AutoGenInstrumentor().activate()
    assert inspect.getattr_static(AssistantAgent, "on_messages_stream") is original
    assert not _modern.ModernAutoGenInstrumentor().is_instrumented_by_opentelemetry
    assert _instrumentation._OWNERS == 0


@pytest.mark.asyncio
async def test_sampled_out_has_no_processor_cache(monkeypatch, client_factory):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    plugin = AutoGenInstrumentor()
    plugin.activate()
    try:
        client, _ = await client_factory()
        await AssistantAgent("assistant", model_client=client).run(task="sampled out")
        assert not _instrumentation._PROCESSOR._native_export_parents
        assert not _instrumentation._PRIVACY_PROCESSOR._content
    finally:
        plugin.deactivate()
        provider.shutdown()


@pytest.mark.asyncio
async def test_structured_output_is_native_typed_result(tracing, client_factory):
    from pydantic import BaseModel

    if "output_content_type" not in inspect.signature(AssistantAgent).parameters:
        pytest.skip("Structured output added after the supported minimum")

    class Answer(BaseModel):
        value: int

    client, _ = await client_factory(response('{"value":0}'))
    result = await AssistantAgent(
        "typed", model_client=client, output_content_type=Answer
    ).run(task="typed fixture")
    assert isinstance(result.messages[-1].content, Answer)
    assert result.messages[-1].content.value == 0
    assert len(spans(tracing[1])) == 2
    assert (
        '"value":0'
        in next(
            s
            for s in spans(tracing[1])
            if s.attributes["respan.entity.log_type"] == "chat"
        ).attributes["gen_ai.completion.0.content"]
    )


@pytest.mark.asyncio
async def test_agent_as_tool_nested_tree(tracing, client_factory):
    try:
        from autogen_agentchat.tools import AgentTool
    except ImportError:
        pytest.skip("AgentTool added after the supported minimum")
    worker_client, _ = await client_factory(response("worker result"))
    worker = AssistantAgent("worker", model_client=worker_client)
    parent_client, _ = await client_factory(
        response(calls=[("nested-call", "worker", '{"task":"nested fixture"}')]),
        response("nested complete"),
    )
    parent = AssistantAgent(
        "parent",
        model_client=parent_client,
        tools=[AgentTool(worker, return_value_as_last_message=True)],
        reflect_on_tool_use=True,
    )
    result = await parent.run(task="delegate")
    assert result.messages[-1].content == "nested complete"
    captured = spans(tracing[1])
    assert len(captured) == 6
    tool = next(s for s in captured if s.attributes["respan.entity.log_type"] == "tool")
    worker_span = next(
        s
        for s in captured
        if s.name == "worker" and s.attributes["respan.entity.log_type"] == "agent"
    )
    assert worker_span.parent.span_id == tool.context.span_id
    assert tool.attributes["gen_ai.tool.call.id"] == "nested-call"


@pytest.mark.asyncio
async def test_invalid_call_is_not_silently_filtered(tracing, client_factory):
    client, requests = await client_factory()
    with pytest.raises(TypeError):
        await client.create(
            [UserMessage(content="fixture", source="user")], unsupported_fixture=True
        )
    assert not requests


@pytest.mark.parametrize("value", [-1, 1.5, True, None])
def test_invalid_usage_is_not_invented(value):
    from types import SimpleNamespace

    attrs = _modern._model_output(
        SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=value, completion_tokens=2)
        ),
        False,
    )
    assert "llm.token_count.prompt" not in attrs
    assert "llm.token_count.total" not in attrs
    assert attrs["llm.token_count.completion"] == 2


def test_conflicting_owner_privacy_settings_rejected(tracing):
    from openinference.instrumentation import TraceConfig

    _, _, first = tracing
    second = AutoGenInstrumentor(
        config=TraceConfig(hide_inputs=True, hide_outputs=True)
    )
    with pytest.raises(ValueError, match="matching instrumentation settings"):
        second.activate()
    assert first._is_instrumented
    assert not second._is_instrumented
    assert _instrumentation._OWNERS == 1
    assert _modern.ModernAutoGenInstrumentor().is_instrumented_by_opentelemetry


@pytest.mark.asyncio
async def test_unstarted_stream_after_deactivation_is_inert(tracing, client_factory):
    _, exporter, plugin = tracing
    model, _ = await client_factory(stream=StreamFixture("complete"))
    stream = model.create_stream([UserMessage(content="fixture", source="user")])
    plugin.deactivate()
    result = [item async for item in stream]
    assert result[-1].content == "fixture answer"
    assert not exporter.get_finished_spans()
