import asyncio
import importlib
import json

import pytest
from agno.agent import Agent
from agno.team import Team
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_agno import AgnoInstrumentor, _otel_emitter
from respan_instrumentation_agno._serialization import json_string
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE

pytest.importorskip(
    "openai", reason="Released OpenAI model client is needed for protocol fixtures"
)
_fixture = importlib.import_module("_fixture_model")
MODEL = _fixture.MODEL
model = _fixture.model


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setenv("AGNO_TELEMETRY", "false")
    spans = []
    provider = TracerProvider()
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(_otel_emitter, "inject_span", lambda span: spans.append(span))
    instrumentor = AgnoInstrumentor()
    instrumentor.activate()
    yield provider, spans
    instrumentor.deactivate()
    provider.shutdown()


def weather(city: str) -> str:
    """Return fixture weather for a city."""
    return f"Sunny in {city}"


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_released_agent_tools_stream_usage_and_parentage(captured, async_mode, stream):
    provider, spans = captured
    agent = Agent(name="Fixture", model=model(), tools=[weather], telemetry=False)
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        if async_mode:

            async def run():
                result = agent.arun("weather", stream=stream)
                if stream:
                    items = []
                    async for item in result:
                        assert _otel_emitter._CURRENT_RUN_CONTEXT.get() is None
                        items.append(item)
                    assert all(getattr(item, "event", None) for item in items)
                    return items
                return await result

            asyncio.run(run())
        else:
            result = agent.run("weather", stream=stream)
            if stream:
                for item in result:
                    assert _otel_emitter._CURRENT_RUN_CONTEXT.get() is None
                    assert getattr(item, "event", None)
    root = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "agent")
    chats = [s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "chat"]
    assert len(chats) == 2
    chat = chats[-1]
    assert (
        json.loads(
            chats[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
        )[0]["function"]["name"]
        == "weather"
    )
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls" not in chats[-1].attributes
    tool = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "tool")
    assert root.parent.span_id == parent.context.span_id
    assert chat.parent.span_id == tool.parent.span_id == root.context.span_id
    assert chat.attributes[SpanAttributes.LLM_REQUEST_MODEL] == MODEL
    assert chat.attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert chat.attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 4
    assert chat.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    assert (
        "Fixture Agno response"
        in chat.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    assert tool.attributes[GenAI.GEN_AI_TOOL_CALL_ID] == "call-fixture"
    assert "Sunny in Paris" in tool.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    assert "fixture-provider-secret" not in json.dumps(
        [dict(s.attributes) for s in spans]
    )
    assert all(SpanAttributes.TRACELOOP_SPAN_KIND not in s.attributes for s in spans)


def test_released_team_delegation(captured):
    _, spans = captured
    member = Agent(id="member", name="Member", model=model(), telemetry=False)
    team = Team(name="Fixture Team", members=[member], model=model(), telemetry=False)
    result = team.run("delegate the fixture task")
    assert result.content
    roots = [s for s in spans if s.attributes[RESPAN_LOG_TYPE] in {"workflow", "agent"}]
    team_span = next(s for s in roots if s.attributes[RESPAN_LOG_TYPE] == "workflow")
    member_span = next(s for s in roots if s.attributes[RESPAN_LOG_TYPE] == "agent")
    assert member_span.parent.span_id == team_span.context.span_id
    assert len({s.context.span_id for s in spans}) == len(spans)


def test_released_stream_early_close_emits_partial_output(captured):
    _, spans = captured
    agent = Agent(name="Early", model=model(), telemetry=False)
    stream = agent.run("hello", stream=True)
    next(stream)
    stream.close()
    roots = [s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "agent"]
    assert len(roots) == 1
    assert (
        roots[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        == "Fixture Agno response."
    )
    assert _otel_emitter._CURRENT_RUN_CONTEXT.get() is None


def test_shared_ownership_retains_patch_until_final_owner():
    original = Agent.run
    first, second = AgnoInstrumentor(), AgnoInstrumentor()
    first.activate()
    second.activate()
    wrapped = Agent.run
    first.deactivate()
    assert Agent.run is wrapped
    second.deactivate()
    assert Agent.run is original


def test_bounded_serialization_redacts_credentials_without_repr():
    class Hostile:
        def __str__(self):
            raise AssertionError("must not stringify arbitrary objects")

    payload = json_string(
        {"api_key": "secret-value", "object": Hostile(), "items": list(range(1000))}
    )
    assert "secret-value" not in payload
    assert len(payload) < 16384
    assert len(json.loads(payload)["items"]) == 64


@pytest.mark.parametrize("async_mode", [False, True])
def test_released_confirmation_continuation_does_not_repeat_previous_turn(
    captured, async_mode
):
    from agno.tools import tool

    _, spans = captured
    approved_weather = tool(requires_confirmation=True)(weather)
    agent = Agent(
        name="Confirm", model=model(), tools=[approved_weather], telemetry=False
    )

    async def run():
        paused = await agent.arun("weather") if async_mode else agent.run("weather")
        assert paused.is_paused
        assert not any(s.attributes[RESPAN_LOG_TYPE] == "tool" for s in spans)
        for requirement in paused.requirements:
            requirement.confirm()
        completed = (
            await agent.acontinue_run(run_response=paused)
            if async_mode
            else agent.continue_run(run_response=paused)
        )
        assert completed.content == "Fixture Agno response."

    asyncio.run(run())
    chats = [s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "chat"]
    assert len(chats) == 2
    assert len([s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "tool"]) == 1


def test_released_async_stream_close_and_cancellation(captured):
    import httpx
    from agno.models.openai import OpenAIChat
    from openai import AsyncOpenAI

    _, spans = captured
    closed = []

    class BlockingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunk = {
                "id": "fixture",
                "model": MODEL,
                "created": 1,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "partial"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def handler(request):
        return httpx.Response(
            200, stream=BlockingStream(), headers={"content-type": "text/event-stream"}
        )

    async def run():
        agent = Agent(name="Async Early", model=model(), telemetry=False)
        early = agent.arun("hello", stream=True)
        await anext(early)
        await early.aclose()
        assert _otel_emitter._CURRENT_RUN_CONTEXT.get() is None
        blocking_model = OpenAIChat(
            id=MODEL,
            async_client=AsyncOpenAI(
                api_key="fixture",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ),
        )
        agent = Agent(name="Cancelled", model=blocking_model, telemetry=False)
        stream = agent.arun("hello", stream=True)
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        pending.cancel("expected Agno cancellation")
        with pytest.raises(asyncio.CancelledError, match="expected Agno cancellation"):
            await pending
        await stream.aclose()
        assert _otel_emitter._CURRENT_RUN_CONTEXT.get() is None

    asyncio.run(run())
    roots = [s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "agent"]
    assert len(roots) == 2
    assert (
        roots[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        == "Fixture Agno response."
    )
    assert roots[-1].status.status_code is trace.StatusCode.ERROR
    assert closed


def test_explicit_final_output_is_preserved(captured):
    _, spans = captured
    from agno.run.agent import RunOutput

    agent = Agent(model=model(), telemetry=False)
    events = list(agent.run("hello", stream=True, yield_run_output=True))
    assert isinstance(events[-1], RunOutput)
    assert events[-1].content == "Fixture Agno response."
    assert len([s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "agent"]) == 1


@pytest.mark.parametrize("async_mode", [False, True])
def test_released_continue_by_run_id_does_not_replay_history(captured, async_mode):
    from agno.db.in_memory import InMemoryDb
    from agno.tools import tool

    _, spans = captured
    agent = Agent(
        name="Stored Confirm",
        model=model(),
        tools=[tool(requires_confirmation=True)(weather)],
        db=InMemoryDb(),
        session_id="fixture-session",
        telemetry=False,
    )

    async def run():
        paused = await agent.arun("weather") if async_mode else agent.run("weather")
        for requirement in paused.requirements:
            requirement.confirm()
        kwargs = {
            "run_id": paused.run_id,
            "session_id": "fixture-session",
            "requirements": paused.requirements,
        }
        result = (
            await agent.acontinue_run(**kwargs)
            if async_mode
            else agent.continue_run(**kwargs)
        )
        assert result.content

    asyncio.run(run())
    assert len([s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "chat"]) == 2


@pytest.mark.parametrize("team_mode", [False, True])
def test_global_and_explicit_instance_owners_are_independent(monkeypatch, team_mode):
    spans = []
    monkeypatch.setattr(_otel_emitter, "inject_span", lambda span: spans.append(span))
    target = (
        Team(name="Owned Team", members=[], model=model(), telemetry=False)
        if team_mode
        else Agent(name="Owned Agent", model=model(), telemetry=False)
    )
    global_owner = AgnoInstrumentor()
    local_owner = AgnoInstrumentor(agent=target)
    global_owner.activate()
    local_owner.activate()
    assert local_owner._is_instrumented
    try:
        target.run("hello")
        root_type = "workflow" if team_mode else "agent"
        assert (
            len([s for s in spans if s.attributes[RESPAN_LOG_TYPE] == root_type]) == 1
        )
        spans.clear()
        global_owner.deactivate()
        target.run("hello")
        assert (
            len([s for s in spans if s.attributes[RESPAN_LOG_TYPE] == root_type]) == 1
        )
    finally:
        local_owner.deactivate()
        global_owner.deactivate()
    assert "run" not in vars(target)
