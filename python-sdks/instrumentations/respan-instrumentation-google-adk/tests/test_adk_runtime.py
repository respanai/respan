"""Real ADK compatibility fixtures; only model responses are deterministic.

The Runner/session/RunConfig, Agent callbacks and Sequential/ParallelAgent graph
mirror johnson7788/MultiAgentPPT at ce8185cee83092290bdb913a528c6e3a72ee879e.
See backend/simpleOutline/{agent,adk_agent_executor}.py and
backend/slide_agent/slide_agent. No ADK or OpenInference modules are stubbed.
"""

import asyncio
import json

import pytest
from google.adk import Runner
from google.adk.agents import Agent, BaseAgent, ParallelAgent, SequentialAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import PrivateAttr

from respan_instrumentation_google_adk import GoogleADKInstrumentor
from respan_instrumentation_google_adk._processor import GOOGLE_ADK_SCOPE_NAME


class ScriptedModel(BaseLlm):
    """Use ADK's actual BaseLlm protocol with provider-shaped response objects."""

    model: str = "gemini-2.0-flash"
    use_tool: bool = True
    fail: bool = False
    _requests: list = PrivateAttr(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
        self._requests.append((llm_request.model_copy(deep=True), stream))
        if self.fail:
            raise RuntimeError("model transport failed")
        if self.use_tool and len(self._requests) == 1:
            parts = [
                types.Part(
                    function_call=types.FunctionCall(
                        id="search-1", name="document_search", args={"query": "slides"}
                    )
                )
            ]
        else:
            if stream:
                yield LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part(text="Outline ")]
                    ),
                    partial=True,
                )
            parts = [types.Part(text="Outline ready")]
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=11, candidates_token_count=3, total_token_count=14
            ),
        )


class SlideLoopAgent(BaseAgent):
    """The target also subclasses BaseAgent and delegates through run_async."""

    async def _run_async_impl(self, ctx):
        for agent in self.sub_agents:
            async for event in agent.run_async(ctx):
                yield event


@pytest.fixture
def capture(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    plugin = GoogleADKInstrumentor()
    plugin.activate()
    assert plugin._is_instrumented
    assert plugin._instrumentor.is_instrumented_by_opentelemetry
    try:
        yield plugin, provider, exporter
    finally:
        plugin.deactivate()
        provider.shutdown()


def make_agent(*, async_tool=False, fail_tool=False, fail_model=False, use_tool=True):
    def document_search(query: str) -> dict:
        """Find sources for an outline."""
        if fail_tool:
            raise RuntimeError("document search failed")
        return {"result": "Sources for " + query}

    async def async_document_search(query: str) -> dict:
        """Find sources for an outline."""
        await asyncio.sleep(0)
        return document_search(query)

    async_document_search.__name__ = "document_search"
    tool = async_document_search if async_tool else document_search
    model = ScriptedModel(fail=fail_model, use_tool=use_tool)
    callback_requests = []

    def before_model(callback_context, llm_request):
        callback_requests.append(llm_request.model)
        assert callback_context.state["metadata"] == {"topic": "slides"}

    agent = Agent(
        name="outline_agent",
        model=model,
        instruction="Use document_search.",
        tools=[tool] if use_tool else [],
        before_model_callback=before_model,
    )
    return agent, model, callback_requests


async def run_agent(agent, *, streaming=StreamingMode.NONE, runner_sync=False):
    sessions = InMemorySessionService()
    await sessions.create_session(
        app_name="ppt",
        user_id="self",
        session_id="session-1",
        state={"metadata": {"topic": "slides"}},
    )
    runner = Runner(agent=agent, app_name="ppt", session_service=sessions)
    kwargs = dict(
        user_id="self",
        session_id="session-1",
        new_message=types.Content(role="user", parts=[types.Part(text="Make slides")]),
        run_config=RunConfig(streaming_mode=streaming),
    )
    if runner_sync:
        # Runner.run drives the real async pipeline from its worker thread.
        return await asyncio.to_thread(lambda: list(runner.run(**kwargs)))
    return [event async for event in runner.run_async(**kwargs)]


def adk_spans(exporter):
    return [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope.name == GOOGLE_ADK_SCOPE_NAME
    ]


@pytest.mark.parametrize("streaming", [StreamingMode.NONE, StreamingMode.SSE])
@pytest.mark.parametrize("async_tool", [False, True])
def test_real_runner_agent_tool_roundtrip(capture, streaming, async_tool):
    _, _, exporter = capture
    agent, model, callbacks = make_agent(async_tool=async_tool)
    root = SequentialAgent(name="slides", sub_agents=[agent])
    events = asyncio.run(run_agent(root, streaming=streaming))
    assert events[-1].content.parts[0].text == "Outline ready"
    assert callbacks == [model.model, model.model]
    assert [stream for _, stream in model._requests] == [
        streaming == StreamingMode.SSE
    ] * 2
    spans = adk_spans(exporter)
    assert len(spans) == 6
    by_id = {span.context.span_id: span for span in spans}
    workflows = [
        s for s in spans if s.attributes.get("respan.entity.log_type") == "workflow"
    ]
    agents = [s for s in spans if s.attributes.get("respan.entity.log_type") == "agent"]
    chats = [s for s in spans if s.attributes.get("respan.entity.log_type") == "chat"]
    tools = [s for s in spans if s.attributes.get("respan.entity.log_type") == "tool"]
    assert len(workflows) == len(tools) == 1
    assert len(agents) == len(chats) == 2
    assert workflows[0].parent is None
    for span in spans:
        assert span.context.trace_id == workflows[0].context.trace_id
        assert span.status.status_code == StatusCode.OK
        assert span.attributes["respan.sessions.session_identifier"] == "session-1"
        assert "traceloop.span.kind" not in span.attributes
        assert not any(key.startswith("gcp.vertex.agent.") for key in span.attributes)
        if span.parent:
            assert span.parent.span_id in by_id
        if span in agents or span in workflows or span in tools:
            assert "gen_ai.request.model" not in span.attributes
            assert "gen_ai.completion.0.tool_calls" not in span.attributes
    for chat in chats:
        attrs = chat.attributes
        assert by_id[chat.parent.span_id] in agents
        assert attrs["gen_ai.request.model"] == model.model
        assert attrs["llm.request.type"] == "chat"
        assert attrs["gen_ai.system"] == "google"
        assert (
            attrs["gen_ai.usage.input_tokens"]
            == attrs["gen_ai.usage.prompt_tokens"]
            == 11
        )
        assert (
            attrs["gen_ai.usage.output_tokens"]
            == attrs["gen_ai.usage.completion_tokens"]
            == 3
        )
        assert attrs["llm.usage.total_tokens"] == 14
        assert attrs["gen_ai.prompt.1.content"] == "Make slides"
        assert (
            json.loads(attrs["llm.request.functions"])[0]["name"] == "document_search"
        )
    first, final = chats
    assert (
        json.loads(first.attributes["gen_ai.completion.0.tool_calls"])[0]["id"]
        == "search-1"
    )
    assert final.attributes["gen_ai.prompt.2.role"] == "assistant"
    assert final.attributes["gen_ai.prompt.3.role"] == "tool"
    assert json.loads(final.attributes["gen_ai.prompt.3.content"])["id"] == "search-1"
    assert final.attributes["gen_ai.completion.0.content"] == "Outline ready"
    assert "gen_ai.completion.0.tool_calls" not in final.attributes
    tool = tools[0]
    assert by_id[tool.parent.span_id] is first
    assert json.loads(tool.attributes["traceloop.entity.input"]) == {
        "name": "document_search",
        "arguments": {"query": "slides"},
    }
    assert json.loads(tool.attributes["traceloop.entity.output"])["response"] == {
        "result": "Sources for slides"
    }


@pytest.mark.parametrize(
    "failure,async_tool", [("model", False), ("tool", False), ("tool", True)]
)
def test_runtime_errors_propagate_and_close_spans(capture, failure, async_tool):
    _, _, exporter = capture
    agent, _, _ = make_agent(
        fail_model=failure == "model",
        fail_tool=failure == "tool",
        async_tool=async_tool,
    )
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(run_agent(agent))
    spans = adk_spans(exporter)
    assert spans
    assert all(s.end_time is not None for s in spans)
    workflow = next(
        s for s in spans if s.attributes.get("respan.entity.log_type") == "workflow"
    )
    assert workflow.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in workflow.events)
    assert not trace.get_current_span().get_span_context().is_valid
    if failure == "model":
        assert not any("gen_ai.completion.0.content" in s.attributes for s in spans)
        assert not any("gen_ai.usage.output_tokens" in s.attributes for s in spans)


def test_parallel_custom_agent_preserves_parents(capture):
    _, _, exporter = capture
    left, _, _ = make_agent(use_tool=False)
    right, _, _ = make_agent(use_tool=False)
    right.name = "slide_agent"
    parallel = ParallelAgent(name="parallel_slides", sub_agents=[left, right])
    root = SlideLoopAgent(name="slide_loop", sub_agents=[parallel])
    asyncio.run(run_agent(root))
    spans = adk_spans(exporter)
    assert len(spans) == 7
    agents = {
        s.attributes["traceloop.entity.name"]: s
        for s in spans
        if s.attributes.get("respan.entity.log_type") == "agent"
    }
    assert (
        agents["parallel_slides"].parent.span_id == agents["slide_loop"].context.span_id
    )
    for name in ("outline_agent", "slide_agent"):
        assert agents[name].parent.span_id == agents["parallel_slides"].context.span_id
    chat_parent_ids = {
        s.parent.span_id
        for s in spans
        if s.attributes.get("respan.entity.log_type") == "chat"
    }
    assert chat_parent_ids == {
        agents[name].context.span_id for name in ("outline_agent", "slide_agent")
    }


def test_sync_runner(capture):
    _, _, exporter = capture
    agent, _, _ = make_agent()
    events = asyncio.run(run_agent(agent, runner_sync=True))
    assert events[-1].content.parts[0].text == "Outline ready"
    assert len(adk_spans(exporter)) == 5


def test_suppression_and_deactivation(capture):
    plugin, provider, exporter = capture
    agent, _, _ = make_agent()
    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        asyncio.run(run_agent(agent))
    finally:
        context.detach(token)
    assert adk_spans(exporter) == []
    plugin.activate()
    assert (
        sum(
            p is plugin._processor
            for p in provider._active_span_processor._span_processors
        )
        == 1
    )
    plugin.deactivate()
    exporter.clear()
    agent, _, _ = make_agent()
    asyncio.run(run_agent(agent))
    assert adk_spans(exporter) == []
    assert not any(
        p.__class__.__name__ == "GoogleADKSpanProcessor"
        for p in provider._active_span_processor._span_processors
    )
    plugin.activate()
    exporter.clear()
    agent, _, _ = make_agent()
    asyncio.run(run_agent(agent))
    assert len(adk_spans(exporter)) == 5


def test_legacy_direct_iterator_alias_and_close(capture, caplog):
    from importlib.metadata import version
    from packaging.version import Version

    if Version(version("google-adk")) >= Version("1.17.0"):
        pytest.skip("Legacy direct iterator bridge is only used below ADK 1.17")
    # Imported before execution, as in MultiAgentPPT's DynamicParallelSearchAgent.
    from google.adk.agents.parallel_agent import _merge_agent_run
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event

    _, provider, exporter = capture
    closed = []
    advanced = []

    class YieldingAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            try:
                advanced.append(self.name)
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role="model", parts=[types.Part(text=self.name)]
                    ),
                )
                advanced.append(self.name + "_second")
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role="model", parts=[types.Part(text="second")]
                    ),
                )
            finally:
                closed.append(self.name)

    async def exercise():
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="ppt", user_id="self")
        agents = [YieldingAgent(name="left"), YieldingAgent(name="right")]
        with provider.get_tracer("test").start_as_current_span("parent") as parent:
            parent_id = parent.get_span_context().span_id
            runs = [
                agent.run_async(
                    InvocationContext(
                        invocation_id="test",
                        agent=agent,
                        session=session,
                        session_service=sessions,
                        run_config=RunConfig(),
                    )
                )
                for agent in agents
            ]
            merged = _merge_agent_run(runs)
            first = await merged.__anext__()
            assert first.author in {"left", "right"}
            assert not any(item.endswith("_second") for item in advanced)
            assert trace.get_current_span().get_span_context().span_id == parent_id
            await merged.aclose()
            for run in runs:
                await run.aclose()
            assert trace.get_current_span().get_span_context().span_id == parent_id
        return parent_id

    parent_id = asyncio.run(exercise())
    assert sorted(closed) == ["left", "right"]
    assert len(adk_spans(exporter)) == 2
    assert all(s.parent.span_id == parent_id for s in adk_spans(exporter))
    assert "Failed to detach context" not in caplog.text


def test_legacy_pending_iteration_cancellation(capture, caplog):
    from importlib.metadata import version
    from packaging.version import Version

    if Version(version("google-adk")) >= Version("1.17.0"):
        pytest.skip("Legacy direct iterator bridge is only used below ADK 1.17")
    from google.adk.agents.invocation_context import InvocationContext

    _, _, exporter = capture
    closed = []

    async def exercise():
        started = asyncio.Event()

        class WaitingAgent(BaseAgent):
            async def _run_async_impl(self, ctx):
                try:
                    started.set()
                    await asyncio.Event().wait()
                    yield
                finally:
                    closed.append(True)

        agent = WaitingAgent(name="waiting")
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="ppt", user_id="self")
        run = agent.run_async(
            InvocationContext(
                invocation_id="test",
                agent=agent,
                session=session,
                session_service=sessions,
                run_config=RunConfig(),
            )
        )
        pending = asyncio.create_task(run.__anext__())
        await started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await run.aclose()
        assert not trace.get_current_span().get_span_context().is_valid

    asyncio.run(exercise())
    assert closed == [True]
    assert len(adk_spans(exporter)) == 1
    assert "Failed to detach context" not in caplog.text


def test_legacy_async_generator_methods(capture, caplog):
    from importlib.metadata import version
    from packaging.version import Version

    if Version(version("google-adk")) >= Version("1.17.0"):
        pytest.skip("Legacy direct iterator bridge is only used below ADK 1.17")
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event

    _, _, exporter = capture

    class OneEventAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text="one")]),
            )

    async def exercise():
        agent = OneEventAgent(name="one_event")
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="ppt", user_id="self")
        run = agent.run_async(
            InvocationContext(
                invocation_id="test",
                agent=agent,
                session=session,
                session_service=sessions,
                run_config=RunConfig(),
            )
        )
        assert (await run.asend(None)).author == "one_event"
        with pytest.raises(ValueError, match="consumer failed"):
            await run.athrow(ValueError("consumer failed"))
        await run.aclose()
        with pytest.raises(StopAsyncIteration):
            await run.__anext__()

    asyncio.run(exercise())
    assert len(adk_spans(exporter)) == 1
    assert adk_spans(exporter)[0].status.status_code == StatusCode.ERROR
    assert "Failed to detach context" not in caplog.text


def test_second_adapter_does_not_take_ownership(capture):
    owner, provider, exporter = capture
    second = GoogleADKInstrumentor()
    second.activate()
    assert not second._is_instrumented
    assert second._instrumentor is None
    assert second._processor is None
    second.deactivate()
    assert owner._instrumentor.is_instrumented_by_opentelemetry
    assert (
        sum(
            p is owner._processor
            for p in provider._active_span_processor._span_processors
        )
        == 1
    )
    agent, _, _ = make_agent()
    asyncio.run(run_agent(agent))
    assert len(adk_spans(exporter)) == 5


def test_external_upstream_instrumentation_is_preserved(capture):
    from openinference.instrumentation.google_adk import (
        GoogleADKInstrumentor as Upstream,
    )

    plugin, provider, _ = capture
    plugin.deactivate()
    external = Upstream()
    external.instrument(tracer_provider=provider)
    try:
        plugin.activate()
        assert not plugin._is_instrumented
        assert plugin._instrumentor is None
        assert plugin._processor is None
        plugin.deactivate()
        assert external.is_instrumented_by_opentelemetry
    finally:
        external.uninstrument()


@pytest.mark.parametrize("parallel", [False, True])
def test_legacy_abandoned_runner_closes_in_owner_context(capture, caplog, parallel):
    from importlib.metadata import version
    from packaging.version import Version

    if Version(version("google-adk")) >= Version("1.17.0"):
        pytest.skip("Legacy iterator shutdown bridge is only used below ADK 1.17")
    _, _, exporter = capture
    left, left_model, _ = make_agent()
    if parallel:
        right, _, _ = make_agent()
        right.name = "slide_agent"
        root = ParallelAgent(name="slides", sub_agents=[left, right])
    else:
        root = left

    async def exercise():
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name="ppt",
            user_id="self",
            session_id="session-1",
            state={"metadata": {"topic": "slides"}},
        )
        runner = Runner(agent=root, app_name="ppt", session_service=sessions)
        # MultiAgentPPT's executor breaks once it receives its final response.
        # Also exercise stopping on a tool request, while model spans are open.
        async for _ in runner.run_async(
            user_id="self",
            session_id="session-1",
            new_message=types.Content(
                role="user", parts=[types.Part(text="Make slides")]
            ),
        ):
            break

    asyncio.run(exercise())
    assert len(left_model._requests) == 1
    spans = adk_spans(exporter)
    assert len(spans) == (6 if parallel else 3)
    assert all(s.end_time is not None for s in spans)
    assert not any(s.attributes.get("respan.entity.log_type") == "tool" for s in spans)
    assert "Failed to detach context" not in caplog.text
    assert "Task was destroyed" not in caplog.text
