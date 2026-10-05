"""Current released SDK privacy, iterator, sampler and Workflow contracts."""

import asyncio
import json
from importlib.metadata import version

import pytest
from google.adk import Runner
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.sessions import InMemorySessionService
from google.genai import types
from openinference.instrumentation import TraceConfig
from opentelemetry import context, trace
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from . import test_adk_runtime
from .test_adk_runtime import (
    adk_spans,
    make_agent,
    run_agent,
)


@pytest.fixture(name="capture")
def current_capture(monkeypatch):
    yield from test_adk_runtime.capture.__wrapped__(monkeypatch)


LATEST = tuple(int(x) for x in version("google-adk").split(".")[:2]) >= (2, 10)


@pytest.mark.parametrize("policy", ["environment", "context", "config"])
def test_content_policy_preserves_usage_without_payloads(capture, monkeypatch, policy):
    plugin, _, exporter = capture
    token = None
    if policy == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    elif policy == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        plugin.deactivate()
        plugin._instrumentor_kwargs["config"] = TraceConfig(
            hide_inputs=True, hide_outputs=True
        )
        plugin.activate()
    try:
        agent, _, _ = make_agent()
        asyncio.run(run_agent(agent, streaming=StreamingMode.SSE))
    finally:
        if token:
            context.detach(token)
    spans = adk_spans(exporter)
    assert spans
    rendered = json.dumps([dict(s.attributes) for s in spans])
    assert "Make slides" not in rendered
    assert "Sources for" not in rendered
    assert "Outline ready" not in rendered
    chats = [s for s in spans if s.attributes.get("respan.entity.log_type") == "chat"]
    assert len(chats) == 2
    assert all(s.attributes["gen_ai.usage.input_tokens"] == 11 for s in chats)
    assert all(
        "traceloop.entity.input" not in s.attributes
        and "traceloop.entity.output" not in s.attributes
        for s in spans
    )


def test_iterator_yield_close_and_sampler_state(capture, caplog):
    plugin, provider, exporter = capture
    sampler = provider.sampler

    async def scenario():
        agent, _, _ = make_agent(use_tool=False)
        sessions = InMemorySessionService()
        session = await sessions.create_session(
            app_name="ppt", user_id="self", state={"metadata": {"topic": "slides"}}
        )
        runner = Runner(agent=agent, app_name="ppt", session_service=sessions)
        with provider.get_tracer("caller").start_as_current_span("parent") as parent:
            stream = runner.run_async(
                user_id="self",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="private stream")]
                ),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
            await anext(stream)
            assert trace.get_current_span() is parent
            await stream.aclose()
            assert trace.get_current_span() is parent

    asyncio.run(scenario())
    assert adk_spans(exporter)
    assert provider.sampler is sampler
    plugin.deactivate()
    assert provider.sampler is sampler
    assert "Failed to detach context" not in caplog.text


@pytest.mark.skipif(not LATEST, reason="Workflow nodes require ADK2.10+")
def test_workflow_nodes_and_tool_confirmation(capture):
    from google.adk.tools import FunctionTool
    from google.adk.workflow import START, Workflow, node
    from google.adk.workflow.utils._workflow_hitl_utils import (
        create_request_input_response,
    )

    _, _, exporter = capture
    executed = []

    def greet(name: str) -> dict:
        """Return a synthetic greeting."""
        executed.append(name)
        return {"greeting": "hello " + name}

    tool = FunctionTool(greet, require_confirmation=True)
    tool_node = node(tool)
    workflow = Workflow(name="fixture_workflow", edges=[(START, tool_node)])

    async def scenario():
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="fixture", user_id="user")
        runner = Runner(node=workflow, app_name="fixture", session_service=sessions)
        events = [
            x
            async for x in runner.run_async(
                user_id="user",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text='{"name":"Ada"}')]
                ),
            )
        ]
        assert executed == []
        assert not [
            s
            for s in adk_spans(exporter)
            if s.attributes.get("respan.entity.log_type") == "tool"
        ]
        call = next(call for event in events for call in event.get_function_calls())
        return [
            x
            async for x in runner.run_async(
                user_id="user",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[create_request_input_response(call.id, {"confirmed": True})],
                ),
            )
        ]

    result = asyncio.run(scenario())
    assert executed == ["Ada"]
    assert any(x.output == {"greeting": "hello Ada"} for x in result)
    spans = adk_spans(exporter)
    tools = [s for s in spans if s.attributes.get("respan.entity.log_type") == "tool"]
    assert len(tools) == 1
    assert tools[0].attributes["gen_ai.tool.call.id"]
    assert json.loads(tools[0].attributes["traceloop.entity.output"]) == {
        "greeting": "hello Ada"
    }
    assert not any(
        s.instrumentation_scope.name != "openinference.instrumentation.google_adk"
        for s in exporter.get_finished_spans()
    )


@pytest.mark.skipif(not LATEST, reason="Workflow/abort APIs require ADK2.10+")
def test_workflow_abort_closes_spans(capture):
    from google.adk.workflow import START, Workflow, node

    _, _, exporter = capture

    async def scenario():
        started = asyncio.Event()
        abort = asyncio.Event()

        @node
        async def waiting(node_input: str):
            started.set()
            await asyncio.Event().wait()
            return "never"

        workflow = Workflow(name="abort_workflow", edges=[(START, waiting)])
        sessions = InMemorySessionService()
        session = await sessions.create_session(app_name="fixture", user_id="user")
        runner = Runner(node=workflow, app_name="fixture", session_service=sessions)

        async def consume():
            return [
                x
                async for x in runner.run_async(
                    user_id="user",
                    session_id=session.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text="abort fixture")]
                    ),
                    abort_signal=abort,
                )
            ]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), 2)
        abort.set()
        await asyncio.wait_for(task, 3)

    asyncio.run(scenario())
    spans = adk_spans(exporter)
    assert spans
    assert all(s.end_time is not None for s in spans)
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.skipif(not LATEST, reason="ModelConsultTool requires ADK2.11")
@pytest.mark.parametrize("reported_usage", [True, False])
def test_model_consult_uses_real_response_usage(capture, reported_usage):
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.model_consult._advisor import call_advisor

    _, provider, exporter = capture

    class Advisor(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="synthetic advice")]
                ),
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=5, candidates_token_count=2, total_token_count=7
                )
                if reported_usage
                else None,
            )

    async def scenario():
        with provider.get_tracer("caller").start_as_current_span("parent"):
            result = await call_advisor(
                Advisor(model="fixture-advisor"),
                [
                    types.Content(
                        role="user", parts=[types.Part(text="synthetic question")]
                    )
                ],
                system_instruction="Synthetic advisor",
            )
            assert result.text == "synthetic advice"

    asyncio.run(scenario())
    chats = [
        s
        for s in adk_spans(exporter)
        if s.attributes.get("respan.entity.log_type") == "chat"
    ]
    assert len(chats) == 1
    attrs = chats[0].attributes
    assert attrs["gen_ai.request.model"] == "fixture-advisor"
    assert attrs["gen_ai.completion.0.content"] == "synthetic advice"
    if reported_usage:
        assert attrs["gen_ai.usage.input_tokens"] == 5
    else:
        assert "gen_ai.usage.input_tokens" not in attrs


def test_later_runner_wrapper_survives_teardown_without_retained_tracing(capture):
    import inspect

    from wrapt import FunctionWrapper

    plugin, _, exporter = capture
    owned = inspect.getattr_static(Runner, "run_async")
    calls = []

    def foreign(wrapped, instance, args, kwargs):
        calls.append(True)
        return wrapped(*args, **kwargs)

    replacement = FunctionWrapper(owned, foreign)
    Runner.run_async = replacement
    try:
        plugin.deactivate()
        assert inspect.getattr_static(Runner, "run_async") is replacement
        exporter.clear()
        agent, _, _ = make_agent(use_tool=False)
        asyncio.run(run_agent(agent))
        assert calls and not adk_spans(exporter)
    finally:
        # The target binding is identified by owner/name (tracer is another binding).
        while hasattr(owned, "__wrapped__"):
            owned = owned.__wrapped__
        Runner.run_async = owned


def test_partial_activation_restores_hooks_and_provider(capture, monkeypatch):
    import inspect

    import respan_instrumentation_google_adk._instrumentation as instrumentation
    from google.adk.agents import BaseAgent

    plugin, provider, _ = capture
    plugin.deactivate()
    originals = [
        inspect.getattr_static(cls, "run_async") for cls in (Runner, BaseAgent)
    ]
    sampler = provider.sampler

    def failure(*args):
        raise RuntimeError("synthetic activation failure")

    monkeypatch.setattr(instrumentation, "patch_legacy_agent_iterator", failure)
    plugin.activate()
    assert not plugin._is_instrumented
    assert provider.sampler is sampler
    assert all(
        inspect.getattr_static(cls, "run_async") is old
        for cls, old in zip((Runner, BaseAgent), originals)
    )
    assert not any(
        type(p).__name__ == "GoogleADKSpanProcessor"
        for p in provider._active_span_processor._span_processors
    )


@pytest.mark.skipif(not LATEST, reason="ModelConsult requires ADK2.11")
@pytest.mark.parametrize("policy", ["environment", "context"])
def test_advisor_start_policy_survives_model_await(capture, monkeypatch, policy):
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.model_consult._advisor import call_advisor

    _, _, exporter = capture

    class Advisor(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
            # Simulate a model adapter changing its own context during the await.
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, True))
            try:
                yield LlmResponse(
                    content=types.Content(
                        role="model", parts=[types.Part(text="private advice")]
                    ),
                    usage_metadata=types.GenerateContentResponseUsageMetadata(
                        prompt_token_count=5,
                        candidates_token_count=2,
                        total_token_count=7,
                    ),
                )
            finally:
                context.detach(token)

    async def scenario():
        token = None
        if policy == "environment":
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        else:
            token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        try:
            result = await call_advisor(
                Advisor(model="fixture-advisor"),
                [
                    types.Content(
                        role="user", parts=[types.Part(text="private question")]
                    )
                ],
                system_instruction="private instruction",
            )
            assert result.text == "private advice"
        finally:
            if token:
                context.detach(token)

    asyncio.run(scenario())
    chats = [
        s
        for s in adk_spans(exporter)
        if s.attributes.get("respan.entity.log_type") == "chat"
    ]
    assert len(chats) == 1
    assert "private" not in json.dumps(dict(chats[0].attributes))
    assert chats[0].attributes["gen_ai.usage.input_tokens"] == 5


def test_conflicting_owner_privacy_configuration_is_rejected(capture):
    from respan_instrumentation_google_adk import GoogleADKInstrumentor

    owner, _, _ = capture
    second = GoogleADKInstrumentor(config=TraceConfig(hide_inputs=True))
    with pytest.raises(ValueError, match="different configuration"):
        second.activate()
    assert not second._is_instrumented
    assert owner._instrumentor.is_instrumented_by_opentelemetry
    assert GoogleADKInstrumentor._owner_count == 1


@pytest.mark.skipif(not LATEST, reason="Workflow tools require ADK2.11")
def test_workflow_telemetry_preserves_unserializable_tool_results(capture):
    from types import SimpleNamespace

    from google.adk.tools import BaseTool
    from google.adk.workflow import node
    from pydantic import BaseModel
    from respan_instrumentation_google_adk._workflow import _NODE_TOOL, _json

    class Hostile(BaseModel):
        def model_dump(self, **kwargs):
            raise ValueError("hostile serialization hook")

    circular = {}
    circular["self"] = circular
    values = [circular, {("tuple", "key"): "value"}, Hostile()]
    failure = RuntimeError("original SDK error")

    class FixtureTool(BaseTool):
        async def run_async(self, *, args, tool_context):
            if args.get("fail"):
                raise failure
            return args["value"]

    tool = FixtureTool(name="fixture", description="synthetic tool")

    async def scenario():
        # Creating the real ToolNode iterator registers its concrete tool class;
        # close before consumption, then exercise that actual SDK method.
        pending = node(tool).run(ctx=None, node_input=circular)
        await pending.aclose()
        token = _NODE_TOOL.set(tool)
        try:
            for value in values:
                assert (
                    await tool.run_async(
                        args={"value": value},
                        tool_context=SimpleNamespace(function_call_id="call-fixture"),
                    )
                    is value
                )
                assert len(_json(value)) <= 16000
            with pytest.raises(RuntimeError) as exc:
                await tool.run_async(
                    args={"fail": True},
                    tool_context=SimpleNamespace(function_call_id="call-fixture"),
                )
            assert exc.value is failure
        finally:
            _NODE_TOOL.reset(token)

    asyncio.run(scenario())


def test_explicit_iterator_close_preserves_native_finally_error():
    from respan_instrumentation_google_adk._compat import _ContextPreservingIterator

    error = RuntimeError("provider close failure")

    async def source():
        try:
            yield 1
        finally:
            raise error

    async def scenario():
        for iterator in (source(), _ContextPreservingIterator(source())):
            assert await anext(iterator) == 1
            with pytest.raises(RuntimeError) as caught:
                await iterator.aclose()
            assert caught.value is error

    asyncio.run(scenario())


def test_iterator_cleanup_does_not_mask_iteration_error():
    from respan_instrumentation_google_adk._compat import _ContextPreservingIterator

    error = RuntimeError("original iteration failure")

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

        async def aclose(self):
            raise ValueError("cleanup failure")

    async def scenario():
        with pytest.raises(RuntimeError) as caught:
            await anext(_ContextPreservingIterator(Source()))
        assert caught.value is error

    asyncio.run(scenario())


def test_iterator_policy_is_snapshotted_before_first_event(capture, monkeypatch):
    _, _, exporter = capture

    async def scenario():
        agent, _, _ = make_agent(use_tool=False)
        sessions = InMemorySessionService()
        session = await sessions.create_session(
            app_name="ppt", user_id="self", state={"metadata": {"topic": "slides"}}
        )
        runner = Runner(agent=agent, app_name="ppt", session_service=sessions)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        iterator = runner.run_async(
            user_id="self",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text="private before consumption")]
            ),
        )
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        assert [event async for event in iterator]

    asyncio.run(scenario())
    rendered = json.dumps([dict(span.attributes) for span in adk_spans(exporter)])
    assert "private before consumption" not in rendered
    assert "Outline ready" not in rendered


@pytest.mark.skipif(not LATEST, reason="ModelConsult requires ADK2.11")
def test_advisor_multiple_final_messages_match_public_result(capture):
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.model_consult._advisor import call_advisor

    _, _, exporter = capture

    class Advisor(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="partial ignored")]
                ),
                partial=True,
            )
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="First fixture. ")]
                ),
                partial=False,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=5, candidates_token_count=2, total_token_count=7
                ),
            )
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="hidden thought", thought=True),
                        types.Part(text="Second fixture."),
                    ],
                ),
                partial=False,
                model_version="actual-advisor",
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=5, candidates_token_count=4, total_token_count=9
                ),
            )

    result = asyncio.run(
        call_advisor(
            Advisor(model="fixture-advisor"),
            [types.Content(role="user", parts=[types.Part(text="synthetic question")])],
            system_instruction="fixture",
        )
    )
    assert result.text == "First fixture. Second fixture."
    attrs = adk_spans(exporter)[0].attributes
    assert attrs["gen_ai.completion.0.content"] == result.text
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 4
    assert json.loads(attrs["traceloop.entity.output"])["content"]["parts"] == [
        {"text": result.text}
    ]
