"""Actual CrewAI agent/flow/native-provider APIs; only HTTP transport is mocked."""

import asyncio
import json
from functools import wraps

import pytest
from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent
from crewai.flow.flow import Flow, listen, start
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.tools import tool
from openai import BadRequestError
from opentelemetry import context, trace
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.semconv_ai import SpanAttributes as A
from respan_instrumentation_crewai import CrewAIInstrumentor
from respan_instrumentation_crewai._event_listener import CrewAIEventListener
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._fixtures import FixtureServer


@pytest.fixture
def runtime(monkeypatch, request):
    provider = TracerProvider(
        **(
            {"sampler": ALWAYS_OFF}
            if getattr(request, "param", None) == "unsampled"
            else {}
        )
    )
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    owners = []

    def activate():
        owner = CrewAIInstrumentor()
        owner.activate()
        assert owner._is_instrumented
        owners.append(owner)
        return owner

    yield activate, exporter, provider
    assert crewai_event_bus.flush()
    for owner in reversed(owners):
        owner.deactivate()
    provider.shutdown()


def finished(exporter, kind=None):
    assert crewai_event_bus.flush()
    return [
        span
        for span in exporter.get_finished_spans()
        if kind is None or span.attributes.get(RESPAN_LOG_TYPE) == kind
    ]


@pytest.mark.parametrize("api", ["completions", "responses"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_provider_apis_usage_response_id_and_original_results(
    runtime, api, asynchronous
):
    activate, exporter, _ = runtime
    activate()
    llm = FixtureServer().llm(api=api)
    before = context.get_current()
    result = (
        asyncio.run(llm.acall("fixture request"))
        if asynchronous
        else llm.call("fixture request")
    )
    assert result == "fixture answer" and context.get_current() is before
    spans = finished(exporter, "chat")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[G.GEN_AI_USAGE_INPUT_TOKENS] == 9
    assert attrs[G.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert attrs[A.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4
    if "response_id" in LLMCallCompletedEvent.model_fields:
        assert attrs[G.GEN_AI_RESPONSE_ID] == (
            "fixture-response-id" if api == "responses" else "fixture-chat-id"
        )
    assert "fixture request" in attrs[A.TRACELOOP_ENTITY_INPUT]
    assert "fixture answer" in attrs[A.TRACELOOP_ENTITY_OUTPUT]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_stream_zero_usage_and_context(runtime, asynchronous):
    activate, exporter, _ = runtime
    owner = activate()
    llm = FixtureServer().llm(stream=True)
    before = context.get_current()
    result = (
        asyncio.run(llm.acall("fixture stream"))
        if asynchronous
        else llm.call("fixture stream")
    )
    assert result == "fixture stream" and context.get_current() is before
    spans = finished(exporter, "chat")
    assert len(spans) == 1
    assert spans[0].attributes[G.GEN_AI_USAGE_INPUT_TOKENS] == 0
    assert spans[0].attributes[G.GEN_AI_USAGE_OUTPUT_TOKENS] == 0
    assert (
        not owner._listener._assembler._open_spans
        and not owner._listener._usage_by_call_id
    )


@tool("fixture_weather")
def fixture_weather(city: str) -> str:
    """Return deterministic weather for a city."""
    return f"{city}: clear"


def test_native_crew_and_agent_tool_history_and_parentage(runtime):
    activate, exporter, provider = runtime
    activate()
    llm = FixtureServer().llm()
    agent = Agent(
        role="FixtureAgent",
        goal="Answer fixture questions",
        backstory="Fixture",
        llm=llm,
        tools=[fixture_weather],
        verbose=False,
    )
    task = Task(
        name="FixtureTask",
        description="Use the weather tool for Tokyo",
        expected_output="An answer",
        agent=agent,
    )
    crew = Crew(name="FixtureCrew", agents=[agent], tasks=[task], verbose=False)
    with provider.get_tracer("fixture.root").start_as_current_span("fixture.root"):
        assert crew.kickoff().raw == "fixture answer"
    spans = finished(exporter)
    chats = finished(exporter, "chat")
    tools = finished(exporter, "tool")
    assert len(chats) == 2 and len(tools) == 1
    first, second = [span.attributes for span in chats]
    assert (
        json.loads(first[f"{A.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
        == "fixture-tool-id"
    )
    assert tools[0].attributes[G.GEN_AI_TOOL_CALL_ID] == "fixture-tool-id"
    assert any(
        key.endswith(".tool_call_id") and value == "fixture-tool-id"
        for key, value in second.items()
    )
    assert not any(
        key.startswith(f"{A.LLM_COMPLETIONS}.") and key.endswith(".tool_calls")
        for key in second
    )
    ids = {span.context.span_id for span in spans}
    assert all(span.parent is None or span.parent.span_id in ids for span in spans)
    assert (
        len(finished(exporter, "agent"))
        == len(finished(exporter, "task"))
        == len(finished(exporter, "workflow"))
        == 1
    )


def test_native_flow_methods_success_and_failure(runtime):
    activate, exporter, _ = runtime
    owner = activate()
    llm = FixtureServer().llm()

    class FixtureFlow(Flow):
        @start()
        def begin(self):
            return llm.call("flow fixture")

        @listen(begin)
        def finish(self, value):
            return value

    assert FixtureFlow().kickoff() == "fixture answer"
    spans = finished(exporter)
    assert len(spans) == 4
    assert (
        len(finished(exporter, "workflow")) == 1
        and len(finished(exporter, "task")) == 2
    )
    exporter.clear()

    class FailureFlow(Flow):
        @start()
        def fail(self):
            raise ValueError("controlled flow failure")

    with pytest.raises(ValueError, match="controlled flow failure"):
        FailureFlow().kickoff()
    finished(exporter)
    owner.deactivate()
    errors = exporter.get_finished_spans()
    assert len(errors) == 2 and all(
        s.status.status_code is trace.StatusCode.ERROR for s in errors
    )
    assert all("http.response.status_code" not in s.attributes for s in errors)


@pytest.mark.parametrize(
    "mode", ["environment", "context", "false_to_true", "true_to_false"]
)
def test_content_policy_at_emission_and_completion(runtime, monkeypatch, mode):
    activate, exporter, _ = runtime
    activate()
    server = FixtureServer()
    if mode in {"environment", "false_to_true"}:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    if mode in {"false_to_true", "true_to_false"}:
        server.before_response = lambda: monkeypatch.setenv(
            "TRACELOOP_TRACE_CONTENT", "true" if mode == "false_to_true" else "false"
        )
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if mode == "context"
        else None
    )
    try:
        assert (
            asyncio.run(server.llm().acall("private fixture request"))
            == "fixture answer"
        )
    finally:
        if token is not None:
            context.detach(token)
    spans = finished(exporter)
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert (
        A.TRACELOOP_ENTITY_INPUT not in attrs and A.TRACELOOP_ENTITY_OUTPUT not in attrs
    )
    assert not any(
        key.startswith((f"{A.LLM_PROMPTS}.", f"{A.LLM_COMPLETIONS}.")) for key in attrs
    )
    assert attrs[G.GEN_AI_USAGE_INPUT_TOKENS] == 9


def test_failed_native_model_has_error_without_fabricated_output_or_usage(runtime):
    activate, exporter, _ = runtime
    activate()
    with pytest.raises(BadRequestError, match="controlled fixture failure"):
        FixtureServer().llm().call("fixture-failure")
    spans = finished(exporter)
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes["error.message"]
    assert A.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert not any(
        key.startswith(("gen_ai.usage.", "llm.usage.", f"{A.LLM_COMPLETIONS}."))
        for key in span.attributes
    )


def test_suppression_finishes_ignored_state(runtime):
    activate, exporter, _ = runtime
    owner = activate()
    with suppress_instrumentation():
        assert FixtureServer().llm().call("suppressed") == "fixture answer"
    assert not finished(exporter)
    assert not owner._listener._assembler._open_spans
    assert not owner._listener._assembler._pending_ends
    assert not owner._listener._usage_by_call_id


@pytest.mark.parametrize("runtime", ["unsampled"], indirect=True)
def test_unsampled_calls_keep_no_open_or_usage_payloads(runtime):
    activate, exporter, _ = runtime
    owner = activate()
    for _ in range(3):
        FixtureServer().llm().call("unsampled")
    assert not finished(exporter)
    assert (
        not owner._listener._assembler._open_spans
        and not owner._listener._usage_by_call_id
    )


def test_two_owners_and_foreign_wrappers_are_preserved_and_inert(runtime):
    activate, exporter, _ = runtime
    original = BaseLLM._track_token_usage_internal
    original_emit = crewai_event_bus.emit
    first, second = activate(), activate()
    first.deactivate()
    assert FixtureServer().llm().call("still active") == "fixture answer"
    assert len(finished(exporter)) == 1
    retained = BaseLLM._track_token_usage_internal
    retained_emit = crewai_event_bus.emit

    @wraps(retained)
    def foreign(*args, **kwargs):
        return retained(*args, **kwargs)

    @wraps(retained_emit)
    def foreign_emit(*args, **kwargs):
        return retained_emit(*args, **kwargs)

    BaseLLM._track_token_usage_internal = foreign
    crewai_event_bus.emit = foreign_emit
    second.deactivate()
    assert (
        BaseLLM._track_token_usage_internal is foreign
        and crewai_event_bus.emit is foreign_emit
    )
    exporter.clear()
    try:
        assert FixtureServer().llm().call("inactive") == "fixture answer"
        assert not finished(exporter)
        third = activate()
        llm = FixtureServer().llm()
        with llm_call_context():
            llm._track_token_usage_internal(
                {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
            )
            assert list(third._listener._usage_by_call_id.values()) == [
                {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
            ]
        third._listener._usage_by_call_id.clear()
        llm.call("reactivated")
        assert len(finished(exporter)) == 1
        third.deactivate()
        assert BaseLLM._track_token_usage_internal is foreign
    finally:
        BaseLLM._track_token_usage_internal = original
        crewai_event_bus.emit = original_emit


def test_partial_listener_installation_rolls_back(runtime, monkeypatch):
    original = crewai_event_bus.emit
    before = {k: set(v) for k, v in crewai_event_bus._sync_handlers.items()}

    def fail(self):
        raise RuntimeError("controlled installation failure")

    monkeypatch.setattr(CrewAIEventListener, "_install_token_usage_patch", fail)
    owner = CrewAIInstrumentor()
    owner.activate()
    assert not owner._is_instrumented
    assert crewai_event_bus.emit == original
    assert {k: set(v) for k, v in crewai_event_bus._sync_handlers.items() if v} == {
        k: v for k, v in before.items() if v
    }


def test_tool_correlation_snapshotted_before_background_handler(runtime):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from crewai.events.types.tool_usage_events import (
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )

    activate, exporter, _ = runtime
    owner = activate()
    source = SimpleNamespace(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "snapshot-call-id",
                        "type": "function",
                        "function": {
                            "name": "fixture_weather",
                            "arguments": '{"city":"Tokyo"}',
                        },
                    }
                ],
            }
        ]
    )
    with owner._listener._lifecycle_lock:
        future = crewai_event_bus.emit(
            source,
            ToolUsageStartedEvent(
                tool_name="fixture_weather", tool_args={"city": "Tokyo"}
            ),
        )
        source.messages.append({"role": "assistant", "content": "later unrelated turn"})
    if future:
        future.result(timeout=10)
    now = datetime.now(timezone.utc)
    crewai_event_bus.emit(
        source,
        ToolUsageFinishedEvent(
            tool_name="fixture_weather",
            tool_args={"city": "Tokyo"},
            started_at=now,
            finished_at=now,
            output="Tokyo: clear",
        ),
    )
    spans = finished(exporter, "tool")
    assert len(spans) == 1
    assert spans[0].attributes[G.GEN_AI_TOOL_CALL_ID] == "snapshot-call-id"


def test_current_tool_failure_result_marks_real_finished_event_error(runtime):
    tool_failure = pytest.importorskip("crewai.tools.tool_failure")
    from crewai.tools.tool_calling import ToolCalling
    from crewai.tools.tool_usage import ToolUsage

    activate, exporter, _ = runtime
    activate()

    @tool("fixture_failed_tool")
    def fixture_failed_tool(city: str) -> str:
        """Return a controlled typed tool failure."""
        return tool_failure.ToolFailure(
            message="controlled reported tool failure", code="fixture_failure"
        )

    structured = fixture_failed_tool.to_structured_tool()
    from types import SimpleNamespace

    agent = Agent(
        role="FailureAgent",
        goal="Run the fixture",
        backstory="Fixture",
        llm=FixtureServer().llm(),
        verbose=False,
    )
    usage = ToolUsage(
        tools_handler=None,
        tools=[structured],
        task=None,
        function_calling_llm=None,
        agent=agent,
        action=SimpleNamespace(
            tool="fixture_failed_tool", tool_input={"city": "Tokyo"}
        ),
    )
    result = usage.use(
        ToolCalling(tool_name="fixture_failed_tool", arguments={"city": "Tokyo"}),
        "fixture_failed_tool",
    )
    assert "controlled reported tool failure" in str(result)
    spans = finished(exporter, "tool")
    assert len(spans) == 1
    assert spans[0].status.status_code is trace.StatusCode.ERROR
    assert spans[0].attributes["error.message"] == "controlled reported tool failure"
    assert (
        "controlled reported tool failure"
        in spans[0].attributes[A.TRACELOOP_ENTITY_OUTPUT]
    )


def test_second_provider_rejected_without_changing_owner(runtime, monkeypatch):
    activate, _exporter, _ = runtime
    first = activate()
    other = TracerProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: other)
    second = CrewAIInstrumentor()
    with pytest.raises(RuntimeError, match="another tracer provider"):
        second.activate()
    assert first._is_instrumented and not second._is_instrumented
    other.shutdown()


@pytest.mark.parametrize("early_close", [False, True])
def test_current_public_stream_session_preserves_context_and_cleans_up(
    runtime, early_close
):
    from concurrent.futures import ThreadPoolExecutor

    activate, exporter, provider = runtime
    owner = activate()
    llm = FixtureServer().llm()
    if not hasattr(llm, "stream_events"):
        pytest.skip("Public StreamSession API was added after CrewAI1.10.1")
    with provider.get_tracer("fixture.root").start_as_current_span(
        "fixture.root"
    ) as parent:
        session = llm.stream_events("fixture public stream")
        iterator = iter(session)
        next(iterator)
        assert trace.get_current_span() is parent
        if early_close:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(session.close).result(timeout=10)
            iterator.close()
        else:
            list(iterator)
            assert session.result == "fixture stream"
        assert trace.get_current_span() is parent
    spans = finished(exporter, "chat")
    assert len(spans) == 1
    assert not owner._listener._assembler._open_spans
    assert not owner._listener._usage_by_call_id


def test_telemetry_serialization_failure_cannot_replace_native_result(
    runtime, monkeypatch
):
    from respan_instrumentation_crewai import _event_assembler

    activate, exporter, _ = runtime
    activate()

    def fail(_):
        raise ValueError("controlled telemetry serialization failure")

    monkeypatch.setattr(_event_assembler, "json_attribute", fail)
    assert FixtureServer().llm().call("normal response") == "fixture answer"
    assert len(finished(exporter, "chat")) == 1


def test_unnamed_task_does_not_use_its_prompt_as_span_name(runtime):
    activate, exporter, _ = runtime
    activate()
    agent = Agent(
        role="FixtureAgent",
        goal="Return fixture answers",
        backstory="Fixture",
        llm=FixtureServer().llm(),
        verbose=False,
    )
    task = Task(
        description="unique-user-prompt-for-this-task",
        expected_output="A fixture answer",
        agent=agent,
    )
    assert (
        Crew(agents=[agent], tasks=[task], verbose=False).kickoff().raw
        == "fixture answer"
    )
    task_span = finished(exporter, "task")[0]
    assert task_span.attributes[A.TRACELOOP_ENTITY_NAME] == "Task"
    assert "unique-user-prompt" not in task_span.name
    assert "unique-user-prompt" in task_span.attributes[A.TRACELOOP_ENTITY_INPUT]
