"""No-network regression coverage for native CrewAI lifecycle translation."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace

from crewai import Agent, Crew, LLM, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.llm_events import LLMCallType
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.tasks.task_output import TaskOutput
from opentelemetry import trace
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from pydantic import BaseModel

from respan_instrumentation_crewai import CrewAIInstrumentor
from respan_instrumentation_crewai import _event_assembler as assembler_module
from respan_instrumentation_crewai._event_assembler import (
    CrewAIEventAssembler,
    SpanEndSpec,
    SpanStartSpec,
)
from respan_instrumentation_crewai._event_listener import CrewAIEventListener
from respan_instrumentation_crewai._serialization import completion_message
from respan_sdk.constants import ERROR_MESSAGE_ATTR
from respan_sdk.constants.llm_logging import (
    LOG_TYPE_AGENT,
    LOG_TYPE_CHAT,
    LOG_TYPE_TASK,
    LOG_TYPE_TOOL,
    LOG_TYPE_WORKFLOW,
    LogMethodChoices,
)
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE


def _emit(source, event):
    future = crewai_event_bus.emit(source, event)
    if future is not None:
        future.result(timeout=10)
    assert crewai_event_bus.flush()
    return event


def _entities():
    llm = LLM(model="gpt-4o-mini", api_key="test-key")
    agent = Agent(
        role="Researcher",
        goal="Answer accurately",
        backstory="A careful researcher",
        llm=llm,
        verbose=False,
    )
    task = Task(
        name="Answer",
        description="Answer the question",
        expected_output="A concise answer",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return crew, task, agent, llm


def _start_lifecycle(crew, task, agent) -> None:
    _emit(
        crew,
        CrewKickoffStartedEvent(
            crew_name="Research Crew",
            inputs={"question": "Why?"},
            crew=crew,
        ),
    )
    _emit(task, TaskStartedEvent(context="Prior context", task=task))
    _emit(
        agent,
        AgentExecutionStartedEvent(
            agent=agent,
            task=task,
            tools=[],
            task_prompt="Answer: Why?",
        ),
    )


def _finish_lifecycle(crew, task, agent, output: str = "Because.") -> None:
    _emit(
        agent,
        AgentExecutionCompletedEvent(
            agent=agent,
            task=task,
            output=output,
        ),
    )
    task_output = TaskOutput(
        name=task.name,
        description=task.description,
        expected_output=task.expected_output,
        raw=output,
        agent=agent.role,
    )
    _emit(task, TaskCompletedEvent(output=task_output, task=task))
    _emit(
        crew,
        CrewKickoffCompletedEvent(
            crew_name="Research Crew",
            output=SimpleNamespace(raw=output),
            crew=crew,
        ),
    )


def _setup_tracing(monkeypatch):
    active_owner = CrewAIInstrumentor._active_owner
    if active_owner is not None:
        active_owner.deactivate()
    CrewAIInstrumentor._active_owner = None

    tracer_provider = TracerProvider()
    span_exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)

    original_token_hook = BaseLLM._track_token_usage_internal
    instrumentor = CrewAIInstrumentor()
    instrumentor.activate()
    assert instrumentor._is_instrumented is True
    assert BaseLLM._track_token_usage_internal is not original_token_hook
    return tracer_provider, span_exporter, instrumentor, original_token_hook


def _finish_tracing(
    tracer_provider,
    instrumentor,
    original_token_hook,
):
    instrumentor.deactivate()
    assert BaseLLM._track_token_usage_internal is original_token_hook
    assert tracer_provider.force_flush()
    tracer_provider.shutdown()


def test_native_listener_exports_canonical_full_lifecycle(monkeypatch):
    tracer_provider, exporter, instrumentor, original_token_hook = _setup_tracing(
        monkeypatch
    )
    try:
        crew, task, agent, llm = _entities()
        _emit(
            crew,
            CrewKickoffStartedEvent(
                crew_name="Research Crew",
                inputs={"question": "Why?"},
                crew=crew,
            ),
        )
        _emit(task, TaskStartedEvent(context="Prior context", task=task))
        _emit(
            agent,
            AgentExecutionStartedEvent(
                agent=agent,
                task=task,
                tools=[],
                task_prompt="Answer: Why?",
            ),
        )

        messages = [{"role": "user", "content": "Why?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a fact",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with llm_call_context():
            llm._emit_call_started_event(
                messages=messages,
                tools=tools,
                from_task=task,
                from_agent=agent,
            )
            assert crewai_event_bus.flush()
            llm._track_token_usage_internal(
                {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                    "cached_prompt_tokens": 2,
                    "cache_creation_tokens": 1,
                }
            )
            llm._emit_call_completed_event(
                response="Because.",
                call_type=LLMCallType.LLM_CALL,
                from_task=task,
                from_agent=agent,
                messages=messages,
            )
            assert crewai_event_bus.flush()

        started_at = datetime.now(timezone.utc)
        _emit(
            agent,
            ToolUsageStartedEvent(
                tool_name="lookup",
                tool_args={"topic": "reason"},
                from_task=task,
                from_agent=agent,
            ),
        )
        _emit(
            agent,
            ToolUsageFinishedEvent(
                tool_name="lookup",
                tool_args={"topic": "reason"},
                output={"answer": "Because."},
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                from_task=task,
                from_agent=agent,
            ),
        )
        _emit(
            agent,
            AgentExecutionCompletedEvent(
                agent=agent,
                task=task,
                output="Because.",
            ),
        )
        task_output = TaskOutput(
            name="Answer",
            description=task.description,
            expected_output=task.expected_output,
            raw="Because.",
            agent=agent.role,
        )
        _emit(task, TaskCompletedEvent(output=task_output, task=task))
        _emit(
            crew,
            CrewKickoffCompletedEvent(
                crew_name="Research Crew",
                output=SimpleNamespace(raw="Because."),
                crew=crew,
                total_tokens=10,
            ),
        )

        assert tracer_provider.force_flush()
        spans = [
            span
            for span in exporter.get_finished_spans()
            if RESPAN_LOG_TYPE in (span.attributes or {})
        ]
        by_type = {(span.attributes or {})[RESPAN_LOG_TYPE]: span for span in spans}
        assert set(by_type) == {
            LOG_TYPE_WORKFLOW,
            LOG_TYPE_TASK,
            LOG_TYPE_AGENT,
            LOG_TYPE_TOOL,
            LOG_TYPE_CHAT,
        }

        workflow_span = by_type[LOG_TYPE_WORKFLOW]
        task_span = by_type[LOG_TYPE_TASK]
        agent_span = by_type[LOG_TYPE_AGENT]
        chat_span = by_type[LOG_TYPE_CHAT]
        tool_span = by_type[LOG_TYPE_TOOL]

        assert task_span.parent.span_id == workflow_span.context.span_id
        assert agent_span.parent.span_id == task_span.context.span_id
        assert chat_span.parent.span_id == agent_span.context.span_id
        assert tool_span.parent.span_id == agent_span.context.span_id
        assert len({span.context.trace_id for span in spans}) == 1

        for span in spans:
            attributes = span.attributes or {}
            assert (
                attributes[RESPAN_LOG_METHOD]
                == LogMethodChoices.TRACING_INTEGRATION.value
            )
            assert SpanAttributes.TRACELOOP_ENTITY_NAME in attributes
            assert SpanAttributes.TRACELOOP_ENTITY_PATH in attributes
            assert SpanAttributes.TRACELOOP_SPAN_KIND not in attributes
            assert not any(key.startswith("crewai.") for key in attributes)

        chat_attributes = chat_span.attributes or {}
        assert chat_attributes[SpanAttributes.LLM_REQUEST_TYPE] == (
            LLMRequestTypeValues.CHAT.value
        )
        assert chat_attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
        assert chat_attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
        assert chat_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
        assert chat_attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Why?"
        assert chat_attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == (
            "assistant"
        )
        assert chat_attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == (
            "Because."
        )
        assert SpanAttributes.LLM_REQUEST_FUNCTIONS in chat_attributes
        assert chat_attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 7
        assert chat_attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
        assert chat_attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
        assert chat_attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3
        assert chat_attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 10
        assert chat_attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 2
        assert (
            chat_attributes[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 1
        )

        banned_aliases = {
            "respan.span.tools",
            "respan.span.tool_calls",
            "tools",
            "tool_calls",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "total_request_tokens",
            "span_tools",
            "has_tool_calls",
        }
        assert banned_aliases.isdisjoint(chat_attributes)

        tool_attributes = tool_span.attributes or {}
        assert (
            '"name":"lookup"'
            in (tool_attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
        )
        assert (
            '"answer":"Because."'
            in (tool_attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
        )
    finally:
        _finish_tracing(tracer_provider, instrumentor, original_token_hook)


def test_native_listener_marks_failed_lifecycles(monkeypatch):
    tracer_provider, exporter, instrumentor, original_token_hook = _setup_tracing(
        monkeypatch
    )
    try:
        crew, task, agent, _ = _entities()
        _emit(
            crew,
            CrewKickoffStartedEvent(
                crew_name="Failure Crew",
                inputs={},
                crew=crew,
            ),
        )
        _emit(task, TaskStartedEvent(context=None, task=task))
        _emit(
            agent,
            AgentExecutionStartedEvent(
                agent=agent,
                task=task,
                tools=[],
                task_prompt="Fail",
            ),
        )
        _emit(
            agent,
            ToolUsageStartedEvent(
                tool_name="explode",
                tool_args={},
                from_task=task,
                from_agent=agent,
            ),
        )
        _emit(
            agent,
            ToolUsageErrorEvent(
                tool_name="explode",
                tool_args={},
                error="tool boom",
                from_task=task,
                from_agent=agent,
            ),
        )
        _emit(
            agent,
            AgentExecutionErrorEvent(agent=agent, task=task, error="agent boom"),
        )
        _emit(task, TaskFailedEvent(error="task boom", task=task))
        _emit(
            crew,
            CrewKickoffFailedEvent(
                crew_name="Failure Crew",
                error="crew boom",
                crew=crew,
            ),
        )

        assert tracer_provider.force_flush()
        spans = [
            span
            for span in exporter.get_finished_spans()
            if RESPAN_LOG_TYPE in (span.attributes or {})
        ]
        assert len(spans) == 4
        for span in spans:
            attributes = span.attributes or {}
            assert span.status.status_code.name == "ERROR"
            assert attributes[ERROR_MESSAGE_ATTR].endswith("boom")
    finally:
        _finish_tracing(tracer_provider, instrumentor, original_token_hook)


def test_tool_call_completion_uses_canonical_indexed_tool_calls(monkeypatch):
    tracer_provider, exporter, instrumentor, original_token_hook = _setup_tracing(
        monkeypatch
    )
    try:
        crew, task, agent, llm = _entities()
        _start_lifecycle(crew, task, agent)
        messages = [{"role": "user", "content": "Use lookup"}]
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"topic":"reason"}',
                },
            }
        ]
        with llm_call_context():
            llm._emit_call_started_event(
                messages=messages,
                from_task=task,
                from_agent=agent,
            )
            assert crewai_event_bus.flush()
            llm._emit_call_completed_event(
                response=tool_calls,
                call_type=LLMCallType.TOOL_CALL,
                from_task=task,
                from_agent=agent,
                messages=messages,
            )
            assert crewai_event_bus.flush()

        _finish_lifecycle(crew, task, agent)
        assert tracer_provider.force_flush()
        chat_span = next(
            span
            for span in exporter.get_finished_spans()
            if (span.attributes or {}).get(RESPAN_LOG_TYPE) == LOG_TYPE_CHAT
        )
        attributes = chat_span.attributes or {}
        assert (
            json.loads(attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
            == tool_calls
        )
        assert attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
        assert "tool_calls" not in attributes
        assert "respan.span.tool_calls" not in attributes
    finally:
        _finish_tracing(tracer_provider, instrumentor, original_token_hook)


def test_serialized_pydantic_completion_preserves_structured_content(monkeypatch):
    class StructuredResponse(BaseModel):
        answer: str
        confidence: float

    tracer_provider, exporter, instrumentor, original_token_hook = _setup_tracing(
        monkeypatch
    )
    try:
        crew, task, agent, llm = _entities()
        _start_lifecycle(crew, task, agent)
        messages = [{"role": "user", "content": "Return JSON"}]
        with llm_call_context():
            llm._emit_call_started_event(
                messages=messages,
                from_task=task,
                from_agent=agent,
            )
            assert crewai_event_bus.flush()
            llm._emit_call_completed_event(
                response=StructuredResponse(answer="Because", confidence=0.9),
                call_type=LLMCallType.LLM_CALL,
                from_task=task,
                from_agent=agent,
                messages=messages,
            )
            assert crewai_event_bus.flush()

        _finish_lifecycle(crew, task, agent)
        assert tracer_provider.force_flush()
        chat_span = next(
            span
            for span in exporter.get_finished_spans()
            if (span.attributes or {}).get(RESPAN_LOG_TYPE) == LOG_TYPE_CHAT
        )
        attributes = chat_span.attributes or {}
        assert json.loads(
            attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        ) == {"answer": "Because", "confidence": 0.9}
    finally:
        _finish_tracing(tracer_provider, instrumentor, original_token_hook)


def test_completion_message_extracts_object_attribute_choices():
    tool_calls = [{"id": "call-1", "function": {"name": "lookup"}}]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="Object response",
                    tool_calls=tool_calls,
                )
            )
        ]
    )

    assert completion_message(response) == {
        "role": "assistant",
        "content": "Object response",
        "tool_calls": tool_calls,
    }
    assert completion_message(
        SimpleNamespace(choices=[SimpleNamespace(text="Legacy text")])
    ) == {"role": "assistant", "content": "Legacy text"}


def test_bedrock_override_captures_camel_case_usage_and_restores(monkeypatch):
    class FakeBedrockCompletion:
        def __init__(self):
            self.seen_usage = None

        def _track_token_usage_internal(self, usage):
            self.seen_usage = usage

    original_bedrock_hook = FakeBedrockCompletion.__dict__[
        "_track_token_usage_internal"
    ]
    monkeypatch.setattr(
        CrewAIEventListener,
        "_token_usage_patch_targets",
        staticmethod(lambda: (BaseLLM, FakeBedrockCompletion)),
    )
    tracer_provider, exporter, instrumentor, original_token_hook = _setup_tracing(
        monkeypatch
    )
    assert (
        FakeBedrockCompletion._track_token_usage_internal is not original_bedrock_hook
    )
    try:
        crew, task, agent, llm = _entities()
        _start_lifecycle(crew, task, agent)
        messages = [{"role": "user", "content": "Why?"}]
        bedrock = FakeBedrockCompletion()
        with llm_call_context():
            llm._emit_call_started_event(
                messages=messages,
                from_task=task,
                from_agent=agent,
            )
            assert crewai_event_bus.flush()
            bedrock._track_token_usage_internal(
                {
                    "inputTokens": 11,
                    "outputTokens": 4,
                    "totalTokens": 15,
                    "cacheReadInputTokenCount": 3,
                }
            )
            llm._emit_call_completed_event(
                response="Because.",
                call_type=LLMCallType.LLM_CALL,
                from_task=task,
                from_agent=agent,
                messages=messages,
            )
            assert crewai_event_bus.flush()

        assert bedrock.seen_usage is not None
        _finish_lifecycle(crew, task, agent)
        assert tracer_provider.force_flush()
        chat_span = next(
            span
            for span in exporter.get_finished_spans()
            if (span.attributes or {}).get(RESPAN_LOG_TYPE) == LOG_TYPE_CHAT
        )
        attributes = chat_span.attributes or {}
        assert attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
        assert attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
        assert attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 11
        assert attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 4
        assert attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 15
        assert attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    finally:
        _finish_tracing(tracer_provider, instrumentor, original_token_hook)
    assert FakeBedrockCompletion._track_token_usage_internal is original_bedrock_hook


def test_suppressed_token_hook_does_not_buffer_usage(monkeypatch):
    tracer_provider, _, instrumentor, original_token_hook = _setup_tracing(monkeypatch)
    try:
        _, _, _, llm = _entities()
        with llm_call_context():
            with suppress_instrumentation():
                llm._track_token_usage_internal(
                    {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    }
                )
        assert instrumentor._listener._usage_by_call_id == {}
    finally:
        _finish_tracing(tracer_provider, instrumentor, original_token_hook)


def test_queued_handler_is_noop_after_listener_shutdown(monkeypatch):
    tracer_provider, exporter, instrumentor, original_token_hook = _setup_tracing(
        monkeypatch
    )
    crew, _, _, _ = _entities()
    queued_handler = next(
        handler
        for event_type, handler in instrumentor._listener._handlers
        if event_type is CrewKickoffStartedEvent
    )
    instrumentor.deactivate()
    queued_handler(
        crew,
        CrewKickoffStartedEvent(
            crew_name="Late Crew",
            inputs={},
            crew=crew,
        ),
    )
    assert tracer_provider.force_flush()
    assert not [
        span
        for span in exporter.get_finished_spans()
        if RESPAN_LOG_TYPE in (span.attributes or {})
    ]
    _finish_tracing(tracer_provider, instrumentor, original_token_hook)


def test_assembler_correlates_end_event_that_arrives_before_start():
    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    assembler = CrewAIEventAssembler(
        tracer_provider.get_tracer("crewai-correlation-test")
    )
    started_at = datetime.now(timezone.utc)

    assembler.end_span(
        SimpleNamespace(started_event_id="start-1", timestamp=started_at),
        SpanEndSpec(output={"ok": True}),
    )
    assembler.start_span(
        SimpleNamespace(
            event_id="start-1",
            parent_event_id=None,
            timestamp=started_at,
        ),
        SpanStartSpec(
            name="out-of-order.task",
            attributes={RESPAN_LOG_TYPE: LOG_TYPE_TASK},
        ),
    )

    assert tracer_provider.force_flush()
    spans = [
        span
        for span in exporter.get_finished_spans()
        if RESPAN_LOG_TYPE in (span.attributes or {})
    ]
    assert len(spans) == 1
    assert spans[0].attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TASK
    assert spans[0].status.status_code.name == "OK"
    assembler.shutdown()
    tracer_provider.shutdown()


def test_assembler_preserves_newer_reused_correlation():
    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    assembler = CrewAIEventAssembler(
        tracer_provider.get_tracer("crewai-reused-correlation-test")
    )
    timestamp = datetime.now(timezone.utc)
    correlation_key = "llm:reused"

    for event_id, name in (("old", "old.chat"), ("new", "new.chat")):
        assembler.start_span(
            SimpleNamespace(
                event_id=event_id,
                parent_event_id=None,
                timestamp=timestamp,
            ),
            SpanStartSpec(
                name=name,
                attributes={RESPAN_LOG_TYPE: LOG_TYPE_CHAT},
                correlation_keys=(correlation_key,),
            ),
        )

    assembler.end_span(
        SimpleNamespace(started_event_id="old", timestamp=timestamp),
        SpanEndSpec(output="old"),
        correlation_keys=(correlation_key,),
    )
    assembler.end_span(
        SimpleNamespace(timestamp=timestamp),
        SpanEndSpec(output="new"),
        correlation_keys=(correlation_key,),
    )

    assert tracer_provider.force_flush()
    spans = [
        span
        for span in exporter.get_finished_spans()
        if (span.attributes or {}).get(RESPAN_LOG_TYPE) == LOG_TYPE_CHAT
    ]
    assert {span.name for span in spans} == {"old.chat", "new.chat"}
    assert all(span.status.status_code.name == "OK" for span in spans)
    assembler.shutdown()
    tracer_provider.shutdown()


def test_assembler_bounds_pending_state_and_buffers_early_scope_close(
    monkeypatch,
):
    monkeypatch.setattr(assembler_module, "MAX_BUFFERED_ENTRIES", 2)
    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    assembler = CrewAIEventAssembler(
        tracer_provider.get_tracer("crewai-bounded-state-test")
    )
    timestamp = datetime.now(timezone.utc)

    for index in range(3):
        assembler.start_span(
            SimpleNamespace(
                event_id=f"child-{index}",
                parent_event_id="missing-parent",
                timestamp=timestamp,
            ),
            SpanStartSpec(
                name=f"child-{index}.task",
                attributes={RESPAN_LOG_TYPE: LOG_TYPE_TASK},
            ),
        )

    assert assembler._pending_start_count == 2
    assert sum(len(queue) for queue in assembler._pending_starts.values()) == 2

    assembler.close_scope(SimpleNamespace(started_event_id="late-scope"))
    assert "late-scope" in assembler._pending_scope_ends
    assembler.open_scope(
        SimpleNamespace(
            event_id="late-scope",
            parent_event_id=None,
            timestamp=timestamp,
        )
    )
    assert "late-scope" not in assembler._pending_scope_ends
    assert "late-scope" not in assembler._transparent_scopes
    assert "late-scope" in assembler._finished_contexts

    for index in range(3):
        assembler.open_scope(
            SimpleNamespace(
                event_id=f"scope-{index}",
                parent_event_id=None,
                timestamp=timestamp,
            )
        )
    assert len(assembler._transparent_scopes) <= 2

    assembler.shutdown()
    tracer_provider.shutdown()
