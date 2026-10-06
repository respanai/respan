import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_agno import (
    AgnoInstrumentor,
    _instrumentation,
    _otel_emitter,
)
from respan_instrumentation_agno._constants import (
    AGNO_AGENT_MODULE,
    AGNO_RUN_ID_ATTR,
    AGNO_TARGET_AGENT,
    AGNO_TARGET_TEAM,
    AGNO_TEAM_MODULE,
)
from respan_sdk.constants.span_attributes import (
    RESPAN_LOG_TYPE,
)
from respan_tracing.core.tracer import RespanTracer

AGNO_PROMPT_PREFIX = f"{SpanAttributes.LLM_PROMPTS}."
AGNO_COMPLETION_PREFIX = f"{SpanAttributes.LLM_COMPLETIONS}."


class FakeMetrics:
    input_tokens = 7
    output_tokens = 5
    total_tokens = 12
    cache_read_tokens = 2


class FakeModel:
    provider = "OpenAI"
    id = "gpt-4o-mini"


class FakeToolExecution:
    tool_call_id = "call_1"
    tool_name = "lookup_weather"
    tool_args: ClassVar[dict] = {"city": "Tokyo"}
    result = "sunny"
    tool_call_error = False


class FakeRunOutput:
    run_id = "run_123"
    agent_id = "agent_123"
    agent_name = "Weather Agent"
    session_id = "session_123"
    user_id = "user_123"
    input = SimpleNamespace(input_content="What is the weather?")
    content = "It is sunny."
    model = "gpt-4o-mini"
    model_provider = "OpenAI"
    metrics = FakeMetrics()
    tools: ClassVar[list] = [FakeToolExecution()]
    metadata: ClassVar[dict] = {"plan": "pro"}
    status = "completed"


class FakeEvent:
    event = "RunContent"
    content = "chunk"


class FakeAgent:
    id = "agent_123"
    name = "Weather Agent"
    model = FakeModel()
    tools: ClassVar[list] = []

    def run(self, input, **kwargs):
        return FakeRunOutput()

    def arun(self, input, **kwargs):
        async def run_async():
            return FakeRunOutput()

        return run_async()


class FakeStreamingAgent(FakeAgent):
    def run(self, input, **kwargs):
        yield FakeEvent()
        yield FakeRunOutput()

    def arun(self, input, **kwargs):
        async def stream_async():
            yield FakeEvent()
            yield FakeRunOutput()

        return stream_async()


class FakeFailingAgent(FakeAgent):
    def run(self, input, **kwargs):
        raise RuntimeError("agno failed")


class FakeTeam:
    id = "team_123"
    name = "Research Team"
    model = FakeModel()
    tools: ClassVar[list] = []

    def run(self, input, **kwargs):
        return SimpleNamespace(
            run_id="team_run_123",
            team_id="team_123",
            team_name="Research Team",
            content="Team answer.",
            model="gpt-4o-mini",
            model_provider="OpenAI",
            metrics=FakeMetrics(),
            status="completed",
        )

    def arun(self, input, **kwargs):
        async def run_async():
            return self.run(input=input, **kwargs)

        return run_async()


class FakeNestedTeam(FakeTeam):
    name = "Nested Team"

    def run(self, input, **kwargs):
        FakeAgent().run(input="Ask the member agent for a draft.")
        return SimpleNamespace(
            run_id="nested_team_run_123",
            team_id="nested_team_123",
            team_name="Nested Team",
            content="Nested team answer.",
            model="gpt-4o-mini",
            model_provider="OpenAI",
            metrics=FakeMetrics(),
            status="completed",
        )


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def captured_spans(monkeypatch):
    spans = []

    def fake_build_readable_span(**kwargs):
        span = SimpleNamespace(**kwargs)
        spans.append(span)
        return span

    monkeypatch.setattr(
        _otel_emitter,
        "build_readable_span",
        fake_build_readable_span,
    )
    monkeypatch.setattr(
        target=_otel_emitter,
        name="inject_span",
        value=lambda span: True,
    )
    return spans


def _install_fake_agno_modules(monkeypatch):
    agno_module = ModuleType("agno")
    agent_package = ModuleType("agno.agent")
    agent_module = ModuleType(AGNO_AGENT_MODULE)
    team_package = ModuleType("agno.team")
    team_module = ModuleType(AGNO_TEAM_MODULE)

    agent_module.Agent = FakeAgent
    team_module.Team = FakeTeam
    agent_package.agent = agent_module
    team_package.team = team_module
    agno_module.agent = agent_package
    agno_module.team = team_package

    monkeypatch.setitem(dic=sys.modules, name="agno", value=agno_module)
    monkeypatch.setitem(dic=sys.modules, name="agno.agent", value=agent_package)
    monkeypatch.setitem(dic=sys.modules, name=AGNO_AGENT_MODULE, value=agent_module)
    monkeypatch.setitem(dic=sys.modules, name="agno.team", value=team_package)
    monkeypatch.setitem(dic=sys.modules, name=AGNO_TEAM_MODULE, value=team_module)
    return SimpleNamespace(agent_class=FakeAgent, team_class=FakeTeam)


def test_activate_patches_agent_and_team_classes(monkeypatch):
    fake_modules = _install_fake_agno_modules(monkeypatch)
    original_agent_run = fake_modules.agent_class.run
    original_team_run = fake_modules.team_class.run

    instrumentor = AgnoInstrumentor()
    instrumentor.activate()

    assert fake_modules.agent_class.run is not original_agent_run
    assert fake_modules.team_class.run is not original_team_run
    assert instrumentor._is_instrumented is True

    instrumentor.deactivate()

    assert fake_modules.agent_class.run is original_agent_run
    assert fake_modules.team_class.run is original_team_run
    assert instrumentor._is_instrumented is False


def test_activate_specific_agent_does_not_patch_class(monkeypatch):
    fake_modules = _install_fake_agno_modules(monkeypatch)
    original_agent_run = fake_modules.agent_class.run
    agent = FakeAgent()

    instrumentor = AgnoInstrumentor(agent=agent)
    instrumentor.activate()

    assert agent.run is not original_agent_run
    assert fake_modules.agent_class.run is original_agent_run

    instrumentor.deactivate()

    assert fake_modules.agent_class.run is original_agent_run


def test_activate_skips_when_respan_tracing_is_disabled(monkeypatch, caplog):
    fake_modules = _install_fake_agno_modules(monkeypatch)
    RespanTracer(is_enabled=False)

    instrumentor = AgnoInstrumentor()
    with caplog.at_level("INFO"):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert fake_modules.agent_class.run is FakeAgent.run
    assert "Agno instrumentation skipped" in caplog.text


def test_activate_logs_warning_when_dependency_is_missing(monkeypatch, caplog):
    def import_module_raises(module_name):
        raise ImportError(module_name)

    monkeypatch.setattr(
        _instrumentation.importlib,
        "import_module",
        import_module_raises,
    )

    instrumentor = AgnoInstrumentor()
    with caplog.at_level("WARNING"):
        instrumentor.activate()

    assert instrumentor._is_instrumented is False
    assert "Failed to activate Agno instrumentation" in caplog.text


def test_sync_run_emits_agent_chat_and_tool_spans(captured_spans):
    agent = FakeAgent()

    _otel_emitter.emit_agno_run(
        target=agent,
        target_kind=AGNO_TARGET_AGENT,
        input_value="What is the weather?",
        output=FakeRunOutput(),
        events=None,
        started_at_ns=100,
        ended_at_ns=200,
    )

    assert [span.name for span in captured_spans] == [
        "agno.agent",
        "agno.model_request",
        "agno.tool",
    ]

    root_attributes = captured_spans[0].attributes
    assert root_attributes[RESPAN_LOG_TYPE] == "agent"
    assert root_attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "Weather Agent"
    assert root_attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == ""
    assert root_attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "Weather Agent"
    assert root_attributes[AGNO_RUN_ID_ATTR] == "run_123"
    assert root_attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '{"input_content":"What is the weather?"}'
    )
    assert root_attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "It is sunny."

    chat_attributes = captured_spans[1].attributes
    assert chat_attributes[RESPAN_LOG_TYPE] == "chat"
    assert chat_attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert chat_attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
    assert chat_attributes[SpanAttributes.LLM_SYSTEM] == "openai"
    assert chat_attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "Weather Agent"
    assert chat_attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 12
    assert chat_attributes[f"{AGNO_PROMPT_PREFIX}0.role"] == "user"
    assert chat_attributes[f"{AGNO_PROMPT_PREFIX}0.content"] == "What is the weather?"
    assert chat_attributes[f"{AGNO_COMPLETION_PREFIX}0.role"] == "assistant"
    assert chat_attributes[f"{AGNO_COMPLETION_PREFIX}0.content"] == "It is sunny."
    assert "gen_ai.completion.0.tool_calls" in chat_attributes
    assert (
        json.loads(chat_attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
            "tool_calls"
        ][0]["function"]["name"]
        == "lookup_weather"
    )
    assert "tool_calls" not in chat_attributes
    assert "tools" not in chat_attributes

    tool_attributes = captured_spans[2].attributes
    assert tool_attributes[RESPAN_LOG_TYPE] == "tool"
    assert tool_attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "lookup_weather"
    assert tool_attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "Weather Agent"
    assert tool_attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "sunny"


def test_instrumented_sync_stream_emits_after_consumption(captured_spans):
    agent = FakeStreamingAgent()
    instrumentor = AgnoInstrumentor(agent=agent)
    instrumentor.activate()

    result = list(agent.run(input="stream please", stream=True))

    assert len(result) == 2
    assert [span.name for span in captured_spans] == [
        "agno.agent",
        "agno.model_request",
        "agno.tool",
    ]


def test_instrumented_run_emits_error_span(captured_spans):
    agent = FakeFailingAgent()
    instrumentor = AgnoInstrumentor(agent=agent)
    instrumentor.activate()

    with pytest.raises(RuntimeError, match="agno failed"):
        agent.run(input="fail")

    assert len(captured_spans) == 1
    assert captured_spans[0].name == "agno.agent"
    assert captured_spans[0].status_code == 500
    assert captured_spans[0].error_message == "agno failed"


def test_instrumented_async_run_emits_spans(captured_spans):
    agent = FakeAgent()
    instrumentor = AgnoInstrumentor(agent=agent)
    instrumentor.activate()

    result = asyncio.run(agent.arun(input="async please"))

    assert result.content == "It is sunny."
    assert [span.name for span in captured_spans] == [
        "agno.agent",
        "agno.model_request",
        "agno.tool",
    ]


def test_instrumented_async_stream_emits_after_consumption(captured_spans):
    agent = FakeStreamingAgent()
    instrumentor = AgnoInstrumentor(agent=agent)
    instrumentor.activate()

    async def collect_stream():
        items = []
        async for item in agent.arun(input="stream please", stream=True):
            items.append(item)
        return items

    result = asyncio.run(collect_stream())

    assert len(result) == 2
    assert [span.name for span in captured_spans] == [
        "agno.agent",
        "agno.model_request",
        "agno.tool",
    ]


def test_team_run_emits_workflow_root(captured_spans):
    team = FakeTeam()

    _otel_emitter.emit_agno_run(
        target=team,
        target_kind=AGNO_TARGET_TEAM,
        input_value="Write a report",
        output=team.run(input="Write a report"),
        events=None,
        started_at_ns=100,
        ended_at_ns=200,
    )

    root_attributes = captured_spans[0].attributes
    assert root_attributes[RESPAN_LOG_TYPE] == "workflow"
    assert root_attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "Research Team"


def test_active_workflow_name_is_preserved_on_all_native_spans(captured_spans):
    token = context_api.attach(
        context_api.set_value(
            SpanAttributes.TRACELOOP_ENTITY_NAME,
            "agno_parent_workflow",
        )
    )
    try:
        agent = FakeAgent()
        run_context = _otel_emitter.create_agno_run_context(
            target=agent,
            target_kind=AGNO_TARGET_AGENT,
            started_at_ns=100,
        )
        with _otel_emitter.use_agno_run_context(run_context=run_context):
            _otel_emitter.emit_agno_run(
                target=agent,
                target_kind=AGNO_TARGET_AGENT,
                input_value="What is the weather?",
                output=FakeRunOutput(),
                events=None,
                started_at_ns=100,
                ended_at_ns=200,
            )
    finally:
        context_api.detach(token)

    assert len(captured_spans) == 3
    assert {
        span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME]
        for span in captured_spans
    } == {"agno_parent_workflow"}


def test_nested_agent_run_uses_team_trace_context(monkeypatch, captured_spans):
    fake_modules = _install_fake_agno_modules(monkeypatch)
    fake_modules.team_class = FakeNestedTeam
    sys.modules[AGNO_TEAM_MODULE].Team = FakeNestedTeam

    instrumentor = AgnoInstrumentor()
    instrumentor.activate()

    team = FakeNestedTeam()
    result = team.run(input="Coordinate a draft.")

    assert result.content == "Nested team answer."

    team_root = next(span for span in captured_spans if span.name == "agno.team")
    agent_root = next(span for span in captured_spans if span.name == "agno.agent")
    assert agent_root.trace_id == team_root.trace_id
    assert agent_root.parent_id == team_root.span_id


def test_callable_tool_definition_includes_json_schema(captured_spans):
    def lookup_weather(city: str, days: int = 1) -> str:
        """Look up weather for a city."""
        return f"{city}: {days}"

    agent = FakeAgent()
    agent.tools = [lookup_weather]

    _otel_emitter.emit_agno_run(
        target=agent,
        target_kind=AGNO_TARGET_AGENT,
        input_value="What is the weather?",
        output=FakeRunOutput(),
        events=None,
        started_at_ns=100,
        ended_at_ns=200,
    )

    chat_attributes = captured_spans[1].attributes
    tool_definitions = json.loads(chat_attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS])
    function_schema = tool_definitions[0]["function"]
    assert function_schema["name"] == "lookup_weather"
    assert function_schema["description"] == "Look up weather for a city."
    assert function_schema["parameters"] == {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
    }
