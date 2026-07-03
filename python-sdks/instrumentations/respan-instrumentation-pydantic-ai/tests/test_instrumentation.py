import logging
import sys
from types import ModuleType, SimpleNamespace

from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes

from respan_instrumentation_pydantic_ai import PydanticAIInstrumentor
from respan_instrumentation_pydantic_ai._processor import (
    PydanticAISpanProcessor,
    enrich_pydantic_ai_span,
)


def _make_fake_tracer_provider():
    return SimpleNamespace(
        _active_span_processor=SimpleNamespace(_span_processors=()),
        add_span_processor=lambda processor: None,
    )


def _install_fake_modules(monkeypatch):
    class FakeInstrumentationSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        _instrument_default = "existing-default"
        instrument_all_calls = []

        @classmethod
        def instrument_all(cls, instrument):
            cls.instrument_all_calls.append(instrument)
            cls._instrument_default = instrument

    pydantic_ai_module = ModuleType("pydantic_ai")
    pydantic_ai_agent_module = ModuleType("pydantic_ai.agent")
    pydantic_ai_agent_module.Agent = FakeAgent
    pydantic_ai_models_module = ModuleType("pydantic_ai.models")
    pydantic_ai_models_instrumented_module = ModuleType(
        "pydantic_ai.models.instrumented"
    )
    pydantic_ai_models_instrumented_module.InstrumentationSettings = (
        FakeInstrumentationSettings
    )

    monkeypatch.setitem(sys.modules, "pydantic_ai", pydantic_ai_module)
    monkeypatch.setitem(sys.modules, "pydantic_ai.agent", pydantic_ai_agent_module)
    monkeypatch.setitem(sys.modules, "pydantic_ai.models", pydantic_ai_models_module)
    monkeypatch.setitem(
        sys.modules,
        "pydantic_ai.models.instrumented",
        pydantic_ai_models_instrumented_module,
    )

    return SimpleNamespace(
        agent_class=FakeAgent,
        instrumentation_settings_class=FakeInstrumentationSettings,
    )


def test_activate_instruments_all_agents_and_restores_previous_global(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_modules(monkeypatch)
    previous_default = fake.agent_class._instrument_default

    instrumentor = PydanticAIInstrumentor(
        include_content=False,
        include_binary_content=False,
        version=4,
    )
    instrumentor.activate()

    assert len(fake.agent_class.instrument_all_calls) == 1
    settings = fake.agent_class.instrument_all_calls[0]
    assert isinstance(settings, fake.instrumentation_settings_class)
    assert settings.kwargs["tracer_provider"] is tracer_provider
    assert settings.kwargs["include_content"] is False
    assert settings.kwargs["include_binary_content"] is False
    assert settings.kwargs["version"] == 4

    active_processors = getattr(
        tracer_provider._active_span_processor, "_span_processors", ()
    )
    assert any(
        isinstance(processor, PydanticAISpanProcessor)
        for processor in active_processors
    )

    instrumentor.deactivate()

    assert fake.agent_class.instrument_all_calls[-1] == previous_default
    assert tracer_provider._active_span_processor._span_processors == ()


def test_activate_specific_agent_restores_existing_agent_instrument(monkeypatch):
    tracer_provider = _make_fake_tracer_provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: tracer_provider)
    fake = _install_fake_modules(monkeypatch)
    agent = SimpleNamespace(instrument="existing-agent-setting")

    instrumentor = PydanticAIInstrumentor(agent=agent, version=4)
    instrumentor.activate()

    assert agent.instrument != "existing-agent-setting"
    assert isinstance(agent.instrument, fake.instrumentation_settings_class)
    assert fake.agent_class.instrument_all_calls == []

    instrumentor.deactivate()

    assert agent.instrument == "existing-agent-setting"
    assert tracer_provider._active_span_processor._span_processors == ()


def test_enrich_pydantic_ai_tool_span_maps_tool_fields():
    span = SimpleNamespace(
        name="execute_tool add",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.tool.name": "add",
            "gen_ai.tool.call.arguments": '{"a":1,"b":2}',
            "gen_ai.tool.call.result": "3",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 7,
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "tool"
    assert span._attributes["respan.entity.log_method"] == "tracing_integration"
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "tool"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "add"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "add"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == '{"a":1,"b":2}'
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == "3"
    assert span._attributes["span_tools"] == ["add"]
    assert span._attributes["input"] == '{"a":1,"b":2}'
    assert span._attributes["output"] == "3"
    assert span._attributes["model"] == "gpt-4o-mini"
    assert span._attributes["prompt_tokens"] == 11
    assert span._attributes["completion_tokens"] == 7
    assert span._attributes["total_request_tokens"] == 18
    assert "gen_ai.tool.name" not in span._attributes


def test_enrich_pydantic_ai_agent_span_sets_workflow_name_and_response_format():
    span = SimpleNamespace(
        name="invoke_agent weather",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.agent.name": "weather",
            "model_request_parameters": '{"output_mode":"native","output_object":{"name":"WeatherAnswer","json_schema":{"type":"object"}}}',
            "gen_ai.request.model": "gpt-4o-mini",
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "agent"
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "agent"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "weather"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "weather"
    assert span._attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "weather"
    assert span._attributes["span_workflow_name"] == "weather"
    assert span._attributes["model"] == "gpt-4o-mini"
    assert span._attributes["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "schema": {"type": "object"},
            "name": "WeatherAnswer",
        },
    }
    assert "gen_ai.agent.name" not in span._attributes


def test_enrich_pydantic_ai_chat_span_maps_messages():
    span = SimpleNamespace(
        name="chat completion",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": '[{"role":"user","content":"hi"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"hello"}]',
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "chat"
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "chat"
    assert span._attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '[{"role": "user", "content": "hi"}]'
    )
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        '[{"role": "assistant", "content": "hello"}]'
    )


def test_enrich_pydantic_ai_response_operation_maps_as_chat_span():
    span = SimpleNamespace(
        name="openai.responses.create",
        _attributes={
            "gen_ai.system": "openai",
            "gen_ai.operation.name": "response",
            "gen_ai.input.messages": '[{"role":"user","content":"hi"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"hello"}]',
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "chat"
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "chat"
    assert span._attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == (
        '[{"role": "user", "content": "hi"}]'
    )
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        '[{"role": "assistant", "content": "hello"}]'
    )


def test_enrich_pydantic_ai_running_tools_span_maps_task_fields():
    span = SimpleNamespace(
        name="running tools",
        _attributes={
            "gen_ai.system": "openai",
            "tools": '["add","multiply"]',
        },
    )

    enrich_pydantic_ai_span(span)

    assert span._attributes["respan.entity.log_type"] == "task"
    assert span._attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "task"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "running_tools"
    assert span._attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "running_tools"
    assert span._attributes["span_tools"] == ["add", "multiply"]


def test_activate_logs_warning_when_dependencies_are_missing(caplog):
    instrumentor = PydanticAIInstrumentor()

    with caplog.at_level(logging.WARNING):
        instrumentor.activate()

    assert "Failed to activate PydanticAI instrumentation" in caplog.text
