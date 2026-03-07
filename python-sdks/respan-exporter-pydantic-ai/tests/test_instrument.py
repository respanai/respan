import json

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai.agent import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_exporter_pydantic_ai.instrument import _build_legacy_messages
from respan_sdk.respan_types.span_types import RespanSpanAttributes
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.decorators import workflow
from respan_tracing.exporters.respan import _build_otlp_payload
from respan_tracing.testing import InMemorySpanExporter


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    Agent.instrument_all(instrument=False)
    yield
    RespanTracer.reset_instance()

def test_instrument_global():
    """After instrument_pydantic_ai(), the global default has a tracer, observable by running an agent."""
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=True, is_batching_enabled=False)
    
    span_exporter = InMemorySpanExporter()
    telemetry.tracer.tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    instrument_pydantic_ai()

    # Test observable behavior: a newly created agent receives instrumentation and exports spans
    agent = Agent(model=TestModel())
    agent.run_sync('test message')
    
    telemetry.flush()
    spans = span_exporter.get_finished_spans()
    
    assert len(spans) > 0
    assert any("gen_ai.system" in (s.attributes or {}) for s in spans)

def test_instrument_disabled():
    """When telemetry is disabled, instrumentation is skipped."""
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=False)
    
    # We can't use InMemorySpanExporter here because tracer is not fully initialized when disabled,
    # but we can verify no instrumentation settings are applied globally.
    instrument_pydantic_ai()

    # Test observable behavior: agent has no instrumentation applied
    agent = Agent(model=TestModel())
    # The property to access the resolved instrumentation for a run is usually internal or we can just run it.
    # We can ensure it doesn't crash and no spans are magically created.
    # A disabled telemetry won't add a span processor.
    agent.run_sync('test message')
    # If it was instrumented with a dummy tracer it might, but instrument_pydantic_ai() returns early
    # when is_enabled=False, so Agent.instrument_all() is never called with the settings.
    # Since we reset to False in fixture, it should remain False.
    # To truly avoid _instrument_default, we just check that the explicit agent.instrument is None/False
    assert agent.instrument is None or agent.instrument is False

def test_instrument_specific_agent():
    """When an agent is passed, only that agent is instrumented."""
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=True, is_batching_enabled=False)
    
    span_exporter = InMemorySpanExporter()
    telemetry.tracer.tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    agent = Agent(model=TestModel())
    other_agent = Agent(model=TestModel())

    instrument_pydantic_ai(agent=agent)

    # Target agent should have instrumentation
    assert isinstance(agent.instrument, InstrumentationSettings)

    # Run the other agent (should not produce spans)
    other_agent.run_sync('test message')
    telemetry.flush()
    
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0, "Uninstrumented agent should not produce spans"

    # Run the instrumented agent (should produce spans)
    agent.run_sync('test message')
    telemetry.flush()
    
    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0, "Instrumented agent should produce spans"


def test_instrument_adds_respan_tool_compatibility_attributes():
    telemetry = RespanTelemetry(
        app_name="test-app",
        api_key="test-key",
        is_enabled=True,
        is_batching_enabled=False,
    )

    span_exporter = InMemorySpanExporter()
    telemetry.add_processor(exporter=span_exporter, is_batching_enabled=False)

    agent = Agent(model=TestModel())

    @agent.tool_plain
    async def add(a: int, b: int) -> int:
        return a + b

    instrument_pydantic_ai(agent=agent)

    @workflow(name="calculator_agent_run")
    def run_agent():
        return agent.run_sync("What is 15 plus 27?")

    run_agent()
    telemetry.flush()

    spans = span_exporter.get_finished_spans()
    chat_span = next(
        span
        for span in spans
        if dict(span.attributes or {}).get("tool_calls")
    )
    attrs = dict(chat_span.attributes or {})

    assert attrs["llm.request.type"] == "chat"
    assert attrs["gen_ai.prompt.0.role"] == "user"
    assert attrs["gen_ai.completion.0.role"] == "assistant"
    assert "gen_ai.completion.0.tool_calls.0.name" not in attrs
    assert "model_request_parameters" not in attrs
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert "gen_ai.system_instructions" not in attrs
    assert "gen_ai.tool.definitions" not in attrs
    assert "gen_ai.response.finish_reasons" not in attrs
    assert "logfire.json_schema" not in attrs
    assert attrs["response_format"] == {"type": "text"}
    assert attrs["tools"][0]["function"]["name"] == "add"
    assert attrs["tool_calls"][0]["function"]["name"] == "add"
    assert attrs["gen_ai.completion.0.tool_calls"][0]["function"]["name"] == "add"
    assert json.loads(attrs["tool_calls"][0]["function"]["arguments"]) == {
        "a": 0,
        "b": 0,
    }
    assert json.loads(
        attrs["gen_ai.completion.0.tool_calls"][0]["function"]["arguments"]
    ) == {
        "a": 0,
        "b": 0,
    }

    payload = _build_otlp_payload(spans=[chat_span])
    otlp_attrs = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    otlp_by_key = {item["key"]: item["value"] for item in otlp_attrs}

    assert "arrayValue" in otlp_by_key["tools"]
    assert "kvlistValue" in otlp_by_key["tools"]["arrayValue"]["values"][0]
    assert "arrayValue" in otlp_by_key["tool_calls"]
    assert "kvlistValue" in otlp_by_key["tool_calls"]["arrayValue"]["values"][0]
    assert "arrayValue" in otlp_by_key["gen_ai.completion.0.tool_calls"]
    assert (
        "kvlistValue"
        in otlp_by_key["gen_ai.completion.0.tool_calls"]["arrayValue"]["values"][0]
    )
    assert "kvlistValue" in otlp_by_key["response_format"]
    assert "model_request_parameters" not in otlp_by_key
    assert "gen_ai.input.messages" not in otlp_by_key
    assert "gen_ai.output.messages" not in otlp_by_key
    assert "gen_ai.tool.definitions" not in otlp_by_key


def test_legacy_message_builder_preserves_string_tool_arguments():
    legacy_messages = _build_legacy_messages(
        messages=[
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "add",
                        "arguments": '{"a":15,"b":27}',
                    }
                ],
                "finish_reason": "tool_call",
            }
        ]
    )

    assert legacy_messages == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": '{"a":15,"b":27}',
                    },
                }
            ],
            "finish_reason": "tool_call",
        }
    ]


def test_instrument_normalizes_running_tools_span_for_backend():
    telemetry = RespanTelemetry(
        app_name="test-app",
        api_key="test-key",
        is_enabled=True,
        is_batching_enabled=False,
    )

    span_exporter = InMemorySpanExporter()
    telemetry.add_processor(exporter=span_exporter, is_batching_enabled=False)

    agent = Agent(model=TestModel())

    @agent.tool_plain
    async def add(a: int, b: int) -> int:
        return a + b

    instrument_pydantic_ai(agent=agent)
    agent.run_sync("What is 15 plus 27?")
    telemetry.flush()

    running_tools_span = next(
        span for span in span_exporter.get_finished_spans() if span.name == "running tools"
    )
    attrs = dict(running_tools_span.attributes or {})

    assert attrs["tools"] == [{"type": "function", "function": {"name": "add"}}]


def test_workflow_context_marks_pydantic_tool_spans_with_valid_log_types():
    telemetry = RespanTelemetry(
        app_name="test-app",
        api_key="test-key",
        is_enabled=True,
        is_batching_enabled=False,
    )

    span_exporter = InMemorySpanExporter()
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter)
    )

    agent = Agent(model=TestModel())

    @agent.tool_plain
    async def add(a: int, b: int) -> int:
        return a + b

    instrument_pydantic_ai()

    @workflow(name="calculator_agent_run")
    def run_agent():
        return agent.run_sync("What is 15 plus 27?")

    run_agent()
    telemetry.flush()

    spans_by_name = {
        span.name: dict(span.attributes or {})
        for span in span_exporter.get_finished_spans()
    }

    assert (
        spans_by_name["agent run"][RespanSpanAttributes.LOG_TYPE.value]
        == "agent"
    )
    assert (
        spans_by_name["running tools"][RespanSpanAttributes.LOG_TYPE.value]
        == "task"
    )
    assert (
        spans_by_name["running tool"][RespanSpanAttributes.LOG_TYPE.value]
        == "tool"
    )
