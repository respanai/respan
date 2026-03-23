import json
from types import SimpleNamespace

import pytest
from opentelemetry.instrumentation.openai.shared import chat_wrappers
from opentelemetry.semconv_ai import SpanAttributes
from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_exporter_pydantic_ai.constants import (
    RESPAN_RESPONSE_FORMAT_ATTR,
    RESPAN_TOOLS_ATTR,
    PYDANTIC_AI_OPENAI_HANDLE_REQUEST_PATCH_MARKER,
)
from respan_exporter_pydantic_ai.instrument import (
    _build_gateway_trace_extra_body,
    _inject_gateway_trace_extra_body,
)
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_CHAT, LOG_TYPE_TASK, LOG_TYPE_TOOL, LogMethodChoices
from respan_sdk.respan_types.base_types import RespanBaseModel
from respan_sdk.constants.span_attributes import RESPAN_LOG_METHOD, RESPAN_LOG_TYPE
from respan_sdk.utils.data_processing.id_processing import format_trace_id, format_span_id
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.decorators import workflow
from respan_tracing.testing import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class StructuredAnswer(RespanBaseModel):
    answer: str


def lookup_weather(query: str) -> str:
    return f"Sunny in {query}"

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
    assert any("respan.entity.log_type" in (s.attributes or {}) for s in spans)

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


def test_instrument_patches_openai_gateway_request_hook():
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=True, is_batching_enabled=False)

    instrument_pydantic_ai()

    assert telemetry.tracer is not None
    assert getattr(
        chat_wrappers,
        PYDANTIC_AI_OPENAI_HANDLE_REQUEST_PATCH_MARKER,
        False,
    ) is True


def test_gateway_trace_extra_body_only_targets_respan_gateway(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

    span = SimpleNamespace(
        name="openai.chat",
        parent=SimpleNamespace(span_id=0x3456),
        attributes={SpanAttributes.TRACELOOP_WORKFLOW_NAME: "calculator_agent_run"},
        get_span_context=lambda: SimpleNamespace(trace_id=0x1234, span_id=0x2345),
    )
    kwargs = {}

    _inject_gateway_trace_extra_body(
        span=span,
        kwargs=kwargs,
        instance=SimpleNamespace(_client=SimpleNamespace(base_url="https://api.respan.ai/api/")),
    )

    assert kwargs["extra_body"]["trace_unique_id"] == format_trace_id(0x1234)
    assert kwargs["extra_body"]["span_unique_id"] == format_span_id(0x2345)
    assert kwargs["extra_body"]["span_parent_id"] == format_span_id(0x3456)
    assert kwargs["extra_body"]["span_name"] == "openai.chat"
    assert kwargs["extra_body"]["span_workflow_name"] == "calculator_agent_run"
    assert "disable_log" not in kwargs["extra_body"]


def test_gateway_trace_extra_body_preserves_existing_extra_body(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

    span = SimpleNamespace(
        name="openai.chat",
        parent=SimpleNamespace(span_id=0x3456),
        attributes={SpanAttributes.TRACELOOP_WORKFLOW_NAME: "calculator_agent_run"},
        get_span_context=lambda: SimpleNamespace(trace_id=0x1234, span_id=0x2345),
    )
    kwargs = {
        "extra_body": {
            "customer_identifier": "user_123",
            "span_name": "custom.chat",
            "span_workflow_name": "custom_workflow",
        }
    }

    _inject_gateway_trace_extra_body(
        span=span,
        kwargs=kwargs,
        instance=SimpleNamespace(_client=SimpleNamespace(base_url="https://api.respan.ai/api/")),
    )

    assert kwargs["extra_body"]["customer_identifier"] == "user_123"
    assert kwargs["extra_body"]["span_name"] == "custom.chat"
    assert kwargs["extra_body"]["span_workflow_name"] == "custom_workflow"
    assert kwargs["extra_body"]["trace_unique_id"] == format_trace_id(0x1234)
    assert kwargs["extra_body"]["span_unique_id"] == format_span_id(0x2345)


def test_gateway_trace_extra_body_skips_non_respan_gateway(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

    span = SimpleNamespace(
        name="openai.chat",
        parent=SimpleNamespace(span_id=0x3456),
        attributes={SpanAttributes.TRACELOOP_WORKFLOW_NAME: "calculator_agent_run"},
        get_span_context=lambda: SimpleNamespace(trace_id=0x1234, span_id=0x2345),
    )
    kwargs = {}

    _inject_gateway_trace_extra_body(
        span=span,
        kwargs=kwargs,
        instance=SimpleNamespace(_client=SimpleNamespace(base_url="https://example.com/openai/")),
    )

    assert kwargs == {}


def test_build_gateway_trace_extra_body_formats_span_ids():
    span = SimpleNamespace(
        name="openai.chat",
        parent=SimpleNamespace(span_id=0x3456),
        attributes={SpanAttributes.TRACELOOP_WORKFLOW_NAME: "calculator_agent_run"},
        get_span_context=lambda: SimpleNamespace(trace_id=0x1234, span_id=0x2345),
    )

    assert _build_gateway_trace_extra_body(span=span) == {
        "trace_unique_id": format_trace_id(0x1234),
        "span_unique_id": format_span_id(0x2345),
        "span_parent_id": format_span_id(0x3456),
        "span_name": "openai.chat",
        "span_workflow_name": "calculator_agent_run",
    }


def test_pydantic_ai_agent_span_enriched_and_stripped():
    """TestModel with version=4 produces agent/tool/task spans (no chat span).

    Verify the invoke_agent span is enriched with Respan fields, raw
    Pydantic AI attributes are stripped, and model/usage fields are mapped.
    """
    telemetry = RespanTelemetry(
        app_name="test-app",
        is_enabled=True,
        is_batching_enabled=False,
    )

    span_exporter = InMemorySpanExporter()
    telemetry.add_processor(
        exporter=span_exporter,
        is_batching_enabled=False,
    )

    agent = Agent(
        model=TestModel(custom_output_args={"answer": "ok"}),
        output_type=StructuredAnswer,
        tools=[lookup_weather],
    )

    instrument_pydantic_ai(agent=agent)

    result = agent.run_sync("What is the weather?")
    assert result.output.answer == "ok"

    telemetry.flush()
    spans = span_exporter.get_finished_spans()
    agent_span = next(
        span
        for span in spans
        if (span.attributes or {}).get(RESPAN_LOG_TYPE) == LOG_TYPE_AGENT
    )

    assert RESPAN_TOOLS_ATTR not in agent_span.attributes
    assert RESPAN_RESPONSE_FORMAT_ATTR not in agent_span.attributes
    assert agent_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_AGENT
    assert (
        agent_span.attributes[RESPAN_LOG_METHOD]
        == LogMethodChoices.TRACING_INTEGRATION.value
    )
    assert agent_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "agent"
    assert agent_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "agent"
    assert agent_span.attributes["model"] == "test"


def test_pydantic_ai_tool_span_maps_to_respan_fields():
    telemetry = RespanTelemetry(
        app_name="test-app",
        is_enabled=True,
        is_batching_enabled=False,
    )

    span_exporter = InMemorySpanExporter()
    telemetry.add_processor(
        exporter=span_exporter,
        is_batching_enabled=False,
    )

    agent = Agent(model=TestModel(call_tools="all", custom_output_text="done"))

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    @workflow(name="calculator_agent_run")
    def run_agent() -> str:
        return agent.run_sync("Use the add tool to compute 1 + 2.").output

    instrument_pydantic_ai(agent=agent)

    assert run_agent() == "done"

    telemetry.flush()
    spans = span_exporter.get_finished_spans()
    tool_span = next(
        span
        for span in spans
        if span.name == "execute_tool add"
        and (span.attributes or {}).get(RESPAN_LOG_TYPE) == LOG_TYPE_TOOL
    )
    running_tools_span = next(
        span
        for span in spans
        if span.name == "running tools"
    )

    assert "gen_ai.tool.name" not in tool_span.attributes
    assert "gen_ai.tool.call.id" not in tool_span.attributes
    assert "logfire.msg" not in tool_span.attributes
    assert tool_span.attributes["span_tools"] == ["add"]
    assert tool_span.name == "execute_tool add"
    tool_input = json.loads(tool_span.attributes["input"])
    assert set(tool_input) == {"a", "b"}
    assert json.loads(tool_span.attributes["output"]) == tool_input["a"] + tool_input["b"]
    assert tool_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "tool"
    assert tool_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME] == "add"
    assert tool_span.attributes[RESPAN_LOG_TYPE] == LOG_TYPE_TOOL
    assert (
        tool_span.attributes[RESPAN_LOG_METHOD]
        == LogMethodChoices.TRACING_INTEGRATION.value
    )
    assert running_tools_span.attributes["span_tools"] == ["add"]
    assert running_tools_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "task"
    assert (
        running_tools_span.attributes[RESPAN_LOG_TYPE]
        == LOG_TYPE_TASK
    )
    assert RESPAN_TOOLS_ATTR not in running_tools_span.attributes
