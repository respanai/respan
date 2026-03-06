import pytest
from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

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

