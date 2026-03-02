import pytest
from pydantic_ai.agent import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer

@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    Agent.instrument_all(instrument=False)
    yield
    RespanTracer.reset_instance()

def test_instrument_global():
    """After instrument_pydantic_ai(), the global default has a tracer."""
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=True)

    instrument_pydantic_ai()

    # Pydantic AI stores the global default in _instrument_default (no public getter).
    # instrument_all() is the public setter; _instrument_default is the only read path.
    default_instrument = Agent._instrument_default
    assert isinstance(default_instrument, InstrumentationSettings)
    assert default_instrument.tracer is not None

def test_instrument_disabled():
    """When telemetry is disabled, instrumentation is skipped."""
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=False)

    instrument_pydantic_ai()

    assert Agent._instrument_default is False

def test_instrument_specific_agent():
    """When an agent is passed, only that agent is instrumented."""
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=True)

    agent = Agent(model='test')
    other_agent = Agent(model='test')

    instrument_pydantic_ai(agent=agent)

    # Target agent should have instrumentation
    assert isinstance(agent.instrument, InstrumentationSettings)
    assert agent.instrument.tracer is not None

    # Global default should NOT be changed
    assert Agent._instrument_default is False

    # Other agent should NOT have instrumentation
    assert other_agent.instrument is None
