import pytest
from pydantic_ai.agent import Agent
from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer

@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    # Reset Pydantic AI agent instrumentation
    Agent._instrument_default = False
    yield
    RespanTracer.reset_instance()

def test_instrument_global():
    # Initialize telemetry
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=True)
    
    # Instrument globally
    instrument_pydantic_ai()
    
    # Check that Agent._instrument_default is set to InstrumentationSettings
    assert Agent._instrument_default is not False
    assert hasattr(Agent._instrument_default, "tracer")
    
    # We just ensure it runs without exception
    assert Agent._instrument_default.tracer is not None

def test_instrument_specific_agent():
    # Initialize telemetry
    telemetry = RespanTelemetry(app_name="test-app", api_key="test-key", is_enabled=True)
    
    # Create an agent
    agent = Agent(model='test')
    
    # By default, not instrumented with specific settings
    assert agent.instrument in (None, False)
    
    # Instrument specific agent
    instrument_pydantic_ai(agent=agent)
    
    # Global should not be changed by this
    assert Agent._instrument_default is False
    
    # Agent should have instrumentation settings
    assert agent.instrument is not None
    assert agent.instrument is not False
    assert hasattr(agent.instrument, "tracer")
    assert agent.instrument.tracer is not None
