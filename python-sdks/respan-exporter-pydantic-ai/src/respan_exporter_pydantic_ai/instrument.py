import logging
from typing import Optional

from pydantic_ai.agent import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from respan_tracing.core.tracer import RespanTracer

logger = logging.getLogger(__name__)

def instrument_pydantic_ai(
    agent: Optional[Agent] = None,
    include_content: bool = True,
    include_binary_content: bool = True,
) -> None:
    """
    Instruments Pydantic AI with Respan telemetry via OpenTelemetry.
    
    If an agent is provided, instruments only that agent.
    Otherwise, instruments all Pydantic AI agents globally.
    
    Args:
        agent: Optional Agent to instrument. If None, instruments globally.
        include_content: Whether to include message content in telemetry.
        include_binary_content: Whether to include binary content in telemetry.
    """
    if not RespanTracer.is_initialized():
        logger.warning(
            "Respan telemetry is not initialized. "
            "Please initialize RespanTelemetry before calling instrument_pydantic_ai()."
        )
        return
    
    tracer = RespanTracer()
    
    if not tracer.is_enabled:
        logger.warning("Respan telemetry is disabled.")
        return
    
    # tracer_provider is guaranteed to exist here: is_initialized() and is_enabled
    # guards above ensure _setup_tracer_provider() has run. Pydantic AI also accepts
    # None (falls back to global provider), but we always have the explicit one.
    settings = InstrumentationSettings(
        tracer_provider=tracer.tracer_provider,
        include_content=include_content,
        include_binary_content=include_binary_content,
        # We use version 2 by default to support standard OTel semantic conventions
        version=2,
    )
    
    if agent is not None:
        agent.instrument = settings
    else:
        Agent.instrument_all(instrument=settings)
