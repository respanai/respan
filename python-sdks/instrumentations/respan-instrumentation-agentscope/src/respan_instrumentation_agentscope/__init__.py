"""AgentScope instrumentation plugin for Respan."""

from respan_instrumentation_agentscope._instrumentation import (
    AGENTSCOPE_INSTRUMENTATION_NAME,
    AgentScopeInstrumentor,
)

__all__ = [
    "AGENTSCOPE_INSTRUMENTATION_NAME",
    "AgentScopeInstrumentor",
]
