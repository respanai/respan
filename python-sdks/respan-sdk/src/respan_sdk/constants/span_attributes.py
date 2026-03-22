"""Respan-specific span attribute keys.

These are attribute keys used by the Respan backend that are NOT part of
OpenTelemetry semantic conventions or OpenLLMetry (semconv_ai).
"""

RESPAN_LOG_TYPE = "respan.entity.log_type"
RESPAN_PROMPT = "respan.prompt"
RESPAN_ENVIRONMENT = "respan.environment"
RESPAN_METADATA_AGENT_NAME = "respan.metadata.agent_name"
RESPAN_METADATA_FROM_AGENT = "respan.metadata.from_agent"
RESPAN_METADATA_TO_AGENT = "respan.metadata.to_agent"
RESPAN_METADATA_GUARDRAIL_NAME = "respan.metadata.guardrail_name"
RESPAN_METADATA_TRIGGERED = "respan.metadata.triggered"
RESPAN_SPAN_TOOLS = "respan.span.tools"
RESPAN_SPAN_HANDOFFS = "respan.span.handoffs"
