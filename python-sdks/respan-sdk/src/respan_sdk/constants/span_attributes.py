"""Span attribute key constants.

All span attribute keys used across the Respan pipeline. Includes:
- Respan-specific keys (respan.*)
- OTEL incubating GenAI keys (gen_ai.*)
- Pydantic AI vendor keys

For attribute VALUES (e.g. log type strings like "chat", "workflow"),
see ``llm_logging.py``.
"""

# ---------------------------------------------------------------------------
# Respan-specific attribute keys (respan.*)
# ---------------------------------------------------------------------------

# Span params
RESPAN_SPAN_CUSTOM_ID = "respan.span_params.custom_identifier"

# Customer params
RESPAN_CUSTOMER_PARAMS_ID = "respan.customer_params.customer_identifier"
RESPAN_CUSTOMER_PARAMS_EMAIL = "respan.customer_params.email"
RESPAN_CUSTOMER_PARAMS_NAME = "respan.customer_params.name"

# Evaluation params
RESPAN_EVALUATION_PARAMS_ID = "respan.evaluation_params.evaluation_identifier"

# Threads
RESPAN_THREADS_ID = "respan.threads.thread_identifier"

# Trace
RESPAN_TRACE_GROUP_ID = "respan.trace.trace_group_identifier"

# Metadata (pattern: "respan.metadata.<key>" where key is customizable)
RESPAN_METADATA = "respan.metadata"

# Prompt & environment
RESPAN_PROMPT = "respan.prompt"
RESPAN_ENVIRONMENT = "respan.environment"

# Span links
RESPAN_LINK_TIMESTAMP = "respan.link.timestamp"

# Logging
RESPAN_LOG_METHOD = "respan.entity.log_method"
RESPAN_LOG_TYPE = "respan.entity.log_type"
RESPAN_LOG_ID = "respan.entity.log_id"
RESPAN_LOG_PARENT_ID = "respan.entity.log_parent_id"
RESPAN_LOG_ROOT_ID = "respan.entity.log_root_id"
RESPAN_LOG_SOURCE = "respan.entity.log_source"

# ---------------------------------------------------------------------------
# Mapping: user-facing kwargs → span attribute keys
# Used by propagate_attributes(), RespanClient, and RespanParams processing.
# ---------------------------------------------------------------------------
RESPAN_SPAN_ATTRIBUTES_MAP = {
    "customer_identifier": RESPAN_CUSTOMER_PARAMS_ID,
    "customer_email": RESPAN_CUSTOMER_PARAMS_EMAIL,
    "customer_name": RESPAN_CUSTOMER_PARAMS_NAME,
    "evaluation_identifier": RESPAN_EVALUATION_PARAMS_ID,
    "thread_identifier": RESPAN_THREADS_ID,
    "custom_identifier": RESPAN_SPAN_CUSTOM_ID,
    "trace_group_identifier": RESPAN_TRACE_GROUP_ID,
    "group_identifier": RESPAN_TRACE_GROUP_ID,
    "metadata": RESPAN_METADATA,
}

# ---------------------------------------------------------------------------
# OTEL incubating GenAI attributes (not yet in semconv_ai.SpanAttributes)
# ---------------------------------------------------------------------------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

# ---------------------------------------------------------------------------
# Pydantic AI vendor attributes (non-standard)
# ---------------------------------------------------------------------------
PYDANTIC_AI_AGENT_NAME = "agent_name"
PYDANTIC_AI_TOOL_ARGUMENTS = "tool_arguments"
PYDANTIC_AI_TOOL_RESPONSE = "tool_response"
