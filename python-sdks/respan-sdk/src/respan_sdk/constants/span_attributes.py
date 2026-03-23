"""Flat convenience aliases for span attribute key constants.

The single source of truth is ``RespanSpanAttributes`` enum in
``respan_sdk.respan_types.span_types``.  This module re-exports the
enum values as plain module-level constants for callers that prefer
``RESPAN_METADATA`` over ``RespanSpanAttributes.RESPAN_METADATA.value``.

For attribute VALUES (e.g. log type strings like "chat", "workflow"),
see ``llm_logging.py``.
"""

from respan_sdk.respan_types.span_types import RespanSpanAttributes

# ---------------------------------------------------------------------------
# Respan-specific attribute keys (respan.*)
# ---------------------------------------------------------------------------

# Span params
RESPAN_SPAN_CUSTOM_ID = RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID.value

# Customer params
RESPAN_CUSTOMER_PARAMS_ID = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID.value
RESPAN_CUSTOMER_PARAMS_EMAIL = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_EMAIL.value
RESPAN_CUSTOMER_PARAMS_NAME = RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_NAME.value

# Evaluation params
RESPAN_EVALUATION_PARAMS_ID = RespanSpanAttributes.RESPAN_EVALUATION_PARAMS_ID.value

# Threads
RESPAN_THREADS_ID = RespanSpanAttributes.RESPAN_THREADS_ID.value

# Trace
RESPAN_TRACE_GROUP_ID = RespanSpanAttributes.RESPAN_TRACE_GROUP_ID.value

# Metadata (pattern: "respan.metadata.<key>" where key is customizable)
RESPAN_METADATA = RespanSpanAttributes.RESPAN_METADATA.value

# Prompt & environment
RESPAN_PROMPT = RespanSpanAttributes.RESPAN_PROMPT.value
RESPAN_ENVIRONMENT = RespanSpanAttributes.RESPAN_ENVIRONMENT.value

# Span links
RESPAN_LINK_TIMESTAMP = RespanSpanAttributes.RESPAN_LINK_TIMESTAMP.value

# Logging
RESPAN_LOG_METHOD = RespanSpanAttributes.LOG_METHOD.value
RESPAN_LOG_TYPE = RespanSpanAttributes.LOG_TYPE.value
RESPAN_LOG_ID = RespanSpanAttributes.LOG_ID.value
RESPAN_LOG_PARENT_ID = RespanSpanAttributes.LOG_PARENT_ID.value
RESPAN_LOG_ROOT_ID = RespanSpanAttributes.LOG_ROOT_ID.value
RESPAN_LOG_SOURCE = RespanSpanAttributes.LOG_SOURCE.value

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
    "prompt": RESPAN_PROMPT,
    "environment": RESPAN_ENVIRONMENT,
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
