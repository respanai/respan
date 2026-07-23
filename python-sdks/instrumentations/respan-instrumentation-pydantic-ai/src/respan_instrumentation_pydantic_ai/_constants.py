"""PydanticAI instrumentation constants.

This module separates:
- native PydanticAI/vendor attributes that we consume as input
- legacy/off-contract Respan override keys that are stripped when present
"""

from opentelemetry.semconv_ai import SpanAttributes


# PydanticAI native / vendor attributes
PYDANTIC_AI_AGENT_NAME_ATTR = "gen_ai.agent.name"
PYDANTIC_AI_OPERATION_NAME_ATTR = "gen_ai.operation.name"
PYDANTIC_AI_TOOL_NAME_ATTR = "gen_ai.tool.name"
PYDANTIC_AI_TOOL_CALL_ARGUMENTS_ATTR = "gen_ai.tool.call.arguments"
PYDANTIC_AI_TOOL_CALL_RESULT_ATTR = "gen_ai.tool.call.result"
PYDANTIC_AI_REQUEST_PARAMETERS_ATTR = "model_request_parameters"
PYDANTIC_AI_TOOL_DEFINITIONS_ATTR = "gen_ai.tool.definitions"
PYDANTIC_AI_INPUT_MESSAGES_ATTR = "gen_ai.input.messages"
PYDANTIC_AI_OUTPUT_MESSAGES_ATTR = "gen_ai.output.messages"
PYDANTIC_AI_PROVIDER_NAME_ATTR = "gen_ai.provider.name"
PYDANTIC_AI_RESPONSE_ID_ATTR = "gen_ai.response.id"
PYDANTIC_AI_RESPONSE_FINISH_REASONS_ATTR = "gen_ai.response.finish_reasons"
PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR = "gen_ai.usage.input_tokens"
PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR = "gen_ai.usage.output_tokens"
PYDANTIC_AI_AGGREGATED_USAGE_INPUT_TOKENS_ATTR = "gen_ai.aggregated_usage.input_tokens"
PYDANTIC_AI_AGGREGATED_USAGE_OUTPUT_TOKENS_ATTR = "gen_ai.aggregated_usage.output_tokens"
PYDANTIC_AI_AGGREGATED_USAGE_TOTAL_TOKENS_ATTR = "gen_ai.aggregated_usage.total_tokens"
PYDANTIC_AI_USAGE_DETAILS_INPUT_TOKENS_ATTR = "gen_ai.usage.details.input_tokens"
PYDANTIC_AI_USAGE_DETAILS_OUTPUT_TOKENS_ATTR = "gen_ai.usage.details.output_tokens"
PYDANTIC_AI_OPERATION_COST_ATTR = "operation.cost"

# Legacy PydanticAI attrs
PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR = "agent_name"
PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR = "tool_arguments"
PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR = "tool_response"

# Other raw attrs seen on spans that we normalize away
PYDANTIC_AI_TOOLS_ATTR = "tools"
LOGFIRE_MESSAGE_ATTR = "logfire.msg"
MODEL_NAME_ATTR = "model_name"
FINAL_RESULT_ATTR = "final_result"
PYDANTIC_ALL_MESSAGES_ATTR = "pydantic_ai.all_messages"
OPENAI_RESPONSE_SERVICE_TIER_ATTR = "openai.response.service_tier"
OTEL_SERVER_ADDRESS_ATTR = "server.address"
OTEL_SERVER_PORT_ATTR = "server.port"
OTEL_SERVICE_NAME_ATTR = "service.name"
OTEL_SCOPE_NAME_ATTR = "otel.scope.name"
OTEL_SCOPE_VERSION_ATTR = "otel.scope.version"

# Backend override keys used by the Respan OTLP pipeline
RESPAN_RESPONSE_FORMAT_ATTR = "response_format"
RESPAN_OVERRIDE_MODEL_ATTR = "model"
RESPAN_OVERRIDE_INPUT_ATTR = "input"
RESPAN_OVERRIDE_OUTPUT_ATTR = "output"
RESPAN_OVERRIDE_SPAN_TOOLS_ATTR = "span_tools"
RESPAN_OVERRIDE_SPAN_WORKFLOW_NAME_ATTR = "span_workflow_name"
RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR = "prompt_tokens"
RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR = "completion_tokens"
RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR = "total_request_tokens"

# Well-known span names emitted by PydanticAI
PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME = "running tools"

PYDANTIC_AI_STRIP_ATTRS = frozenset(
    {
        PYDANTIC_AI_AGENT_NAME_ATTR,
        PYDANTIC_AI_OPERATION_NAME_ATTR,
        PYDANTIC_AI_TOOL_NAME_ATTR,
        PYDANTIC_AI_TOOL_CALL_ARGUMENTS_ATTR,
        PYDANTIC_AI_TOOL_CALL_RESULT_ATTR,
        PYDANTIC_AI_REQUEST_PARAMETERS_ATTR,
        PYDANTIC_AI_TOOL_DEFINITIONS_ATTR,
        PYDANTIC_AI_INPUT_MESSAGES_ATTR,
        PYDANTIC_AI_OUTPUT_MESSAGES_ATTR,
        PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
        PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
        PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
        PYDANTIC_AI_PROVIDER_NAME_ATTR,
        PYDANTIC_AI_RESPONSE_ID_ATTR,
        PYDANTIC_AI_RESPONSE_FINISH_REASONS_ATTR,
        SpanAttributes.GEN_AI_OPENAI_API_BASE,
        PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR,
        PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR,
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
        PYDANTIC_AI_AGGREGATED_USAGE_INPUT_TOKENS_ATTR,
        PYDANTIC_AI_AGGREGATED_USAGE_OUTPUT_TOKENS_ATTR,
        PYDANTIC_AI_AGGREGATED_USAGE_TOTAL_TOKENS_ATTR,
        PYDANTIC_AI_USAGE_DETAILS_INPUT_TOKENS_ATTR,
        PYDANTIC_AI_USAGE_DETAILS_OUTPUT_TOKENS_ATTR,
        PYDANTIC_AI_OPERATION_COST_ATTR,
        PYDANTIC_AI_TOOLS_ATTR,
        LOGFIRE_MESSAGE_ATTR,
        MODEL_NAME_ATTR,
        FINAL_RESULT_ATTR,
        PYDANTIC_ALL_MESSAGES_ATTR,
        SpanAttributes.LLM_HEADERS,
        SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
        SpanAttributes.LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
        OPENAI_RESPONSE_SERVICE_TIER_ATTR,
        SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        SpanAttributes.LLM_USAGE_REASONING_TOKENS,
        OTEL_SERVER_ADDRESS_ATTR,
        OTEL_SERVER_PORT_ATTR,
        OTEL_SERVICE_NAME_ATTR,
        OTEL_SCOPE_NAME_ATTR,
        OTEL_SCOPE_VERSION_ATTR,
        RESPAN_OVERRIDE_MODEL_ATTR,
        RESPAN_OVERRIDE_INPUT_ATTR,
        RESPAN_OVERRIDE_OUTPUT_ATTR,
        RESPAN_OVERRIDE_SPAN_TOOLS_ATTR,
        RESPAN_OVERRIDE_SPAN_WORKFLOW_NAME_ATTR,
        RESPAN_OVERRIDE_PROMPT_TOKENS_ATTR,
        RESPAN_OVERRIDE_COMPLETION_TOKENS_ATTR,
        RESPAN_OVERRIDE_TOTAL_REQUEST_TOKENS_ATTR,
        SpanAttributes.TRACELOOP_SPAN_KIND,
        "respan.span.tools",
        "respan.span.tool_calls",
        "respan.span.handoffs",
        "tool_calls",
        "has_tool_calls",
        "parallel_tool_calls",
    }
)
