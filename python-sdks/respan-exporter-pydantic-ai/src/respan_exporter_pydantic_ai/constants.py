"""Pydantic AI span attribute names and sentinel values."""

from opentelemetry.semconv_ai import SpanAttributes

# ── Pydantic AI native span attributes ──────────────────────────────────────

PYDANTIC_AI_REQUEST_PARAMETERS_ATTR = "model_request_parameters"
PYDANTIC_AI_TOOL_DEFINITIONS_ATTR = "gen_ai.tool.definitions"
PYDANTIC_AI_INPUT_MESSAGES_ATTR = "gen_ai.input.messages"
PYDANTIC_AI_OUTPUT_MESSAGES_ATTR = "gen_ai.output.messages"
PYDANTIC_AI_OPERATION_NAME_ATTR = "gen_ai.operation.name"
PYDANTIC_AI_SYSTEM_ATTR = "gen_ai.system"
PYDANTIC_AI_TOOL_NAME_ATTR = "gen_ai.tool.name"
PYDANTIC_AI_TOOL_CALL_ID_ATTR = "gen_ai.tool.call.id"
PYDANTIC_AI_TOOL_ARGUMENTS_ATTR = "gen_ai.tool.call.arguments"
PYDANTIC_AI_TOOL_RESULT_ATTR = "gen_ai.tool.call.result"
PYDANTIC_AI_AGENT_NAME_ATTR = "gen_ai.agent.name"
PYDANTIC_AI_PROVIDER_NAME_ATTR = "gen_ai.provider.name"
PYDANTIC_AI_RESPONSE_ID_ATTR = "gen_ai.response.id"
PYDANTIC_AI_RESPONSE_FINISH_REASONS_ATTR = "gen_ai.response.finish_reasons"
PYDANTIC_AI_OPENAI_API_BASE_ATTR = "gen_ai.openai.api_base"
PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR = "gen_ai.usage.input_tokens"
PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR = "gen_ai.usage.output_tokens"

# ── Legacy (pre-v4) attribute names ─────────────────────────────────────────

PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR = "tool_arguments"
PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR = "tool_response"
PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR = "agent_name"

# ── Span attribute names from auto-instrumentation / third-party ─────────

RESPAN_TOOLS_ATTR = "tools"
RESPAN_RESPONSE_FORMAT_ATTR = "response_format"
RESPAN_TOOL_CALLS_ATTR = "tool_calls"
LOGFIRE_MSG_ATTR = "logfire.msg"
OTEL_SERVER_ADDRESS_ATTR = "server.address"
OTEL_SERVER_PORT_ATTR = "server.port"
MODEL_NAME_ATTR = "model_name"

# ── Well-known span names emitted by Pydantic AI ───────────────────────────

PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME = "running tools"

# ── Internal patch markers (prevent double-instrumentation) ─────────────────

PYDANTIC_AI_ENRICHMENT_MARKER = "_respan_pydantic_ai_enrichment_installed"
PYDANTIC_AI_ADD_PROCESSOR_PATCH_MARKER = (
    "_respan_pydantic_ai_add_span_processor_patched"
)
PYDANTIC_AI_OPENAI_HANDLE_REQUEST_PATCH_MARKER = (
    "_respan_pydantic_ai_openai_handle_request_patched"
)

# ── Attributes stripped after enrichment ────────────────────────────────────
# These raw provider / OTel attributes are consumed during enrichment and then
# removed so the exported span only carries clean Respan fields.

ENRICHMENT_STRIP_ATTRS = frozenset({
    PYDANTIC_AI_REQUEST_PARAMETERS_ATTR,
    PYDANTIC_AI_TOOL_DEFINITIONS_ATTR,
    PYDANTIC_AI_INPUT_MESSAGES_ATTR,
    PYDANTIC_AI_OUTPUT_MESSAGES_ATTR,
    PYDANTIC_AI_TOOL_ARGUMENTS_ATTR,
    PYDANTIC_AI_TOOL_RESULT_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS_ATTR,
    PYDANTIC_AI_LEGACY_TOOL_RESULT_ATTR,
    PYDANTIC_AI_TOOL_NAME_ATTR,
    PYDANTIC_AI_TOOL_CALL_ID_ATTR,
    PYDANTIC_AI_AGENT_NAME_ATTR,
    PYDANTIC_AI_LEGACY_AGENT_NAME_ATTR,
    PYDANTIC_AI_OPERATION_NAME_ATTR,
    PYDANTIC_AI_SYSTEM_ATTR,
    PYDANTIC_AI_PROVIDER_NAME_ATTR,
    PYDANTIC_AI_RESPONSE_ID_ATTR,
    PYDANTIC_AI_RESPONSE_FINISH_REASONS_ATTR,
    PYDANTIC_AI_OPENAI_API_BASE_ATTR,
    PYDANTIC_AI_USAGE_INPUT_TOKENS_ATTR,
    PYDANTIC_AI_USAGE_OUTPUT_TOKENS_ATTR,
    SpanAttributes.LLM_REQUEST_MODEL,
    SpanAttributes.LLM_RESPONSE_MODEL,
    RESPAN_TOOLS_ATTR,
    RESPAN_RESPONSE_FORMAT_ATTR,
    RESPAN_TOOL_CALLS_ATTR,
    LOGFIRE_MSG_ATTR,
    OTEL_SERVER_ADDRESS_ATTR,
    OTEL_SERVER_PORT_ATTR,
})

# ── Default gateway URL ────────────────────────────────────────────────────

DEFAULT_RESPAN_GATEWAY_BASE_URL = "https://api.respan.ai/api"
