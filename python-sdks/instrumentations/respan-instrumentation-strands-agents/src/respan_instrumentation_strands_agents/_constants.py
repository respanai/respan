"""Strands Agents native telemetry constants."""

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv_ai import SpanAttributes

STRANDS_SYSTEM_NAME = "strands-agents"

STRANDS_OPERATION_INVOKE_AGENT = "invoke_agent"
STRANDS_OPERATION_CHAT = "chat"
STRANDS_OPERATION_EXECUTE_TOOL = "execute_tool"
STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE = "execute_event_loop_cycle"
STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT = "execute_structured_output"
STRANDS_OPERATION_INVOKE_PREFIX = "invoke_"

STRANDS_AGENT_TOOLS_ATTR = "gen_ai.agent.tools"
STRANDS_TOOL_STATUS_ATTR = "gen_ai.tool.status"
STRANDS_TOOL_JSON_SCHEMA_ATTR = "gen_ai.tool.json_schema"
STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR = "gen_ai.usage.cache_write_input_tokens"
STRANDS_EVENT_START_TIME_ATTR = "gen_ai.event.start_time"
STRANDS_EVENT_END_TIME_ATTR = "gen_ai.event.end_time"

STRANDS_EVENT_OPERATION_DETAILS = "gen_ai.client.inference.operation.details"
STRANDS_EVENT_CHOICE = "gen_ai.choice"
STRANDS_EVENT_SYSTEM_MESSAGE = "gen_ai.system.message"
STRANDS_EVENT_TOOL_MESSAGE = "gen_ai.tool.message"
STRANDS_EVENT_MESSAGE_PREFIX = "gen_ai."
STRANDS_EVENT_MESSAGE_SUFFIX = ".message"
STRANDS_EVENT_USER_MESSAGE = (
    f"{STRANDS_EVENT_MESSAGE_PREFIX}user{STRANDS_EVENT_MESSAGE_SUFFIX}"
)

STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN = "gen_ai_tool_definitions"

STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP = frozenset(
    {
        "tools",
        "tool_calls",
        "span_tools",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_request_tokens",
        "has_tool_calls",
        "parallel_tool_calls",
    }
)

STRANDS_RAW_ATTR_PREFIXES_TO_STRIP = frozenset({"event_loop."})

STRANDS_RAW_ATTRS_TO_STRIP = frozenset(
    {
        gen_ai.GEN_AI_AGENT_NAME,
        gen_ai.GEN_AI_OPERATION_NAME,
        gen_ai.GEN_AI_TOOL_NAME,
        gen_ai.GEN_AI_PROVIDER_NAME,
        gen_ai.GEN_AI_AGENT_ID,
        gen_ai.GEN_AI_TOOL_DEFINITIONS,
        gen_ai.GEN_AI_TOOL_DESCRIPTION,
        gen_ai.GEN_AI_TOOL_CALL_ARGUMENTS,
        gen_ai.GEN_AI_TOOL_CALL_RESULT,
        gen_ai.GEN_AI_INPUT_MESSAGES,
        gen_ai.GEN_AI_OUTPUT_MESSAGES,
        gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS,
        STRANDS_AGENT_TOOLS_ATTR,
        STRANDS_TOOL_STATUS_ATTR,
        STRANDS_TOOL_JSON_SCHEMA_ATTR,
        STRANDS_EVENT_START_TIME_ATTR,
        STRANDS_EVENT_END_TIME_ATTR,
        "system_prompt",
    }
)

STRANDS_NON_LLM_ATTRS_TO_STRIP = frozenset(
    {
        gen_ai.GEN_AI_USAGE_INPUT_TOKENS,
        gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS,
        SpanAttributes.GEN_AI_IS_STREAMING,
        SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS,
        SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
        SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
        STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR,
    }
)
