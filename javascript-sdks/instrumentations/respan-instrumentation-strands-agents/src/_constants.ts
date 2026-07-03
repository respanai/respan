export const STRANDS_SYSTEM_NAME = "strands-agents";

export const STRANDS_OPERATION_INVOKE_AGENT = "invoke_agent";
export const STRANDS_OPERATION_CHAT = "chat";
export const STRANDS_OPERATION_EXECUTE_TOOL = "execute_tool";
export const STRANDS_OPERATION_EXECUTE_AGENT_LOOP_CYCLE =
  "execute_agent_loop_cycle";
export const STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE =
  "execute_event_loop_cycle";
export const STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT =
  "execute_structured_output";
export const STRANDS_OPERATION_EXECUTE_NODE = "execute_node";
export const STRANDS_OPERATION_INVOKE_GRAPH = "invoke_graph";
export const STRANDS_OPERATION_INVOKE_SWARM = "invoke_swarm";
export const STRANDS_OPERATION_INVOKE_PREFIX = "invoke_";

export const STRANDS_AGENT_TOOLS_ATTR = "gen_ai.agent.tools";
export const STRANDS_TOOL_STATUS_ATTR = "gen_ai.tool.status";
export const STRANDS_TOOL_JSON_SCHEMA_ATTR = "gen_ai.tool.json_schema";
export const STRANDS_USAGE_TOTAL_TOKENS_ATTR = "gen_ai.usage.total_tokens";
export const STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR =
  "gen_ai.usage.cache_write_input_tokens";
export const STRANDS_EVENT_START_TIME_ATTR = "gen_ai.event.start_time";
export const STRANDS_EVENT_END_TIME_ATTR = "gen_ai.event.end_time";

export const STRANDS_EVENT_MESSAGE_PREFIX = "gen_ai.";
export const STRANDS_EVENT_MESSAGE_SUFFIX = ".message";

export const STRANDS_SEMCONV_TOOL_DEFINITIONS_OPT_IN =
  "gen_ai_tool_definitions";

export const STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP = new Set([
  "tools",
  "tool_calls",
  "span_tools",
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "has_tool_calls",
  "parallel_tool_calls",
]);

export const STRANDS_RAW_ATTR_PREFIXES_TO_STRIP = ["event_loop."] as const;
