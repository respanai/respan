import { RespanLogType } from "@respan/respan-sdk";

export const PACKAGE_VERSION = "0.1.0";
export const EXA_INSTRUMENTATION_NAME = "exa";
export const EXA_INSTRUMENTATION_SCOPE = "@respan/instrumentation-exa";
export const EXA_SYSTEM = "exa";

export const EXA_METADATA_NAMESPACE = "exa";
export const METADATA_OPERATION = "operation";
export const METADATA_LANGUAGE = "language";
export const METADATA_STREAM = "stream";
export const METADATA_STREAM_COMPLETED = "stream_completed";
export const METADATA_RESULT_COUNT = "result_count";
export const METADATA_REQUEST_ID = "request_id";
export const METADATA_RESOLVED_SEARCH_TYPE = "resolved_search_type";
export const METADATA_COST_TOTAL_USD = "cost_total_usd";
export const METADATA_CITATIONS = "citations";
export const METADATA_RESEARCH_LEGACY = "research_legacy";
export const STATUS_CODE_ATTR = "status_code";

export type OperationFamily = "agent" | "chat" | "task" | "tool";

export interface OperationConfig {
  entityName: string;
  family: OperationFamily;
  operation: string;
  alwaysStreaming?: boolean;
  streamFlag?: string;
  streamFamily?: OperationFamily;
  legacyResearch?: boolean;
}

export const LOG_TYPE_BY_FAMILY: Record<OperationFamily, RespanLogType> = {
  agent: RespanLogType.AGENT,
  chat: RespanLogType.CHAT,
  task: RespanLogType.TASK,
  tool: RespanLogType.TOOL,
};

export const OFF_CONTRACT_ALIASES = new Set([
  "tools",
  "tool_calls",
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  "respan.span.tools",
  "respan.span.tool_calls",
  "respan.span.handoffs",
]);
