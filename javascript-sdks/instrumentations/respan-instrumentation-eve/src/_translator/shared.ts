import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_TOOL_NAME,
  GEN_AI_OPERATION_NAME_VALUE_CHAT,
  GEN_AI_OPERATION_NAME_VALUE_EMBEDDINGS,
  GEN_AI_OPERATION_NAME_VALUE_EXECUTE_TOOL,
  GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { VERCEL_PARENT_SPANS, VERCEL_SPAN_CONFIG } from "../constants/index.js";

export type SpanAttributes = Record<string, any>;

export const AI_PREFIX = "ai.";
export const AI_SDK = "ai.sdk";
export const AI_OPERATION_ID = "ai.operationId";
export const AI_MODEL_ID = "ai.model.id";
export const AI_MODEL_PROVIDER = "ai.model.provider";
export const AI_EMBEDDING = "ai.embedding";
export const AI_EMBEDDINGS = "ai.embeddings";
export const AI_VALUE = "ai.value";
export const AI_VALUES = "ai.values";
export const AI_AGENT_ID = "ai.agent.id";
export const AI_WORKFLOW_ID = "ai.workflow.id";
export const AI_TRANSCRIPT = "ai.transcript";
export const AI_SPEECH = "ai.speech";
export const AI_TELEMETRY_METADATA_PREFIX = "ai.telemetry.metadata.";
export const AI_TELEMETRY_FUNCTION_ID = "ai.telemetry.functionId";
export const AI_SETTINGS_CONTEXT_PREFIX = "ai.settings.context.";
export const AI_PROMPT = "ai.prompt";
export const AI_PROMPT_MESSAGES = "ai.prompt.messages";
export const AI_PROMPT_TOOLS = "ai.prompt.tools";
export const AI_PROMPT_TOOL_CHOICE = "ai.prompt.toolChoice";
export const AI_RESPONSE_OBJECT = "ai.response.object";
export const AI_RESPONSE_TEXT = "ai.response.text";
export const AI_RESPONSE_TOOL_CALLS = "ai.response.toolCalls";
export const AI_RESPONSE_MS_TO_FINISH = "ai.response.msToFinish";
export const AI_USAGE_PROMPT_TOKENS = "ai.usage.promptTokens";
export const AI_USAGE_COMPLETION_TOKENS = "ai.usage.completionTokens";
export const AI_USAGE_INPUT_TOKENS = "ai.usage.inputTokens";
export const AI_USAGE_OUTPUT_TOKENS = "ai.usage.outputTokens";
export const AI_USAGE_TOTAL_TOKENS = "ai.usage.totalTokens";
export const AI_USAGE_CACHED_INPUT_TOKENS = "ai.usage.cachedInputTokens";
export const GEN_AI_USAGE_COST = "gen_ai.usage.cost";
export const GEN_AI_USAGE_TTFT = "gen_ai.usage.ttft";
export const GEN_AI_USAGE_GENERATION_TIME = "gen_ai.usage.generation_time";
export const GEN_AI_USAGE_WARNINGS = "gen_ai.usage.warnings";
export const GEN_AI_USAGE_TYPE = "gen_ai.usage.type";
export const AI_TOOL_CALL = "ai.toolCall";
export const AI_TOOL_CALLS = "ai.toolCalls";
export const AI_TOOL_CALL_PREFIX = "ai.toolCall.";
export const AI_TOOL_CALL_ID = "ai.toolCall.id";
export const AI_TOOL_CALL_NAME = "ai.toolCall.name";
export const AI_TOOL_CALL_ARGS = "ai.toolCall.args";
export const AI_TOOL_CALL_RESULT = "ai.toolCall.result";

const VERCEL_AI_SCOPE_NAMES = new Set(["ai", "gen_ai", "@ai-sdk/otel"]);
const MODERN_OPERATION_LOG_TYPES: Record<string, string> = {
  [GEN_AI_OPERATION_NAME_VALUE_CHAT]: RespanLogType.TEXT,
  [GEN_AI_OPERATION_NAME_VALUE_INVOKE_AGENT]: RespanLogType.AGENT,
  [GEN_AI_OPERATION_NAME_VALUE_EMBEDDINGS]: RespanLogType.EMBEDDING,
  [GEN_AI_OPERATION_NAME_VALUE_EXECUTE_TOOL]: RespanLogType.TOOL,
  agent_step: RespanLogType.TASK,
  rerank: RespanLogType.TASK,
};

export function setMetadata(
  attrs: SpanAttributes,
  key: string,
  value: unknown,
): void {
  if (value === undefined || value === null) {
    return;
  }

  const attribute = RespanSpanAttributes.RESPAN_METADATA;
  const existing = safeJsonParse(attrs[attribute]);
  const metadata = isRecord(existing) ? { ...existing } : {};
  if (metadata[key] === undefined) {
    metadata[key] = value;
  }
  attrs[attribute] = safeJsonStr(metadata);

  // Keep the canonical JSON bucket above and also follow the JavaScript
  // tracing SDK's live wire format. The current ingest path promotes
  // respan.metadata.<key> attributes into the request-log metadata object.
  const flattenedAttribute = `${attribute}.${key}`;
  if (attrs[flattenedAttribute] === undefined) {
    attrs[flattenedAttribute] = safeJsonStr(value);
  }
}

export function setDefault(attrs: SpanAttributes, key: string, value: any): void {
  if (attrs[key] === undefined && value !== undefined && value !== null) {
    attrs[key] = value;
  }
}

export function safeJsonStr(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * Map a Vercel embedding span's value(s) + vector(s) onto the universal
 * input/output. Input = the embedded text; output = the embedding vector(s).
 * We capture the full vector (it's debuggable data for RAG — similarity,
 * drift, degenerate-vector detection); size is handled by storage tiering at
 * ingest, not by dropping data here.
 */
export function formatEmbeddingInput(attrs: SpanAttributes): string | undefined {
  const value = attrs[AI_VALUE] ?? attrs[AI_VALUES];
  if (value === undefined || value === null) return undefined;
  return safeJsonStr(normalizeEmbeddingValue(value));
}

export function formatEmbeddingOutput(attrs: SpanAttributes): string | undefined {
  const raw = attrs[AI_EMBEDDING] ?? attrs[AI_EMBEDDINGS];
  if (raw === undefined || raw === null) return undefined;
  return safeJsonStr(normalizeEmbeddingValue(raw));
}

export function safeJsonParse(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export function parseJsonish(value: unknown): unknown {
  if (typeof value !== "string") {
    return value;
  }

  const parsed = safeJsonParse(value);
  return parsed === value ? value : parsed;
}

function normalizeEmbeddingValue(value: unknown): unknown {
  const parsed = parseJsonish(value);
  if (!Array.isArray(parsed)) {
    return parsed;
  }

  return parsed.map((item) => parseJsonish(item));
}

export function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function instrumentationScopeName(span: Partial<ReadableSpan> | any): string | undefined {
  return span.instrumentationScope?.name ?? span.instrumentationScope?.name;
}

export function isVercelAIScope(scopeName: unknown): boolean {
  if (typeof scopeName !== "string") {
    return false;
  }
  return VERCEL_AI_SCOPE_NAMES.has(scopeName) || scopeName.includes("ai-sdk");
}

export function modernOperationName(name: string, attrs: SpanAttributes = {}): string | undefined {
  const fromAttrs = attrs[ATTR_GEN_AI_OPERATION_NAME];
  if (fromAttrs !== undefined && fromAttrs !== null) {
    return String(fromAttrs);
  }

  const [firstToken] = name.trim().split(/\s+/, 1);
  if (firstToken && MODERN_OPERATION_LOG_TYPES[firstToken] !== undefined) {
    return firstToken;
  }

  return undefined;
}

export function isModernVercelAISpanName(name: string): boolean {
  return modernOperationName(name) !== undefined;
}

export function isVercelAISpan(span: ReadableSpan): boolean {
  if (isVercelAIScope(instrumentationScopeName(span))) {
    return true;
  }
  if (span.attributes[AI_SDK] !== undefined) {
    return true;
  }
  if (span.attributes[ATTR_GEN_AI_OPERATION_NAME] !== undefined) {
    return true;
  }
  return span.name.startsWith(AI_PREFIX) || isModernVercelAISpanName(span.name);
}

export function resolveLogType(name: string, attrs: SpanAttributes): string {
  const config = VERCEL_SPAN_CONFIG[name];
  if (config) {
    return config.logType;
  }

  const parentType = VERCEL_PARENT_SPANS[name];
  if (parentType) {
    return parentType;
  }

  const operationId = attrs[AI_OPERATION_ID]?.toString();
  if (operationId) {
    const operationConfig = VERCEL_SPAN_CONFIG[operationId];
    if (operationConfig) {
      return operationConfig.logType;
    }

    const operationParentType = VERCEL_PARENT_SPANS[operationId];
    if (operationParentType) {
      return operationParentType;
    }
  }

  const operationName = modernOperationName(name, attrs);
  if (operationName && MODERN_OPERATION_LOG_TYPES[operationName]) {
    return MODERN_OPERATION_LOG_TYPES[operationName];
  }

  if (
    attrs[AI_EMBEDDING] || attrs[AI_EMBEDDINGS] ||
    name.includes("embed") || operationId?.includes("embed")
  ) {
    return RespanLogType.EMBEDDING;
  }

  if (
    attrs[AI_TOOL_CALL_ID] || attrs[AI_TOOL_CALL_NAME] ||
    attrs[AI_TOOL_CALL_ARGS] || attrs[AI_TOOL_CALL_RESULT] ||
    attrs[ATTR_GEN_AI_TOOL_CALL_ID] || attrs[ATTR_GEN_AI_TOOL_NAME] ||
    attrs[ATTR_GEN_AI_TOOL_CALL_ARGUMENTS] || attrs[ATTR_GEN_AI_TOOL_CALL_RESULT] ||
    attrs[AI_RESPONSE_TOOL_CALLS] ||
    name.includes("tool") || operationId?.includes("tool")
  ) {
    return RespanLogType.TOOL;
  }

  if (
    attrs[AI_AGENT_ID] ||
    name.includes("agent") || operationId?.includes("agent")
  ) {
    return RespanLogType.AGENT;
  }

  if (
    attrs[AI_WORKFLOW_ID] ||
    name.includes("workflow") || operationId?.includes("workflow")
  ) {
    return RespanLogType.WORKFLOW;
  }

  if (
    attrs[AI_TRANSCRIPT] ||
    name.includes("transcript") || operationId?.includes("transcript")
  ) {
    return RespanLogType.TEXT;
  }

  if (
    attrs[AI_SPEECH] ||
    name.includes("speech") || operationId?.includes("speech")
  ) {
    return RespanLogType.TEXT;
  }

  if (name.includes("doGenerate") || name.includes("doStream")) {
    return RespanLogType.TEXT;
  }

  return RespanLogType.TASK;
}

export function normalizeModel(modelId: string): string {
  const model = modelId.toLowerCase();

  if (model.includes("gemini-2.0-flash-001")) return "gemini/gemini-2.0-flash";
  if (model.includes("gemini-2.0-pro")) return "gemini/gemini-2.0-pro-exp-02-05";
  if (model.includes("claude-3-5-sonnet")) return "claude-3-5-sonnet-20241022";
  if (model.includes("deepseek")) return `deepseek/${model}`;
  if (model.includes("o3-mini")) return "o3-mini";

  return model;
}
