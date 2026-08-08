import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes as TraceloopSpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  COHERE_SYSTEM,
  INSTRUMENTATION_NAME,
  MESSAGE_CONTENT_SUFFIX,
  MESSAGE_TOOL_CALLS_SUFFIX,
} from "./_constants.js";
import { safeJson, type SpanAttributes } from "./_utils.js";

const PROMPT_PREFIX = `${TraceloopSpanAttributes.LLM_PROMPTS}.`;
const COMPLETION_PREFIX = `${TraceloopSpanAttributes.LLM_COMPLETIONS}.`;

const OFF_CONTRACT_ALIASES = new Set([
  "llm.system",
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "tools",
  "tool_calls",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  "respan.span.tools",
  "respan.span.tool_calls",
  "respan.span.handoffs",
]);

export function requestTypeForOperation(operation: string): string {
  if (operation === "chat" || operation === "chatStream") return "chat";
  if (operation === "generate" || operation === "generateStream") {
    return "completion";
  }
  if (operation === "embed") return "embedding";
  if (operation === "rerank") return "rerank";
  return "unknown";
}

export function logTypeForOperation(operation: string): string {
  if (operation === "chat" || operation === "chatStream") {
    return RespanLogType.CHAT;
  }
  if (operation === "generate" || operation === "generateStream") {
    return RespanLogType.TEXT;
  }
  if (operation === "embed") return RespanLogType.EMBEDDING;
  if (operation === "rerank") return RespanLogType.TASK;
  return RespanLogType.UNKNOWN;
}

export function isCohereSpan(span: ReadableSpan): boolean {
  const attrs = ((span as any).attributes ?? {}) as SpanAttributes;
  const scopeName =
    (span as any).instrumentationScope?.name ??
    (span as any).instrumentationScope?.name ??
    "";

  if (scopeName === INSTRUMENTATION_NAME) return true;
  if (typeof span.name === "string" && span.name.startsWith("cohere.")) {
    return true;
  }

  const system = attrs[TraceloopSpanAttributes.LLM_SYSTEM] ?? attrs["llm.system"];
  return typeof system === "string" && system.toLowerCase() === COHERE_SYSTEM;
}

function setTokenAliases(attrs: SpanAttributes): void {
  const promptTokens = attrs[TraceloopSpanAttributes.LLM_USAGE_PROMPT_TOKENS];
  const inputTokens = attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS];
  if (promptTokens === undefined && inputTokens !== undefined) {
    attrs[TraceloopSpanAttributes.LLM_USAGE_PROMPT_TOKENS] = inputTokens;
  } else if (promptTokens !== undefined && inputTokens === undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = promptTokens;
  }

  const completionTokens = attrs[TraceloopSpanAttributes.LLM_USAGE_COMPLETION_TOKENS];
  const outputTokens = attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS];
  if (completionTokens === undefined && outputTokens !== undefined) {
    attrs[TraceloopSpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = outputTokens;
  } else if (completionTokens !== undefined && outputTokens === undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = completionTokens;
  }

  const totalTokens = attrs[TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS];
  const resolvedInput = attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS];
  const resolvedOutput = attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS];
  if (
    totalTokens === undefined &&
    typeof resolvedInput === "number" &&
    typeof resolvedOutput === "number"
  ) {
    attrs[TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = resolvedInput + resolvedOutput;
  }
}

function movePromptUserAttrs(attrs: SpanAttributes): void {
  for (const key of Object.keys(attrs)) {
    if (!key.startsWith(PROMPT_PREFIX) || !key.endsWith(".user")) continue;
    const contentKey = `${key.slice(0, -".user".length)}.${MESSAGE_CONTENT_SUFFIX}`;
    if (attrs[contentKey] === undefined) {
      attrs[contentKey] = attrs[key];
    }
    delete attrs[key];
  }
}

function stringifyStructuredCanonicalValues(attrs: SpanAttributes): void {
  if (attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] !== undefined) {
    attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(
      attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS],
    );
  }

  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith(PROMPT_PREFIX) && !key.startsWith(COMPLETION_PREFIX)) {
      continue;
    }
    if (
      key.endsWith(`.${MESSAGE_TOOL_CALLS_SUFFIX}`) ||
      (key.endsWith(`.${MESSAGE_CONTENT_SUFFIX}`) &&
        value !== null &&
        typeof value === "object")
    ) {
      attrs[key] = safeJson(value);
    }
  }
}

export function normalizeCohereAttrs(
  attrs: SpanAttributes,
  operation?: string,
): SpanAttributes {
  attrs[TraceloopSpanAttributes.LLM_SYSTEM] = COHERE_SYSTEM;

  if (operation) {
    attrs[TraceloopSpanAttributes.LLM_REQUEST_TYPE] = requestTypeForOperation(operation);
    attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = logTypeForOperation(operation);
  } else if (attrs[TraceloopSpanAttributes.LLM_REQUEST_TYPE] !== undefined) {
    const requestType = String(attrs[TraceloopSpanAttributes.LLM_REQUEST_TYPE]);
    if (requestType === "chat") attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.CHAT;
    if (requestType === "completion") attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.TEXT;
    if (requestType === "embedding") {
      attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.EMBEDDING;
    }
    if (requestType === "rerank") attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.TASK;
  }

  if (attrs[TraceloopSpanAttributes.LLM_REQUEST_MODEL] !== undefined) {
    attrs[TraceloopSpanAttributes.LLM_REQUEST_MODEL] = String(
      attrs[TraceloopSpanAttributes.LLM_REQUEST_MODEL],
    );
  }

  movePromptUserAttrs(attrs);
  setTokenAliases(attrs);
  stringifyStructuredCanonicalValues(attrs);
  delete attrs[TraceloopSpanAttributes.TRACELOOP_SPAN_KIND];

  for (const key of OFF_CONTRACT_ALIASES) {
    delete attrs[key];
  }

  return attrs;
}

export function normalizeCohereSpan(span: ReadableSpan): void {
  if (!isCohereSpan(span)) return;

  const attrs = ((span as any).attributes ?? {}) as SpanAttributes;
  const operation = span.name?.startsWith("cohere.")
    ? span.name.slice("cohere.".length)
    : undefined;
  normalizeCohereAttrs(attrs, operation);
  (span as any).attributes = attrs;
  (span as any)._attributes = attrs;
}
