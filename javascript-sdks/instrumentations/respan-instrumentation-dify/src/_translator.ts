import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  LLMRequestTypeValues,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

export type AttributeRecord = Record<string, string | number | boolean | Array<string | number | boolean>>;

export interface DifyRequestOptionsLike {
  method?: unknown;
  path?: unknown;
  query?: unknown;
  data?: unknown;
  responseType?: unknown;
}

export interface BuildDifyAttributesOptions {
  request: DifyRequestOptionsLike;
  response?: unknown;
  streamEvents?: unknown[];
  error?: unknown;
  includeContent?: boolean;
}

const OFF_CONTRACT_ALIASES = new Set([
  "completion_tokens",
  "has_tool_calls",
  "model",
  "parallel_tool_calls",
  "prompt_tokens",
  "respan.span.handoffs",
  "respan.span.tool_calls",
  "respan.span.tools",
  "span_tools",
  "tool_calls",
  "tools",
  "total_request_tokens",
]);

const REDACTED = "[REDACTED]";
const SENSITIVE_KEY_FRAGMENTS = [
  "accesskey",
  "apikey",
  "authorization",
  "cookie",
  "credential",
  "password",
  "passwd",
  "passphrase",
  "privatekey",
  "secret",
  "token",
];
const NON_SECRET_TOKEN_KEYS = new Set([
  "cachedtokens",
  "completiontokens",
  "inputtokens",
  "maxtokens",
  "outputtokens",
  "prompttokens",
  "reasoningtokens",
  "tokencount",
  "totaltokens",
]);

const normalizedKey = (key: string): string =>
  key.toLowerCase().replace(/[^a-z0-9]/g, "");

const isSensitiveKey = (key: string): boolean => {
  const normalized = normalizedKey(key);
  if ([...NON_SECRET_TOKEN_KEYS].some((key) => normalized.endsWith(key))) return false;
  return SENSITIVE_KEY_FRAGMENTS.some((fragment) => normalized.includes(fragment));
};

export function redactSensitive(value: unknown, seen = new WeakSet<object>()): unknown {
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Error) return { name: value.name, message: value.message };
  if (value instanceof Date) return value.toISOString();
  if (!value || typeof value !== "object") return value;
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  if (Array.isArray(value)) return value.map((nested) => redactSensitive(nested, seen));
  const redacted: Record<string, unknown> = {};
  for (const [key, nested] of Object.entries(value)) {
    redacted[key] = isSensitiveKey(key) ? REDACTED : redactSensitive(nested, seen);
  }
  return redacted;
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
};

const stringValue = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined;

const integerValue = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isInteger(value)) return value;
  return undefined;
};

export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(redactSensitive(value));
  } catch {
    return JSON.stringify("[Unserializable]");
  }
}

const responsePayload = (response: unknown): Record<string, unknown> => {
  if (!isRecord(response)) return {};
  const looksLikeSdkResponse =
    typeof response.status === "number" &&
    ("headers" in response || "requestId" in response || "url" in response);
  const payload = looksLikeSdkResponse ? response.data : response;
  return isRecord(payload) ? payload : {};
};

const streamPayload = (event: unknown): Record<string, unknown> => {
  if (!isRecord(event)) return {};
  if (isRecord(event.data)) return event.data;
  return event;
};

const endpointKind = (path: string): "chat" | "completion" | "workflow" | "api" => {
  if (path === "/chat-messages") return "chat";
  if (path === "/completion-messages") return "completion";
  if (path === "/workflows" || path.startsWith("/workflows/") || path.endsWith("/pipeline/run")) {
    return "workflow";
  }
  return "api";
};

export const difySpanName = (path: string): string => {
  switch (endpointKind(path)) {
    case "chat":
      return "dify.chat";
    case "completion":
      return "dify.completion";
    case "workflow":
      return "dify.workflow";
    default:
      return "dify.request";
  }
};

const logType = (path: string): RespanLogType => {
  switch (endpointKind(path)) {
    case "chat":
      return RespanLogType.CHAT;
    case "completion":
      return RespanLogType.TEXT;
    case "workflow":
      return RespanLogType.WORKFLOW;
    default:
      return RespanLogType.TASK;
  }
};

const requestBody = (request: DifyRequestOptionsLike): Record<string, unknown> =>
  isRecord(request.data) ? request.data : {};

const outputFromPayload = (payload: Record<string, unknown>): unknown => {
  if (payload.answer !== undefined) return payload.answer;
  if (payload.text !== undefined) return payload.text;
  if (isRecord(payload.data)) {
    if (payload.data.outputs !== undefined) return payload.data.outputs;
    if (payload.data.error) return payload.data.error;
  }
  return payload;
};

const streamOutput = (events: unknown[]): unknown => {
  const text: string[] = [];
  for (const event of events) {
    const payload = streamPayload(event);
    const part = payload.answer ?? payload.text ?? payload.delta;
    if (typeof part === "string") text.push(part);
    if (isRecord(payload.data)) {
      const nested = payload.data.answer ?? payload.data.text ?? payload.data.delta;
      if (typeof nested === "string") text.push(nested);
    }
  }
  if (text.length) return text.join("");
  for (const event of [...events].reverse()) {
    const payload = streamPayload(event);
    const output = outputFromPayload(payload);
    if (output !== payload || Object.keys(payload).length) return output;
  }
  return events;
};

const usageFromPayload = (payload: Record<string, unknown>): Record<string, unknown> => {
  if (isRecord(payload.metadata) && isRecord(payload.metadata.usage)) {
    return payload.metadata.usage;
  }
  if (isRecord(payload.data)) return payload.data;
  return {};
};

const usageFor = (payload: Record<string, unknown>, events: unknown[]): Record<string, unknown> => {
  for (const event of [...events].reverse()) {
    const usage = usageFromPayload(streamPayload(event));
    if (Object.keys(usage).length) return usage;
  }
  return usageFromPayload(payload);
};

const streamMetadataPayload = (events: unknown[]): Record<string, unknown> => {
  const merged: Record<string, unknown> = {};
  for (const event of events) Object.assign(merged, streamPayload(event));
  return merged;
};

const modelFor = (
  payload: Record<string, unknown>,
  usage: Record<string, unknown>,
  body: Record<string, unknown>,
): string | undefined => {
  const metadata = isRecord(payload.metadata) ? payload.metadata : {};
  const data = isRecord(payload.data) ? payload.data : {};
  return [payload.model, metadata.model, usage.model, data.model, body.model]
    .map(stringValue)
    .find(Boolean);
};

const completionPrompt = (body: Record<string, unknown>): string => {
  if (body.query !== undefined) return typeof body.query === "string" ? body.query : safeJson(body.query);
  if (isRecord(body.inputs) && body.inputs.query !== undefined) {
    return typeof body.inputs.query === "string" ? body.inputs.query : safeJson(body.inputs.query);
  }
  return body.inputs === undefined
    ? ""
    : typeof body.inputs === "string"
      ? body.inputs
      : safeJson(body.inputs);
};

const addMetadata = (
  attrs: AttributeRecord,
  request: DifyRequestOptionsLike,
  payload: Record<string, unknown>,
): void => {
  const metadata: Record<string, unknown> = {
    "dify.method": String(request.method ?? "").toUpperCase(),
    "dify.endpoint": String(request.path ?? ""),
  };
  for (const key of [
    "event",
    "task_id",
    "id",
    "message_id",
    "conversation_id",
    "workflow_run_id",
    "mode",
    "status",
  ]) {
    if (payload[key] !== undefined && payload[key] !== null) metadata[`dify.${key}`] = payload[key];
  }
  if (isRecord(payload.data)) {
    for (const key of ["workflow_id", "elapsed_time", "total_steps", "total_tokens"]) {
      if (payload.data[key] !== undefined && payload.data[key] !== null) {
        metadata[`dify.${key}`] = payload.data[key];
      }
    }
  }
  attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson(metadata);
};

export function buildDifySpanAttributes({
  request,
  response,
  streamEvents = [],
  error,
  includeContent = true,
}: BuildDifyAttributesOptions): AttributeRecord {
  const path = String(request.path ?? "");
  const body = requestBody(request);
  const payload = responsePayload(response);
  const semanticPayload = streamEvents.length
    ? { ...payload, ...streamMetadataPayload(streamEvents) }
    : payload;
  const kind = endpointKind(path);
  const name = difySpanName(path);
  const output = error ? String(error instanceof Error ? error.message : error) : streamEvents.length
    ? streamOutput(streamEvents)
    : outputFromPayload(payload);
  const usage = usageFor(payload, streamEvents);

  const attrs: AttributeRecord = {
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType(path),
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
  };
  addMetadata(attrs, request, semanticPayload);
  attrs.status_code = statusCodeFrom(response, error);
  if (error !== undefined) {
    attrs[ATTR_ERROR_MESSAGE] = error instanceof Error ? error.message : String(error);
  }

  const user = body.user ?? (isRecord(request.query) ? request.query.user : undefined);
  if (user !== undefined && user !== null) {
    attrs[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID] = String(user);
  }
  const conversationId = body.conversation_id ??
    (isRecord(request.query) ? request.query.conversation_id : undefined);
  if (conversationId !== undefined && conversationId !== null) {
    attrs[RespanSpanAttributes.RESPAN_THREADS_ID] = String(conversationId);
  }

  const promptTokens = integerValue(usage.prompt_tokens);
  const completionTokens = integerValue(usage.completion_tokens);
  const totalTokens = integerValue(usage.total_tokens) ??
    (promptTokens !== undefined || completionTokens !== undefined
      ? (promptTokens ?? 0) + (completionTokens ?? 0)
      : undefined);
  if (promptTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = promptTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = promptTokens;
  }
  if (completionTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = completionTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = completionTokens;
  }
  if (totalTokens !== undefined) attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;

  if (kind === "chat" || kind === "completion") {
    attrs[ATTR_GEN_AI_SYSTEM] = "dify";
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT;
    const model = modelFor(semanticPayload, usage, body);
    if (model) attrs[ATTR_GEN_AI_REQUEST_MODEL] = model;
  }

  if (includeContent) {
    const input = request.data ?? request.query ?? {};
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(input);
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] =
      typeof output === "string" ? output : safeJson(output);

    if (kind === "chat" || kind === "completion") {
      const prompt = kind === "chat"
        ? (typeof body.query === "string" ? body.query : safeJson(body.query ?? ""))
        : completionPrompt(body);
      if (prompt) {
        attrs[`${SpanAttributes.LLM_PROMPTS}.0.role`] = "user";
        attrs[`${SpanAttributes.LLM_PROMPTS}.0.content`] = prompt;
      }
      const completion = typeof output === "string" ? output : safeJson(output);
      if (completion) {
        attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.role`] = "assistant";
        attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.content`] = completion;
      }
    }
  }

  for (const alias of OFF_CONTRACT_ALIASES) delete attrs[alias];
  return attrs;
}

export function statusCodeFrom(response: unknown, error?: unknown): number {
  if (
    error &&
    typeof error === "object" &&
    typeof (error as { statusCode?: unknown }).statusCode === "number"
  ) {
    return (error as { statusCode: number }).statusCode;
  }
  if (error) return 500;
  if (isRecord(response) && typeof response.status === "number") return response.status;
  return 200;
}
