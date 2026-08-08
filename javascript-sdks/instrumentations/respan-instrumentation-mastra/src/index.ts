import { context, trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  RespanLogType,
  RespanSpanAttributes,
  ToolCallSchema,
} from "@respan/respan-sdk";
import {
  buildReadableSpan,
  ensureTraceId,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "0.1.0";
const INSTRUMENTATION_NAME = "@respan/instrumentation-mastra";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const MASTRA_SPAN_ENDED_EVENT = "span_ended";
const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;
const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";

const DEFAULT_EXCLUDED_SPAN_TYPES = new Set(["model_chunk"]);

export interface MastraInstrumentorOptions {
  excludeSpanTypes?: string[];
}

interface MastraTracingEvent {
  type: string;
  exportedSpan?: MastraExportedSpan;
}

interface MastraExportedSpan {
  id: string;
  traceId: string;
  name: string;
  type: string;
  parentSpanId?: string;
  isRootSpan?: boolean;
  startTime: Date | string | number;
  endTime?: Date | string | number;
  attributes?: Record<string, any>;
  metadata?: Record<string, any>;
  requestContext?: Record<string, any>;
  tags?: string[];
  input?: any;
  output?: any;
  errorInfo?: {
    message?: string;
    name?: string;
    stack?: string;
    details?: Record<string, any>;
  };
  entityId?: string;
  entityName?: string;
  entityType?: string;
}

export class MastraInstrumentor {
  public readonly name = "mastra";

  private _enabled = true;
  private readonly _excludedSpanTypes: Set<string>;
  private readonly _traceIdMap = new Map<string, string>();
  private readonly _emittedSpanIds = new Set<string>();
  private readonly _pendingToolSpans = new Map<string, MastraExportedSpan[]>();

  constructor(options: MastraInstrumentorOptions = {}) {
    this._excludedSpanTypes = new Set([
      ...DEFAULT_EXCLUDED_SPAN_TYPES,
      ...(options.excludeSpanTypes ?? []),
    ]);
  }

  activate(): void {
    this._enabled = true;
  }

  deactivate(): void {
    this._enabled = false;
    this._traceIdMap.clear();
    this._emittedSpanIds.clear();
    this._pendingToolSpans.clear();
  }

  onTracingEvent(event: MastraTracingEvent): void {
    void this.exportTracingEvent(event);
  }

  async exportTracingEvent(event: MastraTracingEvent): Promise<void> {
    if (!this._enabled || event.type !== MASTRA_SPAN_ENDED_EVENT) {
      return;
    }

    const span = event.exportedSpan;
    if (!span || this._excludedSpanTypes.has(span.type)) {
      return;
    }

    if (this._shouldBufferToolSpan(span)) {
      const pendingKey = resolvePendingToolKey(span);
      const pendingSpans = this._pendingToolSpans.get(pendingKey) ?? [];
      pendingSpans.push(span);
      this._pendingToolSpans.set(pendingKey, pendingSpans);
      return;
    }

    const readableSpan = this._buildReadableSpan(span);
    injectSpan(readableSpan);
    this._rememberEmittedSpan(span, readableSpan);

    if (span.type === "agent_run") {
      this._drainPendingToolSpans(span, readableSpan.spanContext().spanId);
    }
  }

  async flush(): Promise<void> {
    // RespanTelemetry owns flushing; this exporter injects synchronously.
  }

  async shutdown(): Promise<void> {
    this._drainAllPendingToolSpans();
    this._traceIdMap.clear();
    this._emittedSpanIds.clear();
    this._pendingToolSpans.clear();
  }

  private _buildReadableSpan(
    span: MastraExportedSpan,
    forcedParentId?: string,
  ): ReadableSpan {
    const activeSpanContext = trace.getSpan(context.active())?.spanContext();
    const activeTraceId = isUsableTraceId(activeSpanContext?.traceId)
      ? ensureTraceId(activeSpanContext?.traceId)
      : undefined;
    const traceId = this._resolveTraceId(span.traceId, activeTraceId);
    const parentId = forcedParentId ?? (span.parentSpanId ?? (span as any).parentSpanContext?.spanId) ?? (
      !(span.parentSpanId ?? (span as any).parentSpanContext?.spanId) && activeTraceId === traceId
        ? activeSpanContext?.spanId
        : undefined
    );

    const readableSpan = buildReadableSpan({
      name: span.name || resolveEntityName(span),
      traceId,
      spanId: span.id,
      parentId,
      startTimeIso: toIsoString(span.startTime),
      endTimeIso: toIsoString(span.endTime ?? new Date()),
      attributes: buildMastraAttributes(span),
      statusCode: span.errorInfo ? 500 : 200,
      errorMessage: span.errorInfo?.message,
    }) as ReadableSpan & {
      instrumentationScope?: { name: string; version?: string };
    };

    readableSpan.instrumentationScope = {
      name: INSTRUMENTATION_NAME,
      version: PACKAGE_VERSION,
    };
    return readableSpan;
  }

  private _resolveTraceId(mastraTraceId: string, activeTraceId?: string): string {
    const existingTraceId = this._traceIdMap.get(mastraTraceId);
    if (existingTraceId) {
      return existingTraceId;
    }

    const resolvedTraceId = activeTraceId ?? ensureTraceId(mastraTraceId);
    this._traceIdMap.set(mastraTraceId, resolvedTraceId);
    return resolvedTraceId;
  }

  private _shouldBufferToolSpan(span: MastraExportedSpan): boolean {
    return isToolSpan(span) && (!(span.parentSpanId ?? (span as any).parentSpanContext?.spanId) || !this._emittedSpanIds.has((span.parentSpanId ?? (span as any).parentSpanContext?.spanId)));
  }

  private _rememberEmittedSpan(span: MastraExportedSpan, readableSpan: ReadableSpan): void {
    this._emittedSpanIds.add(span.id);
    this._emittedSpanIds.add(readableSpan.spanContext().spanId);
  }

  private _drainPendingToolSpans(agentSpan: MastraExportedSpan, parentId: string): void {
    const pendingKey = resolvePendingToolKey(agentSpan);
    const pendingSpans = this._pendingToolSpans.get(pendingKey);
    if (!pendingSpans || pendingSpans.length === 0) {
      return;
    }

    this._pendingToolSpans.delete(pendingKey);
    for (const pendingSpan of pendingSpans) {
      const readableSpan = this._buildReadableSpan(pendingSpan, parentId);
      injectSpan(readableSpan);
      this._rememberEmittedSpan(pendingSpan, readableSpan);
    }
  }

  private _drainAllPendingToolSpans(): void {
    for (const pendingSpans of this._pendingToolSpans.values()) {
      for (const pendingSpan of pendingSpans) {
        const readableSpan = this._buildReadableSpan(pendingSpan);
        injectSpan(readableSpan);
        this._rememberEmittedSpan(pendingSpan, readableSpan);
      }
    }
    this._pendingToolSpans.clear();
  }
}

export { MastraInstrumentor as RespanMastraExporter };

function buildMastraAttributes(span: MastraExportedSpan): Record<string, unknown> {
  const attrs: Record<string, unknown> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: resolveEntityName(span),
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: span.name || resolveEntityName(span),
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: resolveLogType(span.type),
    [metadataKey("mastra_span_type")]: span.type,
    [metadataKey("mastra_span_id")]: span.id,
  };

  if (span.isRootSpan || span.type === "agent_run" || span.type === "workflow_run") {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = resolveEntityName(span);
  }

  if (span.entityId) {
    attrs[metadataKey("mastra_entity_id")] = span.entityId;
  }
  if (span.entityName) {
    attrs[metadataKey("mastra_entity_name")] = span.entityName;
  }
  if (span.entityType) {
    attrs[metadataKey("mastra_entity_type")] = span.entityType;
  }
  if (span.tags && span.tags.length > 0) {
    attrs[metadataKey("mastra_tags")] = safeJson(span.tags);
  }

  mergeMetadata(attrs, span.metadata);
  mergeRequestContext(attrs, span.requestContext);
  addInputOutput(attrs, span);
  addTypeSpecificAttributes(attrs, span);

  return attrs;
}

function addInputOutput(
  attrs: Record<string, unknown>,
  span: MastraExportedSpan,
): void {
  if (span.input !== undefined) {
    const messages = normalizeMessages(span.input);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = messages
      ? safeJson(messages)
      : safeJson(span.input);
    if (isModelSpan(span.type) && messages) {
      addPromptMessageAttributes(attrs, messages);
    }
  }

  if (span.output !== undefined) {
    if (isModelSpan(span.type)) {
      const outputMessage = normalizeAssistantOutput(span.output);
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(outputMessage);
      attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
      attrs[GEN_AI_COMPLETION_CONTENT] = String(outputMessage.content ?? "");
      if (outputMessage.tool_calls) {
        attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(outputMessage.tool_calls);
      }
      return;
    }

    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] =
      typeof span.output === "string" ? span.output : safeJson(span.output);
  }
}

function addTypeSpecificAttributes(
  attrs: Record<string, unknown>,
  span: MastraExportedSpan,
): void {
  const spanAttrs = span.attributes ?? {};

  switch (span.type) {
    case "agent_run":
      attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = resolveEntityName(span);
      break;
    case "model_generation":
    case "model_step":
    case "model_inference":
      attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
      addModelAttributes(attrs, spanAttrs);
      addUsageAttributes(attrs, spanAttrs.usage ?? spanAttrs.internalUsage);
      addToolDefinitions(attrs, spanAttrs.availableTools);
      addToolCalls(attrs, collectToolCalls(span.output));
      break;
    case "rag_embedding":
      attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.EMBEDDING;
      addModelAttributes(attrs, spanAttrs);
      addUsageAttributes(attrs, spanAttrs.usage ?? spanAttrs.internalUsage);
      break;
    case "tool_call":
    case "mcp_tool_call":
      if (spanAttrs.success !== undefined) {
        attrs[metadataKey("mastra_tool_success")] = String(Boolean(spanAttrs.success));
      }
      break;
    default:
      if (spanAttrs.internalUsage) {
        addUsageAttributes(attrs, spanAttrs.internalUsage);
      }
      break;
  }
}

function addModelAttributes(
  attrs: Record<string, unknown>,
  spanAttrs: Record<string, any>,
): void {
  const rawModel = spanAttrs.responseModel ?? spanAttrs.model;
  if (rawModel) {
    const modelInfo = parseModelInfo(String(rawModel), spanAttrs.provider);
    attrs[ATTR_GEN_AI_REQUEST_MODEL] = modelInfo.model;
    if (modelInfo.provider) {
      attrs[ATTR_GEN_AI_SYSTEM] = modelInfo.provider;
    }
  } else if (spanAttrs.provider) {
    attrs[ATTR_GEN_AI_SYSTEM] = String(spanAttrs.provider);
  }

  if (spanAttrs.finishReason) {
    attrs[metadataKey("finish_reason")] = String(spanAttrs.finishReason);
  }
  if (spanAttrs.streaming !== undefined) {
    attrs[metadataKey("stream")] = String(Boolean(spanAttrs.streaming));
  }
}

function addUsageAttributes(
  attrs: Record<string, unknown>,
  usage: unknown,
): void {
  if (!usage || typeof usage !== "object" || Array.isArray(usage)) {
    return;
  }

  const usageRecord = usage as Record<string, any>;
  const inputTokens = coerceInteger(
    usageRecord.inputTokens ?? usageRecord.promptTokens,
  );
  const outputTokens = coerceInteger(
    usageRecord.outputTokens ?? usageRecord.completionTokens,
  );
  const cacheReadTokens = coerceInteger(
    usageRecord.inputDetails?.cacheRead ?? usageRecord.cacheReadInputTokens,
  );

  if (inputTokens !== null) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = inputTokens;
  }
  if (outputTokens !== null) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = outputTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = outputTokens;
  }
  if (inputTokens !== null || outputTokens !== null) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = (inputTokens ?? 0) + (outputTokens ?? 0);
  }
  if (cacheReadTokens !== null) {
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cacheReadTokens;
  }
}

function addToolDefinitions(
  attrs: Record<string, unknown>,
  availableTools: unknown,
): void {
  const tools = normalizeToolDefinitions(availableTools);
  if (tools.length === 0) {
    return;
  }

  const serializedTools = safeJson(tools);
  attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = serializedTools;
}

function addToolCalls(
  attrs: Record<string, unknown>,
  toolCalls: Record<string, unknown>[],
): void {
  if (toolCalls.length === 0) {
    return;
  }

  const serializedToolCalls = safeJson(dedupeToolCalls(toolCalls));
  attrs[GEN_AI_COMPLETION_TOOL_CALLS] = serializedToolCalls;
}

function resolveLogType(spanType: string): string {
  switch (spanType) {
    case "agent_run":
      return RespanLogType.AGENT;
    case "model_generation":
    case "model_step":
    case "model_inference":
      return RespanLogType.CHAT;
    case "rag_embedding":
      return RespanLogType.EMBEDDING;
    case "tool_call":
    case "mcp_tool_call":
      return RespanLogType.TOOL;
    case "workflow_run":
      return RespanLogType.WORKFLOW;
    case "scorer_run":
    case "scorer_step":
      return RespanLogType.GUARDRAIL;
    default:
      return RespanLogType.TASK;
  }
}

function isModelSpan(spanType: string): boolean {
  return ["model_generation", "model_step", "model_inference"].includes(spanType);
}

function isToolSpan(span: MastraExportedSpan): boolean {
  return ["tool_call", "mcp_tool_call"].includes(span.type);
}

function resolvePendingToolKey(span: MastraExportedSpan): string {
  const runId = span.metadata?.runId ?? span.metadata?.run_id ?? span.attributes?.runId;
  return `${span.traceId}:${runId ? String(runId) : ""}`;
}

function resolveEntityName(span: MastraExportedSpan): string {
  return span.entityName || span.name || span.type || "mastra";
}

function normalizeMessages(value: unknown): Array<Record<string, unknown>> | null {
  const payload = unwrapKnownPayload(value);
  if (typeof payload === "string") {
    return [{ role: "user", content: payload }];
  }

  if (Array.isArray(payload)) {
    const messages = payload.flatMap((item) => normalizeMessage(item));
    return messages.length > 0 ? messages : null;
  }

  const message = normalizeMessage(payload);
  return message.length > 0 ? message : null;
}

function normalizeMessage(value: unknown): Array<Record<string, unknown>> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }

  const record = value as Record<string, unknown>;
  if (typeof record.role === "string") {
    const normalized: Record<string, unknown> = {
      role: record.role,
      content: normalizeContent(record.content),
    };
    const toolCalls = normalizeToolCalls(record.tool_calls ?? record.toolCalls);
    if (toolCalls.length > 0) {
      normalized.tool_calls = toolCalls;
    }
    return [normalized];
  }

  for (const key of ["messages", "input", "prompt"]) {
    if (record[key] !== undefined) {
      return normalizeMessages(record[key]) ?? [];
    }
  }

  return [];
}

function normalizeAssistantOutput(value: unknown): Record<string, unknown> {
  const payload = unwrapKnownPayload(value);
  const content = extractText(payload);
  const message: Record<string, unknown> = {
    role: "assistant",
    content: content || (typeof payload === "string" ? payload : safeJson(payload)),
  };
  const toolCalls = collectToolCalls(payload);
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }
  return message;
}

function collectToolCalls(value: unknown): Record<string, unknown>[] {
  const calls: Record<string, unknown>[] = [];

  const visit = (item: unknown): void => {
    if (!item || typeof item !== "object") {
      return;
    }

    if (Array.isArray(item)) {
      item.forEach(visit);
      return;
    }

    const record = item as Record<string, unknown>;
    const directCalls = normalizeToolCalls(
      record.toolCalls ?? record.tool_calls ?? record.toolCall ?? record.tool_call,
    );
    calls.push(...directCalls);

    if (Array.isArray(record.steps)) {
      record.steps.forEach(visit);
    }
    if (Array.isArray(record.content)) {
      record.content.forEach(visit);
    }
  };

  visit(value);
  return dedupeToolCalls(calls);
}

function normalizeToolCalls(value: unknown): Record<string, unknown>[] {
  const rawCalls = Array.isArray(value) ? value : value ? [value] : [];
  return rawCalls
    .map((call) => normalizeToolCall(call))
    .filter((call): call is Record<string, unknown> => call !== null);
}

function normalizeToolCall(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  const record = value as Record<string, unknown>;
  const functionPayload =
    record.function && typeof record.function === "object" && !Array.isArray(record.function)
      ? (record.function as Record<string, unknown>)
      : undefined;
  const toolName =
    functionPayload?.name ??
    record.toolName ??
    record.tool_name ??
    record.name;
  if (typeof toolName !== "string" || !toolName) {
    return null;
  }

  const rawArgs =
    functionPayload?.arguments ??
    functionPayload?.args ??
    record.args ??
    record.arguments ??
    record.input;
  const parsedToolCall = ToolCallSchema.safeParse({
    type: "function",
    id: String(record.id ?? record.toolCallId ?? record.tool_call_id ?? ""),
    name: toolName,
    args: toSerializableValue(rawArgs ?? {}),
  });

  if (parsedToolCall.success) {
    const normalizedToolCall = { ...(parsedToolCall.data as Record<string, unknown>) };
    delete normalizedToolCall.name;
    delete normalizedToolCall.args;
    return normalizedToolCall;
  }

  return {
    id: String(record.id ?? record.toolCallId ?? record.tool_call_id ?? ""),
    type: "function",
    function: {
      name: toolName,
      arguments: typeof rawArgs === "string" ? rawArgs : safeJson(rawArgs ?? {}),
    },
  };
}

function normalizeToolDefinitions(value: unknown): Record<string, unknown>[] {
  const rawTools = Array.isArray(value) ? value : value ? [value] : [];
  const tools = rawTools
    .map((tool): Record<string, unknown> | null => {
      if (typeof tool === "string" && tool) {
        return { type: "function", function: { name: tool } };
      }
      if (!tool || typeof tool !== "object" || Array.isArray(tool)) {
        return null;
      }
      const record = tool as Record<string, unknown>;
      const name = record.id ?? record.name ?? record.toolName;
      if (typeof name !== "string" || !name) {
        return null;
      }
      const normalizedFunction: Record<string, unknown> = { name };
      if (typeof record.description === "string") {
        normalizedFunction.description = record.description;
      }
      const parameters = record.inputSchema ?? record.parameters;
      if (parameters !== undefined) {
        normalizedFunction.parameters = toSerializableValue(parameters);
      }
      return { type: "function", function: normalizedFunction };
    })
    .filter((tool): tool is Record<string, unknown> => tool !== null);

  return dedupeToolDefinitions(tools);
}

function dedupeToolDefinitions(
  tools: Record<string, unknown>[],
): Record<string, unknown>[] {
  const seen = new Set<string>();
  return tools.filter((tool) => {
    const functionPayload =
      tool.function && typeof tool.function === "object" && !Array.isArray(tool.function)
        ? (tool.function as Record<string, unknown>)
        : {};
    const name = String(functionPayload.name ?? "");
    if (!name || seen.has(name)) {
      return false;
    }
    seen.add(name);
    return true;
  });
}

function dedupeToolCalls(
  toolCalls: Record<string, unknown>[],
): Record<string, unknown>[] {
  const seen = new Set<string>();
  return toolCalls.filter((toolCall) => {
    const functionPayload =
      toolCall.function && typeof toolCall.function === "object" && !Array.isArray(toolCall.function)
        ? (toolCall.function as Record<string, unknown>)
        : {};
    const signature = safeJson([
      toolCall.id ?? "",
      functionPayload.name ?? "",
      functionPayload.arguments ?? "",
    ]);
    if (seen.has(signature)) {
      return false;
    }
    seen.add(signature);
    return true;
  });
}

function addPromptMessageAttributes(
  attrs: Record<string, unknown>,
  messages: Array<Record<string, unknown>>,
): void {
  messages.forEach((message, index) => {
    const role = message.role;
    const content = message.content;
    if (typeof role === "string") {
      attrs[`${ATTR_GEN_AI_PROMPT}.${index}.role`] = role;
    }
    if (content !== undefined) {
      attrs[`${ATTR_GEN_AI_PROMPT}.${index}.content`] =
        typeof content === "string" ? content : safeJson(content);
    }
  });
}

function mergeMetadata(
  attrs: Record<string, unknown>,
  metadata?: Record<string, any>,
): void {
  if (!metadata) {
    return;
  }
  for (const [key, value] of Object.entries(metadata)) {
    attrs[metadataKey(key)] = typeof value === "string" ? value : safeJson(value);
  }
}

function mergeRequestContext(
  attrs: Record<string, unknown>,
  requestContext?: Record<string, any>,
): void {
  if (!requestContext) {
    return;
  }

  const customerId = requestContext.customer_identifier ?? requestContext.userId;
  if (customerId !== undefined) {
    attrs[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID] = String(customerId);
  }
  const threadId = requestContext.thread_identifier ?? requestContext.threadId;
  if (threadId !== undefined) {
    attrs[RespanSpanAttributes.RESPAN_THREADS_ID] = String(threadId);
  }
}

function parseModelInfo(
  rawModel: string,
  rawProvider?: unknown,
): { provider?: string; model: string } {
  if (typeof rawProvider === "string" && rawProvider) {
    return { provider: rawProvider, model: stripProviderPrefix(rawModel) };
  }

  const slashIndex = rawModel.indexOf("/");
  if (slashIndex > 0 && slashIndex < rawModel.length - 1) {
    return {
      provider: rawModel.slice(0, slashIndex),
      model: rawModel.slice(slashIndex + 1),
    };
  }

  return { model: rawModel };
}

function stripProviderPrefix(model: string): string {
  const slashIndex = model.indexOf("/");
  if (slashIndex > 0 && slashIndex < model.length - 1) {
    return model.slice(slashIndex + 1);
  }
  return model;
}

function unwrapKnownPayload(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }
  const record = value as Record<string, unknown>;
  return record.response ?? record.result ?? record.output ?? record.text ?? value;
}

function extractText(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(extractText).filter(Boolean).join("\n");
  }
  if (!value || typeof value !== "object") {
    return value === undefined || value === null ? "" : String(value);
  }
  const record = value as Record<string, unknown>;
  for (const key of ["text", "content", "message", "result", "output"]) {
    const text = extractText(record[key]);
    if (text) {
      return text;
    }
  }
  return "";
}

function normalizeContent(value: unknown): unknown {
  if (typeof value === "string") {
    return value;
  }
  const text = extractText(value);
  return text || toSerializableValue(value);
}

function metadataKey(key: string): string {
  return `respan.metadata.${key}`;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(toSerializableValue(value), (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return String(value);
  }
}

function toSerializableValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return undefined;
  }
  if (["string", "number", "boolean"].includes(typeof value)) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => toSerializableValue(item))
      .filter((item) => item !== undefined);
  }
  if (typeof value === "object") {
    const normalizedObject: Record<string, unknown> = {};
    for (const [key, itemValue] of Object.entries(value as Record<string, unknown>)) {
      const normalizedValue = toSerializableValue(itemValue);
      if (normalizedValue !== undefined) {
        normalizedObject[key] = normalizedValue;
      }
    }
    return normalizedObject;
  }
  if (typeof value === "function" || typeof value === "symbol") {
    return undefined;
  }
  return String(value);
}

function coerceInteger(value: unknown): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return null;
  }
  return Math.trunc(numericValue);
}

function toIsoString(value: Date | string | number | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (typeof value === "number") {
    return new Date(value).toISOString();
  }
  return new Date(value).toISOString();
}

function isUsableTraceId(traceId?: string): boolean {
  return Boolean(traceId && !/^0+$/.test(traceId));
}
