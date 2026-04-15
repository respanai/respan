import { context, trace, SpanKind, SpanStatusCode, TraceFlags } from "@opentelemetry/api";
import { hrTime, hrTimeDuration } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { randomBytes } from "node:crypto";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "1.0.0";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";

export interface PendingToolState {
  spanId: string;
  startTime: [number, number];
  toolName: string;
  toolInput: unknown;
}

export interface QueryState {
  agentName: string;
  agentSpanId: string;
  completionTokens?: number;
  errorMessage?: string;
  finalOutput?: unknown;
  inputMessages: Record<string, unknown>[];
  model?: string;
  outputMessages: Record<string, unknown>[];
  parentSpanId?: string;
  pendingTools: Map<string, PendingToolState>;
  prompt: unknown;
  promptCacheCreationTokens?: number;
  promptCacheHitTokens?: number;
  promptTokens?: number;
  sessionId?: string;
  startTime: [number, number];
  statusCode: number;
  toolCalls: Record<string, unknown>[];
  toolDefinitions?: Record<string, unknown>[];
  totalCostUsd?: number;
  totalRequestTokens?: number;
  traceFlags: number;
  traceId: string;
}

interface BuildSpanOptions {
  name: string;
  traceId: string;
  spanId: string;
  traceFlags?: number;
  parentId?: string;
  startTime: [number, number];
  endTime: [number, number];
  attributes: Record<string, unknown>;
  statusCode?: number;
  errorMessage?: string;
}

function randomHex(bytes: number): string {
  return randomBytes(bytes).toString("hex");
}

export function createQueryState({
  prompt,
  options,
  agentName,
}: {
  prompt: unknown;
  options?: Record<string, unknown>;
  agentName?: string;
}): QueryState {
  const activeSpan = trace.getSpan(context.active());
  const activeSpanContext = activeSpan?.spanContext();

  return {
    agentName: agentName?.trim() || inferAgentName(options) || "claude-agent-sdk",
    agentSpanId: randomHex(8),
    inputMessages: normalizeInputMessages(prompt, options),
    outputMessages: [],
    parentSpanId: activeSpanContext?.spanId,
    pendingTools: new Map(),
    prompt,
    startTime: hrTime(),
    statusCode: 200,
    toolCalls: [],
    toolDefinitions: normalizeToolDefinitions(options?.tools),
    traceFlags: activeSpanContext?.traceFlags ?? TraceFlags.SAMPLED,
    traceId: activeSpanContext?.traceId ?? randomHex(16),
  };
}

function inferAgentName(options?: Record<string, unknown>): string | undefined {
  const candidate =
    options?.agentName ??
    options?.name ??
    options?.agent_name;
  return typeof candidate === "string" && candidate.trim() ? candidate.trim() : undefined;
}

function normalizeInputMessages(
  prompt: unknown,
  options?: Record<string, unknown>,
): Record<string, unknown>[] {
  const messages: Record<string, unknown>[] = [];
  const systemInstructions =
    options?.system ??
    options?.systemPrompt ??
    options?.instructions;

  if (typeof systemInstructions === "string" && systemInstructions.trim()) {
    messages.push({
      role: "system",
      content: systemInstructions,
    });
  }

  if (typeof prompt === "string") {
    messages.push({
      role: "user",
      content: prompt,
    });
    return messages;
  }

  const serializedPrompt = toSerializableValue(prompt);
  if (Array.isArray(serializedPrompt)) {
    messages.push({
      role: "user",
      content: serializedPrompt,
    });
    return messages;
  }

  if (serializedPrompt !== undefined) {
    messages.push({
      role: "user",
      content: serializedPrompt,
    });
  }

  return messages;
}

function normalizeToolDefinitions(
  tools: unknown,
): Record<string, unknown>[] | undefined {
  if (!Array.isArray(tools)) {
    return undefined;
  }

  const normalizedTools = tools
    .map((tool) => normalizeToolDefinition(tool))
    .filter((tool): tool is Record<string, unknown> => tool !== null);

  return normalizedTools.length > 0 ? normalizedTools : undefined;
}

function normalizeToolDefinition(
  tool: unknown,
): Record<string, unknown> | null {
  if (typeof tool === "string" && tool) {
    return {
      type: "function",
      function: { name: tool },
    };
  }

  if (!tool || typeof tool !== "object" || Array.isArray(tool)) {
    return null;
  }

  const record = tool as Record<string, unknown>;
  const functionPayload =
    record.function && typeof record.function === "object" && !Array.isArray(record.function)
      ? (record.function as Record<string, unknown>)
      : null;

  if (functionPayload) {
    const functionName = functionPayload.name;
    if (typeof functionName !== "string" || !functionName) {
      return null;
    }

    const normalizedFunction: Record<string, unknown> = { name: functionName };
    for (const key of ["description", "parameters", "strict"]) {
      if (functionPayload[key] !== undefined) {
        normalizedFunction[key] = toSerializableValue(functionPayload[key]);
      }
    }

    return {
      type: record.type ?? "function",
      function: normalizedFunction,
    };
  }

  const toolName = record.name;
  if (typeof toolName !== "string" || !toolName) {
    return null;
  }

  const normalizedFunction: Record<string, unknown> = { name: toolName };
  if (record.description !== undefined) {
    normalizedFunction.description = toSerializableValue(record.description);
  }
  const parameters = record.input_schema ?? record.parameters;
  if (parameters !== undefined) {
    normalizedFunction.parameters = toSerializableValue(parameters);
  }

  return {
    type: record.type ?? "function",
    function: normalizedFunction,
  };
}

export function trackClaudeMessage(state: QueryState, message: unknown): void {
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return;
  }

  const record = message as Record<string, unknown>;
  updateSessionId(state, record.session_id ?? record.sessionId);

  switch (record.type) {
    case "system":
      handleSystemMessage(state, record);
      break;
    case "assistant":
      handleAssistantMessage(state, record);
      break;
    case "result":
      handleResultMessage(state, record);
      break;
    case "stream_event":
      updateSessionId(state, record.session_id ?? record.sessionId);
      break;
    default:
      break;
  }
}

function handleSystemMessage(state: QueryState, message: Record<string, unknown>): void {
  const data =
    message.data && typeof message.data === "object" && !Array.isArray(message.data)
      ? (message.data as Record<string, unknown>)
      : undefined;
  updateSessionId(state, data?.session_id ?? data?.sessionId ?? data?.id);
}

function handleAssistantMessage(state: QueryState, message: Record<string, unknown>): void {
  const payload = resolveAssistantPayload(message);
  const model = payload.model ?? message.model;
  if (typeof model === "string" && model) {
    state.model = model;
  }

  updateUsageFromMessage(
    state,
    payload.usage && typeof payload.usage === "object" && !Array.isArray(payload.usage)
      ? (payload.usage as Record<string, unknown>)
      : message.usage && typeof message.usage === "object" && !Array.isArray(message.usage)
        ? (message.usage as Record<string, unknown>)
      : undefined,
  );

  const rawContent = payload.content ?? message.content;
  const content = Array.isArray(rawContent)
    ? toSerializableValue(rawContent)
    : toSerializableValue(rawContent);
  if (hasMeaningfulContent(content)) {
    state.outputMessages.push({
      role: "assistant",
      content,
    });
  }

  if (Array.isArray(rawContent)) {
    for (const block of rawContent) {
      const toolCall = normalizeToolCall(block);
      if (toolCall) {
        addToolCall(state, toolCall);
      }
    }
  }
}

function handleResultMessage(state: QueryState, message: Record<string, unknown>): void {
  updateUsageFromMessage(
    state,
    message.usage && typeof message.usage === "object" && !Array.isArray(message.usage)
      ? (message.usage as Record<string, unknown>)
      : undefined,
  );

  if (typeof message.total_cost_usd === "number") {
    state.totalCostUsd = message.total_cost_usd;
  }

  if (message.is_error) {
    state.statusCode = 500;
    state.errorMessage = `agent_result_error:${String(message.subtype ?? "error")}`;
  }

  const outputValue =
    message.result ??
    message.structured_output;
  if (
    outputValue !== undefined &&
    !hasRenderableAssistantOutput(state.outputMessages)
  ) {
    state.finalOutput = toSerializableValue(outputValue);
    state.outputMessages.push({
      role: "assistant",
      content: toSerializableValue(outputValue),
    });
  } else if (outputValue !== undefined) {
    state.finalOutput = toSerializableValue(outputValue);
  }
}

function resolveAssistantPayload(
  message: Record<string, unknown>,
): Record<string, unknown> {
  const nestedMessage = message.message;
  if (
    nestedMessage &&
    typeof nestedMessage === "object" &&
    !Array.isArray(nestedMessage)
  ) {
    return nestedMessage as Record<string, unknown>;
  }
  return message;
}

function hasMeaningfulContent(value: unknown): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

function hasRenderableAssistantOutput(
  messages: Record<string, unknown>[],
): boolean {
  return messages.some((message) => hasRenderableContent(message.content));
}

function hasRenderableContent(value: unknown): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.some((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return hasRenderableContent(item);
      }

      const record = item as Record<string, unknown>;
      if (record.type === "thinking" || record.type === "tool_use") {
        return false;
      }
      if (record.type === "text") {
        return typeof record.text === "string" && record.text.trim().length > 0;
      }
      if ("content" in record) {
        return hasRenderableContent(record.content);
      }
      return Object.keys(record).length > 0;
    });
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

function updateUsageFromMessage(
  state: QueryState,
  usage?: Record<string, unknown>,
): void {
  if (!usage) {
    return;
  }

  const rawPromptTokens = coerceInteger(usage.input_tokens ?? usage.prompt_tokens);
  const completionTokens = coerceInteger(
    usage.output_tokens ?? usage.completion_tokens,
  );
  const cacheHitTokens = coerceInteger(usage.cache_read_input_tokens);
  const cacheCreationTokens = coerceInteger(usage.cache_creation_input_tokens);
  const totalTokens = coerceInteger(
    usage.total_tokens ??
      usage.totalTokens,
  );
  let promptTokens = rawPromptTokens;

  if (
    promptTokens !== null &&
    (cacheHitTokens !== null || cacheCreationTokens !== null)
  ) {
    const uncachedPromptTokens =
      promptTokens -
      ((cacheHitTokens ?? 0) + (cacheCreationTokens ?? 0));
    if (uncachedPromptTokens >= 0) {
      promptTokens = uncachedPromptTokens;
    }
  }

  if (promptTokens !== null) {
    state.promptTokens = promptTokens;
  }
  if (completionTokens !== null) {
    state.completionTokens = completionTokens;
  }
  if (cacheHitTokens !== null) {
    state.promptCacheHitTokens = cacheHitTokens;
  }
  if (cacheCreationTokens !== null) {
    state.promptCacheCreationTokens = cacheCreationTokens;
  }

  if (promptTokens !== null || completionTokens !== null) {
    state.totalRequestTokens =
      (promptTokens ?? 0) +
      (completionTokens ?? 0);
    return;
  }

  if (totalTokens !== null) {
    state.totalRequestTokens = totalTokens;
  }
}

export function registerPromptSubmit(
  state: QueryState,
  input: Record<string, unknown>,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
}

export function registerPendingTool(
  state: QueryState,
  input: Record<string, unknown>,
  toolUseId?: string,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
  const resolvedToolUseId = String(input.tool_use_id ?? toolUseId ?? randomHex(8));
  const toolName = String(input.tool_name ?? "tool");
  const toolInput = input.tool_input;

  state.pendingTools.set(resolvedToolUseId, {
    spanId: randomHex(8),
    startTime: hrTime(),
    toolInput,
    toolName,
  });

  addToolCall(state, {
    id: resolvedToolUseId,
    type: "function",
    function: {
      name: toolName,
      arguments: safeJson(toolInput ?? {}),
    },
  });
}

export function emitCompletedTool(
  state: QueryState,
  input: Record<string, unknown>,
  toolUseId?: string,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
  const resolvedToolUseId = String(input.tool_use_id ?? toolUseId ?? randomHex(8));
  const pendingTool =
    state.pendingTools.get(resolvedToolUseId) ?? {
      spanId: randomHex(8),
      startTime: hrTime(),
      toolInput: input.tool_input,
      toolName: String(input.tool_name ?? "tool"),
    };
  state.pendingTools.delete(resolvedToolUseId);

  const toolName = String(input.tool_name ?? pendingTool.toolName ?? "tool");
  const attrs = baseAttrs(toolName, toolName, RespanLogType.TOOL);
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
    pendingTool.toolInput ?? input.tool_input ?? {},
  );
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
    input.tool_response ?? input.tool_result ?? input.output ?? "",
  );
  attrs[RespanSpanAttributes.RESPAN_SPAN_TOOLS] = safeJson([
    {
      type: "function",
      function: { name: toolName },
    },
  ]);

  if (state.model) {
    attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = state.model;
    attrs.model = state.model;
  }
  if (state.sessionId) {
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.sessionId;
  }

  injectSpan(
    buildReadableSpan({
      name: `${toolName}.tool`,
      traceId: state.traceId,
      spanId: pendingTool.spanId,
      traceFlags: state.traceFlags,
      parentId: state.agentSpanId,
      startTime: pendingTool.startTime,
      endTime: hrTime(),
      attributes: attrs,
    }),
  );
}

export function emitAgentSpan(state: QueryState): void {
  const attrs = baseAttrs(state.agentName, state.agentName, RespanLogType.AGENT);
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[RespanSpanAttributes.GEN_AI_SYSTEM] = "anthropic";
  attrs[RespanSpanAttributes.LLM_SYSTEM] = "anthropic";
  attrs[RespanSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = state.agentName;
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = state.agentName;

  if (state.inputMessages.length > 0) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  }
  const formattedOutput = formatAgentOutput(state);
  if (formattedOutput) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = formattedOutput;
  }
  if (state.toolDefinitions && state.toolDefinitions.length > 0) {
    attrs[RespanSpanAttributes.RESPAN_SPAN_TOOLS] = safeJson(state.toolDefinitions);
  }
  if (state.toolCalls.length > 0) {
    attrs[RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS] = safeJson(
      dedupeToolCalls(state.toolCalls),
    );
  }
  if (state.model) {
    attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = state.model;
    attrs.model = state.model;
  }
  if (state.promptTokens !== undefined) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = state.promptTokens;
    attrs.prompt_tokens = state.promptTokens;
  }
  if (state.completionTokens !== undefined) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = state.completionTokens;
    attrs.completion_tokens = state.completionTokens;
  }
  if (state.totalRequestTokens !== undefined) {
    attrs.total_request_tokens = state.totalRequestTokens;
  }
  if (state.promptCacheHitTokens !== undefined) {
    attrs.prompt_cache_hit_tokens = state.promptCacheHitTokens;
  }
  if (state.promptCacheCreationTokens !== undefined) {
    attrs.prompt_cache_creation_tokens = state.promptCacheCreationTokens;
  }
  if (state.totalCostUsd !== undefined) {
    attrs.cost = state.totalCostUsd;
  }
  if (state.sessionId) {
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.sessionId;
  }

  injectSpan(
    buildReadableSpan({
      name: `${state.agentName}.agent`,
      traceId: state.traceId,
      spanId: state.agentSpanId,
      traceFlags: state.traceFlags,
      parentId: state.parentSpanId,
      startTime: state.startTime,
      endTime: hrTime(),
      attributes: attrs,
      statusCode: state.statusCode,
      errorMessage: state.errorMessage,
    }),
  );
}

function formatAgentOutput(state: QueryState): string {
  if (state.finalOutput !== undefined) {
    return stringifyOutputValue(state.finalOutput);
  }

  for (let index = state.outputMessages.length - 1; index >= 0; index -= 1) {
    const formatted = stringifyOutputValue(state.outputMessages[index]?.content);
    if (formatted) {
      return formatted;
    }
  }

  return "";
}

function stringifyOutputValue(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  if (Array.isArray(value)) {
    const parts = value
      .map((item) => stringifyOutputValue(item))
      .filter((part) => part.trim().length > 0);
    return parts.join("\n");
  }

  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (record.type === "thinking" || record.type === "tool_use") {
      return "";
    }
    if (record.type === "text" && typeof record.text === "string") {
      return record.text;
    }
    if ("content" in record) {
      return stringifyOutputValue(record.content);
    }
    return safeJson(record);
  }

  return String(value);
}

function normalizeToolCall(block: unknown): Record<string, unknown> | null {
  if (!block || typeof block !== "object" || Array.isArray(block)) {
    return null;
  }

  const record = block as Record<string, unknown>;
  if (record.type !== "tool_use") {
    return null;
  }

  const toolName = record.name;
  if (typeof toolName !== "string" || !toolName) {
    return null;
  }

  return {
    id: String(record.id ?? record.tool_use_id ?? ""),
    type: "function",
    function: {
      name: toolName,
      arguments: safeJson(record.input ?? {}),
    },
  };
}

function addToolCall(state: QueryState, toolCall: Record<string, unknown>): void {
  state.toolCalls.push(toolCall);
}

function dedupeToolCalls(
  toolCalls: Record<string, unknown>[],
): Record<string, unknown>[] {
  const seen = new Set<string>();
  const deduped: Record<string, unknown>[] = [];

  for (const toolCall of toolCalls) {
    const functionPayload =
      toolCall.function &&
      typeof toolCall.function === "object" &&
      !Array.isArray(toolCall.function)
        ? (toolCall.function as Record<string, unknown>)
        : {};
    const signature = safeJson([
      toolCall.id ?? "",
      functionPayload.name ?? "",
      functionPayload.arguments ?? "",
    ]);
    if (seen.has(signature)) {
      continue;
    }
    seen.add(signature);
    deduped.push(toolCall);
  }

  return deduped;
}

function updateSessionId(state: QueryState, rawSessionId: unknown): void {
  if (rawSessionId === undefined || rawSessionId === null) {
    return;
  }
  const sessionId = String(rawSessionId);
  if (sessionId) {
    state.sessionId = sessionId;
  }
}

function baseAttrs(
  entityName: string,
  entityPath: string,
  logType: string,
): Record<string, unknown> {
  return {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
  };
}

function buildReadableSpan(opts: BuildSpanOptions): ReadableSpan {
  const status =
    opts.statusCode && opts.statusCode >= 400
      ? { code: SpanStatusCode.ERROR, message: opts.errorMessage ?? "" }
      : { code: SpanStatusCode.OK, message: "" };

  return {
    name: opts.name,
    kind: SpanKind.INTERNAL,
    spanContext: () => ({
      traceId: opts.traceId,
      spanId: opts.spanId,
      traceFlags: opts.traceFlags ?? TraceFlags.SAMPLED,
      isRemote: false,
    }),
    parentSpanId: opts.parentId,
    startTime: opts.startTime,
    endTime: opts.endTime,
    duration: hrTimeDuration(opts.startTime, opts.endTime),
    status,
    attributes: opts.attributes,
    links: [],
    events: [],
    resource: { attributes: {} } as any,
    instrumentationLibrary: {
      name: "@respan/instrumentation-claude-agent-sdk",
      version: PACKAGE_VERSION,
    },
    ended: true,
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } as unknown as ReadableSpan;
}

function injectSpan(span: ReadableSpan): void {
  const tracerProvider = trace.getTracerProvider() as any;
  const processor =
    tracerProvider?.activeSpanProcessor ??
    tracerProvider?._delegate?.activeSpanProcessor ??
    tracerProvider?._delegate?._tracerProvider?.activeSpanProcessor;

  if (processor && typeof processor.onEnd === "function") {
    processor.onEnd(span);
  }
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
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value.map((item) => toSerializableValue(item));
  }
  if (typeof value === "object") {
    const normalizedObject: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, itemValue]) => {
      normalizedObject[key] = toSerializableValue(itemValue);
    });
    return normalizedObject;
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
