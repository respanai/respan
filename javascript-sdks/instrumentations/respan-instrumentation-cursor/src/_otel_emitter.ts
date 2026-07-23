import { context, trace } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  WORKFLOW_NAME_KEY,
  buildReadableSpan,
  ensureSpanId,
  ensureTraceId,
  getPropagatedAttributes,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "0.1.0";
const INSTRUMENTATION_NAME = "@respan/instrumentation-cursor";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const CURSOR_SYSTEM = "cursor";

const TL = SpanAttributes as unknown as Record<string, string>;
const ATTR_TRACELOOP_ENTITY_NAME =
  TL.TRACELOOP_ENTITY_NAME ?? "traceloop.entity.name";
const ATTR_TRACELOOP_ENTITY_PATH =
  TL.TRACELOOP_ENTITY_PATH ?? "traceloop.entity.path";
const ATTR_TRACELOOP_ENTITY_INPUT =
  TL.TRACELOOP_ENTITY_INPUT ?? "traceloop.entity.input";
const ATTR_TRACELOOP_ENTITY_OUTPUT =
  TL.TRACELOOP_ENTITY_OUTPUT ?? "traceloop.entity.output";
const ATTR_TRACELOOP_WORKFLOW_NAME =
  TL.TRACELOOP_WORKFLOW_NAME ?? "traceloop.workflow.name";
const ATTR_LLM_REQUEST_TYPE = TL.LLM_REQUEST_TYPE ?? "llm.request.type";
const ATTR_LLM_REQUEST_FUNCTIONS =
  TL.LLM_REQUEST_FUNCTIONS ?? "llm.request.functions";

const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;

type AnyRecord = Record<string, any>;

export interface CursorRunState {
  agentId?: string;
  agentName: string;
  agentSpanId: string;
  callbackEvents: CursorTaskEvent[];
  durationMs?: number;
  emitted: boolean;
  errorMessage?: string;
  inputMessages: AnyRecord[];
  model?: string;
  operation: string;
  outputTextParts: string[];
  parentSpanId?: string;
  pendingTools: Map<string, PendingToolState>;
  requestId?: string;
  result?: string;
  runId?: string;
  startTime: [number, number];
  status: string;
  statusCode: number;
  taskEvents: CursorTaskEvent[];
  toolCalls: AnyRecord[];
  toolDefinitions: AnyRecord[];
  traceId: string;
  workflowName: string;
}

interface PendingToolState {
  args: unknown;
  callId: string;
  name: string;
  startTime: [number, number];
}

interface CursorTaskEvent {
  input?: unknown;
  name: string;
  output?: unknown;
  startTime: [number, number];
  statusCode?: number;
  errorMessage?: string;
}

export function createCursorRunState({
  agent,
  agentName,
  agentOptions,
  message,
  operation,
  options,
}: {
  agent?: unknown;
  agentName?: string;
  agentOptions?: AnyRecord;
  message?: unknown;
  operation: string;
  options?: AnyRecord;
}): CursorRunState {
  const activeSpan = trace.getSpan(context.active());
  const activeSpanContext = activeSpan?.spanContext();
  const agentRecord = asRecord(agent);
  const resolvedAgentName =
    stringValue(agentName) ??
    stringValue(options?.name) ??
    stringValue(agentOptions?.name) ??
    stringValue(agentRecord?.name) ??
    "cursor-agent";
  const propagated = getPropagatedAttributes();
  const propagatedMetadata = asRecord(propagated?.metadata);
  const workflowName =
    stringValue(context.active().getValue(WORKFLOW_NAME_KEY)) ??
    stringValue(propagatedMetadata?.workflow_name) ??
    stringValue(propagated?.trace_group_identifier) ??
    resolvedAgentName;

  const inputMessages = message === undefined ? [] : [normalizeUserMessage(message)];
  const toolDefinitions = dedupeObjects([
    ...normalizeConfiguredTools(agentOptions),
    ...normalizeConfiguredTools(options),
  ]);

  return {
    agentId: stringValue(agentRecord?.agentId),
    agentName: resolvedAgentName,
    agentSpanId: ensureSpanId(),
    callbackEvents: [],
    emitted: false,
    inputMessages,
    model: resolveModel(options?.model ?? agentOptions?.model ?? agentRecord?.model),
    operation,
    outputTextParts: [],
    parentSpanId: activeSpanContext?.spanId,
    pendingTools: new Map(),
    startTime: hrTime(),
    status: "running",
    statusCode: 200,
    taskEvents: [],
    toolCalls: [],
    toolDefinitions,
    traceId: ensureTraceId(activeSpanContext?.traceId),
    workflowName,
  };
}

export function registerCursorToolDefinitions(
  state: CursorRunState,
  tools: AnyRecord[],
): void {
  state.toolDefinitions = dedupeObjects([
    ...state.toolDefinitions,
    ...tools,
  ]);
}

export function trackCursorMessage(
  state: CursorRunState,
  rawMessage: unknown,
): void {
  const message = asRecord(rawMessage);
  if (!message) return;

  state.agentId = stringValue(message.agent_id) ?? state.agentId;
  state.runId = stringValue(message.run_id) ?? state.runId;

  switch (message.type) {
    case "system":
      state.model = resolveModel(message.model) ?? state.model;
      if (Array.isArray(message.tools)) {
        registerCursorToolDefinitions(
          state,
          message.tools
            .map((toolName) => normalizeToolDefinition(toolName))
            .filter((tool): tool is AnyRecord => tool !== null),
        );
      }
      break;
    case "request":
      state.requestId = stringValue(message.request_id) ?? state.requestId;
      break;
    case "user":
      pushMessageIfUseful(state.inputMessages, normalizeSdkUserMessage(message.message));
      break;
    case "assistant":
      trackAssistantMessage(state, message.message);
      break;
    case "tool_call":
      trackToolCallMessage(state, message);
      break;
    case "thinking":
      trackTaskEvent(state, {
        name: "cursor.thinking",
        output: message.text,
        startTime: hrTime(),
      });
      break;
    case "task":
      trackTaskEvent(state, {
        name: stringValue(message.status) ?? "cursor.task",
        input: message.status,
        output: message.text,
        startTime: hrTime(),
      });
      break;
    case "status":
      state.status = stringValue(message.status) ?? state.status;
      if (message.status === "ERROR" || message.status === "CANCELLED") {
        state.statusCode = 500;
        state.errorMessage = stringValue(message.message) ?? state.errorMessage;
      }
      trackTaskEvent(state, {
        name: "cursor.status",
        input: message.status,
        output: message.message,
        startTime: hrTime(),
        statusCode: message.status === "ERROR" ? 500 : 200,
      });
      break;
  }
}

export function trackCursorCallback(
  state: CursorRunState,
  callbackName: string,
  payload: unknown,
): void {
  state.callbackEvents.push({
    name: `cursor.${callbackName}`,
    input: payload,
    startTime: hrTime(),
  });

  const record = asRecord(payload);
  const update = asRecord(record?.update);
  if (update) {
    trackCursorDeltaUpdate(state, update);
  }
}

export function recordCursorRunResult(
  state: CursorRunState,
  result: unknown,
): void {
  const record = asRecord(result);
  if (!record) return;

  state.runId = stringValue(record.id) ?? state.runId;
  state.requestId = stringValue(record.requestId) ?? state.requestId;
  state.status = stringValue(record.status) ?? state.status;
  state.result = stringValue(record.result) ?? state.result;
  state.model = resolveModel(record.model) ?? state.model;
  state.durationMs =
    typeof record.durationMs === "number" ? record.durationMs : state.durationMs;

  if (record.status === "error" || record.status === "cancelled") {
    state.statusCode = 500;
    state.errorMessage = state.result ?? state.errorMessage;
  }

  if (state.result && !state.outputTextParts.includes(state.result)) {
    state.outputTextParts.push(state.result);
  }
}

export function markCursorRunError(
  state: CursorRunState,
  error: unknown,
): void {
  state.status = "error";
  state.statusCode = 500;
  state.errorMessage = error instanceof Error ? error.message : String(error);
}

export function emitCursorToolExecution({
  args,
  callId,
  error,
  result,
  state,
  toolName,
}: {
  args: unknown;
  callId?: string;
  error?: unknown;
  result?: unknown;
  state: CursorRunState;
  toolName: string;
}): void {
  const statusCode = error ? 500 : 200;
  const errorMessage = error instanceof Error ? error.message : error ? String(error) : undefined;

  const toolCall = normalizeToolCall({ id: callId, name: toolName, args });
  if (toolCall) state.toolCalls.push(toolCall);

  emitToolSpan({
    args,
    errorMessage,
    result: error ? errorMessage : result,
    startTime: hrTime(),
    state,
    statusCode,
    toolName,
  });
}

export function emitFinalCursorRunSpans(state: CursorRunState): void {
  if (state.emitted) return;
  state.emitted = true;

  for (const taskEvent of [...state.taskEvents, ...state.callbackEvents]) {
    emitTaskSpan(state, taskEvent);
  }

  emitChatSpan(state);
  emitAgentSpan(state);
}

function trackAssistantMessage(state: CursorRunState, messagePayload: unknown): void {
  const message = asRecord(messagePayload);
  if (!message) return;

  const content = Array.isArray(message.content) ? message.content : [];
  const textParts: string[] = [];
  const assistantToolCalls: AnyRecord[] = [];
  for (const block of content) {
    const blockRecord = asRecord(block);
    if (!blockRecord) continue;
    if (blockRecord.type === "text") {
      const text = stringValue(blockRecord.text);
      if (text) {
        textParts.push(text);
        state.outputTextParts.push(text);
      }
    }
    if (blockRecord.type === "tool_use") {
      const toolCall = normalizeToolCall(blockRecord);
      if (toolCall) {
        assistantToolCalls.push(toolCall);
        state.toolCalls.push(toolCall);
      }
    }
  }

  state.inputMessages.push({
    role: "assistant",
    content: textParts.join(""),
    ...(assistantToolCalls.length > 0 ? { tool_calls: dedupeObjects(assistantToolCalls) } : {}),
  });
}

function trackToolCallMessage(state: CursorRunState, message: AnyRecord): void {
  const callId = stringValue(message.call_id) ?? ensureSpanId();
  const toolName = stringValue(message.name) ?? "cursor_tool";
  const status = stringValue(message.status) ?? "";

  if (status === "running") {
    state.pendingTools.set(callId, {
      args: message.args,
      callId,
      name: toolName,
      startTime: hrTime(),
    });
    return;
  }

  const pending = state.pendingTools.get(callId);
  state.pendingTools.delete(callId);
  const isError = status === "error";
  const result = message.result;
  const errorMessage = isError ? stringifyValue(result ?? "Cursor tool call failed") : undefined;

  const toolCall = normalizeToolCall({
    id: callId,
    name: toolName,
    args: pending?.args ?? message.args,
  });
  if (toolCall) state.toolCalls.push(toolCall);

  emitToolSpan({
    args: pending?.args ?? message.args,
    errorMessage,
    result: isError ? errorMessage : result,
    startTime: pending?.startTime ?? hrTime(),
    state,
    statusCode: isError ? 500 : 200,
    toolName,
  });
}

function trackCursorDeltaUpdate(state: CursorRunState, update: AnyRecord): void {
  if (update.type === "tool-call-started") {
    const callId = stringValue(update.callId) ?? ensureSpanId();
    const toolCall = asRecord(update.toolCall);
    state.pendingTools.set(callId, {
      args: toolCall?.args,
      callId,
      name: stringValue(toolCall?.type) ?? "cursor_tool",
      startTime: hrTime(),
    });
  }

  if (update.type === "tool-call-completed") {
    const callId = stringValue(update.callId) ?? ensureSpanId();
    const pending = state.pendingTools.get(callId);
    state.pendingTools.delete(callId);
    const toolCall = asRecord(update.toolCall);
    const toolName = stringValue(toolCall?.type) ?? pending?.name ?? "cursor_tool";
    const args = pending?.args ?? toolCall?.args;

    emitCursorToolExecution({
      args,
      callId,
      result: toolCall?.result,
      state,
      toolName,
    });
  }
}

function trackTaskEvent(state: CursorRunState, event: CursorTaskEvent): void {
  state.taskEvents.push(event);
}

function emitAgentSpan(state: CursorRunState): void {
  const attrs = baseAttrs(state.agentName, "", RespanLogType.AGENT);
  attrs[ATTR_TRACELOOP_WORKFLOW_NAME] = state.workflowName;
  attrs[ATTR_TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  attrs[ATTR_TRACELOOP_ENTITY_OUTPUT] = state.result ?? state.outputTextParts.join("");
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = state.agentName;
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[ATTR_TRACELOOP_WORKFLOW_NAME] = state.workflowName;
  setIfPresent(attrs, RespanSpanAttributes.RESPAN_SESSION_ID, state.runId);
  setMetadata(attrs, "cursor_agent_id", state.agentId);
  setMetadata(attrs, "cursor_run_id", state.runId);
  setMetadata(attrs, "cursor_request_id", state.requestId);
  setMetadata(attrs, "cursor_operation", state.operation);
  setMetadata(attrs, "cursor_status", state.status);
  setIfPresent(attrs, ATTR_GEN_AI_REQUEST_MODEL, state.model);

  const span = buildCursorReadableSpan({
    name: `${state.agentName}.agent`,
    traceId: state.traceId,
    spanId: state.agentSpanId,
    parentId: state.parentSpanId,
    startTimeHr: state.startTime,
    attributes: attrs,
    statusCode: state.statusCode,
    errorMessage: state.errorMessage,
  });
  injectSpan(span);
}

function emitChatSpan(state: CursorRunState): void {
  const output = state.result ?? state.outputTextParts.join("");
  const toolCalls = dedupeToolCalls(state.toolCalls);
  const tools = dedupeToolDefinitions(state.toolDefinitions);

  const attrs = baseAttrs("cursor.chat", "cursor.chat", RespanLogType.CHAT);
  attrs[ATTR_LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[ATTR_GEN_AI_SYSTEM] = CURSOR_SYSTEM;
  setIfPresent(attrs, ATTR_GEN_AI_REQUEST_MODEL, state.model);
  attrs[ATTR_TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  attrs[ATTR_TRACELOOP_ENTITY_OUTPUT] = output;
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[ATTR_TRACELOOP_WORKFLOW_NAME] = state.workflowName;
  setIfPresent(attrs, RespanSpanAttributes.RESPAN_SESSION_ID, state.runId);

  state.inputMessages.forEach((message, index) => {
    setIfPresent(attrs, `${ATTR_GEN_AI_PROMPT}.${index}.role`, message.role);
    setIfPresent(attrs, `${ATTR_GEN_AI_PROMPT}.${index}.content`, stringifyMessageContent(message.content));
    if (message.tool_calls) {
      attrs[`${ATTR_GEN_AI_PROMPT}.${index}.tool_calls`] = safeJson(message.tool_calls);
    }
  });

  attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
  attrs[GEN_AI_COMPLETION_CONTENT] = output;
  if (toolCalls.length > 0) attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(toolCalls);
  if (tools.length > 0) attrs[ATTR_LLM_REQUEST_FUNCTIONS] = safeJson(tools);

  const span = buildCursorReadableSpan({
    name: "cursor.chat",
    traceId: state.traceId,
    parentId: state.agentSpanId,
    startTimeHr: state.startTime,
    attributes: attrs,
    statusCode: state.statusCode,
    errorMessage: state.errorMessage,
  });
  injectSpan(span);
}

function emitToolSpan({
  args,
  errorMessage,
  result,
  startTime,
  state,
  statusCode,
  toolName,
}: {
  args: unknown;
  errorMessage?: string;
  result?: unknown;
  startTime: [number, number];
  state: CursorRunState;
  statusCode: number;
  toolName: string;
}): void {
  const attrs = baseAttrs(toolName, toolName, RespanLogType.TOOL);
  attrs[ATTR_TRACELOOP_ENTITY_INPUT] = safeJson({
    name: toolName,
    arguments: toSerializableValue(args ?? {}),
  });
  attrs[ATTR_TRACELOOP_ENTITY_OUTPUT] = safeJson(toSerializableValue(result ?? ""));
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[ATTR_TRACELOOP_WORKFLOW_NAME] = state.workflowName;
  setIfPresent(attrs, RespanSpanAttributes.RESPAN_SESSION_ID, state.runId);

  const span = buildCursorReadableSpan({
    name: `${toolName}.tool`,
    traceId: state.traceId,
    parentId: state.agentSpanId,
    startTimeHr: startTime,
    attributes: attrs,
    statusCode,
    errorMessage,
  });
  injectSpan(span);
}

function emitTaskSpan(state: CursorRunState, event: CursorTaskEvent): void {
  const attrs = baseAttrs(event.name, event.name, RespanLogType.TASK);
  if (event.input !== undefined) attrs[ATTR_TRACELOOP_ENTITY_INPUT] = safeJson(toSerializableValue(event.input));
  if (event.output !== undefined) attrs[ATTR_TRACELOOP_ENTITY_OUTPUT] = safeJson(toSerializableValue(event.output));
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[ATTR_TRACELOOP_WORKFLOW_NAME] = state.workflowName;
  setIfPresent(attrs, RespanSpanAttributes.RESPAN_SESSION_ID, state.runId);

  const span = buildCursorReadableSpan({
    name: `${event.name}.task`,
    traceId: state.traceId,
    parentId: state.agentSpanId,
    startTimeHr: event.startTime,
    attributes: attrs,
    statusCode: event.statusCode,
    errorMessage: event.errorMessage,
  });
  injectSpan(span);
}

function baseAttrs(entityName: string, entityPath: string, logType: string): AnyRecord {
  return {
    [ATTR_TRACELOOP_ENTITY_NAME]: entityName,
    [ATTR_TRACELOOP_ENTITY_PATH]: entityPath,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
  };
}

function buildCursorReadableSpan(options: Parameters<typeof buildReadableSpan>[0]): ReadableSpan {
  const span = buildReadableSpan(options) as ReadableSpan & {
    instrumentationLibrary?: { name: string; version?: string };
  };
  span.instrumentationLibrary = { name: INSTRUMENTATION_NAME, version: PACKAGE_VERSION };
  return span;
}

function normalizeConfiguredTools(options?: AnyRecord): AnyRecord[] {
  if (!options) return [];
  const tools: AnyRecord[] = [];
  const local = asRecord(options.local);
  const customTools = asRecord(local?.customTools);
  if (customTools) {
    for (const [toolName, tool] of Object.entries(customTools)) {
      const toolRecord = asRecord(tool);
      const normalized = normalizeToolDefinition({
        name: toolName,
        description: toolRecord?.description,
        inputSchema: toolRecord?.inputSchema,
      });
      if (normalized) tools.push(normalized);
    }
  }

  const mcpServers = asRecord(options.mcpServers);
  if (mcpServers) {
    for (const serverName of Object.keys(mcpServers)) {
      const normalized = normalizeToolDefinition({
        name: `mcp__${serverName}`,
        description: `Cursor MCP server ${serverName}`,
      });
      if (normalized) tools.push(normalized);
    }
  }
  return tools;
}

function normalizeToolDefinition(tool: unknown): AnyRecord | null {
  if (typeof tool === "string" && tool) {
    return { type: "function", function: { name: tool } };
  }
  const record = asRecord(tool);
  if (!record) return null;
  const functionRecord = asRecord(record.function);
  const name = stringValue(functionRecord?.name) ?? stringValue(record.name) ?? stringValue(record.type);
  if (!name) return null;
  const description = stringValue(functionRecord?.description) ?? stringValue(record.description);
  const parameters = functionRecord?.parameters ?? record.parameters ?? record.inputSchema ?? record.input_schema;
  return {
    type: "function",
    function: {
      name,
      ...(description ? { description } : {}),
      ...(parameters !== undefined ? { parameters: toSerializableValue(parameters) } : {}),
    },
  };
}

function normalizeToolCall(rawToolCall: unknown): AnyRecord | null {
  const record = asRecord(rawToolCall);
  if (!record) return null;
  const functionRecord = asRecord(record.function);
  const toolName = stringValue(functionRecord?.name) ?? stringValue(record.name) ?? stringValue(record.toolName);
  if (!toolName) return null;
  const args = functionRecord?.arguments ?? record.args ?? record.arguments ?? record.input ?? {};
  return {
    id: stringValue(record.id) ?? stringValue(record.call_id) ?? stringValue(record.callId) ?? "",
    type: "function",
    function: {
      name: toolName,
      arguments: typeof args === "string" ? args : safeJson(args),
    },
  };
}

function normalizeUserMessage(message: unknown): AnyRecord {
  if (typeof message === "string") return { role: "user", content: message };
  const record = asRecord(message);
  if (record && typeof record.text === "string") {
    return {
      role: "user",
      content: record.text,
      ...(Array.isArray(record.images) ? { images: toSerializableValue(record.images) } : {}),
    };
  }
  return { role: "user", content: toSerializableValue(message) };
}

function normalizeSdkUserMessage(message: unknown): AnyRecord {
  const record = asRecord(message);
  const content = Array.isArray(record?.content) ? record?.content : [];
  const text = content
    .map((block: unknown) => {
      const blockRecord = asRecord(block);
      return stringValue(blockRecord?.text) ?? "";
    })
    .filter(Boolean)
    .join("");
  return { role: "user", content: text || toSerializableValue(message) };
}

function pushMessageIfUseful(messages: AnyRecord[], message: AnyRecord): void {
  const serialized = safeJson(message);
  if (messages.some((existing) => safeJson(existing) === serialized)) return;
  messages.push(message);
}

function resolveModel(model: unknown): string | undefined {
  if (typeof model === "string" && model) return model;
  const record = asRecord(model);
  return stringValue(record?.id) ?? stringValue(record?.model);
}

function setMetadata(attrs: AnyRecord, key: string, value: unknown): void {
  if (value === undefined || value === null || value === "") return;
  attrs[`respan.metadata.${key}`] = String(value);
}

function setIfPresent(attrs: AnyRecord, key: string, value: unknown): void {
  if (value === undefined || value === null || value === "") return;
  attrs[key] = typeof value === "object" ? safeJson(value) : value;
}

function stringifyMessageContent(content: unknown): string {
  if (content === undefined || content === null) return "";
  if (typeof content === "string") return content;
  return safeJson(content);
}

function stringifyValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  return safeJson(value);
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
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) {
    return value.map((item) => toSerializableValue(item)).filter((item) => item !== undefined);
  }
  if (typeof value === "object") {
    const normalizedObject: AnyRecord = {};
    Object.entries(value as AnyRecord).forEach(([key, itemValue]) => {
      if (typeof itemValue === "function" || typeof itemValue === "symbol") return;
      const normalizedValue = toSerializableValue(itemValue);
      if (normalizedValue !== undefined) normalizedObject[key] = normalizedValue;
    });
    return normalizedObject;
  }
  return String(value);
}

function asRecord(value: unknown): AnyRecord | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  return value as AnyRecord;
}

function stringValue(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  const text = String(value);
  return text ? text : undefined;
}

function dedupeObjects(objects: AnyRecord[]): AnyRecord[] {
  const seen = new Set<string>();
  const deduped: AnyRecord[] = [];
  for (const object of objects) {
    const signature = safeJson(object);
    if (seen.has(signature)) continue;
    seen.add(signature);
    deduped.push(object);
  }
  return deduped;
}

function dedupeToolDefinitions(tools: AnyRecord[]): AnyRecord[] {
  const seen = new Set<string>();
  const deduped: AnyRecord[] = [];
  for (const tool of tools) {
    const name = asRecord(tool.function)?.name ?? safeJson(tool);
    if (seen.has(String(name))) continue;
    seen.add(String(name));
    deduped.push(tool);
  }
  return deduped;
}

function dedupeToolCalls(toolCalls: AnyRecord[]): AnyRecord[] {
  const seen = new Set<string>();
  const deduped: AnyRecord[] = [];
  for (const toolCall of toolCalls) {
    const functionRecord = asRecord(toolCall.function);
    const signature = safeJson([
      toolCall.id ?? "",
      functionRecord?.name ?? "",
      functionRecord?.arguments ?? "",
    ]);
    if (seen.has(signature)) continue;
    seen.add(signature);
    deduped.push(toolCall);
  }
  return deduped;
}
