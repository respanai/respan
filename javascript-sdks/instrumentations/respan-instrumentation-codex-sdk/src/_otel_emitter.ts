import { context, trace } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  buildReadableSpan,
  ensureSpanId,
  ensureTraceId,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "1.0.0";
const CODEX_INSTRUMENTATION_NAME = "@respan/instrumentation-codex-sdk";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const DEFAULT_WORKFLOW_NAME = "codex-sdk";
const DEFAULT_AGENT_NAME = "codex-agent";
const STATUS_CODE_ATTR = "status_code";
const ERROR_MESSAGE_ATTR = "error.message";
const GEN_AI_PROMPT_PREFIX = SpanAttributes.LLM_PROMPTS;
const GEN_AI_COMPLETION_PREFIX = SpanAttributes.LLM_COMPLETIONS;
const GEN_AI_PROMPT_ROLE = `${GEN_AI_PROMPT_PREFIX}.0.role`;
const GEN_AI_PROMPT_CONTENT = `${GEN_AI_PROMPT_PREFIX}.0.content`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.0.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.0.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.0.tool_calls`;

type HrTime = [number, number];
type RecordValue = Record<string, unknown>;

export interface CodexEmitterOptions {
  agentName?: string;
  captureItemSpans?: boolean;
  workflowName?: string;
}

export interface CodexTurnState {
  agentName: string;
  agentSpanId: string;
  captureItemSpans: boolean;
  chatSpanId: string;
  emitted: boolean;
  endTime?: HrTime;
  errorMessage?: string;
  finalResponse?: string;
  input: unknown;
  inputMessages: RecordValue[];
  itemOrder: string[];
  items: Map<string, CodexItemState>;
  model?: string;
  parentSpanId?: string;
  startTime: HrTime;
  statusCode: number;
  threadId?: string;
  toolCalls: RecordValue[];
  traceId: string;
  turnId: string;
  usage?: RecordValue;
  workflowName: string;
  workflowSpanId: string;
}

interface CodexItemState {
  endTime?: HrTime;
  errorMessage?: string;
  item: RecordValue;
  spanId: string;
  startTime: HrTime;
  statusCode: number;
}

let turnCounter = 0;

export function createCodexTurnState({
  input,
  thread,
  options = {},
}: {
  input: unknown;
  thread?: unknown;
  options?: CodexEmitterOptions;
}): CodexTurnState {
  const activeSpan = trace.getSpan(context.active());
  const activeContext = activeSpan?.spanContext();
  const threadOptions = readPrivateRecord(thread, "_threadOptions");
  const model = stringValue(threadOptions?.model);
  const workflowName =
    nonEmptyString(options.workflowName) ??
    nonEmptyString(model ? `codex.${model}` : undefined) ??
    DEFAULT_WORKFLOW_NAME;
  const agentName =
    nonEmptyString(options.agentName) ??
    nonEmptyString(model ? `codex.${model}` : undefined) ??
    DEFAULT_AGENT_NAME;
  const turnId = `codex-sdk-${Date.now()}-${++turnCounter}`;
  const traceId = ensureTraceId(activeContext?.traceId);
  const workflowSpanId = ensureSpanId(`${turnId}:workflow`);

  return {
    agentName,
    agentSpanId: ensureSpanId(`${turnId}:agent`),
    captureItemSpans: options.captureItemSpans ?? true,
    chatSpanId: ensureSpanId(`${turnId}:chat`),
    emitted: false,
    input,
    inputMessages: normalizeInputMessages(input),
    itemOrder: [],
    items: new Map(),
    model,
    parentSpanId: activeContext?.spanId,
    startTime: hrTime(),
    statusCode: 200,
    threadId: readThreadId(thread),
    toolCalls: [],
    traceId,
    turnId,
    workflowName,
    workflowSpanId,
  };
}

export function trackCodexEvent(
  state: CodexTurnState,
  event: unknown,
): void {
  if (!event || typeof event !== "object" || Array.isArray(event)) {
    return;
  }

  const record = event as RecordValue;
  switch (record.type) {
    case "thread.started":
      if (typeof record.thread_id === "string" && record.thread_id) {
        state.threadId = record.thread_id;
      }
      break;
    case "turn.completed":
      if (isRecord(record.usage)) {
        state.usage = record.usage;
      }
      state.endTime = hrTime();
      break;
    case "turn.failed":
      state.statusCode = 500;
      state.errorMessage =
        readErrorMessage(record.error) ?? "Codex turn failed.";
      state.endTime = hrTime();
      break;
    case "error":
      state.statusCode = 500;
      state.errorMessage =
        typeof record.message === "string"
          ? record.message
          : "Codex stream error.";
      state.endTime = hrTime();
      break;
    case "item.started":
    case "item.updated":
    case "item.completed":
      trackCodexItemEvent(state, record);
      break;
    default:
      break;
  }
}

export function trackCodexRunResult(
  state: CodexTurnState,
  result: unknown,
  thread?: unknown,
): void {
  if (thread) {
    state.threadId = readThreadId(thread) ?? state.threadId;
  }
  if (!isRecord(result)) {
    state.finalResponse = stringifyOutputValue(result);
    state.endTime = hrTime();
    return;
  }

  if (typeof result.finalResponse === "string") {
    state.finalResponse = result.finalResponse;
  }
  if (isRecord(result.usage)) {
    state.usage = result.usage;
  }
  if (Array.isArray(result.items)) {
    for (const item of result.items) {
      if (isRecord(item)) {
        trackCompletedItem(state, item);
      }
    }
  }
  state.endTime = hrTime();
}

export function markCodexTurnError(
  state: CodexTurnState,
  error: unknown,
): void {
  state.statusCode = 500;
  state.errorMessage = error instanceof Error ? error.message : String(error);
  state.endTime = hrTime();
}

export function finalizeCodexTurnState(
  state: CodexTurnState,
  thread?: unknown,
): void {
  if (thread) {
    state.threadId = readThreadId(thread) ?? state.threadId;
  }
  state.endTime = state.endTime ?? hrTime();
}

export function emitCodexTurnSpans(state: CodexTurnState): void {
  if (state.emitted) {
    return;
  }
  state.emitted = true;
  const endTime = state.endTime ?? hrTime();

  injectSpan(
    buildCodexReadableSpan({
      name: `${state.workflowName}.workflow`,
      traceId: state.traceId,
      spanId: state.workflowSpanId,
      parentId: state.parentSpanId,
      startTimeHr: state.startTime,
      endTimeHr: endTime,
      attributes: workflowAttributes(state),
      statusCode: state.statusCode,
      errorMessage: state.errorMessage,
    }),
  );

  injectSpan(
    buildCodexReadableSpan({
      name: `${state.agentName}.agent`,
      traceId: state.traceId,
      spanId: state.agentSpanId,
      parentId: state.workflowSpanId,
      startTimeHr: state.startTime,
      endTimeHr: endTime,
      attributes: agentAttributes(state),
      statusCode: state.statusCode,
      errorMessage: state.errorMessage,
    }),
  );

  injectSpan(
    buildCodexReadableSpan({
      name: "codex.chat",
      traceId: state.traceId,
      spanId: state.chatSpanId,
      parentId: state.agentSpanId,
      startTimeHr: state.startTime,
      endTimeHr: endTime,
      attributes: chatAttributes(state),
      statusCode: state.statusCode,
      errorMessage: state.errorMessage,
    }),
  );

  if (!state.captureItemSpans) {
    return;
  }

  for (const itemId of state.itemOrder) {
    const itemState = state.items.get(itemId);
    if (!itemState) {
      continue;
    }
    emitItemSpan(state, itemState, endTime);
  }
}

function trackCodexItemEvent(
  state: CodexTurnState,
  event: RecordValue,
): void {
  if (!isRecord(event.item)) {
    return;
  }

  const item = event.item;
  if (event.type === "item.completed") {
    trackCompletedItem(state, item);
    return;
  }

  const itemState = ensureItemState(state, item);
  itemState.item = item;
}

function trackCompletedItem(
  state: CodexTurnState,
  item: RecordValue,
): void {
  const itemState = ensureItemState(state, item);
  itemState.item = item;
  itemState.endTime = hrTime();
  if (isFailedItem(item)) {
    itemState.statusCode = 500;
    itemState.errorMessage = failedItemMessage(item);
  }

  if (item.type === "agent_message" && typeof item.text === "string") {
    state.finalResponse = item.text;
  }

  const toolCall = normalizeToolCall(item);
  if (toolCall) {
    state.toolCalls.push(toolCall);
  }
}

function ensureItemState(
  state: CodexTurnState,
  item: RecordValue,
): CodexItemState {
  const id = itemId(item, state.itemOrder.length);
  const existing = state.items.get(id);
  if (existing) {
    return existing;
  }

  const itemState: CodexItemState = {
    item,
    spanId: ensureSpanId(`${state.turnId}:item:${id}`),
    startTime: hrTime(),
    statusCode: isFailedItem(item) ? 500 : 200,
    errorMessage: failedItemMessage(item),
  };
  state.items.set(id, itemState);
  state.itemOrder.push(id);
  return itemState;
}

function emitItemSpan(
  state: CodexTurnState,
  itemState: CodexItemState,
  fallbackEndTime: HrTime,
): void {
  const item = itemState.item;
  if (item.type === "agent_message") {
    return;
  }

  const itemAttrs = itemAttributes(state, item);
  if (!itemAttrs) {
    return;
  }

  addStatusAttributes(itemAttrs, itemState.statusCode, itemState.errorMessage);

  injectSpan(
    buildCodexReadableSpan({
      name: itemSpanName(item),
      traceId: state.traceId,
      spanId: itemState.spanId,
      parentId: state.agentSpanId,
      startTimeHr: itemState.startTime,
      endTimeHr: itemState.endTime ?? fallbackEndTime,
      attributes: itemAttrs,
      statusCode: itemState.statusCode,
      errorMessage: itemState.errorMessage,
    }),
  );
}

function workflowAttributes(state: CodexTurnState): Record<string, unknown> {
  const attrs = baseAttrs(state.workflowName, "", RespanLogType.WORKFLOW);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = state.finalResponse ?? "";
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = state.workflowName;
  addThreadAttributes(attrs, state);
  addStatusAttributes(attrs, state.statusCode, state.errorMessage);
  return attrs;
}

function agentAttributes(state: CodexTurnState): Record<string, unknown> {
  const attrs = baseAttrs(state.agentName, state.agentName, RespanLogType.AGENT);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = state.finalResponse ?? "";
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = state.workflowName;
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = state.agentName;
  if (state.model) {
    attrs[SpanAttributes.LLM_REQUEST_MODEL] = state.model;
  }
  addThreadAttributes(attrs, state);
  addStatusAttributes(attrs, state.statusCode, state.errorMessage);
  return attrs;
}

function chatAttributes(state: CodexTurnState): Record<string, unknown> {
  const attrs = baseAttrs("codex.response", "codex.response", RespanLogType.CHAT);
  const promptContent = stringifyOutputValue(state.inputMessages[0]?.content ?? "");
  const completionContent = state.finalResponse ?? "";
  attrs[SpanAttributes.LLM_SYSTEM] = "openai";
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = completionContent;
  attrs[GEN_AI_PROMPT_ROLE] = "user";
  attrs[GEN_AI_PROMPT_CONTENT] = promptContent;
  attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
  attrs[GEN_AI_COMPLETION_CONTENT] = completionContent;
  if (state.model) {
    attrs[SpanAttributes.LLM_REQUEST_MODEL] = state.model;
  }
  if (state.toolCalls.length > 0) {
    attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(dedupeToolCalls(state.toolCalls));
  }
  addUsageAttributes(attrs, state.usage);
  addThreadAttributes(attrs, state);
  addStatusAttributes(attrs, state.statusCode, state.errorMessage);
  return attrs;
}

function itemAttributes(
  state: CodexTurnState,
  item: RecordValue,
): Record<string, unknown> | null {
  switch (item.type) {
    case "reasoning": {
      const attrs = baseAttrs("codex.reasoning", "codex.reasoning", RespanLogType.TASK);
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = "";
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = stringValue(item.text) ?? "";
      addThreadAttributes(attrs, state);
      return attrs;
    }
    case "command_execution": {
      const command = stringValue(item.command) ?? "";
      const attrs = baseAttrs("codex.command", "codex.command", RespanLogType.TOOL);
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({ command });
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
        aggregated_output: item.aggregated_output ?? "",
        exit_code: item.exit_code,
        status: item.status,
      });
      addThreadAttributes(attrs, state);
      return attrs;
    }
    case "mcp_tool_call": {
      const server = stringValue(item.server) ?? "mcp";
      const tool = stringValue(item.tool) ?? "tool";
      const attrs = baseAttrs(
        `mcp__${server}__${tool}`,
        `mcp.${server}.${tool}`,
        RespanLogType.TOOL,
      );
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({
        server,
        tool,
        arguments: toSerializableValue(item.arguments),
      });
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
        item.error ?? item.result ?? { status: item.status },
      );
      addThreadAttributes(attrs, state);
      return attrs;
    }
    case "web_search": {
      const attrs = baseAttrs("codex.web_search", "codex.web_search", RespanLogType.TOOL);
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({
        query: item.query ?? "",
      });
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
        query: item.query ?? "",
      });
      addThreadAttributes(attrs, state);
      return attrs;
    }
    case "file_change": {
      const attrs = baseAttrs("codex.file_change", "codex.file_change", RespanLogType.TASK);
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(item.changes ?? []);
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
        changes: item.changes ?? [],
        status: item.status,
      });
      addThreadAttributes(attrs, state);
      return attrs;
    }
    case "todo_list": {
      const attrs = baseAttrs("codex.todo_list", "codex.todo_list", RespanLogType.TASK);
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = "";
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(item.items ?? []);
      addThreadAttributes(attrs, state);
      return attrs;
    }
    case "error": {
      const attrs = baseAttrs("codex.error", "codex.error", RespanLogType.TASK);
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = "";
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
        message: item.message ?? "",
      });
      addThreadAttributes(attrs, state);
      return attrs;
    }
    default:
      return null;
  }
}

function itemSpanName(item: RecordValue): string {
  switch (item.type) {
    case "reasoning":
      return "codex.reasoning";
    case "command_execution":
      return "codex.command";
    case "mcp_tool_call":
      return `codex.mcp.${stringValue(item.server) ?? "mcp"}.${stringValue(item.tool) ?? "tool"}`;
    case "web_search":
      return "codex.web_search";
    case "file_change":
      return "codex.file_change";
    case "todo_list":
      return "codex.todo_list";
    case "error":
      return "codex.error";
    default:
      return `codex.${String(item.type ?? "item")}`;
  }
}

function baseAttrs(
  entityName: string,
  entityPath: string,
  logType: RespanLogType,
): Record<string, unknown> {
  return {
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
    "telemetry.sdk.name": CODEX_INSTRUMENTATION_NAME,
    "telemetry.sdk.version": PACKAGE_VERSION,
  };
}

function buildCodexReadableSpan(
  options: Parameters<typeof buildReadableSpan>[0],
): ReturnType<typeof buildReadableSpan> {
  const span = buildReadableSpan(options) as ReturnType<typeof buildReadableSpan> & {
    instrumentationScope?: {
      name: string;
      version?: string;
    };
  };

  span.instrumentationScope = {
    name: CODEX_INSTRUMENTATION_NAME,
    version: PACKAGE_VERSION,
  };
  return span;
}

function addThreadAttributes(
  attrs: Record<string, unknown>,
  state: CodexTurnState,
): void {
  attrs[RespanSpanAttributes.RESPAN_TRACE_GROUP_ID] = state.workflowName;
  if (state.threadId) {
    attrs[RespanSpanAttributes.RESPAN_THREADS_ID] = state.threadId;
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.threadId;
  }
}

function addStatusAttributes(
  attrs: Record<string, unknown>,
  statusCode: number,
  errorMessage?: string,
): void {
  if (statusCode < 400 && !errorMessage) {
    return;
  }
  attrs[STATUS_CODE_ATTR] = statusCode >= 400 ? statusCode : 500;
  if (errorMessage) {
    attrs[ERROR_MESSAGE_ATTR] = errorMessage;
  }
}

function addUsageAttributes(
  attrs: Record<string, unknown>,
  usage: RecordValue | undefined,
): void {
  if (!usage) {
    return;
  }
  const inputTokens = integerValue(usage.input_tokens);
  const outputTokens = integerValue(usage.output_tokens);
  const totalTokens =
    inputTokens !== undefined || outputTokens !== undefined
      ? (inputTokens ?? 0) + (outputTokens ?? 0)
      : undefined;

  if (inputTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = inputTokens;
  }
  if (outputTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = outputTokens;
  }
  if (totalTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
}

function normalizeInputMessages(input: unknown): RecordValue[] {
  return [
    {
      role: "user",
      content: normalizeInputContent(input),
    },
  ];
}

function normalizeInputContent(input: unknown): string {
  if (typeof input === "string") {
    return input;
  }
  const serialized = toSerializableValue(input);
  if (!Array.isArray(serialized)) {
    return stringifyOutputValue(serialized);
  }

  const parts: string[] = [];
  for (const item of serialized) {
    if (!isRecord(item)) {
      parts.push(stringifyOutputValue(item));
      continue;
    }
    if (item.type === "text" && typeof item.text === "string") {
      parts.push(item.text);
      continue;
    }
    if (item.type === "local_image" && typeof item.path === "string") {
      parts.push(`[local_image:${item.path}]`);
      continue;
    }
    parts.push(stringifyOutputValue(item));
  }
  return parts.filter(Boolean).join("\n\n");
}

function normalizeToolCall(item: RecordValue): RecordValue | null {
  switch (item.type) {
    case "command_execution":
      return createToolCall({
        id: stringValue(item.id),
        name: "command_execution",
        args: { command: item.command ?? "" },
      });
    case "mcp_tool_call": {
      const server = stringValue(item.server) ?? "mcp";
      const tool = stringValue(item.tool) ?? "tool";
      return createToolCall({
        id: stringValue(item.id),
        name: `mcp__${server}__${tool}`,
        args: item.arguments ?? {},
      });
    }
    case "web_search":
      return createToolCall({
        id: stringValue(item.id),
        name: "web_search",
        args: { query: item.query ?? "" },
      });
    default:
      return null;
  }
}

function createToolCall({
  id,
  name,
  args,
}: {
  args: unknown;
  id?: string;
  name: string;
}): RecordValue {
  return {
    id: id ?? ensureSpanId(),
    type: "function",
    function: {
      name,
      arguments: stringifyOutputValue(args),
    },
  };
}

function dedupeToolCalls(toolCalls: RecordValue[]): RecordValue[] {
  const seen = new Set<string>();
  const deduped: RecordValue[] = [];
  for (const toolCall of toolCalls) {
    const key = safeJson(toolCall);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(toolCall);
  }
  return deduped;
}

function isFailedItem(item: RecordValue): boolean {
  return item.status === "failed" || isRecord(item.error);
}

function failedItemMessage(item: RecordValue): string | undefined {
  if (isRecord(item.error)) {
    return readErrorMessage(item.error);
  }
  if (item.status === "failed") {
    return `${String(item.type ?? "codex_item")} failed.`;
  }
  return undefined;
}

function readErrorMessage(error: unknown): string | undefined {
  if (error instanceof Error) {
    return error.message;
  }
  if (isRecord(error) && typeof error.message === "string") {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return undefined;
}

function itemId(item: RecordValue, index: number): string {
  return stringValue(item.id) ?? `${String(item.type ?? "item")}-${index}`;
}

function readThreadId(thread: unknown): string | undefined {
  if (!isRecord(thread)) {
    return undefined;
  }
  return stringValue(thread.id) ?? stringValue(thread._id);
}

function readPrivateRecord(
  target: unknown,
  key: string,
): RecordValue | undefined {
  if (!isRecord(target)) {
    return undefined;
  }
  const value = target[key];
  return isRecord(value) ? value : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function integerValue(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.trunc(value);
}

function stringifyOutputValue(value: unknown): string {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  return safeJson(serialized);
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, (_key, innerValue) =>
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
    const record = value as Record<string, unknown> & {
      toJSON?: () => unknown;
    };
    if (typeof record.toJSON === "function") {
      try {
        return toSerializableValue(record.toJSON());
      } catch {
        // Fall through to shallow structural serialization.
      }
    }
    const normalized: RecordValue = {};
    for (const [key, itemValue] of Object.entries(record)) {
      if (typeof itemValue === "function") {
        continue;
      }
      normalized[key] = toSerializableValue(itemValue);
    }
    return normalized;
  }
  return String(value);
}

function isRecord(value: unknown): value is RecordValue {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
