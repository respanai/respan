/**
 * Respan instrumentation plugin for BeeAI Framework.
 *
 * Wraps `@arizeai/openinference-instrumentation-beeai` in the Respan plugin
 * protocol and normalizes BeeAI event spans for Respan routing.
 *
 * ```typescript
 * import * as beeaiFramework from "beeai-framework";
 * import { Respan } from "@respan/respan";
 * import { BeeAIInstrumentor } from "@respan/instrumentation-beeai";
 *
 * const respan = new Respan({
 *   instrumentations: [new BeeAIInstrumentor({ sdkModule: beeaiFramework })],
 * });
 * await respan.initialize();
 * ```
 */

import { trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { BeeAIInstrumentation } from "@arizeai/openinference-instrumentation-beeai";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

type BeeAIInstrumentationClass = new (...args: any[]) => any;
type ProcessorOnStart = (span: ReadableSpan, parentContext: unknown) => void;
type ProcessorOnEnd = (span: ReadableSpan) => void;

const BEEAI_SCOPE_NAME = "@arizeai/openinference-instrumentation-beeai";
const BEEAI_TARGET = "target";
const BEEAI_DATA = "data";
const BEEAI_METADATA = "metadata";
const BEEAI_TRACE_ID = "traceId";
const BEEAI_VERSION = "beeai.version";
const OTEL_SCOPE_NAME = "otel.scope.name";
const OPENINFERENCE_INPUT_VALUE = "input.value";
const OPENINFERENCE_OUTPUT_VALUE = "output.value";
const MAX_PENDING_CHAT_INPUTS = 20;

const DIRECT_MODEL = "model";
const DIRECT_PROMPT_TOKENS = "prompt_tokens";
const DIRECT_COMPLETION_TOKENS = "completion_tokens";
const DIRECT_TOTAL_REQUEST_TOKENS = "total_request_tokens";

const droppedSpanParentsByTrace = new Map<string, Map<string, string | undefined>>();
const workflowSpanIdsByTrace = new Map<string, string>();

function setDefault(attrs: Record<string, any>, key: string, value: any): void {
  if (attrs[key] === undefined) attrs[key] = value;
}

function safeJsonStr(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function parseJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function asRecord(value: unknown): Record<string, any> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  return value as Record<string, any>;
}

function isMeaningfulStructuredValue(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return value.length > 0;
  if (Array.isArray(value)) return value.some((item) => isMeaningfulStructuredValue(item));
  if (typeof value === "object") {
    return Object.values(value as Record<string, unknown>).some((item) =>
      isMeaningfulStructuredValue(item),
    );
  }
  return true;
}

function firstDefined<T>(...values: Array<T | undefined>): T | undefined {
  for (const value of values) {
    if (value !== undefined) return value;
  }
  return undefined;
}

function getInstrumentationScopeName(span: ReadableSpan): string {
  return (
    ((span as any).instrumentationScope?.name as string | undefined) ??
    ((span as any).instrumentationScope?.name as string | undefined) ??
    ((span as any).attributes?.[OTEL_SCOPE_NAME] as string | undefined) ??
    ""
  );
}

function getBeeAIEventLogType(target: unknown): string | undefined {
  if (typeof target !== "string") return undefined;

  if (target.startsWith("agent.")) return RespanLogType.AGENT;
  if (target.startsWith("tool.")) return RespanLogType.TOOL;

  if (target.startsWith("backend.") && target.includes(".chat.")) {
    return RespanLogType.CHAT;
  }
  if (target.startsWith("backend.") && target.includes(".embedding.")) {
    return RespanLogType.EMBEDDING;
  }

  return undefined;
}

function setTokenAttributes(
  attrs: Record<string, any>,
  usage: Record<string, any> | undefined,
): void {
  if (!usage) return;

  const promptTokens = firstDefined(
    usage.promptTokens,
    usage.prompt_tokens,
    usage.inputTokens,
    usage.input_tokens,
  );
  const completionTokens = firstDefined(
    usage.completionTokens,
    usage.completion_tokens,
    usage.outputTokens,
    usage.output_tokens,
  );
  const totalTokens = firstDefined(
    usage.totalTokens,
    usage.total_tokens,
    promptTokens !== undefined && completionTokens !== undefined
      ? Number(promptTokens) + Number(completionTokens)
      : undefined,
  );

  if (promptTokens !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_USAGE_PROMPT_TOKENS, promptTokens);
    setDefault(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, promptTokens);
    setDefault(attrs, DIRECT_PROMPT_TOKENS, promptTokens);
  }
  if (completionTokens !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_USAGE_COMPLETION_TOKENS, completionTokens);
    setDefault(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, completionTokens);
    setDefault(attrs, DIRECT_COMPLETION_TOKENS, completionTokens);
  }
  if (totalTokens !== undefined) {
    setDefault(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, totalTokens);
    setDefault(attrs, DIRECT_TOTAL_REQUEST_TOKENS, totalTokens);
  }
}

const pendingChatInputsByTrace = new Map<string, unknown[]>();
const pendingChatSpansByTrace = new Map<string, ReadableSpan[]>();

function getSpanTraceKey(span: ReadableSpan, attrs: Record<string, any>): string | undefined {
  const spanContext = typeof (span as any).spanContext === "function"
    ? (span as any).spanContext()
    : undefined;
  const traceId = firstDefined(attrs[BEEAI_TRACE_ID], spanContext?.traceId);
  return typeof traceId === "string" && traceId.length > 0 ? traceId : undefined;
}

function getOtelSpanContext(span: ReadableSpan): { traceId?: string; spanId?: string } | undefined {
  return typeof (span as any).spanContext === "function"
    ? (span as any).spanContext()
    : undefined;
}

function getOtelTraceId(span: ReadableSpan): string | undefined {
  const traceId = getOtelSpanContext(span)?.traceId;
  return typeof traceId === "string" && traceId.length > 0 ? traceId : undefined;
}

function getOtelSpanId(span: ReadableSpan): string | undefined {
  const spanId = getOtelSpanContext(span)?.spanId;
  return typeof spanId === "string" && spanId.length > 0 ? spanId : undefined;
}

function getOtelParentSpanId(span: ReadableSpan): string | undefined {
  const parentSpanId =
    (span as any).parentSpanId ?? (span as any).parentSpanContext?.spanId;
  return typeof parentSpanId === "string" && parentSpanId.length > 0
    ? parentSpanId
    : undefined;
}

function rememberDroppedSpanParent(span: ReadableSpan): void {
  const traceId = getOtelTraceId(span);
  const spanId = getOtelSpanId(span);
  if (!traceId || !spanId) return;

  const traceParents = droppedSpanParentsByTrace.get(traceId) ?? new Map();
  traceParents.set(spanId, getOtelParentSpanId(span));
  droppedSpanParentsByTrace.set(traceId, traceParents);
}

function rememberWorkflowSpan(span: ReadableSpan, attrs: Record<string, any>): void {
  const spanKind = attrs[SpanAttributes.TRACELOOP_SPAN_KIND];
  const isWorkflowSpan =
    (typeof spanKind === "string" && spanKind.toLowerCase() === "workflow") ||
    span.name.endsWith(".workflow.workflow");
  if (!isWorkflowSpan) return;

  const traceId = getOtelTraceId(span);
  const spanId = getOtelSpanId(span);
  if (!traceId || !spanId) return;

  workflowSpanIdsByTrace.set(traceId, spanId);
}

function resolveExportParentSpanId(span: ReadableSpan): string | undefined {
  const traceId = getOtelTraceId(span);
  let parentSpanId = getOtelParentSpanId(span);
  if (!traceId || !parentSpanId) return parentSpanId;

  const traceParents = droppedSpanParentsByTrace.get(traceId);
  if (!traceParents) return parentSpanId;

  const visited = new Set<string>();
  while (parentSpanId && traceParents.has(parentSpanId) && !visited.has(parentSpanId)) {
    visited.add(parentSpanId);
    parentSpanId = traceParents.get(parentSpanId);
  }

  return parentSpanId;
}

function reparentFromDroppedSpans(span: ReadableSpan): void {
  const currentParentSpanId = getOtelParentSpanId(span);
  const traceId = getOtelTraceId(span);
  const workflowSpanId = traceId ? workflowSpanIdsByTrace.get(traceId) : undefined;
  const resolvedParentSpanId = workflowSpanId ?? resolveExportParentSpanId(span);
  if (resolvedParentSpanId === currentParentSpanId) return;

  Object.defineProperty(span, "parentSpanId", {
    value: resolvedParentSpanId,
    writable: false,
    configurable: true,
    enumerable: true,
  });
  // OTEL 2.x reads parentSpanContext on the wire, not parentSpanId.
  Object.defineProperty(span, "parentSpanContext", {
    value: resolvedParentSpanId
      ? {
          traceId,
          spanId: resolvedParentSpanId,
          traceFlags: span.spanContext().traceFlags,
          isRemote: false,
        }
      : undefined,
    writable: false,
    configurable: true,
    enumerable: true,
  });
}

function enqueuePendingChatInput(
  span: ReadableSpan,
  attrs: Record<string, any>,
  input: unknown,
): void {
  if (input === undefined) return;

  const traceKey = getSpanTraceKey(span, attrs);
  if (!traceKey) return;

  const queue = pendingChatInputsByTrace.get(traceKey) ?? [];
  const serializedInput = safeJsonStr(input);
  if (queue.length > 0 && safeJsonStr(queue[queue.length - 1]) === serializedInput) {
    return;
  }

  queue.push(input);
  if (queue.length > MAX_PENDING_CHAT_INPUTS) {
    queue.shift();
  }
  pendingChatInputsByTrace.set(traceKey, queue);
}

function dequeuePendingChatInput(
  span: ReadableSpan,
  attrs: Record<string, any>,
): unknown {
  const traceKey = getSpanTraceKey(span, attrs);
  if (!traceKey) return undefined;

  const queue = pendingChatInputsByTrace.get(traceKey);
  if (!queue || queue.length === 0) return undefined;

  const input = queue.shift();
  if (queue.length === 0) {
    pendingChatInputsByTrace.delete(traceKey);
  }
  return input;
}

function getStateMessages(state: Record<string, any> | undefined): unknown[] | undefined {
  const memory = asRecord(state?.memory);
  return Array.isArray(memory?.messages) ? memory.messages : undefined;
}

function normalizeToolCall(block: Record<string, any>): Record<string, unknown> {
  return {
    id: block.toolCallId,
    type: "function",
    function: {
      name: block.toolName,
      arguments: safeJsonStr(block.args ?? {}),
    },
  };
}

function normalizeToolResult(block: Record<string, any>): Record<string, unknown> {
  return {
    tool_call_id: block.toolCallId,
    name: block.toolName,
    content: block.result,
    is_error: Boolean(block.isError),
  };
}

function normalizeBeeAIMessage(message: unknown): unknown {
  const record = asRecord(message);
  if (!record) return message;

  const normalized: Record<string, unknown> = {};
  if (typeof record.role === "string") {
    normalized.role = record.role;
  }

  if (typeof record.content === "string") {
    normalized.content = record.content;
    return normalized;
  }

  const content = Array.isArray(record.content) ? record.content : undefined;
  if (!content) return normalized.role ? normalized : message;

  const textParts: string[] = [];
  const toolCalls: Record<string, unknown>[] = [];
  const toolResults: Record<string, unknown>[] = [];

  for (const blockValue of content) {
    const block = asRecord(blockValue);
    if (!block) continue;

    if (block.type === "text" && block.text !== undefined) {
      textParts.push(String(block.text));
    } else if (block.type === "tool-call") {
      toolCalls.push(normalizeToolCall(block));
    } else if (block.type === "tool-result") {
      toolResults.push(normalizeToolResult(block));
    }
  }

  if (textParts.length > 0) {
    normalized.content = textParts.join("\n");
  }
  if (toolCalls.length > 0) {
    if (normalized.content === undefined) {
      normalized.content = "";
    }
    normalized.tool_calls = toolCalls;
  }
  if (toolResults.length === 1 && normalized.role === "tool") {
    Object.assign(normalized, toolResults[0]);
  } else if (toolResults.length > 0) {
    normalized.tool_results = toolResults;
  }

  return normalized;
}

function normalizeMessages(messages: unknown[] | undefined): unknown[] | undefined {
  return messages?.map((message) => normalizeBeeAIMessage(message));
}

function getMessageContentValue(message: unknown): unknown {
  const record = asRecord(message);
  const content = record?.content;
  if (!Array.isArray(content) || content.length !== 1) return normalizeBeeAIMessage(message);

  const block = asRecord(content[0]);
  if (!block) return normalizeBeeAIMessage(message);
  if (block.type === "text" && block.text !== undefined) return block.text;
  if (block.type === "tool-result" && block.result !== undefined) {
    return block.isError === undefined
      ? block.result
      : { result: block.result, is_error: Boolean(block.isError) };
  }
  if (block.type === "tool-call") {
    return { tool_call: normalizeToolCall(block) };
  }

  return normalizeBeeAIMessage(message);
}

function getAgentStateInput(state: Record<string, any> | undefined): unknown {
  if (!state) return undefined;
  const messages = getStateMessages(state);
  if (messages) {
    const hasOutput = getStateResultValue(state) !== undefined || getLastStateMessageValue(state) !== undefined;
    const inputMessages = hasOutput && messages.length > 0 ? messages.slice(0, -1) : messages;
    return {
      iteration: state.iteration,
      messages: normalizeMessages(inputMessages),
    };
  }
  return state;
}

function getChatInputFromState(state: Record<string, any> | undefined): unknown {
  const messages = getStateMessages(state);
  return messages ? normalizeMessages(messages) : undefined;
}

function sameToolCalls(left: unknown, right: unknown): boolean {
  if (left === undefined || right === undefined) return false;
  return safeJsonStr(left) === safeJsonStr(right);
}

function matchesAssistantOutput(message: unknown, output: unknown): boolean {
  const messageRecord = asRecord(message);
  const outputRecord = asRecord(output);
  if (!messageRecord || !outputRecord || messageRecord.role !== "assistant") return false;

  if (sameToolCalls(messageRecord.tool_calls, outputRecord.tool_calls)) {
    return true;
  }

  return (
    typeof messageRecord.content === "string" &&
    typeof outputRecord.content === "string" &&
    messageRecord.content === outputRecord.content
  );
}

function getPendingChatInputFromState(
  state: Record<string, any> | undefined,
  pendingSpan: ReadableSpan,
): unknown {
  const messages = normalizeMessages(getStateMessages(state));
  if (!messages || messages.length === 0) return undefined;

  const pendingAttrs = (pendingSpan as any).attributes as Record<string, any> | undefined;
  const output = parseJson(pendingAttrs?.[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]);

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (matchesAssistantOutput(messages[index], output)) {
      return index > 0 ? messages.slice(0, index) : undefined;
    }
  }

  const lastMessage = asRecord(messages[messages.length - 1]);
  if (lastMessage?.role === "assistant") {
    return messages.length > 1 ? messages.slice(0, -1) : undefined;
  }

  return messages;
}

function setChatInputAttributes(span: ReadableSpan, input: unknown): void {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs || input === undefined) return;

  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonStr(input);
  setChatPromptAttributes(attrs, input, true);
}

function queuePendingChatSpan(span: ReadableSpan, attrs: Record<string, any>): void {
  const traceKey = getSpanTraceKey(span, attrs);
  if (!traceKey) return;

  const queue = pendingChatSpansByTrace.get(traceKey) ?? [];
  if (!queue.includes(span)) {
    queue.push(span);
    pendingChatSpansByTrace.set(traceKey, queue);
  }
}

function flushPendingChatSpansFromState(
  span: ReadableSpan,
  attrs: Record<string, any>,
  state: Record<string, any> | undefined,
  exportSpan: ProcessorOnEnd | undefined,
): void {
  if (!exportSpan) return;

  const traceKey = getSpanTraceKey(span, attrs);
  if (!traceKey) return;

  const queue = pendingChatSpansByTrace.get(traceKey);
  if (!queue || queue.length === 0) return;

  const remaining: ReadableSpan[] = [];
  for (const pendingSpan of queue) {
    const pendingAttrs = (pendingSpan as any).attributes as Record<string, any> | undefined;
    const input = getPendingChatInputFromState(state, pendingSpan);
    if (!pendingAttrs || input === undefined) {
      remaining.push(pendingSpan);
      continue;
    }

    setChatInputAttributes(pendingSpan, input);
    exportSpan(pendingSpan);
  }

  if (remaining.length > 0) {
    pendingChatSpansByTrace.set(traceKey, remaining);
  } else {
    pendingChatSpansByTrace.delete(traceKey);
  }
}

function flushAllPendingChatSpans(exportSpan: ProcessorOnEnd | null): void {
  if (!exportSpan) return;

  for (const queue of pendingChatSpansByTrace.values()) {
    for (const pendingSpan of queue) {
      exportSpan(pendingSpan);
    }
  }
  pendingChatSpansByTrace.clear();
}

function shouldDelayMissingChatInputSpan(span: ReadableSpan): boolean {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs) return false;

  return (
    attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] === RespanLogType.CHAT &&
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] === undefined &&
    typeof attrs[BEEAI_TARGET] === "string" &&
    attrs[BEEAI_TARGET].endsWith(".success")
  );
}

function getLastStateMessageValue(state: Record<string, any> | undefined): unknown {
  const messages = getStateMessages(state);
  if (!messages || messages.length === 0) return undefined;
  return getMessageContentValue(messages[messages.length - 1]);
}

function getStateResultValue(state: Record<string, any> | undefined): unknown {
  if (!state || state.result === undefined) return undefined;
  return getMessageContentValue(state.result);
}

function getMatchingToolResult(
  state: Record<string, any> | undefined,
  toolCallMsg: Record<string, any> | undefined,
): unknown {
  const toolCallId = toolCallMsg?.toolCallId;
  if (!toolCallId) return undefined;

  const messages = getStateMessages(state);
  if (!messages) return undefined;

  for (const message of messages) {
    const content = asRecord(message)?.content;
    if (!Array.isArray(content)) continue;

    for (const blockValue of content) {
      const block = asRecord(blockValue);
      if (
        block?.type === "tool-result" &&
        block.toolCallId === toolCallId &&
        block.result !== undefined
      ) {
        return block.isError === undefined
          ? block.result
          : { result: block.result, is_error: Boolean(block.isError) };
      }
    }
  }

  return undefined;
}

function isFinalAnswerTool(toolCallMsg: Record<string, any> | undefined): boolean {
  const name = toolCallMsg?.toolName;
  return typeof name === "string" && name.toLowerCase() === "final_answer";
}

function normalizeInputValue(input: unknown): unknown {
  if (Array.isArray(input)) {
    return normalizeMessages(input);
  }

  const record = asRecord(input);
  if (record && Array.isArray(record.messages)) {
    return normalizeMessages(record.messages);
  }
  return input;
}

function normalizeMeaningfulInputValue(input: unknown): unknown {
  const normalized = normalizeInputValue(input);
  return isMeaningfulStructuredValue(normalized) ? normalized : undefined;
}

function normalizeOutputValue(output: unknown): unknown {
  if (Array.isArray(output)) {
    const messages = normalizeMessages(output);
    return messages && messages.length === 1 ? messages[0] : messages;
  }

  const record = asRecord(output);
  if (record && Array.isArray(record.messages)) {
    const messages = normalizeMessages(record.messages);
    return messages && messages.length === 1 ? messages[0] : messages;
  }
  if (record && record.content !== undefined && record.role !== undefined) {
    return normalizeBeeAIMessage(record);
  }

  return output;
}

function clearChatPromptAttributes(attrs: Record<string, any>): void {
  for (const key of Object.keys(attrs)) {
    if (key.startsWith(`${ATTR_GEN_AI_PROMPT}.`)) {
      delete attrs[key];
    }
  }
}

function setChatPromptAttributes(
  attrs: Record<string, any>,
  input: unknown,
  overwrite = false,
): void {
  const messages = Array.isArray(input)
    ? input
    : asRecord(input)?.role !== undefined
      ? [input]
      : undefined;
  if (!messages) return;

  if (overwrite) {
    clearChatPromptAttributes(attrs);
  }

  const setPromptAttribute = (key: string, value: unknown) => {
    if (overwrite) {
      attrs[key] = value;
    } else {
      setDefault(attrs, key, value);
    }
  };

  for (const [index, message] of messages.entries()) {
    const record = asRecord(message);
    if (!record) continue;

    const prefix = `${ATTR_GEN_AI_PROMPT}.${index}`;
    if (typeof record.role === "string") {
      setPromptAttribute(`${prefix}.role`, record.role);
    }
    if (typeof record.content === "string") {
      setPromptAttribute(`${prefix}.content`, record.content);
    }
    if (typeof record.name === "string") {
      setPromptAttribute(`${prefix}.name`, record.name);
    }
    if (typeof record.tool_call_id === "string") {
      setPromptAttribute(`${prefix}.tool_call_id`, record.tool_call_id);
    }
    if (record.tool_calls !== undefined) {
      setPromptAttribute(`${prefix}.tool_calls`, record.tool_calls);
    }
  }
}

function setChatCompletionAttributes(
  attrs: Record<string, any>,
  output: unknown,
): void {
  const outputRecord = asRecord(output);
  const messages = Array.isArray(output)
    ? output
    : outputRecord?.role === "assistant"
      ? [output]
      : undefined;
  const assistantMessage = messages
    ?.map((message) => asRecord(message))
    .find((message) => message?.role === "assistant");
  if (!assistantMessage) return;

  const completionPrefix = `${ATTR_GEN_AI_COMPLETION}.0`;

  setDefault(attrs, `${completionPrefix}.role`, "assistant");
  setDefault(
    attrs,
    `${completionPrefix}.content`,
    typeof assistantMessage.content === "string" ? assistantMessage.content : "",
  );

  if (assistantMessage.tool_calls !== undefined) {
    setDefault(
      attrs,
      RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
      safeJsonStr(assistantMessage.tool_calls),
    );
    setDefault(
      attrs,
      `${completionPrefix}.tool_calls`,
      assistantMessage.tool_calls,
    );
  }
}

function markSpanDropped(attrs: Record<string, any>): void {
  attrs[RespanSpanAttributes.RESPAN_PROCESSORS] = [];
}

function dropSpan(span: ReadableSpan, attrs: Record<string, any>): void {
  rememberDroppedSpanParent(span);
  markSpanDropped(attrs);
}

function shouldDropTarget(target: string): boolean {
  return target.endsWith(".start") || target.endsWith(".finish");
}

function isBeeAIFrameworkParentSpan(span: ReadableSpan, attrs: Record<string, any>): boolean {
  return span.name === "beeai-framework-main" && (
    attrs[BEEAI_VERSION] !== undefined || attrs.source !== undefined
  );
}

function getOpenInferenceInput(attrs: Record<string, any>): unknown {
  return parseJson(attrs[OPENINFERENCE_INPUT_VALUE]);
}

function getOpenInferenceOutput(attrs: Record<string, any>): unknown {
  return parseJson(attrs[OPENINFERENCE_OUTPUT_VALUE]);
}

function cacheBeeAIStartSpan(span: ReadableSpan): void {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs) return;

  rememberWorkflowSpan(span, attrs);

  if (span.name === "beeai-framework-main" || isBeeAIFrameworkParentSpan(span, attrs)) {
    rememberDroppedSpanParent(span);
    return;
  }

  if (getInstrumentationScopeName(span) !== BEEAI_SCOPE_NAME) return;

  const target = attrs[BEEAI_TARGET];
  const data = asRecord(parseJson(attrs[BEEAI_DATA]));
  const metadata = asRecord(parseJson(attrs[BEEAI_METADATA]));
  const state = asRecord(firstDefined(data?.state, metadata?.state));

  if (typeof target === "string" && target === "agent.toolCalling.start") {
    cacheChatInputFromAgentState(span, attrs, state);
  }

  if (typeof target === "string" && shouldDropTarget(target)) {
    rememberDroppedSpanParent(span);
  }

  if (target !== "backend.openai.chat.start") return;

  const directInput = firstDefined(data?.input, getOpenInferenceInput(attrs));
  if (directInput !== undefined) {
    enqueuePendingChatInput(span, attrs, normalizeMeaningfulInputValue(directInput));
  }
}

function cacheChatInputFromAgentState(
  span: ReadableSpan,
  attrs: Record<string, any>,
  state: Record<string, any> | undefined,
): void {
  enqueuePendingChatInput(span, attrs, getChatInputFromState(state));
}

function cleanupBeeAIRawAttributes(attrs: Record<string, any>): void {
  delete attrs[BEEAI_DATA];
  delete attrs[BEEAI_METADATA];
  delete attrs[BEEAI_VERSION];
  delete attrs.source;
  delete attrs[OPENINFERENCE_INPUT_VALUE];
  delete attrs[OPENINFERENCE_OUTPUT_VALUE];
  delete attrs["input.mime_type"];
  delete attrs["output.mime_type"];

  for (const key of Object.keys(attrs)) {
    if (key.startsWith("llm.input_messages.") || key.startsWith("llm.output_messages.")) {
      delete attrs[key];
    }
  }
}

function setInputOutputAttributes(
  span: ReadableSpan,
  attrs: Record<string, any>,
  logType: string,
  target: unknown,
  data: Record<string, any> | undefined,
  value: Record<string, any> | undefined,
): void {
  const metadata = asRecord(parseJson(attrs[BEEAI_METADATA]));
  const state = asRecord(firstDefined(data?.state, metadata?.state));
  const toolCallMsg = asRecord(firstDefined(data?.toolCallMsg, metadata?.toolCallMsg));
  const targetValue = typeof target === "string" ? target : "";

  const directInput = firstDefined(
    data?.input,
    value?.input,
    getOpenInferenceInput(attrs),
    toolCallMsg?.args,
  );
  const normalizedDirectInput = directInput !== undefined
    ? normalizeMeaningfulInputValue(directInput)
    : undefined;
  const cachedChatInput = logType === RespanLogType.CHAT && targetValue.endsWith(".success")
    ? dequeuePendingChatInput(span, attrs)
    : undefined;
  const stateChatInput = logType === RespanLogType.CHAT
    ? getChatInputFromState(state)
    : undefined;
  const input = logType === RespanLogType.CHAT
    ? firstDefined(cachedChatInput, stateChatInput, normalizedDirectInput)
    : firstDefined(
        normalizedDirectInput,
        logType === RespanLogType.AGENT ? getAgentStateInput(state) : undefined,
      );
  if (input !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonStr(input);
    if (logType === RespanLogType.CHAT) {
      setChatPromptAttributes(attrs, input, true);
    }
  }

  const finalAnswerOutput = targetValue.includes("finalAnswer.") && isFinalAnswerTool(toolCallMsg)
    ? getStateResultValue(state)
    : undefined;
  const directOutput = firstDefined(
    value?.messages,
    getOpenInferenceOutput(attrs),
    data?.output,
    value?.output,
    value?.result,
  );
  const output = firstDefined(
    finalAnswerOutput,
    directOutput !== undefined ? normalizeOutputValue(directOutput) : undefined,
    getMatchingToolResult(state, toolCallMsg),
    logType === RespanLogType.AGENT
      ? firstDefined(getStateResultValue(state), getLastStateMessageValue(state))
      : undefined,
  );
  if (output !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonStr(output);
    if (logType === RespanLogType.CHAT) {
      setChatCompletionAttributes(attrs, output);
    }
  }
}

function translateBeeAIEventSpan(span: ReadableSpan, exportSpan?: ProcessorOnEnd): void {
  const attrs = (span as any).attributes as Record<string, any> | undefined;
  if (!attrs) return;

  if (isBeeAIFrameworkParentSpan(span, attrs)) {
    dropSpan(span, attrs);
    return;
  }

  if (getInstrumentationScopeName(span) !== BEEAI_SCOPE_NAME) return;

  const target = attrs[BEEAI_TARGET];
  const data = asRecord(parseJson(attrs[BEEAI_DATA]));
  const metadata = asRecord(parseJson(attrs[BEEAI_METADATA]));
  const state = asRecord(firstDefined(data?.state, metadata?.state));

  if (typeof target === "string" && target === "backend.openai.chat.start") {
    if (data?.input !== undefined) {
      enqueuePendingChatInput(span, attrs, normalizeInputValue(data.input));
    }
  }

  if (typeof target === "string" && target === "agent.toolCalling.success") {
    flushPendingChatSpansFromState(span, attrs, state, exportSpan);
    cacheChatInputFromAgentState(span, attrs, state);
  }

  if (typeof target === "string" && shouldDropTarget(target)) {
    dropSpan(span, attrs);
    return;
  }

  const logType = getBeeAIEventLogType(target);
  if (!logType) return;

  setDefault(attrs, RespanSpanAttributes.RESPAN_LOG_TYPE, logType);
  if (logType === RespanLogType.CHAT || logType === RespanLogType.EMBEDDING) {
    setDefault(attrs, SpanAttributes.LLM_REQUEST_TYPE, logType);
  }

  setDefault(
    attrs,
    SpanAttributes.TRACELOOP_ENTITY_NAME,
    typeof target === "string" && target.length > 0 ? target : span.name,
  );
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, span.name);

  const value = asRecord(data?.value);
  const usage = asRecord(firstDefined(value?.usage, data?.usage));
  setTokenAttributes(attrs, usage);

  const model = firstDefined(data?.model, value?.model);
  if (model !== undefined) {
    setDefault(attrs, ATTR_GEN_AI_REQUEST_MODEL, model);
    setDefault(attrs, DIRECT_MODEL, model);
  }

  setInputOutputAttributes(span, attrs, logType, target, data, value);
  reparentFromDroppedSpans(span);
  cleanupBeeAIRawAttributes(attrs);
}


interface InstrumentationDelegate {
  activate(): void;
  deactivate(): void;
}

type DelegateFactory = (
  instrumentationClass: BeeAIInstrumentationClass,
  sdkModule: Record<string, unknown>,
) => InstrumentationDelegate | Promise<InstrumentationDelegate>;

interface OpenInferenceModule {
  OpenInferenceInstrumentor: new (
    instrumentationClass: BeeAIInstrumentationClass,
    sdkModule?: Record<string, unknown>,
  ) => InstrumentationDelegate;
}

export interface BeeAIInstrumentorOptions {
  /**
   * Optional BeeAI module object. Pass this in ESM/bundled environments when
   * the OpenInference instrumentor needs to patch a specific module instance.
   */
  sdkModule?: Record<string, unknown>;
  /**
   * Internal extension point for tests and compatible OpenInference subclasses.
   */
  instrumentationClass?: BeeAIInstrumentationClass;
  /**
   * Internal extension point for tests and custom OpenInference delegate wiring.
   */
  delegateFactory?: DelegateFactory;
}

export class BeeAIInstrumentor {
  public readonly name = "beeai";

  private readonly _sdkModule?: Record<string, unknown>;
  private readonly _instrumentationClass: BeeAIInstrumentationClass;
  private readonly _delegateFactory?: DelegateFactory;
  private _delegate: InstrumentationDelegate | null = null;
  private _ownsTranslatorHook = false;

  private static _translatorHookRefCount = 0;
  private static _patchedProcessor: any = null;
  private static _originalProcessorOnStart: ProcessorOnStart | null = null;
  private static _wrappedProcessorOnStart: ProcessorOnStart | null = null;
  private static _originalProcessorOnEnd: ProcessorOnEnd | null = null;
  private static _wrappedProcessorOnEnd: ProcessorOnEnd | null = null;

  constructor(options: BeeAIInstrumentorOptions = {}) {
    this._sdkModule = options.sdkModule;
    this._instrumentationClass =
      options.instrumentationClass ?? BeeAIInstrumentation;
    this._delegateFactory = options.delegateFactory;
  }

  async activate(): Promise<void> {
    if (this._delegate) {
      return;
    }

    const sdkModule = this._sdkModule ?? (await this._loadBeeAIFramework());
    this._delegate = await this._createDelegate(sdkModule);
    this._delegate.activate();

    if (BeeAIInstrumentor._installTranslatorHook()) {
      BeeAIInstrumentor._translatorHookRefCount += 1;
      this._ownsTranslatorHook = true;
    }
  }

  deactivate(): void {
    if (this._ownsTranslatorHook) {
      BeeAIInstrumentor._translatorHookRefCount = Math.max(
        0,
        BeeAIInstrumentor._translatorHookRefCount - 1,
      );
      this._ownsTranslatorHook = false;

      if (BeeAIInstrumentor._translatorHookRefCount === 0) {
        BeeAIInstrumentor._restoreTranslatorHook();
      }
    }

    this._delegate?.deactivate();
    this._delegate = null;
  }

  private static _getActiveSpanProcessor(): any {
    const tracerProvider = trace.getTracerProvider() as any;
    return (
      tracerProvider?.activeSpanProcessor ??
      tracerProvider?._activeSpanProcessor ??
      tracerProvider?._delegate?.activeSpanProcessor ??
      tracerProvider?._delegate?._activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?.activeSpanProcessor ??
      tracerProvider?._delegate?._tracerProvider?._activeSpanProcessor
    );
  }

  private static _installTranslatorHook(): boolean {
    const processor = BeeAIInstrumentor._getActiveSpanProcessor();
    if (!processor || typeof processor.onEnd !== "function") {
      return false;
    }

    if (BeeAIInstrumentor._patchedProcessor === processor) {
      return true;
    }

    BeeAIInstrumentor._restoreTranslatorHook();

    const originalProcessorOnEnd = processor.onEnd.bind(processor);
    const originalProcessorOnStart = typeof processor.onStart === "function"
      ? processor.onStart.bind(processor)
      : null;
    const wrappedProcessorOnStart = originalProcessorOnStart
      ? (span: ReadableSpan, parentContext: unknown) => {
          try {
            cacheBeeAIStartSpan(span);
          } catch {
            // Translation must never block span export.
          }
          return originalProcessorOnStart(span, parentContext);
        }
      : null;
    const wrappedProcessorOnEnd = (span: ReadableSpan) => {
      try {
        translateBeeAIEventSpan(span, originalProcessorOnEnd);
        if (shouldDelayMissingChatInputSpan(span)) {
          queuePendingChatSpan(span, ((span as any).attributes ?? {}) as Record<string, any>);
          return;
        }
      } catch {
        // Translation must never block span export.
      }
      return originalProcessorOnEnd(span);
    };

    if (wrappedProcessorOnStart) {
      processor.onStart = wrappedProcessorOnStart;
    }
    processor.onEnd = wrappedProcessorOnEnd;
    BeeAIInstrumentor._patchedProcessor = processor;
    BeeAIInstrumentor._originalProcessorOnStart = originalProcessorOnStart;
    BeeAIInstrumentor._wrappedProcessorOnStart = wrappedProcessorOnStart;
    BeeAIInstrumentor._originalProcessorOnEnd = originalProcessorOnEnd;
    BeeAIInstrumentor._wrappedProcessorOnEnd = wrappedProcessorOnEnd;
    return true;
  }

  private static _restoreTranslatorHook(): void {
    const processor = BeeAIInstrumentor._patchedProcessor;
    const originalOnStart = BeeAIInstrumentor._originalProcessorOnStart;
    const wrappedOnStart = BeeAIInstrumentor._wrappedProcessorOnStart;
    const originalOnEnd = BeeAIInstrumentor._originalProcessorOnEnd;
    const wrappedOnEnd = BeeAIInstrumentor._wrappedProcessorOnEnd;

    if (processor && originalOnStart) {
      if (!wrappedOnStart || processor.onStart === wrappedOnStart) {
        processor.onStart = originalOnStart;
      } else {
        console.warn(
          "[respan] BeeAIInstrumentor: active span processor onStart was modified externally; original handler could not be restored.",
        );
      }
    }

    if (processor && originalOnEnd) {
      if (!wrappedOnEnd || processor.onEnd === wrappedOnEnd) {
        processor.onEnd = originalOnEnd;
      } else {
        console.warn(
          "[respan] BeeAIInstrumentor: active span processor onEnd was modified externally; original handler could not be restored.",
        );
      }
    }

    flushAllPendingChatSpans(originalOnEnd);
    pendingChatInputsByTrace.clear();
    pendingChatSpansByTrace.clear();
    droppedSpanParentsByTrace.clear();
    workflowSpanIdsByTrace.clear();
    BeeAIInstrumentor._patchedProcessor = null;
    BeeAIInstrumentor._originalProcessorOnStart = null;
    BeeAIInstrumentor._wrappedProcessorOnStart = null;
    BeeAIInstrumentor._originalProcessorOnEnd = null;
    BeeAIInstrumentor._wrappedProcessorOnEnd = null;
  }

  private async _createDelegate(
    sdkModule: Record<string, unknown>,
  ): Promise<InstrumentationDelegate> {
    if (this._delegateFactory) {
      return this._delegateFactory(this._instrumentationClass, sdkModule);
    }

    const { OpenInferenceInstrumentor } = await importOpenInferenceModule();
    return new OpenInferenceInstrumentor(this._instrumentationClass, sdkModule);
  }

  private async _loadBeeAIFramework(): Promise<Record<string, unknown>> {
    try {
      return (await import("beeai-framework")) as Record<string, unknown>;
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(
        `BeeAIInstrumentor requires beeai-framework to be installed, or a sdkModule option to be provided. ${reason}`,
      );
    }
  }
}

async function importOpenInferenceModule(): Promise<OpenInferenceModule> {
  const dynamicImport = new Function("specifier", "return import(specifier)") as (
    specifier: string,
  ) => Promise<OpenInferenceModule>;
  return dynamicImport("@respan/instrumentation-openinference");
}

export { BeeAIInstrumentor as BeeAIInstrumentation };
