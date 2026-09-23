import { SpanStatusCode, type Span } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_EXCEPTION_MESSAGE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_CONVERSATION_ID,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_HTTP_STATUS_CODE,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  AI_TELEMETRY_METADATA_PREFIX,
  N8N_AI_SDK_LLM_SPAN_NAMES,
  N8N_AI_SDK_STRUCTURAL_LLM_SPAN_NAMES,
  N8N_AGENT_METADATA_KEYS,
  N8N_AGENT_SCOPE,
  N8N_ATTRIBUTE_PREFIX,
  N8N_ATTRIBUTES,
  N8N_MEMORY_ATTRIBUTE_PREFIX,
  N8N_MEMORY_OPERATIONS,
  N8N_SPAN_NAMES,
  OFF_CONTRACT_ALIAS_KEYS,
} from "./_constants.js";

type MutableAttributes = Record<string, unknown>;

const BACKEND_STATUS_CODE = "status_code";
// n8n/provider input alias. No pinned semconv package exports it; it is read
// only for status promotion and is always removed before export.
const N8N_RAW_GEN_AI_RESPONSE_STATUS_CODE = "gen_ai.response.status_code";
const AI_MODEL_PROVIDER = "ai.model.provider";
const AI_MODEL_ID = "ai.model.id";
const AI_PROMPT = "ai.prompt";
const AI_PROMPT_MESSAGES = "ai.prompt.messages";
const AI_PROMPT_TOOLS = "ai.prompt.tools";
const AI_RESPONSE_TEXT = "ai.response.text";
const AI_RESPONSE_TOOL_CALLS = "ai.response.toolCalls";
const AI_USAGE_INPUT_TOKENS = "ai.usage.inputTokens";
const AI_USAGE_OUTPUT_TOKENS = "ai.usage.outputTokens";
const AI_USAGE_TOTAL_TOKENS = "ai.usage.totalTokens";
const AI_USAGE_CACHED_INPUT_TOKENS = "ai.usage.cachedInputTokens";
const AI_OPERATION_ID = "ai.operationId";
const AI_FUNCTION_ID = "ai.telemetry.functionId";
const REDACTED_METADATA_VALUE = "[REDACTED]";
// Required by contribution/span-contract.md. The pinned Traceloop JS 0.13
// package has no exported constant for this canonical key yet.
const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";

export type N8nSpanKind = "workflow" | "node" | "agent" | "tool" | "llm" | "memory";

export interface N8nSpanLike {
  readonly name: string;
  readonly attributes: Readonly<Record<string, unknown>>;
  readonly instrumentationScope?: { readonly name?: string };
  readonly instrumentationLibrary?: { readonly name?: string };
}

export function instrumentationScopeName(span: N8nSpanLike): string | undefined {
  return span.instrumentationScope?.name ?? span.instrumentationLibrary?.name;
}

export function classifyN8nSpan(span: N8nSpanLike): N8nSpanKind | undefined {
  const attrs = span.attributes;

  if (
    span.name === N8N_SPAN_NAMES.workflow &&
    (attrs[N8N_ATTRIBUTES.workflowId] !== undefined ||
      attrs[N8N_ATTRIBUTES.workflowName] !== undefined)
  ) {
    return "workflow";
  }

  if (
    span.name === N8N_SPAN_NAMES.node &&
    (attrs[N8N_ATTRIBUTES.nodeId] !== undefined ||
      attrs[N8N_ATTRIBUTES.nodeName] !== undefined)
  ) {
    return "node";
  }

  const operation = stringAttr(attrs, ATTR_GEN_AI_OPERATION_NAME)?.toLowerCase();
  const n8nAgentSpan = isN8nAgentScopedSpan(span);

  if (
    n8nAgentSpan &&
    operation === "invoke_agent" &&
    /\.(?:generate|stream)$/.test(span.name)
  ) {
    return "agent";
  }

  if (
    n8nAgentSpan &&
    operation === "execute_tool" &&
    (span.name.startsWith("execute_tool ") || attrs[ATTR_GEN_AI_TOOL_NAME] !== undefined)
  ) {
    return "tool";
  }

  if (n8nAgentSpan && N8N_MEMORY_OPERATIONS.has(operation ?? "")) {
    return "memory";
  }

  if (n8nAgentSpan && N8N_AI_SDK_LLM_SPAN_NAMES.has(span.name)) {
    return "llm";
  }

  if (n8nAgentSpan && span.name === N8N_SPAN_NAMES.aiToolCall) {
    return "tool";
  }

  return undefined;
}

export function isN8nSpan(span: N8nSpanLike): boolean {
  return classifyN8nSpan(span) !== undefined;
}

export function isN8nStructuralLlmWrapper(span: N8nSpanLike): boolean {
  return (
    isN8nAgentScopedSpan(span) &&
    N8N_AI_SDK_STRUCTURAL_LLM_SPAN_NAMES.has(span.name)
  );
}

export function isN8nAiSdkToolSpan(span: N8nSpanLike): boolean {
  return isN8nAgentScopedSpan(span) && span.name === N8N_SPAN_NAMES.aiToolCall;
}

export function workflowNameFromSpan(span: N8nSpanLike): string | undefined {
  return firstStringAttr(span.attributes, [
    N8N_ATTRIBUTES.workflowName,
    SpanAttributes.TRACELOOP_WORKFLOW_NAME,
  ]);
}

export function enrichLiveN8nSpan(
  span: Span & Partial<N8nSpanLike>,
  workflowName?: string,
): N8nSpanKind | undefined {
  const mutable = span as Span & N8nSpanLike & { attributes: MutableAttributes };
  const kind = classifyN8nSpan(mutable);
  if (!kind) return undefined;

  const next = { ...mutable.attributes };
  enrichAttributes(kind, mutable.name, next, workflowName);

  for (const [key, value] of Object.entries(next)) {
    if (value === undefined || mutable.attributes[key] === value) continue;
    try {
      span.setAttribute(key, value as never);
    } catch {
      // Translation is best effort and must never block workflow execution.
    }
  }

  return kind;
}

export function enrichEndedN8nSpan(
  span: ReadableSpan,
  workflowName?: string,
): N8nSpanKind | undefined {
  const kind = classifyN8nSpan(span as unknown as N8nSpanLike);
  if (!kind) return undefined;

  const attrs = (span as unknown as { attributes: MutableAttributes }).attributes;
  enrichAttributes(kind, span.name, attrs, workflowName);
  return kind;
}

export function prepareN8nSpanForExport(span: ReadableSpan): ReadableSpan {
  const kind = classifyN8nSpan(span as unknown as N8nSpanLike);
  if (!kind) return span;

  const attributes: MutableAttributes = { ...span.attributes };
  enrichAttributes(
    kind,
    span.name,
    attributes,
    firstStringAttr(attributes, [SpanAttributes.TRACELOOP_WORKFLOW_NAME]),
  );
  setBackendStatusCode(span, attributes);
  promoteErrorMessage(span, attributes);
  stripVendorAttributes(kind, attributes);
  stripOffContractAliases(attributes);

  return cloneReadableSpanForExport(
    span,
    attributes,
    kind === "llm" ? stripAiSdkEvents(span.events) : span.events,
  );
}

function promoteErrorMessage(span: ReadableSpan, attrs: MutableAttributes): void {
  if (span.status.code !== SpanStatusCode.ERROR || attrs[ATTR_ERROR_MESSAGE] !== undefined) {
    return;
  }

  for (const event of span.events) {
    if (event.name !== "exception") continue;
    const message = event.attributes?.[ATTR_EXCEPTION_MESSAGE];
    if (message === undefined || message === null || String(message).trim() === "") continue;
    attrs[ATTR_ERROR_MESSAGE] = String(message);
    return;
  }
}

/**
 * Privacy backstop for transform failures. Keep normal fail-open behavior, but
 * never delegate raw n8n Agent `ai.*` attributes or events. The normal path
 * promotes safe values first; the exceptional path prefers privacy over raw
 * vendor diagnostics.
 */
export function sanitizeN8nSpanForFailSafeExport(span: ReadableSpan): ReadableSpan {
  const spanLike = span as unknown as N8nSpanLike;
  if (!isN8nAgentScopedSpan(spanLike)) return span;

  let attributes: MutableAttributes | undefined;
  for (const key of Object.keys(span.attributes)) {
    if (key.startsWith("ai.")) {
      attributes ??= { ...span.attributes };
      delete attributes[key];
    }
  }

  const events = stripAiSdkEvents(span.events);
  if (!attributes && events === span.events) return span;
  return cloneReadableSpanForExport(
    span,
    attributes ?? { ...span.attributes },
    events,
  );
}

function setBackendStatusCode(span: ReadableSpan, attrs: MutableAttributes): void {
  const isError = span.status.code === SpanStatusCode.ERROR;
  const extracted = firstStatusCode(attrs, [
    BACKEND_STATUS_CODE,
    ATTR_HTTP_RESPONSE_STATUS_CODE,
    ATTR_HTTP_STATUS_CODE,
    N8N_RAW_GEN_AI_RESPONSE_STATUS_CODE,
  ]);
  attrs[BACKEND_STATUS_CODE] =
    extracted === undefined ? (isError ? 500 : 200) : isError && extracted < 400 ? 500 : extracted;
}

function firstStatusCode(
  attrs: Readonly<MutableAttributes>,
  keys: readonly string[],
): number | undefined {
  for (const key of keys) {
    const value = attrs[key];
    if (typeof value === "number" && Number.isFinite(value)) return Math.trunc(value);
    if (typeof value !== "string" || value.trim() === "") continue;
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
  }
  return undefined;
}

function enrichAttributes(
  kind: N8nSpanKind,
  spanName: string,
  attrs: MutableAttributes,
  inheritedWorkflowName?: string,
): void {
  const metadata = collectN8nMetadata(attrs);
  const workflowName =
    firstStringAttr(attrs, [N8N_ATTRIBUTES.workflowName, SpanAttributes.TRACELOOP_WORKFLOW_NAME]) ??
    inheritedWorkflowName;

  switch (kind) {
    case "workflow": {
      const entityName =
        firstStringAttr(attrs, [N8N_ATTRIBUTES.workflowName]) ?? "workflow";
      attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.WORKFLOW;
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, "");
      setDefault(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, entityName);
      break;
    }
    case "node": {
      const entityName = firstStringAttr(attrs, [N8N_ATTRIBUTES.nodeName]) ?? "task";
      attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.TASK;
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, workflowName ?? "");
      if (workflowName) setDefault(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflowName);
      break;
    }
    case "agent": {
      const entityName =
        firstStringAttr(attrs, [ATTR_GEN_AI_AGENT_NAME]) ??
        (spanName.replace(/\.(?:generate|stream)$/, "") || "agent");
      attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.AGENT;
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, workflowName ?? "");
      if (workflowName) setDefault(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflowName);

      const threadId = firstStringAttr(attrs, [
        ATTR_GEN_AI_CONVERSATION_ID,
        "thread_id",
        `${AI_TELEMETRY_METADATA_PREFIX}thread_id`,
      ]);
      if (threadId) setDefault(attrs, RespanSpanAttributes.RESPAN_THREADS_ID, threadId);

      const prompt = attrs[ATTR_GEN_AI_PROMPT];
      if (prompt !== undefined) {
        setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, jsonAttribute(prompt));
      }
      break;
    }
    case "tool": {
      const entityName =
        firstStringAttr(attrs, [ATTR_GEN_AI_TOOL_NAME, "ai.toolCall.name"]) ??
        (spanName.replace(/^execute_tool\s+/, "") || "tool");
      attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.TOOL;
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, workflowName ?? "");
      if (workflowName) setDefault(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflowName);

      const args = attrs[ATTR_GEN_AI_TOOL_CALL_ARGUMENTS] ?? attrs["ai.toolCall.args"];
      if (args !== undefined) {
        setDefault(
          attrs,
          SpanAttributes.TRACELOOP_ENTITY_INPUT,
          safeJson({ name: entityName, arguments: parseJsonValue(args) }),
        );
      }

      const result = attrs[ATTR_GEN_AI_TOOL_CALL_RESULT] ?? attrs["ai.toolCall.result"];
      if (result !== undefined) {
        setDefault(
          attrs,
          SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
          safeJson(parseJsonValue(result)),
        );
      }
      break;
    }
    case "llm": {
      attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.TEXT;
      setDefault(
        attrs,
        SpanAttributes.TRACELOOP_ENTITY_NAME,
        spanName.includes("stream") ? "stream" : "generate",
      );
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, workflowName ?? "");
      if (workflowName) setDefault(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflowName);
      setDefault(attrs, SpanAttributes.LLM_REQUEST_TYPE, RespanLogType.CHAT);
      enrichAiSdkLlmAttributes(attrs);
      break;
    }
    case "memory": {
      const operation =
        stringAttr(attrs, ATTR_GEN_AI_OPERATION_NAME)?.toLowerCase() ?? "memory";
      attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.TASK;
      setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, operation);
      setDefault(
        attrs,
        SpanAttributes.TRACELOOP_ENTITY_PATH,
        workflowName ?? firstStringAttr(attrs, [ATTR_GEN_AI_AGENT_NAME]) ?? "",
      );
      if (workflowName) setDefault(attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, workflowName);
      enrichMemoryContent(attrs, operation);
      break;
    }
  }

  delete attrs[SpanAttributes.TRACELOOP_SPAN_KIND];
  mergeN8nMetadata(attrs, metadata);
}

function enrichAiSdkLlmAttributes(attrs: MutableAttributes): void {
  const model = firstStringAttr(attrs, [ATTR_GEN_AI_REQUEST_MODEL, AI_MODEL_ID]);
  if (model) setDefault(attrs, ATTR_GEN_AI_REQUEST_MODEL, model);

  const system = normalizeSystem(attrs[ATTR_GEN_AI_SYSTEM] ?? attrs[AI_MODEL_PROVIDER]);
  if (system) attrs[ATTR_GEN_AI_SYSTEM] = system;

  const prompt = normalizePromptMessages(attrs[AI_PROMPT_MESSAGES] ?? attrs[AI_PROMPT]);
  if (prompt && prompt.length > 0) {
    setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, safeJson(prompt));
    enrichIndexedMessages(attrs, SpanAttributes.LLM_PROMPTS, prompt);
  }

  const toolDefinitions = normalizeToolDefinitions(attrs[AI_PROMPT_TOOLS]);
  if (toolDefinitions && toolDefinitions.length > 0) {
    setDefault(attrs, SpanAttributes.LLM_REQUEST_FUNCTIONS, safeJson(toolDefinitions));
  }

  const completion = completionMessage(attrs);
  if (completion) {
    setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, safeJson(completion));
    enrichIndexedMessages(attrs, SpanAttributes.LLM_COMPLETIONS, [completion]);
  }

  const inputTokens = numberAttr(
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] ??
      attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] ??
      attrs[AI_USAGE_INPUT_TOKENS],
  );
  const outputTokens = numberAttr(
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] ??
      attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] ??
      attrs[AI_USAGE_OUTPUT_TOKENS],
  );
  const totalTokens =
    numberAttr(attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] ?? attrs[AI_USAGE_TOTAL_TOKENS]) ??
    (inputTokens !== undefined && outputTokens !== undefined
      ? inputTokens + outputTokens
      : undefined);
  const cacheReadInputTokens = numberAttr(
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] ??
      attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] ??
      attrs[AI_USAGE_CACHED_INPUT_TOKENS],
  );

  if (inputTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = inputTokens;
  }
  if (outputTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = outputTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = outputTokens;
  }
  if (totalTokens !== undefined) attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  if (cacheReadInputTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cacheReadInputTokens;
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cacheReadInputTokens;
  }

  const finishReason = stringAttr(attrs, "ai.response.finishReason");
  if (finishReason) setDefault(attrs, ATTR_GEN_AI_RESPONSE_FINISH_REASONS, [finishReason]);
  const responseId = stringAttr(attrs, "ai.response.id");
  if (responseId) setDefault(attrs, ATTR_GEN_AI_RESPONSE_ID, responseId);
  const responseModel = stringAttr(attrs, "ai.response.model");
  if (responseModel) setDefault(attrs, ATTR_GEN_AI_RESPONSE_MODEL, responseModel);
}

function enrichMemoryContent(attrs: MutableAttributes, operation: string): void {
  const input: Record<string, unknown> = { operation };
  const output: Record<string, unknown> = {};

  for (const key of ["types", "owners", "store.types", "store.names"]) {
    const value = attrs[`${N8N_MEMORY_ATTRIBUTE_PREFIX}${key}`];
    if (value !== undefined) input[key.replace(".", "_")] = value;
  }
  for (const key of ["ids", "descriptions", "operations"]) {
    const value = attrs[`${N8N_MEMORY_ATTRIBUTE_PREFIX}${key}`];
    if (value !== undefined) output[key] = value;
  }

  if (Object.keys(input).length > 1) {
    setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, safeJson(input));
  }
  if (Object.keys(output).length > 0) {
    setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, safeJson(output));
  }
}

interface CanonicalMessage {
  role: string;
  content: string;
  tool_calls?: Record<string, unknown>[];
  tool_call_id?: string;
  name?: string;
}

function normalizePromptMessages(value: unknown): CanonicalMessage[] | undefined {
  const parsed = parseJsonValue(value);
  if (parsed === undefined || parsed === null) return undefined;

  const messages: unknown[] = [];
  if (isRecord(parsed)) {
    const system = parsed.system ?? parsed.instructions;
    if (system !== undefined && system !== null && String(system) !== "") {
      messages.push({ role: "system", content: system });
    }
    const nested = parsed.messages ?? parsed.prompt;
    if (Array.isArray(nested)) messages.push(...nested);
    else if (nested !== undefined && nested !== null) messages.push(nested);
    else if (parsed.role !== undefined) messages.push(parsed);
  } else if (Array.isArray(parsed)) {
    messages.push(...parsed);
  } else {
    messages.push({ role: "user", content: parsed });
  }

  const normalized = messages
    .map(normalizeMessage)
    .filter((message): message is CanonicalMessage => message !== undefined);
  return normalized.length > 0 ? normalized : undefined;
}

function normalizeMessage(value: unknown): CanonicalMessage | undefined {
  if (!isRecord(value)) {
    if (value === undefined || value === null) return undefined;
    return { role: "user", content: String(value) };
  }

  const role = firstRecordString(value, ["role"]) ?? "user";
  const contentValue = value.content ?? value.text ?? "";
  const content = normalizeMessageContent(contentValue);
  const toolCalls = normalizeToolCalls(value.tool_calls ?? value.toolCalls ?? toolCallsFromContent(contentValue));
  const message: CanonicalMessage = { role, content };
  if (toolCalls && toolCalls.length > 0) message.tool_calls = toolCalls;

  const toolCallId = firstRecordString(value, ["tool_call_id", "toolCallId"]);
  if (toolCallId) message.tool_call_id = toolCallId;
  const name = firstRecordString(value, ["name", "toolName", "tool_name"]);
  if (name) message.name = name;
  return message;
}

function normalizeMessageContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) return value === undefined ? "" : safeJson(value);

  const text: string[] = [];
  const unhandled: unknown[] = [];
  for (const part of value) {
    if (typeof part === "string") {
      text.push(part);
      continue;
    }
    if (
      isRecord(part) &&
      ["text", "input_text", "output_text", "reasoning"].includes(String(part.type ?? ""))
    ) {
      const partText = part.text ?? part.content;
      if (partText !== undefined && partText !== null) text.push(String(partText));
      continue;
    }
    if (isRecord(part) && ["tool-call", "tool-result"].includes(String(part.type ?? ""))) {
      continue;
    }
    unhandled.push(part);
  }
  if (unhandled.length === 0) return text.join("\n");
  return safeJson(value);
}

function toolCallsFromContent(value: unknown): unknown {
  if (!Array.isArray(value)) return undefined;
  const calls = value.filter(
    (part) => isRecord(part) && ["tool-call", "tool_call"].includes(String(part.type ?? "")),
  );
  return calls.length > 0 ? calls : undefined;
}

function normalizeToolCalls(value: unknown): Record<string, unknown>[] | undefined {
  const parsed = parseJsonValue(value);
  const calls = Array.isArray(parsed) ? parsed : parsed !== undefined ? [parsed] : [];
  const normalized = calls.flatMap((call) => {
    if (!isRecord(call)) return [];
    const functionValue = isRecord(call.function) ? call.function : undefined;
    const name =
      firstRecordString(functionValue ?? {}, ["name"]) ??
      firstRecordString(call, ["name", "toolName", "tool_name"]);
    const id = firstRecordString(call, ["id", "toolCallId", "tool_call_id"]);
    const args =
      functionValue?.arguments ??
      functionValue?.args ??
      call.arguments ??
      call.args ??
      call.input;
    if (!name && !id && args === undefined) return [];

    const normalizedCall: Record<string, unknown> = { type: "function" };
    if (id) normalizedCall.id = id;
    const functionPayload: Record<string, unknown> = {};
    if (name) functionPayload.name = name;
    if (args !== undefined) {
      functionPayload.arguments = typeof args === "string" ? args : safeJson(args);
    }
    if (Object.keys(functionPayload).length > 0) normalizedCall.function = functionPayload;
    return [normalizedCall];
  });
  return normalized.length > 0 ? normalized : undefined;
}

function normalizeToolDefinitions(value: unknown): Record<string, unknown>[] | undefined {
  const parsed = parseJsonValue(value);
  const tools = Array.isArray(parsed) ? parsed : parsed !== undefined ? [parsed] : [];
  const normalized = tools.flatMap((tool) => {
    const resolved = parseJsonValue(tool);
    if (!isRecord(resolved)) return [];
    if (resolved.type === "function" && isRecord(resolved.function)) return [resolved];

    const name = firstRecordString(resolved, ["name", "toolName", "tool_name"]);
    if (!name) return [resolved];
    const description = firstRecordString(resolved, ["description"]);
    const parameters = resolved.parameters ?? resolved.inputSchema ?? resolved.input_schema;
    return [
      {
        type: "function",
        function: {
          name,
          ...(description ? { description } : {}),
          ...(parameters !== undefined ? { parameters } : {}),
        },
      },
    ];
  });
  return normalized.length > 0 ? normalized : undefined;
}

function completionMessage(attrs: Readonly<MutableAttributes>): CanonicalMessage | undefined {
  const contentValue = attrs[AI_RESPONSE_TEXT];
  const content = contentValue === undefined || contentValue === null ? "" : String(contentValue);
  const toolCalls = normalizeToolCalls(attrs[AI_RESPONSE_TOOL_CALLS]);
  if (!content && !toolCalls) return undefined;
  return {
    role: "assistant",
    content,
    ...(toolCalls ? { tool_calls: toolCalls } : {}),
  };
}

function enrichIndexedMessages(
  attrs: MutableAttributes,
  prefix: string,
  messages: readonly CanonicalMessage[],
): void {
  messages.forEach((message, index) => {
    attrs[`${prefix}.${index}.role`] = message.role;
    attrs[`${prefix}.${index}.content`] = message.content;
    if (message.tool_calls) {
      attrs[`${prefix}.${index}.tool_calls`] = safeJson(message.tool_calls);
    }
    if (message.tool_call_id) {
      attrs[`${prefix}.${index}.tool_call_id`] = message.tool_call_id;
    }
    if (message.name) attrs[`${prefix}.${index}.name`] = message.name;
  });
}

function normalizeSystem(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  const normalized = String(value).trim().toLowerCase();
  if (!normalized) return undefined;
  if (normalized.includes("openai")) return "openai";
  if (normalized.includes("anthropic")) return "anthropic";
  if (normalized.includes("google") || normalized.includes("gemini")) return "google";
  if (normalized.includes("bedrock")) return "bedrock";
  if (normalized.includes("azure")) return "azure";
  if (normalized.includes("mistral")) return "mistral";
  if (normalized.includes("cohere")) return "cohere";
  if (normalized.includes("groq")) return "groq";
  if (normalized.includes("xai")) return "xai";
  if (normalized.includes("deepseek")) return "deepseek";
  return normalized.split(/[.:/]/, 1)[0] || normalized;
}

function numberAttr(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function collectN8nMetadata(attrs: Readonly<MutableAttributes>): Record<string, unknown> {
  const metadata: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined) continue;
    if (key.startsWith(N8N_ATTRIBUTE_PREFIX)) {
      metadata[key.slice(N8N_ATTRIBUTE_PREFIX.length)] = value;
    }
  }

  for (const key of N8N_AGENT_METADATA_KEYS) {
    const direct = attrs[key];
    const prefixed = attrs[`${AI_TELEMETRY_METADATA_PREFIX}${key}`];
    if (direct !== undefined) metadata[key] = direct;
    else if (prefixed !== undefined) metadata[key] = prefixed;
  }

  const telemetryMetadata: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || !key.startsWith(AI_TELEMETRY_METADATA_PREFIX)) continue;
    const metadataKey = key.slice(AI_TELEMETRY_METADATA_PREFIX.length);
    if (!metadataKey || N8N_AGENT_METADATA_KEYS.some((known) => known === metadataKey)) continue;
    telemetryMetadata[metadataKey] = isSensitiveMetadataKey(metadataKey)
      ? REDACTED_METADATA_VALUE
      : value;
  }
  if (Object.keys(telemetryMetadata).length > 0) metadata.telemetry = telemetryMetadata;

  const model = attrs[ATTR_GEN_AI_REQUEST_MODEL] ?? attrs.model_id;
  if (model !== undefined) metadata.model = model;
  const toolCallId = attrs[ATTR_GEN_AI_TOOL_CALL_ID] ?? attrs["ai.toolCall.id"];
  if (toolCallId !== undefined) metadata.tool_call_id = toolCallId;

  const memory: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && key.startsWith(N8N_MEMORY_ATTRIBUTE_PREFIX)) {
      memory[key.slice(N8N_MEMORY_ATTRIBUTE_PREFIX.length)] = value;
    }
  }
  const memoryOperation = stringAttr(attrs, ATTR_GEN_AI_OPERATION_NAME)?.toLowerCase();
  if (memoryOperation && N8N_MEMORY_OPERATIONS.has(memoryOperation)) {
    memory.operation = memoryOperation;
    const agentName = stringAttr(attrs, ATTR_GEN_AI_AGENT_NAME);
    if (agentName) memory.agent_name = agentName;
  }
  if (Object.keys(memory).length > 0) metadata.memory = memory;

  const aiSdk: Record<string, unknown> = {};
  const aiOperation = attrs[AI_OPERATION_ID];
  const functionId = attrs[AI_FUNCTION_ID];
  if (aiOperation !== undefined) aiSdk.operation = aiOperation;
  if (functionId !== undefined) aiSdk.function_id = functionId;
  for (const [rawKey, metadataKey] of [
    ["ai.response.msToFirstChunk", "time_to_first_output_ms"],
    ["ai.response.msToFinish", "response_time_ms"],
    ["ai.response.avgOutputTokensPerSecond", "output_tokens_per_second"],
  ] as const) {
    const value = attrs[rawKey];
    if (value !== undefined) aiSdk[metadataKey] = value;
  }
  if (Object.keys(aiSdk).length > 0) metadata.ai_sdk = aiSdk;

  return metadata;
}

function mergeN8nMetadata(attrs: MutableAttributes, n8nMetadata: Record<string, unknown>): void {
  if (Object.keys(n8nMetadata).length === 0) return;

  const current = parseMetadata(attrs[RespanSpanAttributes.RESPAN_METADATA]);
  const currentN8n = isRecord(current.n8n) ? current.n8n : {};
  attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
    ...current,
    n8n: mergeRecords(currentN8n, n8nMetadata),
  });
}

function mergeRecords(
  current: Readonly<Record<string, unknown>>,
  incoming: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  const merged: Record<string, unknown> = { ...current };
  for (const [key, value] of Object.entries(incoming)) {
    merged[key] = isRecord(merged[key]) && isRecord(value)
      ? mergeRecords(merged[key], value)
      : value;
  }
  return merged;
}

function parseMetadata(value: unknown): Record<string, unknown> {
  if (isRecord(value)) return { ...value };
  if (typeof value !== "string" || value.trim() === "") return {};

  try {
    const parsed = JSON.parse(value) as unknown;
    if (isRecord(parsed)) return { ...parsed };
    return { value: parsed };
  } catch {
    return { value };
  }
}

function stripVendorAttributes(kind: N8nSpanKind, attrs: MutableAttributes): void {
  for (const key of Object.keys(attrs)) {
    if (key.startsWith(N8N_ATTRIBUTE_PREFIX)) {
      delete attrs[key];
      continue;
    }

    if (
      (kind === "agent" || kind === "tool" || kind === "llm" || kind === "memory") &&
      (key.startsWith(AI_TELEMETRY_METADATA_PREFIX) || key.startsWith("ai."))
    ) {
      delete attrs[key];
    }
  }

  if (kind === "agent" || kind === "tool" || kind === "llm" || kind === "memory") {
    for (const key of N8N_AGENT_METADATA_KEYS) delete attrs[key];
    delete attrs["operation.name"];
    delete attrs["resource.name"];
  }

  if (kind === "agent" || kind === "tool" || kind === "memory") {
    for (const key of Object.keys(attrs)) {
      if (key.startsWith("gen_ai.")) delete attrs[key];
    }
  } else if (kind === "llm") {
    delete attrs[ATTR_GEN_AI_OPERATION_NAME];
    delete attrs[ATTR_GEN_AI_PROMPT];
    delete attrs[N8N_RAW_GEN_AI_RESPONSE_STATUS_CODE];
    for (const key of Object.keys(attrs)) {
      if (key.startsWith("gen_ai.tool.")) delete attrs[key];
    }
  }
}

function stripOffContractAliases(attrs: MutableAttributes): void {
  for (const key of OFF_CONTRACT_ALIAS_KEYS) delete attrs[key];
}

function cloneReadableSpanForExport(
  span: ReadableSpan,
  attributes: MutableAttributes,
  events: ReadableSpan["events"],
): ReadableSpan {
  const clone = Object.create(Object.getPrototypeOf(span));
  Object.assign(clone, span);
  Object.defineProperty(clone, "attributes", {
    value: attributes,
    enumerable: true,
    configurable: true,
  });
  if (events !== span.events) {
    Object.defineProperty(clone, "events", {
      value: events,
      enumerable: true,
      configurable: true,
    });
  }
  return clone as ReadableSpan;
}

function stripAiSdkEvents(events: ReadableSpan["events"]): ReadableSpan["events"] {
  let changed = false;
  const filtered = events.flatMap((event) => {
    if (event.name.startsWith("ai.")) {
      changed = true;
      return [];
    }

    const attributes = event.attributes;
    if (!attributes) return [event];
    const cleanAttributes = Object.fromEntries(
      Object.entries(attributes).filter(([key]) => !key.startsWith("ai.")),
    );
    if (Object.keys(cleanAttributes).length === Object.keys(attributes).length) return [event];
    changed = true;
    return [{ ...event, attributes: cleanAttributes }];
  });
  return changed ? filtered : events;
}

function setDefault(attrs: MutableAttributes, key: string, value: unknown): void {
  if (attrs[key] === undefined && value !== undefined) attrs[key] = value;
}

function firstStringAttr(
  attrs: Readonly<MutableAttributes>,
  keys: readonly string[],
): string | undefined {
  for (const key of keys) {
    const value = stringAttr(attrs, key);
    if (value) return value;
  }
  return undefined;
}

function firstRecordString(
  value: Readonly<Record<string, unknown>>,
  keys: readonly string[],
): string | undefined {
  return firstStringAttr(value, keys);
}

function stringAttr(attrs: Readonly<MutableAttributes>, key: string): string | undefined {
  const value = attrs[key];
  if (value === undefined || value === null) return undefined;
  const text = String(value).trim();
  return text || undefined;
}

function jsonAttribute(value: unknown): string {
  if (typeof value === "string") {
    try {
      JSON.parse(value);
      return value;
    } catch {
      return safeJson(value);
    }
  }
  return safeJson(value);
}

function parseJsonValue(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return JSON.stringify(String(value));
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSensitiveMetadataKey(key: string): boolean {
  const normalized = key.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  return (
    normalized.includes("api_key") ||
    normalized.includes("apikey") ||
    normalized.includes("private_key") ||
    normalized.includes("access_key") ||
    /(^|_)(authorization|auth|token|secret|password|credential|cookie)($|_)/.test(
      normalized,
    )
  );
}

function isN8nAgentScopedSpan(span: N8nSpanLike): boolean {
  const attrs = span.attributes;
  return (
    instrumentationScopeName(span) === N8N_AGENT_SCOPE ||
    N8N_AGENT_METADATA_KEYS.some(
      (key) =>
        attrs[key] !== undefined ||
        attrs[`${AI_TELEMETRY_METADATA_PREFIX}${key}`] !== undefined,
    )
  );
}
