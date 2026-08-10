/**
 * Emit OpenAI Agents SDK spans as OTEL ReadableSpan objects.
 *
 * Each per-type emitter converts an OpenAI Agents SDK Trace/Span into a
 * ReadableSpan with traceloop.* and gen_ai.* attributes, then injects it
 * into the OTEL pipeline via the active TracerProvider's span processor.
 */

import { trace, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { hrTime, hrTimeDuration } from "@opentelemetry/core";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  LLMRequestTypeValues,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import type { Span, Trace } from "@openai/agents";

const PACKAGE_VERSION = "1.0.6";
const GEN_AI_USAGE_INPUT_TOKENS = ATTR_GEN_AI_USAGE_INPUT_TOKENS;
const GEN_AI_USAGE_OUTPUT_TOKENS = ATTR_GEN_AI_USAGE_OUTPUT_TOKENS;
const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";
const GEN_AI_COMPLETION_PREFIX = `${SpanAttributes.LLM_COMPLETIONS}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;

function safeJson(obj: any): string {
  try {
    return JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value,
    );
  } catch {
    return String(obj);
  }
}

function toSerializableValue(value: any): any {
  if (value === null || value === undefined) return undefined;
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
  try {
    return JSON.parse(
      JSON.stringify(value, (_key, innerValue) =>
        typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
      ),
    );
  } catch {
    // Fall back to recursive structural cloning below.
  }
  if (Array.isArray(value)) {
    return value.map((item) => toSerializableValue(item));
  }
  if (typeof value === "object") {
    if (typeof value.toJSON === "function") {
      try {
        return toSerializableValue(value.toJSON());
      } catch {
        // Ignore and continue to shallow recursive copy.
      }
    }
    const normalized: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, itemValue]) => {
      normalized[key] = toSerializableValue(itemValue);
    });
    return normalized;
  }
  return String(value);
}

function stringifyStructured(value: any): string {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  return safeJson(serialized);
}

function contentBlocksToText(contentBlocks: any): string {
  const serialized = toSerializableValue(contentBlocks);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  if (!Array.isArray(serialized)) {
    return stringifyStructured(serialized);
  }

  const textParts: string[] = [];
  for (const block of serialized) {
    if (typeof block === "string") {
      textParts.push(block);
      continue;
    }
    if (!block || typeof block !== "object" || Array.isArray(block)) {
      textParts.push(String(block ?? ""));
      continue;
    }

    const blockType = (block as any).type ?? "";
    if (blockType === "input_image" || blockType === "image") {
      textParts.push("[image]");
      continue;
    }
    if (blockType === "input_file" || blockType === "file") {
      textParts.push("[file]");
      continue;
    }
    if (blockType === "audio" || blockType === "input_audio") {
      textParts.push("[audio]");
      continue;
    }
    if (blockType === "refusal" && typeof (block as any).refusal === "string") {
      textParts.push((block as any).refusal);
      continue;
    }
    if (typeof (block as any).text === "string") {
      textParts.push((block as any).text);
      continue;
    }
    textParts.push(stringifyStructured(block));
  }

  return textParts.join("\n");
}

function normalizeMessageContent(content: any): string {
  if (content === undefined || content === null) {
    return "";
  }
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return contentBlocksToText(content);
  }
  return stringifyStructured(content);
}

function stringifyToolResult(value: any): string {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  if (Array.isArray(serialized)) {
    return contentBlocksToText(serialized);
  }
  if (typeof serialized === "object") {
    const blockType = (serialized as any).type ?? "";
    if (
      (blockType === "text" ||
        blockType === "output_text" ||
        blockType === "input_text") &&
      typeof (serialized as any).text === "string"
    ) {
      return (serialized as any).text;
    }
    if (
      (serialized as any).output !== undefined ||
      (serialized as any).result !== undefined
    ) {
      return stringifyToolResult(
        (serialized as any).output ?? (serialized as any).result,
      );
    }
  }
  return stringifyStructured(serialized);
}

function normalizeToolSpanOutput(value: any): any {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  if (Array.isArray(serialized)) {
    const text = contentBlocksToText(serialized);
    return text || serialized;
  }
  if (typeof serialized === "object") {
    const blockType = (serialized as any).type ?? "";
    if (
      (blockType === "text" ||
        blockType === "output_text" ||
        blockType === "input_text") &&
      typeof (serialized as any).text === "string"
    ) {
      return (serialized as any).text;
    }
    if (blockType === "image") return "[image]";
    if (blockType === "file") return "[file]";
    if (
      (serialized as any).output !== undefined ||
      (serialized as any).result !== undefined
    ) {
      return normalizeToolSpanOutput(
        (serialized as any).output ?? (serialized as any).result,
      );
    }
  }
  return serialized;
}

function setJsonStructuredAttr(
  attrs: Record<string, any>,
  key: string,
  value: any,
): void {
  if (value === undefined || value === null) return;
  if (value === "") return;
  if (Array.isArray(value) && value.length === 0) return;
  attrs[key] = safeJson(value);
}

function normalizeToolCall(rawToolCall: any): Record<string, any> | null {
  const toolCall = toSerializableValue(rawToolCall);
  if (!toolCall || typeof toolCall !== "object" || Array.isArray(toolCall)) {
    return null;
  }

  const toolCallType = String((toolCall as any).type ?? "");
  const functionName =
    (toolCall as any).name ??
    (toolCall as any).function?.name ??
    (toolCall as any).functionName ??
    (toolCall as any).toolName ??
    (toolCallType.endsWith("_call") ? toolCallType : "") ??
    "";
  const functionArguments =
    (toolCall as any).arguments ??
    (toolCall as any).function?.arguments ??
    (toolCall as any).action ??
    (toolCall as any).actions ??
    (toolCall as any).operation ??
    "";

  if (
    !functionName &&
    !(toolCall as any).function &&
    toolCallType !== "function_call" &&
    !toolCallType.endsWith("_call")
  ) {
    return null;
  }

  const normalized: Record<string, any> = {
    id:
      (toolCall as any).call_id ??
      (toolCall as any).callId ??
      (toolCall as any).tool_call_id ??
      (toolCall as any).id ??
      "",
    type: "function",
    function: {
      name: functionName,
      arguments: stringifyStructured(functionArguments),
    },
  };
  if ((toolCall as any).namespace !== undefined) {
    normalized.function.namespace = (toolCall as any).namespace;
  }
  if ((toolCall as any).status !== undefined) {
    normalized.status = (toolCall as any).status;
  }
  if (toolCallType && toolCallType !== "function_call" && toolCallType !== "function") {
    normalized.openai_agents_type = toolCallType;
  }
  return normalized;
}

function extractToolCalls(output: any): Record<string, any>[] {
  const serialized = toSerializableValue(output);
  if (serialized === undefined || serialized === null) return [];

  const items = Array.isArray(serialized) ? serialized : [serialized];
  const result: Record<string, any>[] = [];

  for (const item of items) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }

    if (Array.isArray((item as any).choices)) {
      for (const choice of (item as any).choices) {
        if ((choice as any)?.message?.tool_calls) {
          for (const toolCall of (choice as any).message.tool_calls) {
            const normalized = normalizeToolCall(toolCall);
            if (normalized) result.push(normalized);
          }
        }
      }
    }

    const itemType = (item as any).type ?? "";
    if (
      itemType === "function_call" ||
      itemType === "function" ||
      itemType === "hosted_tool_call" ||
      itemType === "tool_search_call" ||
      itemType === "computer_call" ||
      itemType === "shell_call" ||
      itemType === "apply_patch_call"
    ) {
      const toolCall = normalizeToolCall(item);
      if (toolCall) result.push(toolCall);
      continue;
    }

    if (Array.isArray((item as any).tool_calls)) {
      for (const toolCall of (item as any).tool_calls) {
        const normalized = normalizeToolCall(toolCall);
        if (normalized) result.push(normalized);
      }
    }
  }

  return result;
}

function extractTools(tools: any): Record<string, any>[] {
  const serialized = toSerializableValue(tools);
  if (!Array.isArray(serialized)) return [];

  const result: Record<string, any>[] = [];
  for (const tool of serialized) {
    if (!tool || typeof tool !== "object" || Array.isArray(tool)) {
      continue;
    }

    const toolType = (tool as any).type ?? "";
    if (toolType === "namespace" && Array.isArray((tool as any).tools)) {
      result.push({
        ...tool,
        tools: extractTools((tool as any).tools),
      } as Record<string, any>);
      continue;
    }

    if (toolType === "function") {
      const func: Record<string, any> = {
        name: (tool as any).name ?? (tool as any).function?.name ?? "",
      };
      const description =
        (tool as any).description ??
        (tool as any).function?.description;
      if (description) func.description = description;

      const parameters =
        (tool as any).parameters ??
        (tool as any).function?.parameters;
      if (parameters !== undefined) func.parameters = parameters;

      result.push({ type: "function", function: func });
      continue;
    }

    result.push(tool as Record<string, any>);
  }

  return result;
}

function normalizeChatMessage(rawMessage: any): Record<string, any> {
  const message = toSerializableValue(rawMessage);
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return { role: "user", content: stringifyStructured(rawMessage) };
  }

  const normalized: Record<string, any> = {
    role: (message as any).role ?? "user",
    content: normalizeMessageContent((message as any).content),
  };

  const rawToolCalls = (message as any).tool_calls ?? (message as any).toolCalls;
  if (Array.isArray(rawToolCalls)) {
    const toolCalls = extractToolCalls(rawToolCalls);
    if (toolCalls.length) {
      normalized.tool_calls = toolCalls;
      if (!normalized.content) normalized.content = "";
    }
  }

  const toolCallId =
    (message as any).tool_call_id ??
    (message as any).toolCallId ??
    (message as any).call_id ??
    (message as any).callId;
  if (toolCallId) {
    normalized.tool_call_id = toolCallId;
  }

  return normalized;
}

function responsesApiItemToMessage(rawItem: any): Record<string, any> | null {
  const item = toSerializableValue(rawItem);
  if (!item || typeof item !== "object" || Array.isArray(item)) {
    return null;
  }

  const itemType = (item as any).type ?? "";
  if (itemType === "message") {
    return {
      role: (item as any).role ?? "user",
      content: normalizeMessageContent((item as any).content),
    };
  }

  if (
    itemType === "function_call" ||
    itemType === "hosted_tool_call" ||
    itemType === "tool_search_call" ||
    itemType === "computer_call" ||
    itemType === "shell_call" ||
    itemType === "apply_patch_call"
  ) {
    const toolCall = normalizeToolCall(item);
    if (!toolCall) return null;
    return {
      role: "assistant",
      content: "",
      tool_calls: [toolCall],
    };
  }

  if (
    itemType === "function_call_output" ||
    itemType === "function_call_result" ||
    itemType === "tool_search_output" ||
    itemType === "computer_call_result" ||
    itemType === "shell_call_output" ||
    itemType === "apply_patch_call_output"
  ) {
    return {
      role: "tool",
      content: stringifyToolResult(
        (item as any).output ?? (item as any).result ?? (item as any).tools ?? "",
      ),
      tool_call_id:
        (item as any).call_id ??
        (item as any).callId ??
        (item as any).tool_call_id ??
        "",
    };
  }

  if ((item as any).role) {
    return normalizeChatMessage(item);
  }

  if (itemType === "reasoning") {
    return {
      role: "assistant",
      content: normalizeMessageContent((item as any).content ?? item),
    };
  }

  return null;
}

function parseISOToHrTime(iso: string | undefined): [number, number] | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime();
  if (!Number.isFinite(ms)) return null;

  const secs = Math.floor(ms / 1000);
  const nanos = (ms % 1000) * 1_000_000;
  return [secs, nanos];
}

function numberFromUsage(value: any): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function usageCandidateFrom(value: any): Record<string, any> | null {
  const serialized = toSerializableValue(value);
  if (!serialized || typeof serialized !== "object") return null;
  if (Array.isArray(serialized)) {
    for (const item of serialized) {
      const usage = usageCandidateFrom(item);
      if (usage) return usage;
    }
    return null;
  }

  const direct = serialized as Record<string, any>;
  if (direct.usage && typeof direct.usage === "object") {
    return direct.usage as Record<string, any>;
  }

  if (
    direct.input_tokens !== undefined ||
    direct.output_tokens !== undefined ||
    direct.prompt_tokens !== undefined ||
    direct.completion_tokens !== undefined ||
    direct.inputTokens !== undefined ||
    direct.outputTokens !== undefined ||
    direct.promptTokens !== undefined ||
    direct.completionTokens !== undefined
  ) {
    return direct;
  }

  return null;
}

function setUsageAttrs(attrs: Record<string, any>, ...sources: any[]): void {
  const usage = sources.map(usageCandidateFrom).find(Boolean);
  if (!usage) return;

  const inputTokens = numberFromUsage(
    usage.input_tokens ??
      usage.prompt_tokens ??
      usage.inputTokens ??
      usage.promptTokens,
  );
  const outputTokens = numberFromUsage(
    usage.output_tokens ??
      usage.completion_tokens ??
      usage.outputTokens ??
      usage.completionTokens,
  );
  const totalTokens = numberFromUsage(
    usage.total_tokens ??
      usage.totalTokens ??
      (inputTokens !== undefined && outputTokens !== undefined
        ? inputTokens + outputTokens
        : undefined),
  );
  const cacheReadInputTokens = numberFromUsage(
    usage.cache_read_input_tokens ??
      usage.cacheReadInputTokens ??
      usage.input_tokens_details?.cached_tokens ??
      usage.prompt_tokens_details?.cached_tokens ??
      usage.details?.cache_read_input_tokens ??
      usage.details?.cached_tokens,
  );

  if (inputTokens !== undefined) {
    attrs[GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
    attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = inputTokens;
  }
  if (outputTokens !== undefined) {
    attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = outputTokens;
    attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = outputTokens;
  }
  if (totalTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
  if (cacheReadInputTokens !== undefined) {
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cacheReadInputTokens;
  }
}

function setPromptAttrs(
  attrs: Record<string, any>,
  messages: Record<string, any>[],
): void {
  for (const [index, message] of messages.entries()) {
    const prefix = `${SpanAttributes.LLM_PROMPTS}.${index}`;
    attrs[`${prefix}.role`] = String(message.role ?? "user");
    attrs[`${prefix}.content`] = normalizeMessageContent(message.content);
    if (message.tool_calls !== undefined) {
      attrs[`${prefix}.tool_calls`] = safeJson(message.tool_calls);
    }
  }
}

function setCompletionAttrs(
  attrs: Record<string, any>,
  content: string,
  toolCalls: Record<string, any>[],
): void {
  attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
  attrs[GEN_AI_COMPLETION_CONTENT] = content;
  if (toolCalls.length) {
    attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(toolCalls);
  }
}

function firstModelFromOutput(output: any): string | undefined {
  const serialized = toSerializableValue(output);
  const items = Array.isArray(serialized) ? serialized : [serialized];
  for (const item of items) {
    if (item && typeof item === "object" && !Array.isArray(item)) {
      const model = (item as any).model ?? (item as any)._response?.model;
      if (typeof model === "string" && model) return model;
    }
  }
  return undefined;
}

function formatChatCompletionOutput(rawResponse: any): string {
  const response = toSerializableValue(rawResponse);
  if (!response || typeof response !== "object" || Array.isArray(response)) {
    return "";
  }

  const choices = (response as any).choices;
  if (!Array.isArray(choices)) return "";

  const textParts: string[] = [];
  for (const choice of choices) {
    const message = (choice as any)?.message ?? (choice as any)?.delta;
    if (!message || typeof message !== "object") continue;
    const content = normalizeMessageContent((message as any).content);
    if (content) textParts.push(content);
    if (typeof (message as any).refusal === "string" && (message as any).refusal) {
      textParts.push((message as any).refusal);
    }
  }
  return textParts.join("\n");
}

function hashStringToHexId(s: string, length: number): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  }
  const hex = Math.abs(hash).toString(16).padStart(8, "0");
  return (hex + hex + hex + hex).slice(0, length);
}

function ensureTraceId(id: string): string {
  if (/^[0-9a-f]{32}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 32);
}

function ensureSpanId(id: string): string {
  if (/^[0-9a-f]{16}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 16);
}

function resolveSpanTimes(item: Span<any>): {
  startTimeHr: [number, number] | null;
  endTimeHr: [number, number] | null;
} {
  const serialized = toSerializableValue(item) as Record<string, any> | undefined;
  return {
    startTimeHr: parseISOToHrTime(
      serialized?.started_at ?? serialized?.startedAt ?? (item as any).startedAt,
    ),
    endTimeHr: parseISOToHrTime(
      serialized?.ended_at ?? serialized?.endedAt ?? (item as any).endedAt,
    ),
  };
}

function baseAttrs(
  entityName: string,
  entityPath: string,
  logType: string,
): Record<string, any> {
  return {
    // Leave traceloop.span.kind unset for injected OpenAI Agents spans.
    // In the JS tracing pipeline, that attribute is treated as a root-span
    // marker and would flatten the parent/child tree.
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
  };
}

interface BuildSpanOptions {
  name: string;
  traceId: string;
  spanId: string;
  parentId?: string;
  startTimeHr?: [number, number] | null;
  endTimeHr?: [number, number] | null;
  attributes: Record<string, any>;
  statusCode?: number;
  errorMessage?: string;
}

function buildReadableSpan(opts: BuildSpanOptions): ReadableSpan {
  const startTime = opts.startTimeHr ?? hrTime();
  const endTime = opts.endTimeHr ?? startTime;

  const traceId = ensureTraceId(opts.traceId);
  const spanId = ensureSpanId(opts.spanId);
  const parentSpanId = opts.parentId ? ensureSpanId(opts.parentId) : undefined;

  const status =
    opts.statusCode && opts.statusCode >= 400
      ? { code: SpanStatusCode.ERROR, message: opts.errorMessage ?? "" }
      : { code: SpanStatusCode.OK, message: "" };

  return {
    name: opts.name,
    kind: SpanKind.INTERNAL,
    spanContext: () => ({
      traceId,
      spanId,
      traceFlags: 1,
      isRemote: false,
    }),
    // OTEL 2.x: parent is a SpanContext (parentSpanId was removed in 2.0)
    parentSpanContext: parentSpanId
      ? { traceId, spanId: parentSpanId, traceFlags: 1, isRemote: false }
      : undefined,
    startTime,
    endTime,
    duration: hrTimeDuration(startTime, endTime),
    status,
    attributes: opts.attributes,
    links: [],
    events: [],
    resource: { attributes: {} } as any,
    instrumentationScope: {
      name: "@respan/instrumentation-openai-agents",
      version: PACKAGE_VERSION,
    },
    ended: true,
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } satisfies ReadableSpan;
}

function injectSpan(span: ReadableSpan): void {
  const tp = trace.getTracerProvider() as any;
  const processor =
    tp?.activeSpanProcessor ??
      tp?._activeSpanProcessor ??
      tp?._delegate?.activeSpanProcessor ??
      tp?._delegate?._activeSpanProcessor ??
      tp?._delegate?._tracerProvider?.activeSpanProcessor ??
      tp?._delegate?._tracerProvider?._activeSpanProcessor;
  if (processor && typeof processor.onEnd === "function") {
    processor.onEnd(span);
  }
}

function formatInputMessages(input: any): any[] | null {
  const serialized = toSerializableValue(input);
  if (serialized === undefined || serialized === null) return null;

  if (Array.isArray(serialized)) {
    const hasResponsesApiItems = serialized.some(
      (item) =>
        item &&
        typeof item === "object" &&
        !Array.isArray(item) &&
        "type" in item,
    );

    if (hasResponsesApiItems) {
      const messages: Record<string, any>[] = [];
      for (const item of serialized) {
        if (!item || typeof item !== "object" || Array.isArray(item)) {
          continue;
        }
        if ("type" in item) {
          const message = responsesApiItemToMessage(item);
          if (message) messages.push(message);
        } else if ("role" in item) {
          messages.push(normalizeChatMessage(item));
        }
      }
      return messages.length ? messages : serialized;
    }

    if (
      serialized.length > 0 &&
      serialized.every(
        (item) =>
          item &&
          typeof item === "object" &&
          !Array.isArray(item) &&
          "role" in item,
      )
    ) {
      return serialized.map((item) => normalizeChatMessage(item));
    }

    return serialized.map((item) => ({
      role: "user",
      content: stringifyStructured(item),
    }));
  }

  if (typeof serialized === "string") {
    return [{ role: "user", content: serialized }];
  }

  if (typeof serialized === "object") {
    return [{ role: "user", content: safeJson(serialized) }];
  }

  return [{ role: "user", content: String(serialized) }];
}

function formatOutput(output: any): string {
  const serialized = toSerializableValue(output);
  if (serialized === undefined || serialized === null) return "";

  if (typeof serialized === "string") {
    return serialized;
  }

  if (typeof serialized === "object" && !Array.isArray(serialized)) {
    const chatCompletionText = formatChatCompletionOutput(serialized);
    if (chatCompletionText) {
      return chatCompletionText;
    }
    if ((serialized as any).content === undefined) {
      return safeJson(serialized);
    }
    return normalizeMessageContent((serialized as any).content);
  }

  if (Array.isArray(serialized)) {
    const textParts: string[] = [];
    for (const item of serialized) {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        textParts.push(stringifyStructured(item));
        continue;
      }

      const chatCompletionText = formatChatCompletionOutput(item);
      if (chatCompletionText) {
        textParts.push(chatCompletionText);
        continue;
      }

      const itemType = (item as any).type ?? "";
      if (
        itemType === "function_call" ||
        itemType === "hosted_tool_call" ||
        itemType === "tool_search_call" ||
        itemType === "computer_call" ||
        itemType === "shell_call" ||
        itemType === "apply_patch_call" ||
        itemType === "function_call_output" ||
        itemType === "function_call_result" ||
        itemType === "tool_search_output" ||
        itemType === "computer_call_result" ||
        itemType === "shell_call_output" ||
        itemType === "apply_patch_call_output" ||
        itemType === "reasoning"
      ) {
        continue;
      }
      if (
        itemType === "output_text" ||
        itemType === "text" ||
        itemType === "input_text"
      ) {
        textParts.push((item as any).text ?? "");
        continue;
      }
      if (itemType === "message") {
        textParts.push(normalizeMessageContent((item as any).content));
        continue;
      }
      if ("content" in item) {
        textParts.push(normalizeMessageContent((item as any).content));
        continue;
      }
      textParts.push(stringifyStructured(item));
    }
    return textParts.filter(Boolean).join("\n");
  }

  return stringifyStructured(serialized);
}

function isTrace(item: any): item is Trace {
  return "traceId" in item && "name" in item && !("spanId" in item);
}

function isSpan(item: any): item is Span<any> {
  return "spanId" in item && "spanData" in item;
}

function emitTrace(traceObj: Trace): void {
  const traceName = traceObj.name || "trace";
  const attrs = baseAttrs(traceName, "", RespanLogType.WORKFLOW);
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = traceName;
  const metadata: Record<string, any> = {};
  if (traceObj.groupId) {
    attrs[RespanSpanAttributes.RESPAN_TRACE_GROUP_ID] = traceObj.groupId;
    metadata.group_id = traceObj.groupId;
  }
  const traceMetadata = toSerializableValue(traceObj.metadata);
  if (
    traceMetadata &&
    typeof traceMetadata === "object" &&
    !Array.isArray(traceMetadata)
  ) {
    Object.assign(metadata, traceMetadata);
  }
  if (Object.keys(metadata).length > 0) {
    attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson(metadata);
  }

  const span = buildReadableSpan({
    name: `${traceName}.workflow`,
    traceId: traceObj.traceId,
    spanId: traceObj.traceId,
    attributes: attrs,
  });
  injectSpan(span);
}

function emitAgent(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = data.name || "agent";

  const attrs = baseAttrs(name, name, RespanLogType.AGENT);
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = name;
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = name;

  const span = buildReadableSpan({
    name: `${name}.agent`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitResponse(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);

  const attrs = baseAttrs("chat", "chat", RespanLogType.CHAT);
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT;
  attrs[SpanAttributes.LLM_SYSTEM] = "openai";

  const inputMsgs = formatInputMessages(data._input);
  if (inputMsgs) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);
    setPromptAttrs(attrs, inputMsgs);
  }

  const resp = toSerializableValue(data._response);
  if (resp && typeof resp === "object" && !Array.isArray(resp)) {
    if ((resp as any).model) {
      attrs[SpanAttributes.LLM_REQUEST_MODEL] = (resp as any).model;
    }

    if ("output" in resp) {
      const outputText = formatOutput((resp as any).output);
      const toolCalls = extractToolCalls((resp as any).output);
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = outputText;
      setCompletionAttrs(attrs, outputText, toolCalls);
    }

    if ("tools" in resp) {
      setJsonStructuredAttr(
        attrs,
        SpanAttributes.LLM_REQUEST_FUNCTIONS,
        extractTools((resp as any).tools),
      );
    }

    setUsageAttrs(attrs, (resp as any).usage);
  }

  const span = buildReadableSpan({
    name: "openai.chat",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitFunction(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = data.name || "function";

  const attrs = baseAttrs(name, name, RespanLogType.TOOL);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({
    name,
    arguments: toSerializableValue(data.input) ?? "",
  });
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
    normalizeToolSpanOutput(data.output),
  );
  if (data.mcp_data !== undefined) {
    attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
      mcp_data: toSerializableValue(data.mcp_data),
    });
  }

  const span = buildReadableSpan({
    name: `${name}.tool`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitGeneration(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);

  const attrs = baseAttrs("chat", "chat", RespanLogType.CHAT);
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT;
  attrs[SpanAttributes.LLM_SYSTEM] = "openai";

  const model = data.model ?? firstModelFromOutput(data.output);
  if (model) attrs[SpanAttributes.LLM_REQUEST_MODEL] = model;

  const inputMsgs = formatInputMessages(data.input);
  if (inputMsgs) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);
    setPromptAttrs(attrs, inputMsgs);
  }

  const outputText = formatOutput(data.output);
  const toolCalls = extractToolCalls(data.output);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = outputText;
  setCompletionAttrs(attrs, outputText, toolCalls);

  setUsageAttrs(attrs, data.usage, data.output);

  const span = buildReadableSpan({
    name: "openai.chat",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitHandoff(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);

  const fromAgent = data.from_agent || "";
  const toAgent = data.to_agent || "";

  const attrs = baseAttrs("handoff", "handoff", RespanLogType.TASK);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({ from_agent: fromAgent });
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({ to_agent: toAgent });
  attrs[RespanSpanAttributes.RESPAN_METADATA_FROM_AGENT] = fromAgent;
  attrs[RespanSpanAttributes.RESPAN_METADATA_TO_AGENT] = toAgent;

  const span = buildReadableSpan({
    name: "handoff.task",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitGuardrail(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = `guardrail:${data.name}`;

  const attrs = baseAttrs(name, name, RespanLogType.GUARDRAIL);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({ name: data.name });
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
    triggered: Boolean(data.triggered),
  });
  attrs[RespanSpanAttributes.RESPAN_METADATA_GUARDRAIL_NAME] = data.name;
  attrs[RespanSpanAttributes.RESPAN_METADATA_TRIGGERED] = String(data.triggered);

  const span = buildReadableSpan({
    name: `${name}.task`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitCustom(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = data.name || data.data?.name || "custom";

  const attrs = baseAttrs(name, name, RespanLogType.TASK);
  const customData = data.data || {};
  const metadata: Record<string, any> = {};
  for (const [key, value] of Object.entries(customData)) {
    if (key === "model") {
      attrs[SpanAttributes.LLM_REQUEST_MODEL] = value;
    } else if (key === "prompt_tokens") {
      attrs[GEN_AI_USAGE_INPUT_TOKENS] = value;
      attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = value;
    } else if (key === "completion_tokens") {
      attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = value;
      attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = value;
    } else if (key === "name") {
      continue;
    } else if (key === "input") {
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(value);
    } else if (key === "output") {
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(value);
    } else {
      metadata[key] = toSerializableValue(value);
    }
  }
  if (Object.keys(metadata).length > 0) {
    attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson(metadata);
  }

  const span = buildReadableSpan({
    name: `${name}.task`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitMcpTools(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const server = data.server || "mcp";
  const name = `${server}.list_tools`;

  const attrs = baseAttrs(name, name, RespanLogType.TOOL);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({ server });
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
    toSerializableValue(data.result ?? []),
  );

  const span = buildReadableSpan({
    name: `${name}.tool`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

export function emitSdkItem(item: Trace | Span<any>): void {
  if (isTrace(item)) {
    emitTrace(item);
    return;
  }

  if (!isSpan(item)) return;

  const spanData = item.spanData as any;
  const type = spanData?.type;

  try {
    if (type === "response") emitResponse(item);
    else if (type === "function") emitFunction(item);
    else if (type === "generation") emitGeneration(item);
    else if (type === "agent") emitAgent(item);
    else if (type === "handoff") emitHandoff(item);
    else if (type === "custom") emitCustom(item);
    else if (type === "mcp_tools") emitMcpTools(item);
    else if (typeof spanData?.triggered === "boolean") emitGuardrail(item);
    else if (spanData?.name && spanData?.data) emitCustom(item);
    else {
      console.debug(`[Respan] Unknown OpenAI Agents span data type: ${type}`);
    }
  } catch (error) {
    console.error(`[Respan] Error emitting OpenAI Agents span:`, error);
  }
}
