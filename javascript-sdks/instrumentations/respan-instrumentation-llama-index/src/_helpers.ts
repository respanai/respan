import type { SpanContext } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { context, trace } from "@opentelemetry/api";
import { TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import {
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
} from "./_constants.js";

export type SpanAttributes = Record<string, any>;
export type HrTime = [number, number];

export function safeJson(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function normalizeText(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    const textParts = value
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) {
          return String((part as Record<string, unknown>).text ?? "");
        }
        return "";
      })
      .filter(Boolean);
    if (textParts.length > 0) {
      return textParts.join("\n");
    }
  }
  return safeJson(value);
}

export function normalizeModel(model: unknown): string | undefined {
  if (!model) return undefined;
  return String(model).toLowerCase();
}

export function inferGenAISystem(model: unknown): string | undefined {
  const normalized = normalizeModel(model);
  if (!normalized) return undefined;
  if (normalized.includes("gpt") || normalized.includes("o1") || normalized.includes("o3")) {
    return "openai";
  }
  if (normalized.includes("claude")) return "anthropic";
  if (normalized.includes("gemini")) return "google";
  if (normalized.includes("bedrock")) return "bedrock";
  return "llamaindex";
}

export function formatToolCall(toolCall: unknown): Record<string, unknown> {
  const value =
    toolCall && typeof toolCall === "object"
      ? (toolCall as Record<string, unknown>)
      : {};
  const id = value.id ?? value.call_id ?? value.toolCallId;
  const name = value.name ?? value.tool_name;
  const input = value.input ?? value.arguments ?? value.args ?? {};
  return {
    id: id ? String(id) : undefined,
    type: "function",
    function: {
      name: name ? String(name) : "unknown_tool",
      arguments: typeof input === "string" ? input : safeJson(input),
    },
  };
}

export function formatMessage(message: unknown): Record<string, unknown> {
  const value =
    message && typeof message === "object"
      ? (message as Record<string, any>)
      : { content: message };
  const role = String(value.role ?? "user");
  const formatted: Record<string, unknown> = {
    role,
    content: normalizeText(value.content),
  };

  const options =
    value.options && typeof value.options === "object"
      ? (value.options as Record<string, unknown>)
      : {};
  const toolCalls = options.toolCall;
  if (Array.isArray(toolCalls) && toolCalls.length > 0) {
    formatted.tool_calls = toolCalls.map(formatToolCall);
  }
  const toolResult = options.toolResult;
  if (toolResult && typeof toolResult === "object") {
    const result = toolResult as Record<string, unknown>;
    formatted.role = "tool";
    formatted.tool_call_id = result.id ? String(result.id) : undefined;
    formatted.name = result.name ? String(result.name) : undefined;
    formatted.content = normalizeText(result.result);
  }

  return formatted;
}

export function formatMessages(messages: unknown): Record<string, unknown>[] {
  if (!Array.isArray(messages)) {
    return [];
  }
  return messages.map(formatMessage);
}

export function extractResponseMessage(response: unknown): Record<string, unknown> {
  const value =
    response && typeof response === "object"
      ? (response as Record<string, any>)
      : {};
  return formatMessage(value.message ?? { role: "assistant", content: response });
}

export function extractResponseText(response: unknown): string {
  const message = extractResponseMessage(response);
  return String(message.content ?? "");
}

export function extractResponseModel(response: unknown): string | undefined {
  const value =
    response && typeof response === "object"
      ? (response as Record<string, any>)
      : {};
  const raw = value.raw;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    const rawRecord = raw as Record<string, any>;
    const model = rawRecord.model ?? rawRecord.response?.model;
    if (model) return String(model);
  }
  if (Array.isArray(raw)) {
    for (let index = raw.length - 1; index >= 0; index -= 1) {
      const item = raw[index];
      if (item && typeof item === "object" && "model" in item) {
        return String((item as Record<string, unknown>).model);
      }
    }
  }
  return undefined;
}

export function extractUsage(response: unknown): {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
} {
  const value =
    response && typeof response === "object"
      ? (response as Record<string, any>)
      : {};

  const candidates: Record<string, any>[] = [];
  if (value.raw && typeof value.raw === "object" && !Array.isArray(value.raw)) {
    candidates.push(value.raw.usage ?? value.raw.response?.usage ?? {});
  }
  if (Array.isArray(value.raw)) {
    for (const item of value.raw) {
      if (item && typeof item === "object") {
        candidates.push((item as Record<string, any>).usage ?? {});
      }
    }
  }
  const options = value.message?.options;
  if (options && typeof options === "object") {
    candidates.push((options as Record<string, any>).usage ?? {});
  }

  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    const usage = candidates[index];
    const inputTokens =
      usage.prompt_tokens ?? usage.input_tokens ?? usage.promptTokens ?? usage.inputTokens;
    const outputTokens =
      usage.completion_tokens ??
      usage.output_tokens ??
      usage.completionTokens ??
      usage.outputTokens;
    const totalTokens = usage.total_tokens ?? usage.totalTokens;
    if (
      inputTokens !== undefined ||
      outputTokens !== undefined ||
      totalTokens !== undefined
    ) {
      return {
        inputTokens: toNumber(inputTokens),
        outputTokens: toNumber(outputTokens),
        totalTokens: toNumber(totalTokens),
      };
    }
  }

  return {};
}

export function toNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function randomHex(length: number): string {
  return Array.from({ length }, () =>
    Math.floor(Math.random() * 16).toString(16)
  ).join("");
}

export function activeSpanContext(): SpanContext | undefined {
  return trace.getSpan(context.active())?.spanContext();
}

export function emitReadableSpan(opts: {
  name: string;
  traceId?: string;
  spanId?: string;
  parentId?: string;
  startTime: HrTime;
  endTime?: HrTime;
  attributes: SpanAttributes;
  errorMessage?: string;
}): void {
  const activeContext = activeSpanContext();
  const span = buildReadableSpan({
    name: opts.name,
    traceId: opts.traceId ?? activeContext?.traceId,
    spanId: opts.spanId,
    parentId: opts.parentId ?? activeContext?.spanId,
    startTimeHr: opts.startTime,
    endTimeHr: opts.endTime ?? hrTime(),
    attributes: opts.attributes,
    errorMessage: opts.errorMessage,
    mergePropagated: true,
  }) as ReadableSpan & {
    instrumentationScope?: { name: string; version?: string };
    spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
  };

  const originalSpanContext = span.spanContext.bind(span);
  span.spanContext = () => ({
    ...originalSpanContext(),
    traceFlags: activeContext?.traceFlags ?? TraceFlags.SAMPLED,
  });
  span.instrumentationScope = {
    name: INSTRUMENTATION_LIBRARY_NAME,
    version: PACKAGE_VERSION,
  };
  injectSpan(span);
}
