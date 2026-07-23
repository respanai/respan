import type { SpanContext } from "@opentelemetry/api";
import { context, trace } from "@opentelemetry/api";

export type SpanAttributes = Record<string, any>;

export function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function safeJson(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function safeJsonParse(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export function setIfPresent(
  attrs: SpanAttributes,
  key: string,
  value: unknown,
): void {
  if (value !== undefined && value !== null) {
    attrs[key] = value;
  }
}

export function setDefault(
  attrs: SpanAttributes,
  key: string,
  value: unknown,
): void {
  if (attrs[key] === undefined && value !== undefined && value !== null) {
    attrs[key] = value;
  }
}

export function activeSpanContext(): SpanContext | undefined {
  return trace.getSpan(context.active())?.spanContext();
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export function normalizeRole(role: unknown): string {
  const value = String(role ?? "user").toLowerCase();
  if (value === "chatbot" || value === "assistant") return "assistant";
  if (value === "system") return "system";
  if (value === "tool") return "tool";
  return "user";
}

export function cohereContentToString(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const textParts = value
      .filter(isRecord)
      .filter((item) => item.type === "text" && typeof item.text === "string")
      .map((item) => item.text);
    if (textParts.length === value.length && textParts.length > 0) {
      return textParts.join("");
    }
  }
  return safeJson(value);
}
