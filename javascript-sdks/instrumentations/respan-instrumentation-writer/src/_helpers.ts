import { snapshot } from "./_privacy.js";
export const INSTRUMENTATION_LIBRARY_NAME = "@respan/instrumentation-writer";
export const PACKAGE_VERSION = "0.1.0";
export const WRITER_CHAT_ENTITY_NAME = "writer.chat";
export const WRITER_COMPLETION_ENTITY_NAME = "writer.completion";
export type SpanAttributesRecord = Record<string, any>;
export function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
export function safeJsonString(value: unknown): string {
  return JSON.stringify(snapshot(value)) ?? "null";
}
export function stringifyContent(value: unknown): string {
  return typeof value === "string" ? value : safeJsonString(value);
}
export function toNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}
export function setIfPresent(
  attrs: SpanAttributesRecord,
  key: string,
  value: unknown,
): void {
  if (
    value !== undefined &&
    (typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean")
  )
    attrs[key] = value;
}
export function normalizeMessages(messages: unknown): Record<string, any>[] {
  return Array.isArray(messages) ? snapshot(messages) : [];
}
export function formatTools(tools: unknown): any[] | undefined {
  return Array.isArray(tools) ? snapshot(tools) : undefined;
}
export function formatChatOutput(response: any): any[] {
  return Array.isArray(response?.choices)
    ? response.choices
        .map((choice: any) => snapshot(choice.message))
        .filter((message: any) => message !== undefined)
    : [];
}
export function formatTextOutput(response: any): string {
  return Array.isArray(response?.choices)
    ? response.choices
        .map((choice: any) => stringifyContent(choice.text))
        .join("\n")
    : "";
}
export function extractUsage(response: any): {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
} {
  const usage = response?.usage;
  return {
    promptTokens: toNumber(usage?.prompt_tokens),
    completionTokens: toNumber(usage?.completion_tokens),
    totalTokens: toNumber(usage?.total_tokens),
    cacheReadInputTokens: toNumber(usage?.prompt_token_details?.cached_tokens),
  };
}
export function extractErrorMessage(error: unknown): string | undefined {
  const copy = snapshot(error);
  return typeof copy?.message === "string" ? copy.message : undefined;
}

export interface ChatStreamState {
  id?: string;
  model?: string;
  created?: number;
  content: string;
  toolCalls: Map<number, Record<string, any>>;
  choices: Map<number, any>;
  usage?: Record<string, any>;
}
export function createChatStreamState(body: any = {}): ChatStreamState {
  return {
    ...(typeof body.model === "string" ? { model: body.model } : {}),
    content: "",
    toolCalls: new Map(),
    choices: new Map(),
  };
}
export function updateChatStreamState(
  state: ChatStreamState,
  chunk: any,
): void {
  if (!isRecord(chunk)) return;
  for (const key of ["id", "model", "created"] as const)
    if (chunk[key] !== undefined) (state as any)[key] = chunk[key];
  if (isRecord(chunk.usage)) state.usage = chunk.usage;
  if (!Array.isArray(chunk.choices)) return;
  for (const choice of chunk.choices) {
    if (!isRecord(choice) || typeof choice.index !== "number") continue;
    let stored = state.choices.get(choice.index);
    if (!stored) {
      stored = { index: choice.index, message: {}, calls: new Map() };
      state.choices.set(choice.index, stored);
    }
    if (choice.finish_reason !== undefined)
      stored.finish_reason = choice.finish_reason;
    const delta = choice.delta;
    if (!isRecord(delta)) continue;
    if (typeof delta.content === "string") state.content += delta.content;
    for (const key of Object.keys(delta)) {
      if (key === "tool_calls") continue;
      if (
        typeof delta[key] === "string" &&
        ["content", "refusal"].includes(key)
      )
        stored.message[key] = (stored.message[key] ?? "") + delta[key];
      else stored.message[key] = delta[key];
    }
    if (Array.isArray(delta.tool_calls))
      for (const call of delta.tool_calls) {
        if (!isRecord(call) || typeof call.index !== "number") continue;
        const current = stored.calls.get(call.index) ?? {};
        for (const key of ["id", "type"])
          if (call[key] !== undefined) current[key] = call[key];
        if (isRecord(call.function)) {
          current.function ??= {};
          for (const [key, value] of Object.entries(call.function))
            current.function[key] =
              typeof value === "string"
                ? (current.function[key] ?? "") + value
                : value;
        }
        stored.calls.set(call.index, current);
        state.toolCalls.set(call.index, current);
      }
  }
}
export function buildChatCompletionFromStreamState(
  state: ChatStreamState,
  _body: any = {},
): any {
  const result: any = {
    choices: [...state.choices.values()]
      .sort((a, b) => a.index - b.index)
      .map(({ calls, ...choice }) => {
        if (calls.size)
          choice.message.tool_calls = [...calls.entries()]
            .sort(([a], [b]) => a - b)
            .map(([, call]) => call);
        return choice;
      }),
  };
  for (const key of ["id", "model", "created", "usage"] as const)
    if (state[key] !== undefined) result[key] = state[key];
  return result;
}
export interface TextStreamState {
  model?: string;
  text: string;
  seen?: boolean;
  usage?: any;
}
export function createTextStreamState(body: any = {}): TextStreamState {
  return {
    ...(typeof body.model === "string" ? { model: body.model } : {}),
    text: "",
  };
}
export function updateTextStreamState(
  state: TextStreamState,
  chunk: any,
): void {
  if (typeof chunk?.value === "string") {
    state.text += chunk.value;
    state.seen = true;
  }
  if (typeof chunk?.model === "string") state.model = chunk.model;
  if (isRecord(chunk?.usage)) state.usage = chunk.usage;
}
export function buildCompletionFromStreamState(
  state: TextStreamState,
  _body: any = {},
): any {
  return {
    ...(state.model !== undefined ? { model: state.model } : {}),
    choices:
      state.seen === false || (state.seen === undefined && state.text === "")
        ? []
        : [{ text: state.text }],
    ...(state.usage !== undefined ? { usage: state.usage } : {}),
  };
}
