export const INSTRUMENTATION_LIBRARY_NAME = "@respan/instrumentation-writer";
export const PACKAGE_VERSION = "0.1.0";
export const WRITER_CHAT_ENTITY_NAME = "writer.chat";
export const WRITER_COMPLETION_ENTITY_NAME = "writer.completion";
export const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
export const STATUS_CODE_ATTR = "status_code";
export const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";

export type SpanAttributesRecord = Record<string, any>;

export function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function safeJsonString(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function stringifyContent(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    const parts = value.map((part) => {
      if (typeof part === "string") return part;
      if (isRecord(part) && part.type === "text") return String(part.text ?? "");
      return safeJsonString(part);
    });
    return parts.filter(Boolean).join("\n");
  }
  return safeJsonString(value);
}

export function toNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

export function setIfPresent(
  attrs: SpanAttributesRecord,
  key: string,
  value: unknown,
): void {
  if (value === undefined || value === null || value === "") return;
  attrs[key] = value;
}

export function normalizeMessage(message: unknown): Record<string, unknown> {
  const value = isRecord(message) ? message : { role: "user", content: message };
  const formatted: Record<string, unknown> = {
    role: String(value.role ?? "user"),
    content: stringifyContent(value.content),
  };

  if (value.name !== undefined) formatted.name = String(value.name);
  if (value.tool_call_id !== undefined && value.tool_call_id !== null) {
    formatted.tool_call_id = String(value.tool_call_id);
  }
  if (Array.isArray(value.tool_calls)) {
    formatted.tool_calls = value.tool_calls.map(formatToolCall);
  }

  const content = value.content;
  if (Array.isArray(content) && content.some((part) => isRecord(part) && part.type !== "text")) {
    formatted.content_parts = content;
  }

  return formatted;
}

export function normalizeMessages(messages: unknown): Record<string, unknown>[] {
  if (!Array.isArray(messages)) return [];
  return messages.map(normalizeMessage);
}

export function formatTool(tool: unknown): Record<string, unknown> {
  if (!isRecord(tool)) {
    return { type: "function", function: { name: "unknown_tool" } };
  }

  if (tool.type === "function") {
    const fn = isRecord(tool.function) ? tool.function : {};
    return {
      type: "function",
      function: {
        name: String(fn.name ?? "unknown_tool"),
        description: fn.description === undefined ? undefined : String(fn.description),
        parameters: fn.parameters,
      },
    };
  }

  return { ...tool };
}

export function formatTools(tools: unknown): Record<string, unknown>[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  return tools.map(formatTool);
}

export function formatToolCall(toolCall: unknown): Record<string, unknown> {
  const value = isRecord(toolCall) ? toolCall : {};
  const fn = isRecord(value.function) ? value.function : {};
  const args = fn.arguments ?? value.arguments ?? {};
  return {
    id: value.id === undefined || value.id === null ? undefined : String(value.id),
    type: String(value.type ?? "function"),
    function: {
      name: fn.name === undefined || fn.name === null ? "unknown_tool" : String(fn.name),
      arguments: typeof args === "string" ? args : safeJsonString(args),
    },
  };
}

export function extractCompletionToolCalls(response: unknown): Record<string, unknown>[] | undefined {
  if (!isRecord(response) || !Array.isArray(response.choices)) return undefined;
  const toolCalls: Record<string, unknown>[] = [];
  for (const choice of response.choices) {
    if (!isRecord(choice)) continue;
    const message = choice.message;
    if (!isRecord(message) || !Array.isArray(message.tool_calls)) continue;
    for (const toolCall of message.tool_calls) {
      toolCalls.push(formatToolCall(toolCall));
    }
  }
  return toolCalls.length > 0 ? toolCalls : undefined;
}

export function formatChatOutput(response: unknown): Record<string, unknown>[] {
  if (!isRecord(response) || !Array.isArray(response.choices)) {
    return [{ role: "assistant", content: stringifyContent(response) }];
  }

  return response.choices.map((choice) => {
    const message = isRecord(choice) && isRecord(choice.message) ? choice.message : {};
    const formatted: Record<string, unknown> = {
      role: String(message.role ?? "assistant"),
      content: stringifyContent(message.content),
    };
    if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      formatted.tool_calls = message.tool_calls.map(formatToolCall);
    }
    for (const key of ["graph_data", "llm_data", "translation_data", "web_search_data"]) {
      if (message[key] !== undefined && message[key] !== null) {
        formatted[key] = message[key];
      }
    }
    return formatted;
  });
}

export function formatTextOutput(response: unknown): string {
  if (!isRecord(response) || !Array.isArray(response.choices)) {
    return stringifyContent(response);
  }
  return response.choices
    .map((choice) => (isRecord(choice) ? stringifyContent(choice.text) : ""))
    .filter(Boolean)
    .join("\n");
}

export function extractUsage(response: unknown): {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
} {
  const usage = isRecord(response) && isRecord(response.usage) ? response.usage : {};
  const promptTokens = toNumber(usage.prompt_tokens ?? usage.input_tokens);
  const completionTokens = toNumber(usage.completion_tokens ?? usage.output_tokens);
  const totalTokens = toNumber(usage.total_tokens);
  const promptTokenDetails = isRecord(usage.prompt_token_details)
    ? usage.prompt_token_details
    : {};
  const cacheReadInputTokens = toNumber(promptTokenDetails.cached_tokens);
  return {
    promptTokens,
    completionTokens,
    totalTokens,
    cacheReadInputTokens,
  };
}

export function extractErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

export function extractStatusCode(err: unknown): number {
  if (!isRecord(err)) return 500;
  return toNumber(err.status ?? err.statusCode ?? err.code) ?? 500;
}

export interface ToolExecution {
  id?: string;
  name: string;
  arguments: unknown;
  output: unknown;
}

export function extractToolExecutions(messages: unknown): ToolExecution[] {
  if (!Array.isArray(messages)) return [];

  const toolCallsById = new Map<string, ReturnType<typeof formatToolCall>>();
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.tool_calls)) continue;
    for (const rawToolCall of message.tool_calls) {
      const formatted = formatToolCall(rawToolCall);
      if (formatted.id) {
        toolCallsById.set(String(formatted.id), formatted);
      }
    }
  }

  const executions: ToolExecution[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== "tool") continue;
    const id = message.tool_call_id === undefined || message.tool_call_id === null
      ? undefined
      : String(message.tool_call_id);
    const formattedCall = id ? toolCallsById.get(id) : undefined;
    const fn = isRecord(formattedCall?.function) ? formattedCall.function : {};
    executions.push({
      id,
      name: String(fn.name ?? message.name ?? "writer_tool"),
      arguments: fn.arguments ?? {},
      output: message.content,
    });
  }
  return executions;
}

export interface ChatStreamState {
  id?: string;
  model?: string;
  created?: number;
  content: string;
  toolCalls: Map<number, Record<string, any>>;
  usage?: Record<string, any>;
}

export function createChatStreamState(body: Record<string, any>): ChatStreamState {
  return {
    model: typeof body.model === "string" ? body.model : undefined,
    content: "",
    toolCalls: new Map(),
  };
}

export function updateChatStreamState(state: ChatStreamState, chunk: unknown): void {
  if (!isRecord(chunk)) return;
  if (typeof chunk.id === "string") state.id = chunk.id;
  if (typeof chunk.model === "string") state.model = chunk.model;
  if (typeof chunk.created === "number") state.created = chunk.created;
  if (isRecord(chunk.usage)) state.usage = chunk.usage;

  if (!Array.isArray(chunk.choices)) return;
  for (const choice of chunk.choices) {
    if (!isRecord(choice) || !isRecord(choice.delta)) continue;
    const delta = choice.delta;
    if (typeof delta.content === "string") {
      state.content += delta.content;
    }
    if (!Array.isArray(delta.tool_calls)) continue;
    for (const rawToolCall of delta.tool_calls) {
      if (!isRecord(rawToolCall)) continue;
      const index = toNumber(rawToolCall.index) ?? state.toolCalls.size;
      const existing = state.toolCalls.get(index) ?? {
        id: undefined,
        type: "function",
        function: { name: "", arguments: "" },
      };
      if (rawToolCall.id) existing.id = String(rawToolCall.id);
      if (rawToolCall.type) existing.type = String(rawToolCall.type);
      const rawFn = isRecord(rawToolCall.function) ? rawToolCall.function : {};
      existing.function ??= { name: "", arguments: "" };
      if (rawFn.name) existing.function.name = String(rawFn.name);
      if (rawFn.arguments) {
        existing.function.arguments = `${existing.function.arguments ?? ""}${rawFn.arguments}`;
      }
      state.toolCalls.set(index, existing);
    }
  }
}

export function buildChatCompletionFromStreamState(
  state: ChatStreamState,
  body: Record<string, any>,
): Record<string, any> {
  const toolCalls = Array.from(state.toolCalls.entries())
    .sort(([left], [right]) => left - right)
    .map(([, toolCall]) => formatToolCall(toolCall));
  return {
    id: state.id ?? "writer-stream",
    model: state.model ?? body.model,
    created: state.created ?? Math.floor(Date.now() / 1000),
    object: "chat.completion",
    choices: [
      {
        index: 0,
        finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop",
        message: {
          role: "assistant",
          content: state.content,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        },
      },
    ],
    usage: state.usage,
  };
}

export interface TextStreamState {
  model?: string;
  text: string;
}

export function createTextStreamState(body: Record<string, any>): TextStreamState {
  return {
    model: typeof body.model === "string" ? body.model : undefined,
    text: "",
  };
}

export function updateTextStreamState(state: TextStreamState, chunk: unknown): void {
  if (!isRecord(chunk)) return;
  if (typeof chunk.model === "string") state.model = chunk.model;
  if (typeof chunk.value === "string") state.text += chunk.value;
}

export function buildCompletionFromStreamState(
  state: TextStreamState,
  body: Record<string, any>,
): Record<string, any> {
  return {
    model: state.model ?? body.model,
    choices: [{ text: state.text }],
  };
}
