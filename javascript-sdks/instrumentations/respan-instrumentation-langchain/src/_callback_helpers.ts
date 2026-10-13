import { types } from "node:util";
import type { Span } from "@opentelemetry/api";
import { data, snapshot, type CapturePolicy } from "./_privacy.js";

export const LANGCHAIN_RUN_ID_ATTR = "langchain.run_id";
export const LANGCHAIN_PARENT_RUN_ID_ATTR = "langchain.parent_run_id";
export const LANGCHAIN_FRAMEWORK_ATTR = "langchain.framework";
export interface LangChainCallbackConfig {
  callbacks?: unknown;
}
export interface RespanCallbackHandlerOptions {
  includeContent?: boolean;
  includeMetadata?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
  groupLangflowRootRuns?: boolean;
  maxCachedRuns?: number;
}
export type SpanAttributesRecord = Record<string, any>;
export interface RunRecord {
  runId: string;
  span: Span;
  policy: CapturePolicy;
  name: string;
  path: string;
  workflowName: string;
  logType: string;
  model?: string;
  tokens: string[];
  indexed: SpanAttributesRecord;
}
export function isPlainRecord(
  value: unknown,
): value is Record<string, unknown> {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    !types.isProxy(value)
  );
}
export const toSerializableValue = snapshot;
export function safeJsonString(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  try {
    return JSON.stringify(snapshot(value));
  } catch {
    return undefined;
  }
}
export function setIfPresent(
  attrs: SpanAttributesRecord,
  key: string,
  value: unknown,
): void {
  if (value !== undefined) attrs[key] = value;
}
export function extractName(
  serialized: unknown,
  fallback: string,
  explicit?: unknown,
): string {
  if (typeof explicit === "string" && explicit) return explicit;
  const name = data(serialized, "name");
  if (typeof name === "string" && name) return name;
  const ids = data(serialized, "id");
  const last = Array.isArray(ids) ? data(ids, ids.length - 1) : undefined;
  return typeof last === "string" && last ? last : fallback;
}
export function detectFramework({
  tags,
  metadata,
}: {
  tags?: unknown;
  metadata?: unknown;
  serialized?: unknown;
  name?: string;
}): "langchain" | "langgraph" | "langflow" {
  if (
    data(metadata, "framework") === "langflow" ||
    typeof data(metadata, "langflow_component") === "string"
  )
    return "langflow";
  if (
    data(metadata, "framework") === "langgraph" ||
    data(metadata, "langgraph_node") !== undefined
  )
    return "langgraph";
  if (Array.isArray(tags) && !types.isProxy(tags)) {
    for (let i = 0; i < tags.length; i++) {
      const tag = data(tags, i);
      if (tag === "langflow") return "langflow";
      if (tag === "langgraph") return "langgraph";
    }
  }
  return "langchain";
}
export function extractModel(
  serialized?: unknown,
  response?: unknown,
  metadata?: unknown,
  extra?: unknown,
): string | undefined {
  const invocation =
    data(extra, "invocation_params") ?? data(extra, "invocationParams");
  const kwargs = data(serialized, "kwargs");
  const output = data(response, "llmOutput") ?? data(response, "llm_output");
  for (const [obj, keys] of [
    [metadata, ["ls_model_name"]],
    [invocation, ["model", "model_name", "modelName"]],
    [kwargs, ["model", "modelName", "model_name"]],
    [output, ["model_name", "modelName", "model"]],
  ] as const) {
    for (const key of keys) {
      const value = data(obj, key);
      if (typeof value === "string" && value) return value;
    }
  }
  return undefined;
}
function messageToDict(message: unknown): Record<string, any> {
  let role = data(message, "role") ?? data(message, "type");
  if (
    typeof role !== "string" &&
    message &&
    typeof message === "object" &&
    !types.isProxy(message)
  ) {
    const ctor = data(Object.getPrototypeOf(message), "constructor");
    const name = data(ctor, "name");
    role = (
      {
        HumanMessage: "user",
        HumanMessageChunk: "user",
        AIMessage: "assistant",
        AIMessageChunk: "assistant",
        SystemMessage: "system",
        SystemMessageChunk: "system",
        ToolMessage: "tool",
        ToolMessageChunk: "tool",
        FunctionMessage: "tool",
      } as Record<string, string>
    )[name];
  }
  role =
    role === "human"
      ? "user"
      : role === "ai"
        ? "assistant"
        : role === "function"
          ? "tool"
          : role;
  const result: Record<string, any> = {};
  if (typeof role === "string") result.role = role;
  const content = data(message, "content");
  if (content !== undefined) result.content = snapshot(content);
  for (const key of ["id", "name", "tool_call_id", "tool_calls"]) {
    const value = data(message, key);
    if (value !== undefined) result[key] = snapshot(value);
  }
  const additional = data(message, "additional_kwargs");
  for (const key of ["tool_calls", "function_call"])
    if (result[key] === undefined && data(additional, key) !== undefined)
      result[key] = snapshot(data(additional, key));
  return result;
}
function arrayData(value: unknown): unknown[] {
  if (!Array.isArray(value) || types.isProxy(value)) return [];
  const items: unknown[] = [];
  for (let index = 0; index < data(value, "length"); index++)
    items.push(data(value, index));
  return items;
}
export function normalizeChatMessages(
  messages: unknown,
): Record<string, any>[][] {
  return arrayData(messages).map((conversation) =>
    (Array.isArray(conversation)
      ? arrayData(conversation)
      : [conversation]
    ).map(messageToDict),
  );
}
export function normalizeOutputForLogging(value: unknown): unknown {
  return snapshot(value);
}
export function extractLlmOutput(response: unknown): {
  outputPayload: unknown;
  completionMessages: Record<string, any>[];
} {
  const generations = data(response, "generations");
  if (!Array.isArray(generations) || types.isProxy(generations))
    return { outputPayload: snapshot(response), completionMessages: [] };
  const batches = arrayData(generations).map((batch) =>
    (Array.isArray(batch) ? arrayData(batch) : [batch]).map((generation) => {
      const message = data(generation, "message");
      return message !== undefined
        ? messageToDict(message)
        : { role: "assistant", content: snapshot(data(generation, "text")) };
    }),
  );
  return { outputPayload: batches, completionMessages: batches.flat() };
}
export function extractToolCallsFromMessages(
  messages: Record<string, any>[],
): Record<string, any>[] | undefined {
  const calls: Record<string, any>[] = [];
  for (const message of messages) {
    const raw = data(message, "tool_calls");
    if (!Array.isArray(raw)) continue;
    for (const call of raw) {
      if (!isPlainRecord(call)) continue;
      if (data(call, "function")) {
        calls.push(snapshot(call));
        continue;
      }
      const name = data(call, "name");
      if (typeof name !== "string") continue;
      calls.push({
        id: data(call, "id"),
        type: "function",
        function: {
          name,
          arguments: safeJsonString(
            data(call, "args") ?? data(call, "arguments"),
          ),
        },
      });
    }
  }
  return calls.length ? calls : undefined;
}
function integer(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0
    ? value
    : undefined;
}
export function extractUsage(response: unknown): {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
} {
  const llm = data(response, "llmOutput") ?? data(response, "llm_output");
  const generations = data(response, "generations");
  const firstBatch = data(generations, 0);
  const message = data(
    data(Array.isArray(firstBatch) ? firstBatch : [firstBatch], 0),
    "message",
  );
  const meta = data(message, "response_metadata");
  const candidates = [
    data(llm, "tokenUsage"),
    data(llm, "token_usage"),
    data(llm, "usage"),
    data(llm, "usage_metadata"),
    data(response, "usage"),
    data(response, "usage_metadata"),
    data(message, "usage_metadata"),
    data(meta, "tokenUsage"),
    data(meta, "token_usage"),
    data(meta, "usage"),
  ];
  const out: Record<string, number> = {};
  for (const candidate of candidates) {
    for (const [name, keys] of [
      ["promptTokens", ["prompt_tokens", "promptTokens", "input_tokens"]],
      [
        "completionTokens",
        ["completion_tokens", "completionTokens", "output_tokens"],
      ],
      ["totalTokens", ["total_tokens", "totalTokens"]],
    ] as const) {
      if (out[name] !== undefined) continue;
      for (const key of keys) {
        const value = integer(data(candidate, key));
        if (value !== undefined) {
          out[name] = value;
          break;
        }
      }
    }
    const details =
      data(candidate, "input_token_details") ??
      data(candidate, "prompt_tokens_details");
    const cached = integer(
      data(details, "cache_read") ?? data(details, "cached_tokens"),
    );
    if (cached !== undefined) out.cacheReadInputTokens = cached;
  }
  return out;
}
export function extractTools({
  serialized,
  extraParams,
}: {
  serialized?: unknown;
  extraParams?: unknown;
}): Record<string, any>[] | undefined {
  const candidates: unknown[] = [];
  const seen = new WeakSet<object>();
  function collect(value: unknown) {
    if (!isPlainRecord(value) || seen.has(value)) return;
    seen.add(value);
    const tools = data(value, "tools") ?? data(value, "functions");
    if (Array.isArray(tools) && !types.isProxy(tools))
      candidates.push(...arrayData(tools));
    for (const key of [
      "invocation_params",
      "invocationParams",
      "kwargs",
      "options",
      "config",
      "model_kwargs",
      "modelKwargs",
    ])
      collect(data(value, key));
  }
  collect(extraParams);
  collect(serialized);
  const tools = candidates
    .map((tool) => {
      const fn = data(tool, "function");
      if (typeof data(fn, "name") === "string") return snapshot(tool);
      const name = data(tool, "name");
      if (typeof name !== "string") return undefined;
      const functionDefinition: Record<string, any> = { name };
      const description = data(tool, "description");
      if (description !== undefined)
        functionDefinition.description = snapshot(description);
      const parameters = data(tool, "parameters") ?? data(tool, "schema");
      if (parameters !== undefined)
        functionDefinition.parameters = snapshot(parameters);
      return { type: "function", function: functionDefinition };
    })
    .filter(Boolean);
  const unique = [
    ...new Map(tools.map((tool) => [JSON.stringify(tool), tool])).values(),
  ];
  return unique.length ? unique : undefined;
}
export function extractToolNamesFromSerialized(
  serialized: unknown,
): string[] | undefined {
  const names = extractTools({ serialized })
    ?.map((tool) => tool.function?.name)
    .filter((name): name is string => typeof name === "string");
  return names?.length ? names : undefined;
}
export function normalizeTags(tags: unknown): string[] | undefined {
  return Array.isArray(tags)
    ? snapshot(tags)?.filter((tag: unknown) => typeof tag === "string")
    : undefined;
}
export function normalizeMetadata(
  metadata: unknown,
): Record<string, any> | undefined {
  return isPlainRecord(metadata) ? snapshot(metadata) : undefined;
}
export function getErrorMessage(error: unknown): string | undefined {
  const message = data(error, "message");
  return typeof message === "string"
    ? message
    : typeof error === "string"
      ? error
      : undefined;
}
export function trimMap<K, V>(map: Map<K, V>, maxSize: number): void {
  while (map.size > maxSize) {
    const first = map.keys().next();
    if (first.done) return;
    map.delete(first.value);
  }
}
