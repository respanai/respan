import { context, trace } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  RespanLogType,
  RespanSpanAttributes,
  ToolCallSchema,
} from "@respan/respan-sdk";
import {
  buildReadableSpan,
  ensureSpanId,
  ensureTraceId,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

const PACKAGE_VERSION = "1.0.0";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const CLAUDE_AGENT_INSTRUMENTATION_NAME = "@respan/instrumentation-claude-agent-sdk";
const GEN_AI_PROMPT_PREFIX = SpanAttributes.LLM_PROMPTS;
const GEN_AI_COMPLETION_PREFIX = SpanAttributes.LLM_COMPLETIONS;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.0.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.0.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.0.tool_calls`;
const GEN_AI_USAGE_INPUT_TOKENS = ATTR_GEN_AI_USAGE_INPUT_TOKENS;
const GEN_AI_USAGE_OUTPUT_TOKENS = ATTR_GEN_AI_USAGE_OUTPUT_TOKENS;
const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";
const GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS =
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS;

export interface PendingToolState {
  spanId: string;
  startTime: [number, number];
  toolName: string;
  toolInput: unknown;
}

export interface QueryState {
  agentName: string;
  agentSpanId: string;
  chatSpanId: string;
  completionTokens?: number;
  errorMessage?: string;
  finalOutput?: unknown;
  inputMessages: Record<string, unknown>[];
  model?: string;
  outputMessages: Record<string, unknown>[];
  parentSpanId?: string;
  pendingTools: Map<string, PendingToolState>;
  prompt: unknown;
  promptCacheCreationTokens?: number;
  promptCacheHitTokens?: number;
  promptTokens?: number;
  sessionId?: string;
  startTime: [number, number];
  statusCode: number;
  streamBlocks: Map<number, StreamingContentBlockState>;
  toolCalls: Record<string, unknown>[];
  toolDefinitions?: Record<string, unknown>[];
  totalCostUsd?: number;
  totalRequestTokens?: number;
  traceId: string;
}

interface StreamingContentBlockState {
  emittedToolCall?: boolean;
  id?: string;
  input?: unknown;
  inputJson: string;
  name?: string;
  text: string;
  thinking: string;
  type?: string;
}

export function createQueryState({
  prompt,
  options,
  agentName,
}: {
  prompt: unknown;
  options?: Record<string, unknown>;
  agentName?: string;
}): QueryState {
  const activeSpan = trace.getSpan(context.active());
  const activeSpanContext = activeSpan?.spanContext();

  return {
    agentName: agentName?.trim() || inferAgentName(options) || "claude-agent-sdk",
    agentSpanId: ensureSpanId(),
    chatSpanId: ensureSpanId(),
    inputMessages: normalizeInputMessages(prompt, options),
    outputMessages: [],
    parentSpanId: activeSpanContext?.spanId,
    pendingTools: new Map(),
    prompt,
    startTime: hrTime(),
    statusCode: 200,
    streamBlocks: new Map(),
    toolCalls: [],
    toolDefinitions: normalizeConfiguredToolDefinitions(options),
    traceId: ensureTraceId(activeSpanContext?.traceId),
  };
}

function inferAgentName(options?: Record<string, unknown>): string | undefined {
  const candidate =
    options?.agentName ??
    options?.name ??
    options?.agent_name;
  return typeof candidate === "string" && candidate.trim() ? candidate.trim() : undefined;
}

function normalizeInputMessages(
  prompt: unknown,
  options?: Record<string, unknown>,
): Record<string, unknown>[] {
  const messages: Record<string, unknown>[] = [];
  const systemInstructions =
    options?.system ??
    options?.systemPrompt ??
    options?.instructions;

  if (typeof systemInstructions === "string" && systemInstructions.trim()) {
    messages.push({
      role: "system",
      content: systemInstructions,
    });
  }

  if (typeof prompt === "string") {
    messages.push({
      role: "user",
      content: prompt,
    });
    return messages;
  }

  if (isAsyncIterable(prompt)) {
    return messages;
  }

  const serializedPrompt = toSerializableValue(prompt);
  if (Array.isArray(serializedPrompt)) {
    messages.push(...normalizeMessageArray(serializedPrompt));
    return messages;
  }

  if (serializedPrompt !== undefined) {
    messages.push({
      role: "user",
      content: serializedPrompt,
    });
  }

  return messages;
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return Boolean(
    value &&
      typeof value === "object" &&
      Symbol.asyncIterator in (value as Record<PropertyKey, unknown>),
  );
}

function normalizeMessageArray(
  messages: unknown[],
): Record<string, unknown>[] {
  const normalizedMessages: Record<string, unknown>[] = [];

  for (const message of messages) {
    if (!message || typeof message !== "object" || Array.isArray(message)) {
      normalizedMessages.push({
        role: "user",
        content: message,
      });
      continue;
    }

    const record = message as Record<string, unknown>;
    const nestedMessage =
      record.message && typeof record.message === "object" && !Array.isArray(record.message)
        ? (record.message as Record<string, unknown>)
        : record;
    normalizedMessages.push(...normalizeConversationMessage(nestedMessage));
  }

  return normalizedMessages;
}

function normalizeConfiguredToolDefinitions(
  options?: Record<string, unknown>,
): Record<string, unknown>[] | undefined {
  const normalizedTools = [
    ...normalizeToolDefinitions(options?.allowedTools),
    ...normalizeToolDefinitions(options?.allowed_tools),
    ...normalizeToolDefinitions(options?.tools),
    ...normalizeMcpServerToolDefinitions(options?.mcpServers),
    ...normalizeMcpServerToolDefinitions(options?.mcp_servers),
  ];

  if (normalizedTools.length === 0) {
    return undefined;
  }

  return dedupeToolDefinitions(normalizedTools);
}

function normalizeToolDefinitions(
  tools: unknown,
): Record<string, unknown>[] {
  if (!Array.isArray(tools)) {
    return [];
  }

  return tools
    .map((tool) => normalizeToolDefinition(tool))
    .filter((tool): tool is Record<string, unknown> => tool !== null);
}

function normalizeMcpServerToolDefinitions(
  mcpServers: unknown,
): Record<string, unknown>[] {
  if (!mcpServers || typeof mcpServers !== "object" || Array.isArray(mcpServers)) {
    return [];
  }

  const normalizedTools: Record<string, unknown>[] = [];
  for (const [serverAlias, serverConfig] of Object.entries(
    mcpServers as Record<string, unknown>,
  )) {
    if (
      !serverConfig ||
      typeof serverConfig !== "object" ||
      Array.isArray(serverConfig)
    ) {
      continue;
    }

    const serverRecord = serverConfig as Record<string, unknown>;
    const registeredTools =
      serverRecord.instance &&
      typeof serverRecord.instance === "object" &&
      !Array.isArray(serverRecord.instance)
        ? (serverRecord.instance as Record<string, unknown>)._registeredTools
        : undefined;

    if (
      !registeredTools ||
      typeof registeredTools !== "object" ||
      Array.isArray(registeredTools)
    ) {
      continue;
    }

    for (const [toolName, toolConfig] of Object.entries(
      registeredTools as Record<string, unknown>,
    )) {
      const normalizedTool = normalizeToolDefinition({
        ...(toolConfig && typeof toolConfig === "object" && !Array.isArray(toolConfig)
          ? (toolConfig as Record<string, unknown>)
          : {}),
        name: `mcp__${serverAlias}__${toolName}`,
      });

      if (normalizedTool) {
        normalizedTools.push(normalizedTool);
      }
    }
  }

  return normalizedTools;
}

function normalizeToolDefinition(
  tool: unknown,
): Record<string, unknown> | null {
  if (typeof tool === "string" && tool) {
    return {
      type: "function",
      function: { name: tool },
    };
  }

  if (!tool || typeof tool !== "object" || Array.isArray(tool)) {
    return null;
  }

  const record = tool as Record<string, unknown>;
  const functionPayload =
    record.function && typeof record.function === "object" && !Array.isArray(record.function)
      ? (record.function as Record<string, unknown>)
      : null;

  if (functionPayload) {
    const functionName = functionPayload.name;
    if (typeof functionName !== "string" || !functionName) {
      return null;
    }

    const normalizedFunction: Record<string, unknown> = { name: functionName };
    for (const key of ["description", "parameters", "strict"]) {
      if (functionPayload[key] !== undefined) {
        normalizedFunction[key] =
          key === "parameters"
            ? normalizeToolParameters(functionPayload[key])
            : toSerializableValue(functionPayload[key]);
      }
    }

    return {
      type: record.type ?? "function",
      function: normalizedFunction,
    };
  }

  const toolName = record.name;
  if (typeof toolName !== "string" || !toolName) {
    return null;
  }

  const normalizedFunction: Record<string, unknown> = { name: toolName };
  if (record.description !== undefined) {
    normalizedFunction.description = toSerializableValue(record.description);
  }
  const parameters = record.input_schema ?? record.inputSchema ?? record.parameters;
  if (parameters !== undefined) {
    normalizedFunction.parameters = normalizeToolParameters(parameters);
  }

  return {
    type: record.type ?? "function",
    function: normalizedFunction,
  };
}

function normalizeToolParameters(parameters: unknown): unknown {
  const jsonSchema =
    extractJsonSchema(parameters) ?? extractZodLikeJsonSchema(parameters);
  return toSerializableValue(jsonSchema ?? parameters);
}

function extractJsonSchema(parameters: unknown): unknown | undefined {
  if (!parameters || typeof parameters !== "object" || Array.isArray(parameters)) {
    return undefined;
  }

  const record = parameters as Record<string, unknown> & {
    toJSONSchema?: (...args: unknown[]) => unknown;
  };

  if (typeof record.toJSONSchema === "function") {
    try {
      const jsonSchema = record.toJSONSchema();
      if (jsonSchema !== undefined) {
        return jsonSchema;
      }
    } catch {
      // Fall back to serializing a stripped-down version of the original object.
    }
  }

  return undefined;
}

function extractZodLikeJsonSchema(parameters: unknown): unknown | undefined {
  if (!isZodLikeSchema(parameters)) {
    return undefined;
  }

  const schemaRecord = parameters as Record<string, unknown>;
  const schemaDef =
    schemaRecord.def && typeof schemaRecord.def === "object" && !Array.isArray(schemaRecord.def)
      ? (schemaRecord.def as Record<string, unknown>)
      : {};
  const schemaType =
    typeof schemaRecord.type === "string"
      ? schemaRecord.type
      : typeof schemaDef.type === "string"
        ? schemaDef.type
        : undefined;

  switch (schemaType) {
    case "object": {
      const shape =
        schemaDef.shape &&
        typeof schemaDef.shape === "object" &&
        !Array.isArray(schemaDef.shape)
          ? (schemaDef.shape as Record<string, unknown>)
          : {};
      const properties: Record<string, unknown> = {};
      const required: string[] = [];

      for (const [key, fieldSchema] of Object.entries(shape)) {
        const normalizedField =
          extractZodLikeJsonSchema(fieldSchema) ?? toSerializableValue(fieldSchema);
        if (normalizedField !== undefined) {
          properties[key] = normalizedField;
        }

        if (!isOptionalZodLikeSchema(fieldSchema)) {
          required.push(key);
        }
      }

      return {
        type: "object",
        properties,
        ...(required.length > 0 ? { required } : {}),
      };
    }
    case "optional":
      return extractZodLikeJsonSchema(schemaDef.innerType);
    case "string":
    case "number":
    case "boolean":
      return { type: schemaType };
    case "array": {
      const items =
        extractZodLikeJsonSchema(schemaDef.element) ??
        extractZodLikeJsonSchema(schemaDef.innerType);
      return {
        type: "array",
        ...(items !== undefined ? { items } : {}),
      };
    }
    default:
      return undefined;
  }
}

function isOptionalZodLikeSchema(schema: unknown): boolean {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    return false;
  }

  const record = schema as Record<string, unknown>;
  if (record.type === "optional") {
    return true;
  }

  return Boolean(
    record.def &&
    typeof record.def === "object" &&
    !Array.isArray(record.def) &&
    (record.def as Record<string, unknown>).type === "optional",
  );
}

function isZodLikeSchema(schema: unknown): boolean {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    return false;
  }

  const record = schema as Record<string, unknown>;
  const standard =
    record["~standard"] &&
    typeof record["~standard"] === "object" &&
    !Array.isArray(record["~standard"])
      ? (record["~standard"] as Record<string, unknown>)
      : undefined;
  if (standard?.vendor === "zod") {
    return true;
  }

  const def =
    record.def && typeof record.def === "object" && !Array.isArray(record.def)
      ? (record.def as Record<string, unknown>)
      : undefined;
  if (!def || typeof def.type !== "string") {
    return false;
  }

  return true;
}

export function trackClaudeMessage(state: QueryState, message: unknown): void {
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return;
  }

  const record = message as Record<string, unknown>;
  updateSessionId(state, record.session_id ?? record.sessionId);

  switch (record.type) {
    case "system":
      handleSystemMessage(state, record);
      break;
    case "user":
      handleUserMessage(state, record);
      break;
    case "assistant":
      handleAssistantMessage(state, record);
      break;
    case "stream_event":
      handleStreamEventMessage(state, record);
      break;
    case "result":
      handleResultMessage(state, record);
      break;
    case "error":
      handleErrorMessage(state, record);
      break;
    default:
      break;
  }
}

function handleSystemMessage(state: QueryState, message: Record<string, unknown>): void {
  const data =
    message.data && typeof message.data === "object" && !Array.isArray(message.data)
      ? (message.data as Record<string, unknown>)
      : undefined;
  updateSessionId(state, data?.session_id ?? data?.sessionId ?? data?.id);

  const model = message.model ?? data?.model;
  if (typeof model === "string" && model) {
    state.model = model;
  }

  const toolDefinitions = [
    ...normalizeToolDefinitions(message.tools),
    ...normalizeToolDefinitions(data?.tools),
  ];
  if (toolDefinitions.length > 0) {
    state.toolDefinitions = dedupeToolDefinitions([
      ...(state.toolDefinitions ?? []),
      ...toolDefinitions,
    ]);
  }
}

function handleUserMessage(state: QueryState, message: Record<string, unknown>): void {
  const payload =
    message.message && typeof message.message === "object" && !Array.isArray(message.message)
      ? (message.message as Record<string, unknown>)
      : message;
  const normalizedMessages = normalizeConversationMessage(payload);
  if (normalizedMessages.length > 0) {
    state.inputMessages.push(...normalizedMessages);
  }

  if (
    message.tool_use_result !== undefined &&
    !normalizedMessages.some((normalized) => normalized.role === "tool")
  ) {
    state.inputMessages.push({
      role: "tool",
      content: toSerializableValue(message.tool_use_result),
      ...(typeof message.parent_tool_use_id === "string"
        ? { tool_call_id: message.parent_tool_use_id }
        : {}),
    });
  }
}

function handleAssistantMessage(state: QueryState, message: Record<string, unknown>): void {
  const payload = resolveAssistantPayload(message);
  const model = payload.model ?? message.model;
  if (typeof model === "string" && model) {
    state.model = model;
  }
  const assistantError = message.error ?? payload.error;
  if (typeof assistantError === "string" && assistantError) {
    state.statusCode = 500;
    state.errorMessage = assistantError;
  }

  updateUsageFromMessage(
    state,
    payload.usage && typeof payload.usage === "object" && !Array.isArray(payload.usage)
      ? (payload.usage as Record<string, unknown>)
      : message.usage && typeof message.usage === "object" && !Array.isArray(message.usage)
        ? (message.usage as Record<string, unknown>)
      : undefined,
  );

  const rawContent = payload.content ?? message.content;
  const content = toSerializableValue(rawContent);
  if (hasMeaningfulContent(content)) {
    state.outputMessages.push({
      role: "assistant",
      content,
    });
  }

  if (Array.isArray(rawContent)) {
    for (const block of rawContent) {
      const toolCall = normalizeToolCall(block);
      if (toolCall) {
        addToolCall(state, toolCall);
      }
    }
  }
}

function handleStreamEventMessage(
  state: QueryState,
  message: Record<string, unknown>,
): void {
  const event =
    message.event && typeof message.event === "object" && !Array.isArray(message.event)
      ? (message.event as Record<string, unknown>)
      : undefined;
  if (!event) {
    return;
  }

  updateSessionId(state, message.session_id ?? message.sessionId);

  switch (event.type) {
    case "message_start":
      handleStreamMessageStart(state, event);
      break;
    case "content_block_start":
      handleStreamContentBlockStart(state, event);
      break;
    case "content_block_delta":
      handleStreamContentBlockDelta(state, event);
      break;
    case "content_block_stop":
      handleStreamContentBlockStop(state, event);
      break;
    case "message_delta":
      handleStreamMessageDelta(state, event);
      break;
    case "message_stop":
      flushStreamBlocks(state);
      break;
    default:
      break;
  }
}

function handleStreamMessageStart(
  state: QueryState,
  event: Record<string, unknown>,
): void {
  const message =
    event.message && typeof event.message === "object" && !Array.isArray(event.message)
      ? (event.message as Record<string, unknown>)
      : undefined;
  if (!message) {
    return;
  }

  const model = message.model;
  if (typeof model === "string" && model) {
    state.model = model;
  }

  updateUsageFromMessage(
    state,
    message.usage && typeof message.usage === "object" && !Array.isArray(message.usage)
      ? (message.usage as Record<string, unknown>)
      : undefined,
  );
}

function handleStreamContentBlockStart(
  state: QueryState,
  event: Record<string, unknown>,
): void {
  const index = coerceInteger(event.index) ?? state.streamBlocks.size;
  const contentBlock =
    event.content_block &&
    typeof event.content_block === "object" &&
    !Array.isArray(event.content_block)
      ? (event.content_block as Record<string, unknown>)
      : {};
  const block: StreamingContentBlockState = {
    id: stringValue(contentBlock.id),
    input: contentBlock.input,
    inputJson:
      contentBlock.input !== undefined && contentBlock.input !== null
        ? safeJson(contentBlock.input)
        : "",
    name: stringValue(contentBlock.name),
    text: stringValue(contentBlock.text) ?? "",
    thinking: stringValue(contentBlock.thinking) ?? "",
    type: stringValue(contentBlock.type),
  };
  state.streamBlocks.set(index, block);
}

function handleStreamContentBlockDelta(
  state: QueryState,
  event: Record<string, unknown>,
): void {
  const index = coerceInteger(event.index) ?? state.streamBlocks.size;
  const block: StreamingContentBlockState = state.streamBlocks.get(index) ?? {
    inputJson: "",
    text: "",
    thinking: "",
  };
  const delta =
    event.delta && typeof event.delta === "object" && !Array.isArray(event.delta)
      ? (event.delta as Record<string, unknown>)
      : {};

  switch (delta.type) {
    case "text_delta":
      block.type = block.type ?? "text";
      block.text += stringValue(delta.text) ?? "";
      break;
    case "thinking_delta":
      block.type = block.type ?? "thinking";
      block.thinking += stringValue(delta.thinking) ?? "";
      break;
    case "input_json_delta":
      block.type = block.type ?? "tool_use";
      block.inputJson += stringValue(delta.partial_json) ?? "";
      break;
    default:
      break;
  }

  state.streamBlocks.set(index, block);
}

function handleStreamContentBlockStop(
  state: QueryState,
  event: Record<string, unknown>,
): void {
  const index = coerceInteger(event.index);
  if (index === null) {
    return;
  }
  const block = state.streamBlocks.get(index);
  if (!block || block.type !== "tool_use" || block.emittedToolCall) {
    return;
  }
  const toolName = block.name;
  if (!toolName) {
    return;
  }
  addToolCall(
    state,
    createToolCall({
      id: block.id || ensureSpanId(),
      toolName,
      args: parseJsonObject(block.inputJson, block.input ?? {}),
    }),
  );
  block.emittedToolCall = true;
}

function handleStreamMessageDelta(
  state: QueryState,
  event: Record<string, unknown>,
): void {
  updateUsageFromMessage(
    state,
    event.usage && typeof event.usage === "object" && !Array.isArray(event.usage)
      ? (event.usage as Record<string, unknown>)
      : undefined,
  );
}

function flushStreamBlocks(state: QueryState): void {
  if (state.streamBlocks.size === 0) {
    return;
  }

  const content = [...state.streamBlocks.entries()]
    .sort(([leftIndex], [rightIndex]) => leftIndex - rightIndex)
    .map(([_index, block]) => streamBlockToContent(block))
    .filter((block): block is Record<string, unknown> => block !== null);

  if (content.length > 0) {
    state.outputMessages.push({
      role: "assistant",
      content,
    });
  }

  state.streamBlocks.clear();
}

function streamBlockToContent(
  block: StreamingContentBlockState,
): Record<string, unknown> | null {
  switch (block.type) {
    case "text":
      return block.text.trim()
        ? {
            type: "text",
            text: block.text,
          }
        : null;
    case "thinking":
      return block.thinking.trim()
        ? {
            type: "thinking",
            thinking: block.thinking,
          }
        : null;
    case "tool_use": {
      if (!block.name) {
        return null;
      }
      return {
        type: "tool_use",
        id: block.id,
        name: block.name,
        input: parseJsonObject(block.inputJson, block.input ?? {}),
      };
    }
    default:
      return null;
  }
}

function handleResultMessage(state: QueryState, message: Record<string, unknown>): void {
  updateUsageFromMessage(
    state,
    message.usage && typeof message.usage === "object" && !Array.isArray(message.usage)
      ? (message.usage as Record<string, unknown>)
      : undefined,
  );
  updateUsageFromModelUsage(state, message.modelUsage ?? message.model_usage);

  if (typeof message.total_cost_usd === "number") {
    state.totalCostUsd = message.total_cost_usd;
  }

  const apiErrorStatus = coerceInteger(message.api_error_status);
  if (apiErrorStatus !== null) {
    state.statusCode = apiErrorStatus;
  }

  if (message.is_error || String(message.subtype ?? "").startsWith("error")) {
    state.statusCode = state.statusCode >= 400 ? state.statusCode : 500;
    state.errorMessage =
      firstString(message.errors) ??
      stringValue(message.error) ??
      `agent_result_error:${String(message.subtype ?? "error")}`;
  }

  const outputValue =
    message.result ??
    message.structured_output;
  if (
    outputValue !== undefined &&
    !hasRenderableAssistantOutput(state.outputMessages)
  ) {
    state.finalOutput = toSerializableValue(outputValue);
    state.outputMessages.push({
      role: "assistant",
      content: toSerializableValue(outputValue),
    });
  } else if (outputValue !== undefined) {
    state.finalOutput = toSerializableValue(outputValue);
  }
}

function handleErrorMessage(state: QueryState, message: Record<string, unknown>): void {
  state.statusCode = 500;
  state.errorMessage =
    stringValue(message.error) ??
    stringValue(message.message) ??
    stringValue(message.reason) ??
    "claude_agent_sdk_error";
}

function resolveAssistantPayload(
  message: Record<string, unknown>,
): Record<string, unknown> {
  const nestedMessage = message.message;
  if (
    nestedMessage &&
    typeof nestedMessage === "object" &&
    !Array.isArray(nestedMessage)
  ) {
    return nestedMessage as Record<string, unknown>;
  }
  return message;
}

function normalizeConversationMessage(
  message: Record<string, unknown>,
): Record<string, unknown>[] {
  const role = stringValue(message.role) ?? "user";
  const content = message.content;

  if (Array.isArray(content)) {
    const normalizedMessages: Record<string, unknown>[] = [];
    const nonToolResultBlocks: unknown[] = [];

    for (const block of content) {
      if (isToolResultBlock(block)) {
        const blockRecord = block as Record<string, unknown>;
        normalizedMessages.push({
          role: "tool",
          content: toSerializableValue(blockRecord.content ?? blockRecord),
          ...(typeof blockRecord.tool_use_id === "string"
            ? { tool_call_id: blockRecord.tool_use_id }
            : {}),
        });
      } else {
        const normalizedBlock = toSerializableValue(block);
        if (normalizedBlock !== undefined) {
          nonToolResultBlocks.push(normalizedBlock);
        }
      }
    }

    if (nonToolResultBlocks.length > 0) {
      normalizedMessages.unshift({
        role,
        content:
          nonToolResultBlocks.length === 1
            ? nonToolResultBlocks[0]
            : nonToolResultBlocks,
      });
    }

    return normalizedMessages;
  }

  const normalizedContent = toSerializableValue(content);
  if (normalizedContent === undefined) {
    return [];
  }

  return [
    {
      role,
      content: normalizedContent,
    },
  ];
}

function isToolResultBlock(block: unknown): boolean {
  return Boolean(
    block &&
      typeof block === "object" &&
      !Array.isArray(block) &&
      (block as Record<string, unknown>).type === "tool_result",
  );
}

function hasMeaningfulContent(value: unknown): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

function hasRenderableAssistantOutput(
  messages: Record<string, unknown>[],
): boolean {
  return messages.some((message) => hasRenderableContent(message.content));
}

function hasRenderableContent(value: unknown): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.some((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return hasRenderableContent(item);
      }

      const record = item as Record<string, unknown>;
      if (record.type === "thinking" || record.type === "tool_use") {
        return false;
      }
      if (record.type === "text") {
        return typeof record.text === "string" && record.text.trim().length > 0;
      }
      if ("content" in record) {
        return hasRenderableContent(record.content);
      }
      return Object.keys(record).length > 0;
    });
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

function updateUsageFromMessage(
  state: QueryState,
  usage?: Record<string, unknown>,
): void {
  if (!usage) {
    return;
  }

  const promptTokenDetails =
    usage.prompt_tokens_details &&
    typeof usage.prompt_tokens_details === "object" &&
    !Array.isArray(usage.prompt_tokens_details)
      ? (usage.prompt_tokens_details as Record<string, unknown>)
      : usage.promptTokensDetails &&
          typeof usage.promptTokensDetails === "object" &&
          !Array.isArray(usage.promptTokensDetails)
        ? (usage.promptTokensDetails as Record<string, unknown>)
        : undefined;
  const rawPromptTokens = coerceInteger(
    usage.input_tokens ??
      usage.inputTokens ??
      usage.prompt_tokens ??
      usage.promptTokens,
  );
  const completionTokens = coerceInteger(
    usage.output_tokens ??
      usage.outputTokens ??
      usage.completion_tokens ??
      usage.completionTokens,
  );
  const cacheHitTokens = coerceInteger(
    usage.cache_read_input_tokens ??
      usage.cacheReadInputTokens ??
      promptTokenDetails?.cached_tokens ??
      promptTokenDetails?.cachedTokens ??
      promptTokenDetails?.cache_read_tokens ??
      promptTokenDetails?.cacheReadTokens,
  );
  const cacheCreationTokens = coerceInteger(
    usage.cache_creation_input_tokens ??
      usage.cacheCreationInputTokens ??
      promptTokenDetails?.cache_creation_tokens ??
      promptTokenDetails?.cacheCreationTokens,
  );
  const totalTokens = coerceInteger(
    usage.total_tokens ??
      usage.totalTokens,
  );
  const resolvedTotalTokens =
    totalTokens ??
    (rawPromptTokens !== null && completionTokens !== null
      ? rawPromptTokens + completionTokens
      : null);
  let promptTokens = rawPromptTokens;

  if (
    promptTokens !== null &&
    (cacheHitTokens !== null || cacheCreationTokens !== null)
  ) {
    const uncachedPromptTokens =
      promptTokens -
      ((cacheHitTokens ?? 0) + (cacheCreationTokens ?? 0));
    if (uncachedPromptTokens >= 0) {
      promptTokens = uncachedPromptTokens;
    }
  }

  if (promptTokens !== null) {
    state.promptTokens = promptTokens;
  }
  if (completionTokens !== null) {
    state.completionTokens = completionTokens;
  }
  if (cacheHitTokens !== null) {
    state.promptCacheHitTokens = cacheHitTokens;
  }
  if (cacheCreationTokens !== null) {
    state.promptCacheCreationTokens = cacheCreationTokens;
  }

  if (resolvedTotalTokens !== null) {
    state.totalRequestTokens = resolvedTotalTokens;
  }
}

function updateUsageFromModelUsage(
  state: QueryState,
  modelUsage: unknown,
): void {
  if (!modelUsage || typeof modelUsage !== "object" || Array.isArray(modelUsage)) {
    return;
  }

  const usageRecords = Object.entries(modelUsage as Record<string, unknown>)
    .filter((entry): entry is [string, Record<string, unknown>] => {
      const value = entry[1];
      return Boolean(value && typeof value === "object" && !Array.isArray(value));
    });
  if (usageRecords.length === 0) {
    return;
  }

  if (!state.model && usageRecords.length === 1) {
    state.model = usageRecords[0][0];
  }

  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadInputTokens = 0;
  let cacheCreationInputTokens = 0;
  let sawUsage = false;
  let sawCacheReadInputTokens = false;
  let sawCacheCreationInputTokens = false;

  for (const [_model, usage] of usageRecords) {
    const modelInputTokens = coerceInteger(usage.inputTokens ?? usage.input_tokens);
    const modelOutputTokens = coerceInteger(usage.outputTokens ?? usage.output_tokens);
    const modelCacheReadTokens = coerceInteger(
      usage.cacheReadInputTokens ?? usage.cache_read_input_tokens,
    );
    const modelCacheCreationTokens = coerceInteger(
      usage.cacheCreationInputTokens ?? usage.cache_creation_input_tokens,
    );

    if (modelInputTokens !== null) {
      inputTokens += modelInputTokens;
      sawUsage = true;
    }
    if (modelOutputTokens !== null) {
      outputTokens += modelOutputTokens;
      sawUsage = true;
    }
    if (modelCacheReadTokens !== null) {
      cacheReadInputTokens += modelCacheReadTokens;
      sawUsage = true;
      sawCacheReadInputTokens = true;
    }
    if (modelCacheCreationTokens !== null) {
      cacheCreationInputTokens += modelCacheCreationTokens;
      sawUsage = true;
      sawCacheCreationInputTokens = true;
    }
  }

  if (!sawUsage) {
    return;
  }

  updateUsageFromMessage(state, {
    input_tokens: inputTokens + cacheReadInputTokens + cacheCreationInputTokens,
    output_tokens: outputTokens,
    ...(sawCacheReadInputTokens
      ? { cache_read_input_tokens: cacheReadInputTokens }
      : {}),
    ...(sawCacheCreationInputTokens
      ? { cache_creation_input_tokens: cacheCreationInputTokens }
      : {}),
    total_tokens:
      inputTokens + cacheReadInputTokens + cacheCreationInputTokens + outputTokens,
  });
}

function resolveToolUseId(
  input: Record<string, unknown>,
  toolUseId?: string,
  pendingTools?: Map<string, PendingToolState>,
): string {
  const directToolUseId = input.tool_use_id ?? input.toolUseID ?? input.toolUseId ?? toolUseId;
  if (directToolUseId !== undefined && directToolUseId !== null) {
    const normalizedToolUseId = String(directToolUseId);
    if (normalizedToolUseId) {
      return normalizedToolUseId;
    }
  }

  if (pendingTools && pendingTools.size === 1) {
    const pendingTool = pendingTools.keys().next().value;
    if (pendingTool) {
      return pendingTool;
    }
  }

  return ensureSpanId();
}

export function registerPromptSubmit(
  state: QueryState,
  input: Record<string, unknown>,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
}

export function registerPendingTool(
  state: QueryState,
  input: Record<string, unknown>,
  toolUseId?: string,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
  const resolvedToolUseId = resolveToolUseId(input, toolUseId);
  const toolName = String(input.tool_name ?? input.toolName ?? "tool");
  const toolInput = input.tool_input ?? input.toolInput;

  state.pendingTools.set(resolvedToolUseId, {
    spanId: ensureSpanId(),
    startTime: hrTime(),
    toolInput,
    toolName,
  });

  addToolCall(
    state,
    createToolCall({
      id: resolvedToolUseId,
      toolName,
      args: toolInput ?? {},
    }),
  );
}

export function emitCompletedTool(
  state: QueryState,
  input: Record<string, unknown>,
  toolUseId?: string,
): void {
  updateSessionId(state, input.session_id ?? input.sessionId);
  const resolvedToolUseId = resolveToolUseId(
    input,
    toolUseId,
    state.pendingTools,
  );
  const pendingTool =
    state.pendingTools.get(resolvedToolUseId) ?? {
      spanId: ensureSpanId(),
      startTime: hrTime(),
      toolInput: input.tool_input ?? input.toolInput,
      toolName: String(input.tool_name ?? input.toolName ?? "tool"),
    };
  state.pendingTools.delete(resolvedToolUseId);

  const toolName = String(input.tool_name ?? input.toolName ?? pendingTool.toolName ?? "tool");
  const toolError =
    typeof input.error === "string" && input.error.trim().length > 0
      ? input.error
      : undefined;
  const toolOutput =
    toolError ??
    input.tool_response ??
    input.toolResponse ??
    input.tool_result ??
    input.toolResult ??
    input.output ??
    "";
  const attrs = baseAttrs(toolName, toolName, RespanLogType.TOOL);
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
    {
      name: toolName,
      arguments: pendingTool.toolInput ?? input.tool_input ?? input.toolInput ?? {},
    },
  );
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(toolOutput);

  if (state.sessionId) {
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.sessionId;
  }

  injectSpan(
    buildClaudeReadableSpan({
      name: `${toolName}.tool`,
      traceId: state.traceId,
      spanId: pendingTool.spanId,
      parentId: state.agentSpanId,
      startTimeHr: pendingTool.startTime,
      endTimeHr: hrTime(),
      attributes: attrs,
      statusCode: toolError ? 500 : undefined,
      errorMessage: toolError,
    }),
  );
}

export function emitAgentSpan(state: QueryState): void {
  flushStreamBlocks(state);
  const endTime = hrTime();
  const agentAttrs = buildAgentAttributes(state);
  const chatAttrs = buildChatAttributes(state);

  injectSpan(
    buildClaudeReadableSpan({
      name: `${state.agentName}.agent`,
      traceId: state.traceId,
      spanId: state.agentSpanId,
      parentId: state.parentSpanId,
      startTimeHr: state.startTime,
      endTimeHr: endTime,
      attributes: agentAttrs,
      statusCode: state.statusCode,
      errorMessage: state.errorMessage,
    }),
  );

  injectSpan(
    buildClaudeReadableSpan({
      name: `${state.agentName}.chat`,
      traceId: state.traceId,
      spanId: state.chatSpanId,
      parentId: state.agentSpanId,
      startTimeHr: state.startTime,
      endTimeHr: endTime,
      attributes: chatAttrs,
      statusCode: state.statusCode,
      errorMessage: state.errorMessage,
    }),
  );
}

function buildAgentAttributes(state: QueryState): Record<string, unknown> {
  const attrs = baseAttrs(state.agentName, state.agentName, RespanLogType.AGENT);
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = state.agentName;
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = state.agentName;

  if (state.inputMessages.length > 0) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
  }
  const formattedOutput = formatAgentOutput(state);
  if (formattedOutput) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = formattedOutput;
  }
  if (state.sessionId) {
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.sessionId;
  }

  return attrs;
}

function buildChatAttributes(state: QueryState): Record<string, unknown> {
  const attrs = baseAttrs(`${state.agentName}.chat`, `${state.agentName}.chat`, RespanLogType.CHAT);
  const dedupedToolCalls = dedupeToolCalls(state.toolCalls);
  const formattedOutput = formatAgentOutput(state);

  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[SpanAttributes.LLM_SYSTEM] = "anthropic";
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = state.agentName;

  if (state.inputMessages.length > 0) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(state.inputMessages);
    setPromptAttributes(attrs, state.inputMessages);
  }
  if (formattedOutput) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = formattedOutput;
  }
  attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
  attrs[GEN_AI_COMPLETION_CONTENT] = formattedOutput;
  if (state.model) {
    attrs[SpanAttributes.LLM_REQUEST_MODEL] = state.model;
  }
  addUsageAttributes(attrs, state);
  if (state.totalCostUsd !== undefined) {
    attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
      response_cost: state.totalCostUsd,
    });
  }
  if (state.sessionId) {
    attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = state.sessionId;
  }
  if (state.toolDefinitions && state.toolDefinitions.length > 0) {
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(state.toolDefinitions);
  }
  if (dedupedToolCalls.length > 0) {
    attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(dedupedToolCalls);
  }

  return attrs;
}

function addUsageAttributes(
  attrs: Record<string, unknown>,
  state: QueryState,
): void {
  if (state.promptTokens !== undefined) {
    attrs[GEN_AI_USAGE_INPUT_TOKENS] = state.promptTokens;
    attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = state.promptTokens;
  }
  if (state.completionTokens !== undefined) {
    attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = state.completionTokens;
    attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = state.completionTokens;
  }
  if (state.totalRequestTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = state.totalRequestTokens;
  }
  if (state.promptCacheHitTokens !== undefined) {
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = state.promptCacheHitTokens;
  }
  if (state.promptCacheCreationTokens !== undefined) {
    attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] =
      state.promptCacheCreationTokens;
  }
}

function setPromptAttributes(
  attrs: Record<string, unknown>,
  messages: Record<string, unknown>[],
): void {
  messages.forEach((message, index) => {
    const prefix = `${GEN_AI_PROMPT_PREFIX}.${index}`;
    const role = message.role;
    const content = message.content;
    const toolCalls = message.tool_calls ?? message.toolCalls;

    if (role !== undefined) {
      attrs[`${prefix}.role`] = String(role);
    }
    if (content !== undefined) {
      attrs[`${prefix}.content`] = stringifyPromptContent(content);
    }
    if (toolCalls !== undefined) {
      attrs[`${prefix}.tool_calls`] = safeJson(toolCalls);
    }
  });
}

function stringifyPromptContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  return safeJson(content);
}

function formatAgentOutput(state: QueryState): string {
  if (state.finalOutput !== undefined) {
    return stringifyOutputValue(state.finalOutput);
  }

  for (let index = state.outputMessages.length - 1; index >= 0; index -= 1) {
    const formatted = stringifyOutputValue(state.outputMessages[index]?.content);
    if (formatted) {
      return formatted;
    }
  }

  return "";
}

function stringifyOutputValue(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  if (Array.isArray(value)) {
    const parts = value
      .map((item) => stringifyOutputValue(item))
      .filter((part) => part.trim().length > 0);
    return parts.join("\n");
  }

  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (record.type === "thinking" || record.type === "tool_use") {
      return "";
    }
    if (record.type === "text" && typeof record.text === "string") {
      return record.text;
    }
    if ("content" in record) {
      return stringifyOutputValue(record.content);
    }
    return safeJson(record);
  }

  return String(value);
}

function createToolCall({
  id,
  toolName,
  args,
}: {
  id: string;
  toolName: string;
  args: unknown;
}): Record<string, unknown> {
  const parsedToolCall = ToolCallSchema.safeParse({
    type: "function",
    id,
    name: toolName,
    args: toSerializableValue(args),
  });
  if (parsedToolCall.success) {
    const normalizedToolCall = {
      ...(parsedToolCall.data as Record<string, unknown>),
    };
    delete normalizedToolCall.name;
    delete normalizedToolCall.args;
    return normalizedToolCall;
  }

  return {
    id,
    type: "function",
    function: {
      name: toolName,
      arguments: safeJson(args ?? {}),
    },
  };
}

function normalizeToolCall(block: unknown): Record<string, unknown> | null {
  if (!block || typeof block !== "object" || Array.isArray(block)) {
    return null;
  }

  const record = block as Record<string, unknown>;
  if (record.type !== "tool_use") {
    return null;
  }

  const toolName = record.name;
  if (typeof toolName !== "string" || !toolName) {
    return null;
  }

  return createToolCall({
    id: String(record.id ?? record.tool_use_id ?? record.toolUseId ?? ensureSpanId()),
    toolName,
    args: record.input ?? record.tool_input ?? record.toolInput ?? {},
  });
}

function addToolCall(state: QueryState, toolCall: Record<string, unknown>): void {
  state.toolCalls.push(toolCall);
}

function dedupeToolDefinitions(
  toolDefinitions: Record<string, unknown>[],
): Record<string, unknown>[] {
  const seen = new Set<string>();
  const deduped: Record<string, unknown>[] = [];

  for (const toolDefinition of toolDefinitions) {
    const functionPayload =
      toolDefinition.function &&
      typeof toolDefinition.function === "object" &&
      !Array.isArray(toolDefinition.function)
        ? (toolDefinition.function as Record<string, unknown>)
        : {};
    const name = functionPayload.name;
    if (typeof name !== "string" || !name) {
      deduped.push(toolDefinition);
      continue;
    }
    if (seen.has(name)) {
      continue;
    }
    seen.add(name);
    deduped.push(toolDefinition);
  }

  return deduped;
}

function dedupeToolCalls(
  toolCalls: Record<string, unknown>[],
): Record<string, unknown>[] {
  const seen = new Map<string, number>();
  const deduped: Record<string, unknown>[] = [];

  for (const toolCall of toolCalls) {
    const functionPayload =
      toolCall.function &&
      typeof toolCall.function === "object" &&
      !Array.isArray(toolCall.function)
        ? (toolCall.function as Record<string, unknown>)
        : {};
    const id = typeof toolCall.id === "string" ? toolCall.id : "";
    const name = typeof functionPayload.name === "string" ? functionPayload.name : "";
    const key =
      id || name
        ? safeJson([id, name])
        : safeJson([id, name, functionPayload.arguments ?? ""]);
    const existingIndex = seen.get(key);
    if (existingIndex !== undefined) {
      const existing = deduped[existingIndex];
      const existingArgs =
        existing?.function &&
        typeof existing.function === "object" &&
        !Array.isArray(existing.function)
          ? (existing.function as Record<string, unknown>).arguments
          : undefined;
      if (toolCallArgumentsScore(functionPayload.arguments) > toolCallArgumentsScore(existingArgs)) {
        deduped[existingIndex] = toolCall;
      }
      continue;
    }
    seen.set(key, deduped.length);
    deduped.push(toolCall);
  }

  return deduped;
}

function toolCallArgumentsScore(args: unknown): number {
  if (args === undefined || args === null) {
    return 0;
  }
  const text = typeof args === "string" ? args.trim() : safeJson(args).trim();
  return text && text !== "{}" && text !== "[]" ? text.length : 0;
}

function updateSessionId(state: QueryState, rawSessionId: unknown): void {
  if (rawSessionId === undefined || rawSessionId === null) {
    return;
  }
  const sessionId = String(rawSessionId);
  if (sessionId) {
    state.sessionId = sessionId;
  }
}

function baseAttrs(
  entityName: string,
  entityPath: string,
  logType: string,
): Record<string, unknown> {
  return {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
  };
}

function buildClaudeReadableSpan(
  options: Parameters<typeof buildReadableSpan>[0],
): ReadableSpan {
  const span = buildReadableSpan(options) as ReadableSpan & {
    instrumentationLibrary?: {
      name: string;
      version?: string;
    };
  };

  span.instrumentationLibrary = {
    name: CLAUDE_AGENT_INSTRUMENTATION_NAME,
    version: PACKAGE_VERSION,
  };
  return span;
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

function parseJsonObject(value: string, fallback: unknown): unknown {
  if (!value.trim()) {
    return toSerializableValue(fallback);
  }
  try {
    return JSON.parse(value);
  } catch {
    return toSerializableValue(fallback) ?? value;
  }
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function firstString(value: unknown): string | undefined {
  if (typeof value === "string" && value) {
    return value;
  }
  if (!Array.isArray(value)) {
    return undefined;
  }
  return value.find((item): item is string => typeof item === "string" && item.length > 0);
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
    return value
      .map((item) => toSerializableValue(item))
      .filter((item) => item !== undefined);
  }
  if (typeof value === "object") {
    const normalizedObject: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, itemValue]) => {
      const normalizedValue = toSerializableValue(itemValue);
      if (normalizedValue !== undefined) {
        normalizedObject[key] = normalizedValue;
      }
    });
    return normalizedObject;
  }
  if (typeof value === "function" || typeof value === "symbol") {
    return undefined;
  }
  return String(value);
}

function coerceInteger(value: unknown): number | null {
  if (value === undefined || value === null) {
    return null;
  }
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return null;
  }
  return Math.trunc(numericValue);
}
