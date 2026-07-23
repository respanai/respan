import {
  ASSISTANT_ROLE,
  BODY_KEY,
  CONTENT_KEY,
  CONVERSE_OPERATION,
  CONVERSE_STREAM_OPERATION,
  DESCRIPTION_KEY,
  FUNCTION_KEY,
  FUNCTION_TOOL_TYPE,
  INPUT_KEY,
  INPUT_SCHEMA_KEY,
  INVOKE_MODEL_OPERATION,
  INVOKE_MODEL_STREAM_OPERATION,
  MESSAGE_KEY,
  MESSAGES_KEY,
  MODEL_ID_KEY,
  NAME_KEY,
  OUTPUT_KEY,
  ROLE_KEY,
  SYSTEM_KEY,
  SYSTEM_ROLE,
  TEXT_KEY,
  TOOL_CONFIG_KEY,
  TOOL_ROLE,
  TOOLS_KEY,
  TYPE_KEY,
  USAGE_KEY,
  USER_ROLE,
} from "./_constants.js";

export interface BedrockRequest {
  operationName: string;
  modelId?: string;
  messages: Record<string, unknown>[];
  tools: Record<string, unknown>[];
  rawPayload?: unknown;
}

export interface BedrockResponse {
  content: string;
  role: string;
  toolCalls: Record<string, unknown>[];
  usage: Record<string, number>;
  rawPayload?: unknown;
}

export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(toSerializableValue(value), (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return String(value);
  }
}

export function toJsonAttr(value: unknown): string {
  return typeof value === "string" ? value : safeJson(value);
}

export function toSerializableValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
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
  if (value instanceof Uint8Array) {
    const decoded = decodeBytes(value);
    return loadJson(decoded) ?? decoded;
  }
  if (value instanceof ArrayBuffer) {
    const decoded = decodeBytes(new Uint8Array(value));
    return loadJson(decoded) ?? decoded;
  }
  if (Array.isArray(value)) {
    return value.map((item) => toSerializableValue(item));
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const output: Record<string, unknown> = {};
    for (const [key, innerValue] of Object.entries(record)) {
      if (typeof innerValue === "function") {
        continue;
      }
      output[key] = toSerializableValue(innerValue);
    }
    return output;
  }
  return String(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function field(value: unknown, name: string, defaultValue?: unknown): unknown {
  return isRecord(value) ? value[name] ?? defaultValue : defaultValue;
}

function coerceInt(value: unknown): number | undefined {
  if (value === undefined || value === null || typeof value === "boolean") {
    return undefined;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? undefined : parsed;
  }
  return undefined;
}

function decodeBytes(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

function loadJson(value: unknown): unknown {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (value instanceof Uint8Array) {
    return loadJson(decodeBytes(value));
  }
  if (value instanceof ArrayBuffer) {
    return loadJson(new Uint8Array(value));
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return undefined;
    }
    try {
      return JSON.parse(trimmed);
    } catch {
      return value;
    }
  }
  if (Array.isArray(value) || isRecord(value)) {
    return value;
  }
  return toSerializableValue(value);
}

export function captureInvokeResponsePayload(response: unknown): unknown {
  if (!isRecord(response)) {
    return response;
  }
  const body = response[BODY_KEY];
  if (body === undefined || body === null) {
    return response;
  }
  return loadJson(body) ?? response;
}

function normalizeTextContent(content: unknown): unknown {
  if (content === undefined || content === null) {
    return "";
  }
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const normalized = content
      .map((item) => normalizeTextContent(item))
      .filter((item) => item !== undefined && item !== null && item !== "" && !(Array.isArray(item) && item.length === 0));
    if (normalized.length === 0) {
      return "";
    }
    if (normalized.every((item) => typeof item === "string")) {
      return normalized.join("\n");
    }
    return normalized;
  }
  if (isRecord(content)) {
    const text = content[TEXT_KEY];
    if (typeof text === "string") {
      return text;
    }
    if (content.json !== undefined) {
      return toSerializableValue(content.json);
    }
    if (isRecord(content.toolUse)) {
      return normalizeToolCall(content.toolUse);
    }
    if (isRecord(content.toolResult)) {
      return normalizeToolResult(content.toolResult);
    }
    if (content[TYPE_KEY] === "text") {
      return String(content[TEXT_KEY] ?? "");
    }
    if (content[TYPE_KEY] === "tool_use") {
      return normalizeAnthropicToolUse(content);
    }
    if (content[TYPE_KEY] === "tool_result") {
      return normalizeAnthropicToolResult(content);
    }
    return toSerializableValue(content);
  }
  return toSerializableValue(content);
}

function normalizeMessage(
  message: unknown,
  defaultRole: string = USER_ROLE,
): Record<string, unknown> {
  const role = field(message, ROLE_KEY, defaultRole) ?? defaultRole;
  let content = field(message, CONTENT_KEY);
  if (content === undefined) {
    content = field(message, "contentBlocks");
  }
  if (content === undefined && typeof message === "string") {
    content = message;
  }

  const normalized: Record<string, unknown> = {
    [ROLE_KEY]: normalizeBedrockRole(role),
    [CONTENT_KEY]: normalizeTextContent(content),
  };
  const toolCalls = extractToolCallsFromContent(content);
  if (toolCalls.length > 0 && normalized[ROLE_KEY] === ASSISTANT_ROLE) {
    normalized.tool_calls = toolCalls;
  }
  return normalized;
}

function normalizeBedrockRole(role: unknown): string {
  if (role === "assistant" || role === "model") {
    return ASSISTANT_ROLE;
  }
  if (role === SYSTEM_ROLE) {
    return SYSTEM_ROLE;
  }
  if (role === TOOL_ROLE) {
    return TOOL_ROLE;
  }
  return typeof role === "string" && role ? role : USER_ROLE;
}

function normalizeSystemMessages(system: unknown): Record<string, unknown>[] {
  if (system === undefined || system === null) {
    return [];
  }
  if (typeof system === "string") {
    return [{ [ROLE_KEY]: SYSTEM_ROLE, [CONTENT_KEY]: system }];
  }
  const content = normalizeTextContent(system);
  return content ? [{ [ROLE_KEY]: SYSTEM_ROLE, [CONTENT_KEY]: content }] : [];
}

function normalizePromptFromBody(body: unknown): Record<string, unknown>[] {
  if (!isRecord(body)) {
    return [];
  }

  const messages: Record<string, unknown>[] = [
    ...normalizeSystemMessages(body[SYSTEM_KEY]),
  ];
  const rawMessages = body[MESSAGES_KEY];
  if (Array.isArray(rawMessages)) {
    messages.push(...rawMessages.map((message) => normalizeMessage(message)));
    return messages;
  }

  for (const key of ["prompt", "inputText", "input_text", INPUT_KEY]) {
    const value = body[key];
    if (value) {
      messages.push({
        [ROLE_KEY]: USER_ROLE,
        [CONTENT_KEY]: normalizeTextContent(value),
      });
      return messages;
    }
  }
  return messages;
}

function normalizeConverseMessages(apiParams: Record<string, unknown>): Record<string, unknown>[] {
  const messages: Record<string, unknown>[] = [
    ...normalizeSystemMessages(apiParams[SYSTEM_KEY]),
  ];
  const rawMessages = apiParams[MESSAGES_KEY];
  if (Array.isArray(rawMessages)) {
    messages.push(...rawMessages.map((message) => normalizeMessage(message)));
  }
  return messages;
}

function normalizeAnthropicToolUse(block: Record<string, unknown>): Record<string, unknown> {
  return {
    id: block.id ?? "",
    [TYPE_KEY]: FUNCTION_TOOL_TYPE,
    [FUNCTION_KEY]: {
      [NAME_KEY]: block[NAME_KEY] ?? "",
      arguments: toJsonAttr(block[INPUT_KEY] ?? {}),
    },
  };
}

function normalizeToolCall(block: Record<string, unknown>): Record<string, unknown> {
  return {
    id: block.toolUseId ?? "",
    [TYPE_KEY]: FUNCTION_TOOL_TYPE,
    [FUNCTION_KEY]: {
      [NAME_KEY]: block[NAME_KEY] ?? "",
      arguments: toJsonAttr(block[INPUT_KEY] ?? {}),
    },
  };
}

function normalizeAnthropicToolResult(block: Record<string, unknown>): Record<string, unknown> {
  return {
    [ROLE_KEY]: TOOL_ROLE,
    tool_call_id: block.tool_use_id ?? "",
    [CONTENT_KEY]: normalizeTextContent(block[CONTENT_KEY] ?? ""),
  };
}

function normalizeToolResult(block: Record<string, unknown>): Record<string, unknown> {
  return {
    [ROLE_KEY]: TOOL_ROLE,
    tool_call_id: block.toolUseId ?? "",
    [CONTENT_KEY]: normalizeTextContent(block[CONTENT_KEY] ?? ""),
  };
}

function extractToolCallsFromContent(content: unknown): Record<string, unknown>[] {
  const toolCalls: Record<string, unknown>[] = [];
  const blocks = Array.isArray(content) ? content : [content];
  for (const block of blocks) {
    if (!isRecord(block)) {
      continue;
    }

    let toolCall: Record<string, unknown> | undefined;
    if (isRecord(block.toolUse)) {
      toolCall = normalizeToolCall(block.toolUse);
    } else if (block[TYPE_KEY] === "tool_use") {
      toolCall = normalizeAnthropicToolUse(block);
    }

    const functionPayload = isRecord(toolCall?.[FUNCTION_KEY])
      ? toolCall[FUNCTION_KEY] as Record<string, unknown>
      : undefined;
    if (toolCall && functionPayload?.[NAME_KEY]) {
      toolCalls.push(toolCall);
    }
  }
  return toolCalls;
}

function normalizeToolDefinition(tool: unknown): Record<string, unknown> | undefined {
  if (!isRecord(tool)) {
    return undefined;
  }

  const toolSpec = isRecord(tool.toolSpec) ? tool.toolSpec : tool;
  const name = toolSpec[NAME_KEY];
  if (typeof name !== "string" || !name) {
    return undefined;
  }

  let schema = toolSpec[INPUT_SCHEMA_KEY] ?? toolSpec.input_schema;
  if (isRecord(schema) && schema.json !== undefined) {
    schema = schema.json;
  }
  if (schema === undefined) {
    schema = { type: "object" };
  }

  const functionPayload: Record<string, unknown> = {
    [NAME_KEY]: name,
    parameters: toSerializableValue(schema),
  };
  if (toolSpec[DESCRIPTION_KEY]) {
    functionPayload[DESCRIPTION_KEY] = toolSpec[DESCRIPTION_KEY];
  }

  return {
    [TYPE_KEY]: FUNCTION_TOOL_TYPE,
    [FUNCTION_KEY]: functionPayload,
  };
}

function extractToolsFromBody(body: unknown): Record<string, unknown>[] {
  if (!isRecord(body) || !Array.isArray(body[TOOLS_KEY])) {
    return [];
  }
  return body[TOOLS_KEY]
    .map((tool) => normalizeToolDefinition(tool))
    .filter((tool): tool is Record<string, unknown> => tool !== undefined);
}

function extractToolsFromConverse(apiParams: Record<string, unknown>): Record<string, unknown>[] {
  const toolConfig = apiParams[TOOL_CONFIG_KEY];
  if (!isRecord(toolConfig) || !Array.isArray(toolConfig[TOOLS_KEY])) {
    return [];
  }
  return toolConfig[TOOLS_KEY]
    .map((tool) => normalizeToolDefinition(tool))
    .filter((tool): tool is Record<string, unknown> => tool !== undefined);
}

export function parseBedrockRequest(params: {
  operationName: string;
  apiParams?: Record<string, unknown>;
}): BedrockRequest {
  const apiParams = params.apiParams ?? {};
  const rawModelId = apiParams[MODEL_ID_KEY];
  const modelId = typeof rawModelId === "string" ? rawModelId : undefined;

  if (
    params.operationName === INVOKE_MODEL_OPERATION ||
    params.operationName === INVOKE_MODEL_STREAM_OPERATION
  ) {
    const body = loadJson(apiParams[BODY_KEY]);
    return {
      operationName: params.operationName,
      modelId,
      messages: normalizePromptFromBody(body),
      tools: extractToolsFromBody(body),
      rawPayload: body,
    };
  }

  if (
    params.operationName === CONVERSE_OPERATION ||
    params.operationName === CONVERSE_STREAM_OPERATION
  ) {
    return {
      operationName: params.operationName,
      modelId,
      messages: normalizeConverseMessages(apiParams),
      tools: extractToolsFromConverse(apiParams),
      rawPayload: toSerializableValue(apiParams),
    };
  }

  return {
    operationName: params.operationName,
    modelId,
    messages: [],
    tools: [],
    rawPayload: toSerializableValue(apiParams),
  };
}

function usageFromMapping(value: unknown): Record<string, number> {
  if (!isRecord(value)) {
    return {};
  }

  const promptTokens =
    coerceInt(value.input_tokens) ??
    coerceInt(value.inputTokens) ??
    coerceInt(value.prompt_tokens) ??
    coerceInt(value.promptTokens) ??
    coerceInt(value.inputTextTokenCount);
  const completionTokens =
    coerceInt(value.output_tokens) ??
    coerceInt(value.outputTokens) ??
    coerceInt(value.completion_tokens) ??
    coerceInt(value.completionTokens);
  let totalTokens =
    coerceInt(value.total_tokens) ??
    coerceInt(value.totalTokens) ??
    coerceInt(value.total_token_count);

  const result: Record<string, number> = {};
  if (promptTokens !== undefined) {
    result.input_tokens = promptTokens;
  }
  if (completionTokens !== undefined) {
    result.output_tokens = completionTokens;
  }
  if (
    totalTokens === undefined &&
    (promptTokens !== undefined || completionTokens !== undefined)
  ) {
    totalTokens = (promptTokens ?? 0) + (completionTokens ?? 0);
  }
  if (totalTokens !== undefined) {
    result.total_tokens = totalTokens;
  }
  return result;
}

function mergeUsage(target: Record<string, number>, source: Record<string, number>): void {
  for (const [key, value] of Object.entries(source)) {
    if (Number.isInteger(value)) {
      target[key] = value;
    }
  }
}

function responseFromAnthropicPayload(payload: Record<string, unknown>): BedrockResponse {
  const content = payload[CONTENT_KEY];
  return {
    content: extractTextFromResponseContent(content),
    role: normalizeBedrockRole(payload[ROLE_KEY] ?? ASSISTANT_ROLE),
    toolCalls: extractToolCallsFromContent(content),
    usage: usageFromMapping(payload[USAGE_KEY]),
    rawPayload: payload,
  };
}

function responseFromConversePayload(payload: Record<string, unknown>): BedrockResponse {
  const output = payload[OUTPUT_KEY];
  const message = isRecord(output) ? output[MESSAGE_KEY] : undefined;
  if (!isRecord(message)) {
    return {
      content: "",
      role: ASSISTANT_ROLE,
      toolCalls: [],
      usage: {},
      rawPayload: payload,
    };
  }
  const content = message[CONTENT_KEY];
  return {
    content: extractTextFromResponseContent(content),
    role: normalizeBedrockRole(message[ROLE_KEY] ?? ASSISTANT_ROLE),
    toolCalls: extractToolCallsFromContent(content),
    usage: usageFromMapping(payload[USAGE_KEY]),
    rawPayload: payload,
  };
}

function responseFromTitanPayload(payload: Record<string, unknown>): BedrockResponse {
  let content = "";
  const usage: Record<string, number> = {};
  const inputTokens = coerceInt(payload.inputTextTokenCount);
  if (inputTokens !== undefined) {
    usage.input_tokens = inputTokens;
  }

  const results = payload.results;
  if (Array.isArray(results) && isRecord(results[0])) {
    content = String(results[0].outputText ?? results[0][TEXT_KEY] ?? "");
    const outputTokens = coerceInt(results[0].tokenCount);
    if (outputTokens !== undefined) {
      usage.output_tokens = outputTokens;
    }
  }
  if (Object.keys(usage).length > 0 && usage.total_tokens === undefined) {
    usage.total_tokens = (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0);
  }
  return {
    content,
    role: ASSISTANT_ROLE,
    toolCalls: [],
    usage,
    rawPayload: payload,
  };
}

function extractTextFromResponseContent(content: unknown): string {
  if (content === undefined || content === null) {
    return "";
  }
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((item) => extractTextFromResponseContent(item))
      .filter(Boolean)
      .join("\n");
  }
  if (isRecord(content)) {
    const text = content[TEXT_KEY];
    if (typeof text === "string") {
      return text;
    }
    if (content[TYPE_KEY] === "text") {
      return String(content[TEXT_KEY] ?? "");
    }
  }
  return "";
}

export function parseBedrockResponse(params: {
  operationName: string;
  responsePayload: unknown;
}): BedrockResponse {
  const payload = loadJson(params.responsePayload);
  if (!isRecord(payload)) {
    return {
      content: "",
      role: ASSISTANT_ROLE,
      toolCalls: [],
      usage: {},
      rawPayload: payload,
    };
  }

  if (params.operationName === CONVERSE_OPERATION) {
    return responseFromConversePayload(payload);
  }

  if (CONTENT_KEY in payload && Array.isArray(payload[CONTENT_KEY])) {
    return responseFromAnthropicPayload(payload);
  }

  if ("results" in payload || "inputTextTokenCount" in payload) {
    return responseFromTitanPayload(payload);
  }

  for (const key of ["generation", "outputText", "completion"]) {
    const value = payload[key];
    if (typeof value === "string") {
      return {
        content: value,
        role: ASSISTANT_ROLE,
        toolCalls: [],
        usage: usageFromMapping(payload[USAGE_KEY] ?? payload),
        rawPayload: payload,
      };
    }
  }

  const outputs = payload.outputs;
  if (Array.isArray(outputs) && outputs.length > 0) {
    return {
      content: extractTextFromResponseContent(outputs),
      role: ASSISTANT_ROLE,
      toolCalls: [],
      usage: usageFromMapping(payload[USAGE_KEY] ?? payload),
      rawPayload: payload,
    };
  }

  return {
    content: extractTextFromResponseContent(payload),
    role: ASSISTANT_ROLE,
    toolCalls: [],
    usage: usageFromMapping(payload[USAGE_KEY] ?? payload),
    rawPayload: payload,
  };
}

function parseChunkPayload(event: Record<string, unknown>): unknown {
  const chunk = event.chunk;
  if (!isRecord(chunk)) {
    return undefined;
  }
  const bytes = chunk.bytes;
  if (bytes === undefined || bytes === null) {
    return undefined;
  }
  return loadJson(bytes);
}

export function parseBedrockStreamResponse(params: {
  operationName: string;
  events: unknown[];
}): BedrockResponse {
  const textParts: string[] = [];
  const toolCalls: Record<string, unknown>[] = [];
  const usage: Record<string, number> = {};
  const rawPayloads: unknown[] = [];

  for (const event of params.events) {
    if (!isRecord(event)) {
      rawPayloads.push(toSerializableValue(event));
      continue;
    }
    rawPayloads.push(toSerializableValue(event));

    if (params.operationName === CONVERSE_STREAM_OPERATION) {
      const contentBlockDelta = event.contentBlockDelta;
      const delta = isRecord(contentBlockDelta) ? contentBlockDelta.delta : undefined;
      const text = isRecord(delta) ? delta[TEXT_KEY] : undefined;
      if (typeof text === "string") {
        textParts.push(text);
      }

      const contentBlockStart = event.contentBlockStart;
      const start = isRecord(contentBlockStart) ? contentBlockStart.start : undefined;
      const toolUse = isRecord(start) ? start.toolUse : undefined;
      if (isRecord(toolUse)) {
        const toolCall = normalizeToolCall(toolUse);
        const functionPayload = toolCall[FUNCTION_KEY];
        if (isRecord(functionPayload) && functionPayload[NAME_KEY]) {
          toolCalls.push(toolCall);
        }
      }

      if (isRecord(event.metadata)) {
        mergeUsage(usage, usageFromMapping(event.metadata[USAGE_KEY]));
      }
      continue;
    }

    const payload = parseChunkPayload(event);
    if (!isRecord(payload)) {
      continue;
    }
    rawPayloads.push(payload);

    const payloadType = payload[TYPE_KEY];
    if (payloadType === "content_block_delta") {
      const delta = payload.delta;
      const text = isRecord(delta) ? delta[TEXT_KEY] : undefined;
      if (typeof text === "string") {
        textParts.push(text);
      }
    } else if (payloadType === "content_block_start") {
      const contentBlock = payload.content_block;
      if (isRecord(contentBlock) && contentBlock[TYPE_KEY] === "tool_use") {
        const toolCall = normalizeAnthropicToolUse(contentBlock);
        const functionPayload = toolCall[FUNCTION_KEY];
        if (isRecord(functionPayload) && functionPayload[NAME_KEY]) {
          toolCalls.push(toolCall);
        }
      }
    } else if (payloadType === "message_start" && isRecord(payload[MESSAGE_KEY])) {
      mergeUsage(usage, usageFromMapping((payload[MESSAGE_KEY] as Record<string, unknown>)[USAGE_KEY]));
    } else if (payloadType === "message_delta") {
      mergeUsage(usage, usageFromMapping(payload[USAGE_KEY]));
    } else {
      for (const key of ["completion", "generation", "outputText"]) {
        const value = payload[key];
        if (typeof value === "string") {
          textParts.push(value);
          break;
        }
      }
      mergeUsage(usage, usageFromMapping(payload[USAGE_KEY] ?? payload));
    }
  }

  return {
    content: textParts.join(""),
    role: ASSISTANT_ROLE,
    toolCalls,
    usage,
    rawPayload: rawPayloads,
  };
}
