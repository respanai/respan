import {
  AI_PROMPT,
  AI_PROMPT_MESSAGES,
  AI_PROMPT_TOOLS,
  AI_PROMPT_TOOL_CHOICE,
  AI_RESPONSE_OBJECT,
  AI_RESPONSE_TEXT,
  AI_RESPONSE_TOOL_CALLS,
  AI_TOOL_CALL,
  AI_TOOL_CALLS,
  AI_TOOL_CALL_ARGS,
  AI_TOOL_CALL_ID,
  AI_TOOL_CALL_NAME,
  AI_TOOL_CALL_PREFIX,
  AI_TOOL_CALL_RESULT,
  RESPAN_SPAN_TOOLS,
  RESPAN_SPAN_TOOL_CALLS,
  isRecord,
  safeJsonParse,
  safeJsonStr,
  type SpanAttributes,
} from "./shared.js";

type MessagePayload = Record<string, any>;

function normalizeToolCallShape(call: unknown): Record<string, any> | undefined {
  if (!isRecord(call)) {
    return undefined;
  }

  const callFunction = isRecord(call.function) ? call.function : undefined;
  const rawName =
    callFunction?.name ??
    call.name ??
    call.toolName ??
    call.tool_name;
  const rawArguments =
    callFunction?.arguments ??
    callFunction?.args ??
    call.args ??
    call.arguments ??
    call.input;
  const rawId = call.id ?? call.toolCallId ?? call.tool_call_id;

  const normalized: Record<string, any> = { type: "function" };
  if (rawId !== undefined) {
    normalized.id = String(rawId);
  }

  const functionPayload: Record<string, any> = {};
  if (rawName !== undefined) {
    functionPayload.name = String(rawName);
  }
  if (rawArguments !== undefined) {
    functionPayload.arguments =
      typeof rawArguments === "string"
        ? rawArguments
        : safeJsonStr(rawArguments);
  }
  if (Object.keys(functionPayload).length > 0) {
    normalized.function = functionPayload;
  }

  return normalized;
}

function normalizeToolCallList(value: unknown): Record<string, any>[] | undefined {
  const parsed = safeJsonParse(value);
  const rawCalls = Array.isArray(parsed) ? parsed : parsed !== undefined ? [parsed] : [];
  const normalized = rawCalls
    .map((call) => normalizeToolCallShape(call))
    .filter((call): call is Record<string, any> => Boolean(call));
  return normalized.length > 0 ? normalized : undefined;
}

export function parseToolCalls(attrs: SpanAttributes): Record<string, any>[] | undefined {
  for (const key of [AI_RESPONSE_TOOL_CALLS, AI_TOOL_CALL, AI_TOOL_CALLS]) {
    if (!attrs[key]) {
      continue;
    }

    const normalized = normalizeToolCallList(attrs[key]);
    if (normalized) {
      return normalized;
    }
  }

  if (attrs[AI_TOOL_CALL_ID] || attrs[AI_TOOL_CALL_NAME] || attrs[AI_TOOL_CALL_ARGS]) {
    const toolCall: Record<string, any> = { type: "function" };
    for (const [key, value] of Object.entries(attrs)) {
      if (key.startsWith(AI_TOOL_CALL_PREFIX)) {
        toolCall[key.replace(AI_TOOL_CALL_PREFIX, "")] = value;
      }
    }
    return normalizeToolCallList(toolCall);
  }

  return undefined;
}

function parseToolDefinition(tool: unknown): unknown {
  const parsedTool = typeof tool === "string" ? safeJsonParse(tool) : tool;
  if (!isRecord(parsedTool) || parsedTool.type !== "function") {
    return parsedTool;
  }

  if (isRecord(parsedTool.function)) {
    if (parsedTool.inputSchema && !parsedTool.function.parameters) {
      const { inputSchema, ...rest } = parsedTool;
      return {
        ...rest,
        function: { ...parsedTool.function, parameters: inputSchema },
      };
    }

    return parsedTool;
  }

  const { name, description, parameters, inputSchema, ...rest } = parsedTool;
  const resolvedParameters = parameters ?? inputSchema;
  return {
    ...rest,
    type: "function",
    function: {
      name,
      ...(description ? { description } : {}),
      ...(resolvedParameters ? { parameters: resolvedParameters } : {}),
    },
  };
}

export function parseToolsValue(attrs: SpanAttributes): unknown[] | undefined {
  try {
    const tools = attrs[AI_PROMPT_TOOLS];
    if (!tools) {
      return undefined;
    }

    const rawTools = Array.isArray(tools) ? tools : [tools];
    const parsedTools = rawTools
      .map((tool) => parseToolDefinition(tool))
      .filter(Boolean);

    return parsedTools.length > 0 ? parsedTools : undefined;
  } catch {
    return undefined;
  }
}

export function extractToolNames(tools: unknown[] | undefined): string[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  const names = tools
    .map((tool) => {
      const parsed = typeof tool === "string" ? safeJsonParse(tool) : tool;
      if (!isRecord(parsed)) {
        return undefined;
      }

      const functionName = isRecord(parsed.function) ? parsed.function.name : undefined;
      const name = functionName ?? parsed.name ?? parsed.toolName ?? parsed.tool_name;
      return name === undefined ? undefined : String(name);
    })
    .filter((name): name is string => Boolean(name));

  return names.length > 0 ? Array.from(new Set(names)) : undefined;
}

export function enrichToolDefinitionAttrs(attrs: SpanAttributes, tools: unknown[]): void {
  attrs[RESPAN_SPAN_TOOLS] = tools;
  attrs.tools = tools;

  const toolNames = extractToolNames(tools);
  if (toolNames) {
    attrs.span_tools = toolNames;
  }
}

export function parseToolChoice(attrs: SpanAttributes): string | undefined {
  try {
    const toolChoice = attrs[AI_PROMPT_TOOL_CHOICE];
    if (!toolChoice) {
      return undefined;
    }

    const parsed = typeof toolChoice === "string" ? JSON.parse(toolChoice) : toolChoice;
    if (parsed.function?.name) {
      return safeJsonStr({
        type: String(parsed.type),
        function: { name: String(parsed.function.name) },
      });
    }

    return safeJsonStr({ type: String(parsed.type) });
  } catch {
    return undefined;
  }
}

function isMessageLike(value: unknown): value is MessagePayload {
  if (!isRecord(value) || typeof value.role !== "string") {
    return false;
  }

  return (
    "content" in value ||
    "tool_calls" in value ||
    "toolCalls" in value ||
    "tool_call_id" in value ||
    "toolCallId" in value
  );
}

function isMessageArray(value: unknown): value is MessagePayload[] {
  return Array.isArray(value) && value.length > 0 && value.every((item) => isMessageLike(item));
}

function unwrapKnownResponseWrapper(value: unknown): unknown {
  if (!isRecord(value)) {
    return value;
  }

  return value.response ?? value.object ?? value.output ?? value.result ?? value;
}

function extractMessagePayload(value: unknown): unknown {
  const parsed = safeJsonParse(value);

  if (typeof parsed === "string") {
    const nested = safeJsonParse(parsed);
    return nested === parsed ? undefined : extractMessagePayload(nested);
  }

  if (isMessageLike(parsed) || isMessageArray(parsed)) {
    return parsed;
  }

  if (Array.isArray(parsed)) {
    for (const item of parsed) {
      const nested = extractMessagePayload(item);
      if (nested !== undefined) {
        return nested;
      }
    }
    return undefined;
  }

  if (!isRecord(parsed)) {
    return undefined;
  }

  const unwrapped = unwrapKnownResponseWrapper(parsed);
  if (unwrapped !== parsed) {
    const nested = extractMessagePayload(unwrapped);
    if (nested !== undefined) {
      return nested;
    }
  }

  for (const key of ["message", "messages", "text", "content", "value"]) {
    if (!(key in parsed)) {
      continue;
    }

    const nested = extractMessagePayload(parsed[key]);
    if (nested !== undefined) {
      return nested;
    }
  }

  return undefined;
}

function collapseNestedMessageWrapper(value: unknown): unknown {
  if (!isMessageLike(value)) {
    return value;
  }

  const hasTopLevelMessageFields =
    "tool_calls" in value ||
    "toolCalls" in value ||
    "tool_call_id" in value ||
    "toolCallId" in value ||
    "name" in value;
  if (hasTopLevelMessageFields) {
    return value;
  }

  const nested = extractMessagePayload(value.content);
  return nested ?? value;
}

function isContentBlockType(block: unknown, type: string): block is MessagePayload {
  return isRecord(block) && block.type === type;
}

function normalizeToolResultMessage(block: MessagePayload): MessagePayload {
  const result = block.result ?? block.content ?? block.output ?? "";
  const message: MessagePayload = {
    role: "tool",
    content: typeof result === "string" ? result : safeJsonStr(result),
  };

  const toolCallId = block.toolCallId ?? block.tool_call_id ?? block.id;
  if (toolCallId !== undefined) {
    message.tool_call_id = String(toolCallId);
  }

  const toolName = block.toolName ?? block.tool_name ?? block.name;
  if (toolName !== undefined) {
    message.name = String(toolName);
  }

  return message;
}

function normalizeMessageForBackend(message: unknown): MessagePayload[] {
  if (!isRecord(message)) {
    return [];
  }

  const normalizedToolCalls =
    normalizeToolCallList(message.tool_calls) ??
    normalizeToolCallList(message.toolCalls);

  if (!Array.isArray(message.content)) {
    const normalized = { ...message };
    if (normalizedToolCalls) {
      normalized.tool_calls = normalizedToolCalls;
    }
    delete normalized.toolCalls;

    if (normalized.role === "tool") {
      if (normalized.toolCallId !== undefined && normalized.tool_call_id === undefined) {
        normalized.tool_call_id = String(normalized.toolCallId);
      }
      delete normalized.toolCallId;
      if (normalized.content !== undefined && typeof normalized.content !== "string") {
        normalized.content = safeJsonStr(normalized.content);
      }
    }

    return [normalized];
  }

  const textParts: string[] = [];
  const toolCallBlocks = message.content.filter((block) => isContentBlockType(block, "tool-call"));
  const toolResultBlocks = message.content.filter((block) => isContentBlockType(block, "tool-result"));
  const unknownBlocks = message.content.filter((block) => {
    if (typeof block === "string") {
      return false;
    }
    if (!isRecord(block)) {
      return true;
    }
    return !["text", "output_text", "tool-call", "tool-result"].includes(String(block.type ?? ""));
  });

  for (const block of message.content) {
    if (typeof block === "string") {
      textParts.push(block);
      continue;
    }
    if (!isRecord(block)) {
      continue;
    }
    if ((block.type === "text" || block.type === "output_text") && typeof block.text === "string") {
      textParts.push(block.text);
    }
  }

  const textContent = textParts.join("\n");

  if (message.role === "assistant" && (toolCallBlocks.length > 0 || normalizedToolCalls)) {
    const normalizedMessage: MessagePayload = {
      ...message,
      content: textContent,
    };
    const toolCalls = normalizeToolCallList(toolCallBlocks) ?? normalizedToolCalls;
    if (toolCalls) {
      normalizedMessage.tool_calls = toolCalls;
    }
    delete normalizedMessage.toolCalls;
    return [normalizedMessage];
  }

  if (message.role === "tool" && toolResultBlocks.length > 0) {
    return toolResultBlocks.map((block) => normalizeToolResultMessage(block));
  }

  if (unknownBlocks.length === 0) {
    return [{ ...message, content: textContent }];
  }

  return [{ ...message, content: message.content }];
}

function normalizeMessageCollection(value: unknown): MessagePayload[] | undefined {
  const parsed = safeJsonParse(value);
  if (parsed === undefined || parsed === null) {
    return undefined;
  }

  if (Array.isArray(parsed)) {
    const normalized = parsed.flatMap((message) => normalizeMessageForBackend(message));
    return normalized.length > 0 ? normalized : undefined;
  }

  if (isRecord(parsed)) {
    const normalized = normalizeMessageForBackend(parsed);
    return normalized.length > 0 ? normalized : undefined;
  }

  return undefined;
}

function collapseSingleMessagePayload(messages: MessagePayload[]): unknown {
  return messages.length === 1 ? messages[0] : messages;
}

function buildToolResultMessage(attrs: SpanAttributes): MessagePayload | undefined {
  if (!attrs[AI_TOOL_CALL_RESULT]) {
    return undefined;
  }

  return {
    role: "tool",
    tool_call_id: String(attrs[AI_TOOL_CALL_ID] || ""),
    content: String(attrs[AI_TOOL_CALL_RESULT] || ""),
  };
}

function hasToolResultMessage(payload: unknown, toolResult: MessagePayload): boolean {
  const messages = Array.isArray(payload) ? payload : [payload];
  return messages.some((message) => {
    if (!isRecord(message) || message.role !== "tool") {
      return false;
    }

    if (toolResult.tool_call_id) {
      return message.tool_call_id === toolResult.tool_call_id || message.toolCallId === toolResult.tool_call_id;
    }

    return message.content === toolResult.content;
  });
}

function appendToolResultMessage(payload: unknown, attrs: SpanAttributes): unknown {
  const toolResult = buildToolResultMessage(attrs);
  if (!toolResult || hasToolResultMessage(payload, toolResult)) {
    return payload;
  }

  return Array.isArray(payload) ? [...payload, toolResult] : [payload, toolResult];
}

function selectPrimaryAssistantMessage(value: unknown): MessagePayload | undefined {
  if (isMessageLike(value)) {
    return value.role === "assistant" ? value : undefined;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      if (isMessageLike(item) && item.role === "assistant") {
        return item;
      }
    }
  }

  return undefined;
}

function enrichCompletionAttrs(attrs: SpanAttributes, payload: unknown): void {
  const message = selectPrimaryAssistantMessage(payload);
  const responseToolCalls = message
    ? normalizeToolCallList(message.tool_calls) ||
      normalizeToolCallList(message.toolCalls) ||
      parseToolCalls(attrs)
    : parseToolCalls(attrs);

  if (message) {
    attrs["gen_ai.completion.0.role"] = "assistant";
    attrs["gen_ai.completion.0.content"] =
      typeof message.content === "string"
        ? message.content
        : message.content !== undefined
          ? safeJsonStr(message.content)
          : "";
  }

  if (responseToolCalls && responseToolCalls.length > 0) {
    attrs["gen_ai.completion.0.tool_calls"] = responseToolCalls;
    attrs[RESPAN_SPAN_TOOL_CALLS] = responseToolCalls;
    attrs.tool_calls = responseToolCalls;
    attrs["has_tool_calls"] = true;
    attrs["parallel_tool_calls"] = responseToolCalls.length > 1;
  }
}

function parsePromptInputValue(attrs: SpanAttributes): Record<string, any>[] | undefined {
  const normalizedMessages = normalizeMessageCollection(attrs[AI_PROMPT_MESSAGES]);
  if (normalizedMessages && normalizedMessages.length > 0) {
    return normalizedMessages;
  }

  const prompt = attrs[AI_PROMPT];
  if (prompt) {
    return [{ role: "user", content: String(prompt) }];
  }

  return undefined;
}

export function formatPromptInput(attrs: SpanAttributes): string | undefined {
  const messages = parsePromptInputValue(attrs);
  return messages ? safeJsonStr(messages) : undefined;
}

export function formatCompletionOutput(attrs: SpanAttributes): string | undefined {
  // `ai.response.text` may itself be a JSON-encoded chat message. Normalize that
  // shape first so assistant messages and tool calls are preserved for the backend.
  const existingPayload = extractMessagePayload(attrs[AI_RESPONSE_TEXT]);
  if (existingPayload !== undefined) {
    const normalizedMessages = normalizeMessageCollection(collapseNestedMessageWrapper(existingPayload));
    const normalizedPayload = normalizedMessages
      ? collapseSingleMessagePayload(normalizedMessages)
      : collapseNestedMessageWrapper(existingPayload);
    const outputPayload = appendToolResultMessage(normalizedPayload, attrs);
    enrichCompletionAttrs(attrs, outputPayload);
    return safeJsonStr(outputPayload);
  }

  let content = "";

  if (attrs[AI_RESPONSE_OBJECT]) {
    const parsed = unwrapKnownResponseWrapper(safeJsonParse(attrs[AI_RESPONSE_OBJECT]));
    content = safeJsonStr(parsed);
  } else {
    content = String(attrs[AI_RESPONSE_TEXT] ?? "");
  }

  const toolCalls = parseToolCalls(attrs);
  if (!content && (!toolCalls || toolCalls.length === 0)) {
    return undefined;
  }

  const message: Record<string, any> = { role: "assistant", content };
  if (toolCalls && toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  const normalizedPayload = appendToolResultMessage(message, attrs);
  enrichCompletionAttrs(attrs, normalizedPayload);
  return safeJsonStr(normalizedPayload);
}

export function formatToolInput(attrs: SpanAttributes): string | undefined {
  const name = attrs[AI_TOOL_CALL_NAME];
  const args = attrs[AI_TOOL_CALL_ARGS];
  if (!name && !args) {
    return undefined;
  }

  const input: Record<string, any> = {};
  if (name) {
    input.name = name;
  }
  if (args) {
    input.args = typeof args === "string" ? safeJsonParse(args) : args;
  }
  return safeJsonStr(input);
}

export function formatToolOutput(attrs: SpanAttributes): string | undefined {
  const result = attrs[AI_TOOL_CALL_RESULT];
  if (result === undefined) {
    return undefined;
  }
  return safeJsonStr(typeof result === "string" ? safeJsonParse(result) : result);
}

export function enrichToolSpanAttrs(attrs: SpanAttributes): void {
  const name = attrs[AI_TOOL_CALL_NAME];
  if (name) {
    attrs.span_tools = [String(name)];
  }

  const toolCalls = parseToolCalls(attrs);
  if (toolCalls) {
    attrs[RESPAN_SPAN_TOOL_CALLS] = toolCalls;
    attrs.tool_calls = toolCalls;
  }
}
