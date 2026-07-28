import {
  ARGS_KEY,
  ASSISTANT_ROLE,
  CANDIDATES_KEY,
  CANDIDATES_TOKEN_COUNT_KEY,
  CANDIDATES_TOKEN_COUNT_SNAKE_KEY,
  THOUGHTS_TOKEN_COUNT_KEY,
  THOUGHTS_TOKEN_COUNT_SNAKE_KEY,
  CONTENT_KEY,
  FUNCTION_CALL_KEY,
  FUNCTION_CALL_SNAKE_KEY,
  FUNCTION_DECLARATIONS_KEY,
  FUNCTION_DECLARATIONS_SNAKE_KEY,
  FUNCTION_KEY,
  FUNCTION_RESPONSE_KEY,
  FUNCTION_RESPONSE_SNAKE_KEY,
  FUNCTION_TOOL_TYPE,
  GENERATION_CONFIG_KEY,
  GENERATION_CONFIG_SNAKE_KEY,
  ID_KEY,
  MODEL_ROLE,
  NAME_KEY,
  PARAMETERS_KEY,
  PARTS_KEY,
  PROMPT_TOKEN_COUNT_KEY,
  PROMPT_TOKEN_COUNT_SNAKE_KEY,
  RESPONSE_KEY,
  ROLE_KEY,
  STREAM_KEY,
  SYSTEM_INSTRUCTION_KEY,
  SYSTEM_INSTRUCTION_SNAKE_KEY,
  SYSTEM_ROLE,
  TEXT_KEY,
  TOOL_CONFIG_KEY,
  TOOL_CONFIG_SNAKE_KEY,
  TOOL_ROLE,
  TOOLS_KEY,
  TOTAL_TOKEN_COUNT_KEY,
  TOTAL_TOKEN_COUNT_SNAKE_KEY,
  TYPE_KEY,
  USER_ROLE,
  USAGE_METADATA_KEY,
  USAGE_METADATA_SNAKE_KEY,
} from "./_constants.js";

export interface VertexAIRequestPayload {
  model?: string;
  contents?: unknown;
  generationConfig?: unknown;
  toolConfig?: unknown;
  tools?: unknown;
  stream?: boolean;
  systemInstruction?: unknown;
}

export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return String(value);
  }
}

export function toSerializableValue(value: any): any {
  if (value === null) return null;
  if (value === undefined) return undefined;
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
    return value.map((item) => toSerializableValue(item));
  }
  if (typeof value === "object") {
    if (typeof value.toJSON === "function") {
      try {
        return toSerializableValue(value.toJSON());
      } catch {
        // Continue with a structural copy.
      }
    }
    if (typeof value.toObject === "function") {
      try {
        return toSerializableValue(value.toObject());
      } catch {
        // Continue with a structural copy.
      }
    }

    const normalized: Record<string, unknown> = {};
    for (const [key, itemValue] of Object.entries(value as Record<string, unknown>)) {
      if (itemValue !== undefined) {
        normalized[key] = toSerializableValue(itemValue);
      }
    }
    return normalized;
  }
  return String(value);
}

export function toJsonAttr(value: unknown): string {
  if (typeof value === "string") return value;
  return safeJson(toSerializableValue(value));
}

function getField(value: any, ...names: string[]): any {
  if (value === undefined || value === null) return undefined;
  for (const name of names) {
    if (typeof value === "object" && name in value) {
      return value[name];
    }
    const nestedValue = value?.[name];
    if (nestedValue !== undefined) {
      return nestedValue;
    }
  }
  return undefined;
}

function firstDefined<T>(...values: Array<T | undefined>): T | undefined {
  for (const value of values) {
    if (value !== undefined) return value;
  }
  return undefined;
}

function role(value: unknown, defaultRole = USER_ROLE): string {
  const rawRole = getField(value, ROLE_KEY) ?? defaultRole;
  if (rawRole === MODEL_ROLE) return ASSISTANT_ROLE;
  if (rawRole === FUNCTION_KEY) return TOOL_ROLE;
  return String(rawRole);
}

function isContentLike(value: unknown): boolean {
  return Boolean(value && typeof value === "object" && (
    getField(value, PARTS_KEY) !== undefined ||
    getField(value, ROLE_KEY) !== undefined
  ));
}

function isPartLike(value: unknown): boolean {
  if (typeof value === "string") return true;
  return Boolean(value && typeof value === "object" && (
    getField(value, TEXT_KEY) !== undefined ||
    getField(value, FUNCTION_CALL_KEY, FUNCTION_CALL_SNAKE_KEY) !== undefined ||
    getField(value, FUNCTION_RESPONSE_KEY, FUNCTION_RESPONSE_SNAKE_KEY) !== undefined ||
    getField(value, "inlineData", "inline_data") !== undefined ||
    getField(value, "fileData", "file_data") !== undefined
  ));
}

function normalizeFunctionCall(value: any): Record<string, unknown> {
  const functionCall: Record<string, any> = {
    [TYPE_KEY]: FUNCTION_TOOL_TYPE,
    [FUNCTION_KEY]: {},
  };
  const callId = getField(value, ID_KEY);
  if (callId) functionCall[ID_KEY] = callId;

  const name = getField(value, NAME_KEY);
  if (name) functionCall[FUNCTION_KEY][NAME_KEY] = name;

  const args = getField(value, ARGS_KEY);
  if (args !== undefined) {
    functionCall[FUNCTION_KEY].arguments = toJsonAttr(toSerializableValue(args));
  }
  return functionCall;
}

function normalizeFunctionResponse(value: any): Record<string, unknown> {
  const result: Record<string, any> = {
    [TYPE_KEY]: "function_response",
    [FUNCTION_KEY]: {},
  };
  const name = getField(value, NAME_KEY);
  if (name) result[FUNCTION_KEY][NAME_KEY] = name;

  const response = getField(value, RESPONSE_KEY);
  if (response !== undefined) {
    result[FUNCTION_KEY][RESPONSE_KEY] = toSerializableValue(response);
  }
  return result;
}

function normalizePart(part: unknown): unknown {
  if (typeof part === "string") return part;

  const text = getField(part, TEXT_KEY);
  if (text !== undefined) return text;

  const functionCall = getField(part, FUNCTION_CALL_KEY, FUNCTION_CALL_SNAKE_KEY);
  if (functionCall !== undefined) {
    return normalizeFunctionCall(functionCall);
  }

  const functionResponse = getField(part, FUNCTION_RESPONSE_KEY, FUNCTION_RESPONSE_SNAKE_KEY);
  if (functionResponse !== undefined) {
    return normalizeFunctionResponse(functionResponse);
  }

  const inlineData = getField(part, "inlineData", "inline_data");
  if (inlineData !== undefined) {
    return { [TYPE_KEY]: "inline_data", inline_data: toSerializableValue(inlineData) };
  }

  const fileData = getField(part, "fileData", "file_data");
  if (fileData !== undefined) {
    return { [TYPE_KEY]: "file_data", file_data: toSerializableValue(fileData) };
  }

  return toSerializableValue(part);
}

function normalizeParts(parts: unknown): unknown {
  if (parts === undefined || parts === null) return "";
  if (typeof parts === "string") return parts;
  if (!Array.isArray(parts)) return normalizePart(parts);

  const normalizedParts = parts
    .map((part) => normalizePart(part))
    .filter((part) => part !== undefined && part !== null && part !== "");
  if (normalizedParts.length === 0) return "";
  if (normalizedParts.length === 1) return normalizedParts[0];
  if (normalizedParts.every((part) => typeof part === "string")) {
    return normalizedParts.join("\n");
  }
  return normalizedParts;
}

function normalizeContent(content: unknown, defaultRole = USER_ROLE): Record<string, unknown> {
  if (typeof content === "string") {
    return { [ROLE_KEY]: defaultRole, [CONTENT_KEY]: content };
  }
  if (isPartLike(content) && !isContentLike(content)) {
    return { [ROLE_KEY]: defaultRole, [CONTENT_KEY]: normalizePart(content) };
  }

  return {
    [ROLE_KEY]: role(content, defaultRole),
    [CONTENT_KEY]: normalizeParts(getField(content, PARTS_KEY)),
  };
}

export function normalizeInputMessages(
  contents: unknown,
  systemInstruction?: unknown,
): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];

  if (systemInstruction !== undefined && systemInstruction !== null) {
    messages.push(
      isContentLike(systemInstruction)
        ? normalizeContent(systemInstruction, SYSTEM_ROLE)
        : { [ROLE_KEY]: SYSTEM_ROLE, [CONTENT_KEY]: normalizeParts(systemInstruction) },
    );
  }

  if (contents === undefined || contents === null) return messages;
  if (typeof contents === "string") {
    messages.push({ [ROLE_KEY]: USER_ROLE, [CONTENT_KEY]: contents });
    return messages;
  }
  if (isContentLike(contents)) {
    messages.push(normalizeContent(contents));
    return messages;
  }
  if (isPartLike(contents)) {
    messages.push({ [ROLE_KEY]: USER_ROLE, [CONTENT_KEY]: normalizePart(contents) });
    return messages;
  }
  if (Array.isArray(contents)) {
    if (contents.every((item) => isContentLike(item))) {
      messages.push(...contents.map((item) => normalizeContent(item)));
      return messages;
    }
    messages.push({ [ROLE_KEY]: USER_ROLE, [CONTENT_KEY]: normalizeParts(contents) });
    return messages;
  }

  messages.push({ [ROLE_KEY]: USER_ROLE, [CONTENT_KEY]: toSerializableValue(contents) });
  return messages;
}

export function formatInput(contents: unknown, systemInstruction?: unknown): string {
  return safeJson(normalizeInputMessages(contents, systemInstruction));
}

function candidateContents(response: unknown): unknown[] {
  const candidates = getField(response, CANDIDATES_KEY) ?? [];
  if (!Array.isArray(candidates)) return [];
  return candidates
    .map((candidate) => getField(candidate, CONTENT_KEY))
    .filter((content) => content !== undefined && content !== null);
}

function responseText(response: unknown): string {
  if (response === undefined || response === null) return "";

  const textProp = getField(response, TEXT_KEY);
  if (typeof textProp === "string") return textProp;
  if (typeof textProp === "function") {
    try {
      const text = textProp.call(response);
      if (typeof text === "string") return text;
    } catch {
      // Fall back to candidates.
    }
  }

  const parts: string[] = [];
  for (const content of candidateContents(response)) {
    const normalized = normalizeContent(content, ASSISTANT_ROLE)[CONTENT_KEY];
    if (typeof normalized === "string") {
      parts.push(normalized);
    }
  }
  return parts.join("\n");
}

export function formatOutput(responseOrChunks: unknown): string {
  if (Array.isArray(responseOrChunks)) {
    return responseOrChunks.map((chunk) => responseText(chunk)).join("");
  }
  return responseText(responseOrChunks);
}

export function extractUsage(responseOrChunks: unknown): Record<string, number> {
  let response = responseOrChunks;
  if (Array.isArray(responseOrChunks)) {
    response = [...responseOrChunks]
      .reverse()
      .find((chunk) => getField(chunk, USAGE_METADATA_KEY, USAGE_METADATA_SNAKE_KEY) !== undefined)
      ?? responseOrChunks.at(-1);
  }

  const usage = getField(response, USAGE_METADATA_KEY, USAGE_METADATA_SNAKE_KEY);
  if (!usage) return {};

  const result: Record<string, number> = {};
  const promptTokens = getField(usage, PROMPT_TOKEN_COUNT_KEY, PROMPT_TOKEN_COUNT_SNAKE_KEY);
  const completionTokens = getField(
    usage,
    CANDIDATES_TOKEN_COUNT_KEY,
    CANDIDATES_TOKEN_COUNT_SNAKE_KEY,
  );
  const thoughtsTokens = getField(
    usage,
    THOUGHTS_TOKEN_COUNT_KEY,
    THOUGHTS_TOKEN_COUNT_SNAKE_KEY,
  );
  const totalTokens = getField(usage, TOTAL_TOKEN_COUNT_KEY, TOTAL_TOKEN_COUNT_SNAKE_KEY);

  if (typeof promptTokens === "number") result[PROMPT_TOKEN_COUNT_KEY] = promptTokens;
  if (typeof completionTokens === "number") {
    // Gemini reports thinking tokens separately from candidatesTokenCount and bills
    // them at the output rate, so they belong in the output count. This matches the
    // google-adk instrumentation, which already folds them in.
    result[CANDIDATES_TOKEN_COUNT_KEY] =
      completionTokens + (typeof thoughtsTokens === "number" ? thoughtsTokens : 0);
  }
  if (typeof totalTokens === "number") result[TOTAL_TOKEN_COUNT_KEY] = totalTokens;
  return result;
}

export function extractToolCalls(responseOrChunks: unknown): Array<Record<string, unknown>> {
  const chunks = Array.isArray(responseOrChunks) ? responseOrChunks : [responseOrChunks];
  const toolCalls: Array<Record<string, unknown>> = [];
  const seen = new Set<string>();

  for (const response of chunks) {
    for (const content of candidateContents(response)) {
      const parts = getField(content, PARTS_KEY) ?? [];
      if (!Array.isArray(parts)) continue;
      for (const part of parts) {
        const functionCall = getField(part, FUNCTION_CALL_KEY, FUNCTION_CALL_SNAKE_KEY);
        if (functionCall === undefined || functionCall === null) continue;
        const normalized = normalizeFunctionCall(functionCall);
        const functionValue = normalized[FUNCTION_KEY];
        if (
          !functionValue ||
          typeof functionValue !== "object" ||
          !(functionValue as Record<string, unknown>)[NAME_KEY]
        ) {
          continue;
        }
        const signature = safeJson(normalized);
        if (seen.has(signature)) continue;
        seen.add(signature);
        toolCalls.push(normalized);
      }
    }
  }

  return toolCalls;
}

function functionDeclarationTool(definition: unknown): Record<string, unknown> | undefined {
  const name = getField(definition, NAME_KEY);
  if (!name) return undefined;

  const functionValue: Record<string, unknown> = { [NAME_KEY]: name };
  const description = getField(definition, "description");
  if (description) functionValue.description = description;

  const parameters = getField(definition, PARAMETERS_KEY);
  if (parameters !== undefined) {
    functionValue[PARAMETERS_KEY] = toSerializableValue(parameters);
  }

  return { [TYPE_KEY]: FUNCTION_TOOL_TYPE, [FUNCTION_KEY]: functionValue };
}

export function extractTools(tools: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(tools)) return [];

  const normalizedTools: Array<Record<string, unknown>> = [];
  for (const tool of tools) {
    const declarations = getField(
      tool,
      FUNCTION_DECLARATIONS_KEY,
      FUNCTION_DECLARATIONS_SNAKE_KEY,
    );
    if (Array.isArray(declarations)) {
      for (const declaration of declarations) {
        const normalized = functionDeclarationTool(declaration);
        if (normalized) normalizedTools.push(normalized);
      }
      continue;
    }

    const normalized = functionDeclarationTool(tool);
    if (normalized) normalizedTools.push(normalized);
  }

  return normalizedTools;
}

function firstArg(args: unknown[]): unknown {
  return args.length > 0 ? args[0] : undefined;
}

function requestObjectFromArgs(args: unknown[]): Record<string, any> {
  const value = firstArg(args);
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, any>
    : {};
}

function modelName(instance: any): string | undefined {
  const value = firstDefined(
    getField(instance, "model"),
    getField(instance, "modelName", "model_name"),
    getField(instance, "_model", "_modelName", "_model_name"),
    getField(instance, "modelId", "model_id", "_modelId", "_model_id"),
  );
  if (typeof value === "string" && value.length > 0) return value;
  if (value && typeof value === "object" && value !== instance) {
    const nested = modelName(value);
    if (nested) return nested;
  }

  const nestedModel = getField(instance, "generativeModel", "modelInstance");
  if (nestedModel && nestedModel !== instance) return modelName(nestedModel);
  return undefined;
}

function systemInstruction(instance: any, request: Record<string, any>): unknown {
  const requestValue = getField(request, SYSTEM_INSTRUCTION_KEY, SYSTEM_INSTRUCTION_SNAKE_KEY);
  if (requestValue !== undefined) return requestValue;

  const instanceValue = getField(
    instance,
    SYSTEM_INSTRUCTION_KEY,
    SYSTEM_INSTRUCTION_SNAKE_KEY,
    "_systemInstruction",
    "_system_instruction",
  );
  if (instanceValue !== undefined) return instanceValue;

  const nestedModel = getField(instance, "model", "generativeModel", "modelInstance");
  if (nestedModel && nestedModel !== instance) {
    return systemInstruction(nestedModel, request);
  }
  return undefined;
}

function tools(instance: any, request: Record<string, any>): unknown {
  const requestTools = getField(request, TOOLS_KEY);
  if (requestTools !== undefined) return requestTools;

  const instanceTools = getField(instance, TOOLS_KEY, "_tools");
  if (instanceTools !== undefined) return instanceTools;

  const nestedModel = getField(instance, "model", "generativeModel", "modelInstance");
  if (nestedModel && nestedModel !== instance) {
    return tools(nestedModel, request);
  }
  return undefined;
}

function generationConfig(instance: any, request: Record<string, any>): unknown {
  return firstDefined(
    getField(request, GENERATION_CONFIG_KEY, GENERATION_CONFIG_SNAKE_KEY),
    getField(instance, GENERATION_CONFIG_KEY, GENERATION_CONFIG_SNAKE_KEY),
  );
}

export function requestPayloadFromCall(
  instance: unknown,
  args: unknown[],
  opts: { isChatMethod?: boolean; isStreamMethod?: boolean } = {},
): VertexAIRequestPayload {
  const request = requestObjectFromArgs(args);
  const rawFirstArg = firstArg(args);
  const contents = opts.isChatMethod
    ? rawFirstArg
    : firstDefined(getField(request, "contents", CONTENT_KEY), rawFirstArg);

  return {
    model: modelName(instance),
    contents,
    generationConfig: generationConfig(instance, request),
    toolConfig: getField(request, TOOL_CONFIG_KEY, TOOL_CONFIG_SNAKE_KEY),
    tools: tools(instance, request),
    stream: opts.isStreamMethod || Boolean(getField(request, STREAM_KEY)),
    systemInstruction: systemInstruction(instance, request),
  };
}
