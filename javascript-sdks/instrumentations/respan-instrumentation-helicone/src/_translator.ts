import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_ERROR_TYPE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_REQUEST_STREAM,
  ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  RESPAN_SPAN_ATTRIBUTES_MAP,
  RespanLogType,
  RespanSpanAttributes,
} from "@respan/respan-sdk";
import { buildReadableSpan } from "@respan/tracing";
import {
  LLMRequestTypeValues,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import {
  HeliconeEventTypes,
  HeliconeFields,
  HeliconeHeaders,
  MAX_ERROR_MESSAGE_CHARS,
  MAX_INDEXED_PROMPT_MESSAGES,
  MAX_SERIALIZED_BYTES,
  TraceloopCompatibilityFields,
} from "./_constants.js";

export interface HeliconeParentContext {
  traceId?: string;
  parentId?: string;
  entityPath?: string;
}

export interface HeliconeCapture {
  request: unknown;
  response?: unknown;
  options?: unknown;
  loggerHeaders?: unknown;
  propagatedAttributes?: unknown;
  error?: unknown;
  parent: HeliconeParentContext;
  fallbackOperation?: string;
  traceContent: boolean;
  instrumentationScope: {
    name: string;
    version: string;
  };
}

type AnyRecord = Record<string, any>;
type Attributes = Record<string, string | number | boolean | Array<string | number | boolean>>;

interface ParsedResponse {
  raw: unknown;
  record?: AnyRecord;
  text?: string;
  message?: AnyRecord;
  toolCalls?: AnyRecord[];
  usage?: AnyRecord;
}

interface OperationShape {
  logType: RespanLogType;
  entityName: string;
  requestType?: LLMRequestTypeValues;
}

const TRACE_METHOD = "ts_tracing";

export function buildHeliconeSpan(capture: HeliconeCapture): ReadableSpan {
  const request = asRecord(capture.request);
  const options = asRecord(capture.options);
  const parsedResponse = parseResponse(capture.response);
  const shape = resolveOperationShape(request, capture.fallbackOperation);
  const resolvedErrorMessage = resolveErrorMessage(
    capture.error,
    parsedResponse,
    options.status,
  );
  const errorMessage = resolvedErrorMessage
    ? redactSensitiveText(resolvedErrorMessage)
    : undefined;
  const attributes: Attributes = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: shape.entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: capture.parent.entityPath ?? "",
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: TRACE_METHOD,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: shape.logType,
  };

  if (request[HeliconeFields.EVENT_TYPE] === HeliconeEventTypes.TOOL) {
    attachToolEvent(attributes, request, parsedResponse, capture.traceContent);
  } else if (request[HeliconeFields.EVENT_TYPE] === HeliconeEventTypes.VECTOR_DB) {
    attachVectorEvent(attributes, request, parsedResponse, capture.traceContent);
  } else if (request[HeliconeFields.EVENT_TYPE] === HeliconeEventTypes.DATA) {
    attachDataEvent(attributes, request, parsedResponse, capture.traceContent);
  } else {
    attachLlmAttributes(
      attributes,
      request,
      parsedResponse,
      options,
      shape,
      capture.traceContent,
    );
    const streaming = resolveStreaming(
      request,
      options,
      capture.fallbackOperation,
    );
    if (streaming !== undefined) {
      attributes[ATTR_GEN_AI_REQUEST_STREAM] = streaming;
    }
  }

  attachSafeHeliconeHeaders(
    attributes,
    capture.loggerHeaders,
    options.additionalHeaders,
  );
  attachOperationalMetadata(
    attributes,
    options,
    capture.fallbackOperation,
    capture.propagatedAttributes,
  );
  if (errorMessage) {
    const statusCode = errorStatusCode(options.status);
    attributes[ATTR_ERROR_TYPE] = errorType(capture.error, options.status);
    attributes[ATTR_HTTP_RESPONSE_STATUS_CODE] = statusCode;
    attributes[ATTR_ERROR_MESSAGE] = errorMessage;
    attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
      error: errorMessage,
    });
    removeCompletionAttributes(attributes);
  }

  const { startTimeIso, endTimeIso } = resolveTimes(options);
  const span = buildReadableSpan({
    name: shape.entityName,
    traceId: capture.parent.traceId,
    parentId: capture.parent.parentId,
    startTimeIso,
    endTimeIso,
    attributes,
    statusCode: errorMessage ? 500 : 200,
    errorMessage,
    mergePropagated: capture.propagatedAttributes === undefined,
  }) as ReadableSpan & {
    instrumentationScope?: { name: string; version?: string };
  };
  span.instrumentationScope = capture.instrumentationScope;
  return span;
}

function resolveOperationShape(
  request: AnyRecord,
  fallbackOperation?: string,
): OperationShape {
  if (request[HeliconeFields.EVENT_TYPE] === HeliconeEventTypes.TOOL) {
    return {
      logType: RespanLogType.TOOL,
      entityName: stringValue(request[HeliconeFields.TOOL_NAME]) || "helicone.tool",
    };
  }

  if (request[HeliconeFields.EVENT_TYPE] === HeliconeEventTypes.VECTOR_DB) {
    const operation = sanitizeNamePart(request[HeliconeFields.OPERATION]) || "operation";
    return {
      logType: RespanLogType.TASK,
      entityName: `helicone.vector_db.${operation}`,
    };
  }

  if (request[HeliconeFields.EVENT_TYPE] === HeliconeEventTypes.DATA) {
    const name = sanitizeNamePart(request[HeliconeFields.NAME]) || "event";
    return {
      logType: RespanLogType.TASK,
      entityName: `helicone.data.${name}`,
    };
  }

  if (Array.isArray(request.messages) || Array.isArray(request.contents)) {
    return {
      logType: RespanLogType.CHAT,
      entityName: "helicone.chat",
      requestType: LLMRequestTypeValues.CHAT,
    };
  }

  if (request.prompt !== undefined || request.input !== undefined) {
    return {
      logType: RespanLogType.TEXT,
      entityName: "helicone.text",
      requestType: LLMRequestTypeValues.CHAT,
    };
  }

  const fallback = sanitizeNamePart(fallbackOperation);
  return {
    logType: RespanLogType.CHAT,
    entityName: fallback ? `helicone.${fallback}` : "helicone.chat",
    requestType: LLMRequestTypeValues.CHAT,
  };
}

function attachLlmAttributes(
  attributes: Attributes,
  request: AnyRecord,
  response: ParsedResponse,
  options: AnyRecord,
  shape: OperationShape,
  traceContent: boolean,
): void {
  const requestModel = request.model;
  const responseModel = firstDefined(
    response.record?.model,
    response.record?.response?.model,
  );
  if (requestModel !== undefined) {
    attributes[SpanAttributes.LLM_REQUEST_MODEL] = String(requestModel);
  }
  if (responseModel !== undefined) {
    attributes[SpanAttributes.LLM_RESPONSE_MODEL] = String(responseModel);
  }

  const provider = resolveProvider(
    options.provider,
    request,
    firstDefined(requestModel, responseModel),
  );
  if (provider) attributes[SpanAttributes.LLM_SYSTEM] = provider;
  if (shape.requestType) {
    attributes[SpanAttributes.LLM_REQUEST_TYPE] = shape.requestType;
  }

  attachRequestSettings(attributes, request);
  attachUsage(attributes, response.usage);

  if (!traceContent) return;

  attachToolDefinitions(attributes, request.tools ?? request.functions);

  const requestMessages = normalizeRequestMessages(request);
  if (requestMessages) {
    const systemContent = firstDefined(
      request.system,
      asRecord(request.systemInstruction).parts,
      request.systemInstruction,
    );
    const promptMessages = systemContent !== undefined
      ? [{ role: "system", content: systemContent }, ...requestMessages]
      : requestMessages;
    attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(promptMessages);
    attachPromptMessages(attributes, promptMessages);
  } else {
    const scalarPrompt = firstDefined(request.prompt, request.input);
    if (scalarPrompt !== undefined) {
      const promptMessages = [
        ...(request.system !== undefined
          ? [{ role: "system", content: request.system }]
          : []),
        { role: "user", content: scalarPrompt },
      ];
      attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(promptMessages);
      attachPromptMessages(attributes, promptMessages);
    } else {
      attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request);
    }
  }

  const output = response.message ?? response.raw ?? null;
  attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(output);
  attachCompletion(attributes, response);
}

function normalizeRequestMessages(request: AnyRecord): AnyRecord[] | undefined {
  if (Array.isArray(request.messages)) {
    return request.messages.map((value: unknown) => {
      const message = asRecord(value);
      return {
        ...message,
        role: String(message.role ?? "user"),
      };
    });
  }
  if (!Array.isArray(request.contents)) return undefined;
  return request.contents.map((value: unknown) => {
    const content = asRecord(value);
    return {
      role: content.role === "model" ? "assistant" : String(content.role ?? "user"),
      content: firstDefined(content.parts, content.content, []),
    };
  });
}

function attachRequestSettings(attributes: Attributes, request: AnyRecord): void {
  setPrimitive(attributes, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request.max_tokens ?? request.maxTokens);
  setPrimitive(attributes, SpanAttributes.LLM_REQUEST_TEMPERATURE, request.temperature);
  setPrimitive(attributes, SpanAttributes.LLM_REQUEST_TOP_P, request.top_p ?? request.topP);
  setPrimitive(attributes, SpanAttributes.LLM_TOP_K, request.top_k ?? request.topK);
  setPrimitive(
    attributes,
    SpanAttributes.LLM_FREQUENCY_PENALTY,
    request.frequency_penalty ?? request.frequencyPenalty,
  );
  setPrimitive(
    attributes,
    SpanAttributes.LLM_PRESENCE_PENALTY,
    request.presence_penalty ?? request.presencePenalty,
  );
  if (request.stop !== undefined || request.stopSequences !== undefined) {
    attributes[SpanAttributes.LLM_CHAT_STOP_SEQUENCES] = safeJson(
      request.stop ?? request.stopSequences,
    );
  }
}

function attachToolDefinitions(attributes: Attributes, value: unknown): void {
  const tools = normalizeToolDefinitions(value);
  if (tools.length > 0) {
    attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
  }
}

function normalizeToolDefinitions(value: unknown): AnyRecord[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = asRecord(item);
    if (Array.isArray(record.functionDeclarations)) {
      return normalizeToolDefinitions(record.functionDeclarations);
    }
    if (isRecord(record.function)) {
      return [{
        type: String(record.type ?? "function"),
        function: {
          name: String(record.function.name ?? ""),
          ...(record.function.description !== undefined
            ? { description: String(record.function.description) }
            : {}),
          ...(record.function.parameters !== undefined
            ? { parameters: record.function.parameters }
            : {}),
        },
      }];
    }

    if (record.name !== undefined) {
      const parameters = firstDefined(record.parameters, record.input_schema);
      return [{
        type: "function",
        function: {
          name: String(record.name),
          ...(record.description !== undefined
            ? { description: String(record.description) }
            : {}),
          ...(parameters !== undefined
            ? { parameters }
            : {}),
        },
      }];
    }
    return [];
  });
}

function normalizeToolCalls(value: unknown): AnyRecord[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = asRecord(item);
    const fn = asRecord(record.function);
    const functionCall = asRecord(record.functionCall);
    const name = firstDefined(fn.name, functionCall.name, record.name);
    if (name === undefined) return [];
    const rawArguments = firstDefined(
      fn.arguments,
      functionCall.args,
      record.input,
      record.arguments,
      {},
    );
    return [{
      ...(record.id !== undefined ? { id: String(record.id) } : {}),
      type: "function",
      function: {
        name: String(name),
        arguments: normalizeToolArguments(rawArguments),
      },
    }];
  });
}

function normalizeToolArguments(value: unknown): string {
  if (typeof value !== "string") return safeJson(value);
  const parsed = parseJson(value);
  return parsed === undefined ? redactSensitiveText(value) : safeJson(parsed);
}

function toolCallsFromContent(value: unknown): AnyRecord[] {
  if (!Array.isArray(value)) return [];
  return normalizeToolCalls(
    value.filter((item) => {
      const record = asRecord(item);
      return record.type === "tool_use" || isRecord(record.functionCall);
    }),
  );
}

function attachPromptMessages(attributes: Attributes, messages: unknown): void {
  if (!Array.isArray(messages)) return;
  messages.slice(0, MAX_INDEXED_PROMPT_MESSAGES).forEach((rawMessage, index) => {
    const message = asRecord(rawMessage);
    const prefix = `${SpanAttributes.LLM_PROMPTS}.${index}`;
    attributes[`${prefix}.role`] = String(message.role ?? "user");
    if (message.content !== undefined) {
      attributes[`${prefix}.content`] = contentString(message.content);
    }
    const explicitToolCalls = normalizeToolCalls(firstDefined(
      message.tool_calls,
      message.toolCalls,
    ));
    const toolCalls = explicitToolCalls.length > 0
      ? explicitToolCalls
      : toolCallsFromContent(message.content);
    if (toolCalls.length > 0) {
      attributes[`${prefix}.tool_calls`] = safeJson(toolCalls);
    }
  });
}

function attachCompletion(attributes: Attributes, response: ParsedResponse): void {
  const message = response.message;
  const hasOutput =
    response.text !== undefined ||
    message?.content !== undefined ||
    (response.toolCalls?.length ?? 0) > 0;
  if (!hasOutput) return;

  const prefix = `${SpanAttributes.LLM_COMPLETIONS}.0`;
  attributes[`${prefix}.role`] = String(message?.role ?? "assistant");
  attributes[`${prefix}.content`] = contentString(
    firstDefined(message?.content, response.text, ""),
  );
  if (response.toolCalls && response.toolCalls.length > 0) {
    attributes[`${prefix}.tool_calls`] = safeJson(response.toolCalls);
  }
}

function attachUsage(attributes: Attributes, usageValue: unknown): void {
  const usage = asRecord(usageValue);
  const input = integerValue(firstDefined(
    usage.input_tokens,
    usage.prompt_tokens,
    usage.inputTokens,
    usage.promptTokens,
    usage.promptTokenCount,
  ));
  const output = integerValue(firstDefined(
    usage.output_tokens,
    usage.completion_tokens,
    usage.outputTokens,
    usage.completionTokens,
    usage.candidatesTokenCount,
  ));
  const explicitTotal = integerValue(firstDefined(
    usage.total_tokens,
    usage.totalTokens,
    usage.totalTokenCount,
  ));
  const cacheRead = integerValue(firstDefined(
    usage.cache_read_input_tokens,
    usage.cacheReadInputTokens,
    usage.cache_read_tokens,
    usage.cacheReadTokens,
    usage.cachedContentTokenCount,
  ));

  if (input !== undefined) {
    attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = input;
    attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = input;
  }
  if (output !== undefined) {
    attributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = output;
    attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output;
  }
  if (cacheRead !== undefined) {
    attributes[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cacheRead;
    attributes[
      TraceloopCompatibilityFields.LLM_USAGE_CACHE_READ_INPUT_TOKENS
    ] = cacheRead;
  }
  const total = explicitTotal ??
    (input !== undefined || output !== undefined ? (input ?? 0) + (output ?? 0) : undefined);
  if (total !== undefined) {
    attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = total;
  }
}

function attachToolEvent(
  attributes: Attributes,
  request: AnyRecord,
  response: ParsedResponse,
  traceContent: boolean,
): void {
  if (!traceContent) return;
  const args = firstDefined(
    request[HeliconeFields.INPUT],
    request[HeliconeFields.ARGUMENTS],
    omitKeys(request, [HeliconeFields.EVENT_TYPE, HeliconeFields.TOOL_NAME]),
  );
  attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({
    name: String(request[HeliconeFields.TOOL_NAME] ?? "tool"),
    arguments: args ?? {},
  });
  attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(response.raw ?? null);
}

function attachVectorEvent(
  attributes: Attributes,
  request: AnyRecord,
  response: ParsedResponse,
  traceContent: boolean,
): void {
  setPrimitive(
    attributes,
    SpanAttributes.VECTOR_DB_QUERY_TOP_K,
    request[HeliconeFields.TOP_K],
  );
  setPrimitive(
    attributes,
    SpanAttributes.VECTOR_DB_TABLE_NAME,
    request[HeliconeFields.DATABASE_NAME],
  );
  if (!traceContent) return;
  attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request);
  attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(response.raw ?? null);
}

function attachDataEvent(
  attributes: Attributes,
  request: AnyRecord,
  response: ParsedResponse,
  traceContent: boolean,
): void {
  if (!traceContent) return;
  attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
    firstDefined(
      request[HeliconeFields.META],
      omitKeys(request, [HeliconeFields.EVENT_TYPE, HeliconeFields.NAME]),
    ),
  );
  attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(response.raw ?? null);
}

function attachSafeHeliconeHeaders(
  attributes: Attributes,
  ...headerSources: unknown[]
): void {
  const headers = new Map<string, string>();
  for (const source of headerSources) {
    for (const [key, value] of normalizeHeaders(source)) headers.set(key, value);
  }
  const userId = headers.get(HeliconeHeaders.USER_ID);
  const sessionId = headers.get(HeliconeHeaders.SESSION_ID);
  if (userId) {
    attributes[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID] = userId;
  }
  if (sessionId) {
    attributes[RespanSpanAttributes.RESPAN_THREADS_ID] = sessionId;
  }

  const properties: Record<string, string> = {};
  for (const [key, value] of headers.entries()) {
    if (!key.startsWith(HeliconeHeaders.PROPERTY_PREFIX)) continue;
    const propertyName = key.slice(HeliconeHeaders.PROPERTY_PREFIX.length);
    if (propertyName && value) properties[propertyName] = value;
  }
  if (Object.keys(properties).length > 0) {
    attributes[SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES] = safeJson(properties);
  }
}

function attachOperationalMetadata(
  attributes: Attributes,
  options: AnyRecord,
  fallbackOperation?: string,
  propagatedAttributes?: unknown,
): void {
  const propagated = asRecord(propagatedAttributes);
  const propagatedMetadata = asRecord(propagated.metadata);
  const status = finiteNumber(options.status);
  const timeToFirstToken = finiteNumber(options.timeToFirstToken);
  if (timeToFirstToken !== undefined) {
    attributes[ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] =
      timeToFirstToken / 1_000;
  }
  const heliconeMetadata: AnyRecord = {
    ...(status !== undefined ? { status } : {}),
    ...(timeToFirstToken !== undefined
      ? { time_to_first_token_ms: timeToFirstToken }
      : {}),
    ...(fallbackOperation ? { operation: fallbackOperation } : {}),
    ...(
      timeToFirstToken !== undefined ||
      fallbackOperation?.toLowerCase().includes("stream")
        ? { streaming: true }
        : {}
    ),
  };
  const metadata = {
    ...propagatedMetadata,
    helicone: heliconeMetadata,
  };
  attributes[RespanSpanAttributes.RESPAN_METADATA] = safeJson(metadata);

  if (propagatedAttributes !== undefined) {
    attachPropagatedSnapshot(attributes, propagated);
  }
}

function attachPropagatedSnapshot(
  attributes: Attributes,
  propagated: AnyRecord,
): void {
  for (const [key, value] of Object.entries(propagated)) {
    if (key === "metadata" || value === undefined || value === null) continue;
    const attributeKey = (RESPAN_SPAN_ATTRIBUTES_MAP as Record<string, string>)[key];
    if (!attributeKey || attributes[attributeKey] !== undefined) continue;
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      attributes[attributeKey] = value;
    } else {
      attributes[attributeKey] = safeJson(value);
    }
  }
}

function resolveStreaming(
  request: AnyRecord,
  options: AnyRecord,
  fallbackOperation?: string,
): boolean | undefined {
  if (typeof request.stream === "boolean") return request.stream;
  if (options.timeToFirstToken !== undefined) return true;
  if (fallbackOperation?.toLowerCase().includes("stream")) return true;
  return undefined;
}

function parseResponse(value: unknown): ParsedResponse {
  if (typeof value !== "string") {
    const record = asRecord(value);
    if (isGoogleResponse(record)) return aggregateGoogleChunks([record], value);
    const message = extractMessage(record);
    return {
      raw: value,
      record,
      text: extractText(record),
      message,
      toolCalls: extractToolCalls(record, message),
      usage: findUsage(record),
    };
  }

  const parsedWhole = parseJson(value);
  if (parsedWhole !== undefined) return parseResponse(parsedWhole);

  const chunks = value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.startsWith("data:") ? line.slice(5).trim() : line)
    .filter((line) => line !== "[DONE]")
    .map(parseJson)
    .filter((item): item is unknown => item !== undefined);
  if (chunks.length === 0) {
    return { raw: value, text: value };
  }
  return aggregateChunks(chunks, value);
}

function aggregateChunks(chunks: unknown[], raw: unknown): ParsedResponse {
  if (chunks.some((chunk) => isAnthropicStreamChunk(asRecord(chunk)))) {
    return aggregateAnthropicChunks(chunks, raw);
  }
  if (chunks.some((chunk) => isGoogleResponse(asRecord(chunk)))) {
    return aggregateGoogleChunks(chunks, raw);
  }

  const textParts: string[] = [];
  const toolCalls = new Map<string, AnyRecord>();
  let usage: AnyRecord | undefined;
  let role = "assistant";
  let model: unknown;

  chunks.forEach((chunkValue) => {
    const chunk = asRecord(chunkValue);
    model = chunk.model ?? model;
    usage = isRecord(chunk.usage) ? chunk.usage : usage;
    const choice = asRecord(Array.isArray(chunk.choices) ? chunk.choices[0] : undefined);
    const message = asRecord(choice.delta ?? choice.message);
    if (message.role) role = String(message.role);
    if (message.content !== undefined) textParts.push(contentString(message.content));
    mergeStreamingToolCalls(toolCalls, message.tool_calls ?? message.toolCalls);
  });

  const mergedTools = normalizeToolCalls([...toolCalls.values()]);
  const text = textParts.join("");
  const message: AnyRecord = { role, content: text };
  if (mergedTools.length > 0) message.tool_calls = mergedTools;
  return {
    raw,
    record: { model, usage, choices: [{ message }] },
    text,
    message,
    toolCalls: mergedTools,
    usage,
  };
}

function isAnthropicStreamChunk(chunk: AnyRecord): boolean {
  return typeof chunk.type === "string" && (
    chunk.type.startsWith("message_") ||
    chunk.type.startsWith("content_block_")
  );
}

function aggregateAnthropicChunks(
  chunks: unknown[],
  raw: unknown,
): ParsedResponse {
  const blocks = new Map<number, AnyRecord>();
  const partialToolInputs = new Map<number, string>();
  let role = "assistant";
  let model: unknown;
  let usage: AnyRecord = {};

  for (const chunkValue of chunks) {
    const chunk = asRecord(chunkValue);
    if (chunk.type === "message_start") {
      const message = asRecord(chunk.message);
      role = String(message.role ?? role);
      model = firstDefined(message.model, model);
      usage = mergeUsage(usage, message.usage);
      if (Array.isArray(message.content)) {
        message.content.forEach((block: unknown, index: number) => {
          blocks.set(index, { ...asRecord(block) });
        });
      }
      continue;
    }
    if (chunk.type === "content_block_start") {
      const index = integerValue(chunk.index) ?? blocks.size;
      const block = { ...asRecord(chunk.content_block) };
      blocks.set(index, block);
      if (block.type === "tool_use" && block.input !== undefined) {
        if (typeof block.input === "string" && block.input) {
          partialToolInputs.set(index, block.input);
        } else if (isRecord(block.input) && Object.keys(block.input).length > 0) {
          partialToolInputs.set(index, safeJson(block.input));
        }
      }
      continue;
    }
    if (chunk.type === "content_block_delta") {
      const index = integerValue(chunk.index) ?? 0;
      const delta = asRecord(chunk.delta);
      const block = blocks.get(index) ?? {
        type: delta.type === "input_json_delta" ? "tool_use" : "text",
      };
      if (delta.type === "text_delta" || delta.text !== undefined) {
        block.text = String(block.text ?? "") + String(delta.text ?? "");
      }
      if (delta.type === "input_json_delta" || delta.partial_json !== undefined) {
        partialToolInputs.set(
          index,
          (partialToolInputs.get(index) ?? "") + String(delta.partial_json ?? ""),
        );
      }
      blocks.set(index, block);
      continue;
    }
    if (chunk.type === "message_delta") {
      usage = mergeUsage(usage, chunk.usage);
    }
  }

  const content = [...blocks.entries()]
    .sort(([left], [right]) => left - right)
    .map(([index, rawBlock]) => {
      const block = { ...rawBlock };
      if (block.type === "tool_use" && partialToolInputs.has(index)) {
        const rawInput = partialToolInputs.get(index) ?? "";
        block.input = parseJson(rawInput) ?? rawInput;
      }
      return block;
    });
  const message: AnyRecord = { role, content };
  const toolCalls = toolCallsFromContent(content);
  return {
    raw,
    record: { model, usage, role, content },
    text: contentString(content),
    message,
    toolCalls,
    usage,
  };
}

function isGoogleResponse(record: AnyRecord): boolean {
  return Array.isArray(record.candidates) ||
    isRecord(record.usageMetadata) ||
    record.modelVersion !== undefined;
}

function aggregateGoogleChunks(
  chunks: unknown[],
  raw: unknown,
): ParsedResponse {
  const parts: AnyRecord[] = [];
  let role = "assistant";
  let model: unknown;
  let usage: AnyRecord = {};

  for (const chunkValue of chunks) {
    const chunk = asRecord(chunkValue);
    model = firstDefined(chunk.modelVersion, chunk.model, model);
    usage = mergeUsage(usage, chunk.usageMetadata ?? chunk.usage);
    const candidate = asRecord(
      Array.isArray(chunk.candidates) ? chunk.candidates[0] : undefined,
    );
    const content = asRecord(candidate.content);
    if (content.role) role = content.role === "model" ? "assistant" : String(content.role);
    if (!Array.isArray(content.parts)) continue;
    for (const rawPart of content.parts) {
      const part = { ...asRecord(rawPart) };
      const previous = parts.at(-1);
      if (
        typeof part.text === "string" &&
        previous &&
        typeof previous.text === "string" &&
        Object.keys(part).length === 1 &&
        Object.keys(previous).length === 1
      ) {
        previous.text += part.text;
      } else {
        parts.push(part);
      }
    }
  }

  const message: AnyRecord = { role, content: parts };
  const toolCalls = toolCallsFromContent(parts);
  return {
    raw,
    record: { model, usage, candidates: [{ content: { role, parts } }] },
    text: contentString(parts),
    message,
    toolCalls,
    usage,
  };
}

function mergeUsage(current: AnyRecord, next: unknown): AnyRecord {
  return isRecord(next) ? { ...current, ...next } : current;
}

function mergeStreamingToolCalls(target: Map<string, AnyRecord>, value: unknown): void {
  if (!Array.isArray(value)) return;
  value.forEach((rawCall, index) => {
    const call = asRecord(rawCall);
    const fn = asRecord(call.function);
    const key = String(call.index ?? call.id ?? index);
    const existing = target.get(key) ?? {
      ...(call.id !== undefined ? { id: String(call.id) } : {}),
      type: String(call.type ?? "function"),
      function: { name: "", arguments: "" },
    };
    if (call.id !== undefined) existing.id = String(call.id);
    if (fn.name !== undefined) existing.function.name += String(fn.name);
    if (fn.arguments !== undefined) existing.function.arguments += String(fn.arguments);
    target.set(key, existing);
  });
}

function extractMessage(record: AnyRecord): AnyRecord | undefined {
  const choice = asRecord(Array.isArray(record.choices) ? record.choices[0] : undefined);
  const message = asRecord(firstDefined(
    choice.message,
    record.message,
    record.output,
    record.role !== undefined || record.content !== undefined ? record : undefined,
  ));
  if (Object.keys(message).length > 0) return message;
  if (choice.text !== undefined || record.text !== undefined) {
    return {
      role: "assistant",
      content: String(choice.text ?? record.text),
    };
  }
  return undefined;
}

function extractText(record: AnyRecord): string | undefined {
  const message = extractMessage(record);
  const value = firstDefined(
    message?.content,
    asRecord(Array.isArray(record.choices) ? record.choices[0] : undefined).text,
    record.text,
  );
  return value === undefined ? undefined : contentString(value);
}

function extractToolCalls(record: AnyRecord, message?: AnyRecord): AnyRecord[] {
  const explicitToolCalls = normalizeToolCalls(firstDefined(
    message?.tool_calls,
    message?.toolCalls,
    record.tool_calls,
    record.toolCalls,
  ));
  return explicitToolCalls.length > 0
    ? explicitToolCalls
    : toolCallsFromContent(message?.content);
}

function findUsage(record: AnyRecord): AnyRecord | undefined {
  const candidates = [
    record.usage,
    record.metrics?.usage,
    record.response?.usage,
  ];
  return candidates.find(isRecord) as AnyRecord | undefined;
}

function resolveProvider(
  provider: unknown,
  request: AnyRecord,
  model: unknown,
): string | undefined {
  const explicit = firstDefined(provider, request.provider);
  if (explicit !== undefined) return normalizeProvider(String(explicit));
  const modelName = String(model ?? "").toLowerCase();
  if (modelName.includes("claude")) return "anthropic";
  if (modelName.includes("gemini")) return "google";
  if (modelName.includes("command")) return "cohere";
  if (modelName.includes("gpt") || /^o[134](?:-|$)/.test(modelName)) return "openai";
  return undefined;
}

function resolveErrorMessage(
  error: unknown,
  response: ParsedResponse,
  statusValue: unknown,
): string | undefined {
  if (error !== undefined && error !== null) return errorMessage(error);
  const status = Number(statusValue);
  if (Number.isFinite(status) && (status >= 400 || status < 0)) {
    const responseError = firstDefined(
      response.record?.error?.message,
      response.record?.error,
      response.record?.message,
    );
    if (responseError !== undefined) return errorMessage(responseError);
    if (response.text) return response.text.split(/\r?\n/, 1)[0];
    if (status < 0) return "Helicone operation cancelled";
    return `Helicone operation failed with status ${status}`;
  }
  return undefined;
}

function errorType(error: unknown, status: unknown): string {
  if (error instanceof Error && error.name) return error.name;
  if (error !== undefined && error !== null) return typeof error;
  const numericStatus = Number(status);
  if (numericStatus < 0) return "cancelled";
  if (Number.isFinite(numericStatus)) return `status_${numericStatus}`;
  return "error";
}

function errorStatusCode(status: unknown): number {
  const numericStatus = Number(status);
  if (Number.isFinite(numericStatus) && numericStatus >= 400) {
    return Math.trunc(numericStatus);
  }
  if (Number.isFinite(numericStatus) && numericStatus < 0) return 499;
  return 500;
}

function removeCompletionAttributes(attributes: Attributes): void {
  const prefix = `${SpanAttributes.LLM_COMPLETIONS}.0.`;
  for (const key of Object.keys(attributes)) {
    if (key.startsWith(prefix)) delete attributes[key];
  }
}

function resolveTimes(options: AnyRecord): {
  startTimeIso: string;
  endTimeIso: string;
} {
  const now = Date.now();
  const start = finiteNumber(options.startTime) ?? now;
  const end = Math.max(start, finiteNumber(options.endTime) ?? now);
  return {
    startTimeIso: new Date(start).toISOString(),
    endTimeIso: new Date(end).toISOString(),
  };
}

function normalizeHeaders(value: unknown): Map<string, string> {
  const out = new Map<string, string>();
  if (value instanceof Headers) {
    value.forEach((headerValue, key) => out.set(key.toLowerCase(), headerValue));
    return out;
  }
  if (!isRecord(value)) return out;
  Object.entries(value).forEach(([key, headerValue]) => {
    if (headerValue !== undefined && headerValue !== null) {
      out.set(key.toLowerCase(), String(headerValue));
    }
  });
  return out;
}

function contentString(value: unknown): string {
  if (typeof value === "string") {
    const parsed = parseJson(value);
    return isRecord(parsed) || Array.isArray(parsed) ? safeJson(parsed) : value;
  }
  // The span contract requires every structured content value to remain JSON,
  // including arrays made only of text blocks; concatenating loses boundaries.
  return value === undefined || value === null ? "" : safeJson(value);
}

function errorMessage(value: unknown): string {
  if (value instanceof Error) return value.message;
  if (typeof value === "string") return value;
  if (isRecord(value) && value.message !== undefined) return String(value.message);
  return safeJson(value);
}

function redactSensitiveText(value: string): string {
  const parsed = parseJson(value);
  if (isRecord(parsed) || Array.isArray(parsed)) {
    return safeJson(parsed).slice(0, MAX_ERROR_MESSAGE_CHARS);
  }
  return value
    .replace(
      /((?:"|')?(?:authorization|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|bearer[_ -]?token|id[_ -]?token|session[_ -]?token|private[_ -]?key|client[_ -]?secret|credential|credentials|helicone[_ -]?auth|token|password|secret|cookie)(?:"|')?\s*[:=]\s*)(?:(?:bearer|basic)\s+)?(?:"[^"]*"|'[^']*'|[^\s,;]+)/gi,
      "$1[REDACTED]",
    )
    .replace(/\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer [REDACTED]")
    .replace(/\bsk-(?:proj-)?[A-Za-z0-9_-]{8,}\b/g, "[REDACTED]")
    .slice(0, MAX_ERROR_MESSAGE_CHARS);
}

function normalizeProvider(value: string): string | undefined {
  const normalized = value.toLowerCase().trim().replace(/^@/, "");
  return normalized.replace(/[^a-z0-9._-]+/g, "_") || undefined;
}

function sanitizeNamePart(value: unknown): string {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function setPrimitive(attributes: Attributes, key: string, value: unknown): void {
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    attributes[key] = value;
  }
}

function integerValue(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : undefined;
}

function finiteNumber(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function parseJson(value: string): unknown | undefined {
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

export function safeJson(value: unknown): string {
  const sanitized = sanitizeForSerialization(value);
  let serialized: string;
  try {
    serialized = JSON.stringify(sanitized);
  } catch {
    serialized = JSON.stringify(String(sanitized));
  }
  const bytes = new TextEncoder().encode(serialized).byteLength;
  if (bytes <= MAX_SERIALIZED_BYTES) return serialized;
  return JSON.stringify({
    truncated: true,
    original_bytes: bytes,
    preview: serialized.slice(0, Math.floor(MAX_SERIALIZED_BYTES / 2)),
  });
}

function sanitizeForSerialization(value: unknown): unknown {
  const seen = new WeakSet<object>();
  let visited = 0;

  const visit = (current: unknown, depth: number, key?: string): unknown => {
    visited += 1;
    if (visited > 50_000 || depth > 32) return "[Truncated]";
    if (key && isSensitiveKey(key)) return "[REDACTED]";
    if (typeof current === "string") {
      const parsed = parseJson(current);
      if (isRecord(parsed) || Array.isArray(parsed)) {
        try {
          return JSON.stringify(visit(parsed, depth + 1));
        } catch {
          return "[Truncated]";
        }
      }
      return current;
    }
    if (
      current === null ||
      typeof current === "number" ||
      typeof current === "boolean"
    ) return current;
    if (typeof current === "bigint") return current.toString();
    if (current === undefined) return null;
    if (current instanceof Date) return current.toISOString();
    if (current instanceof Error) {
      return {
        name: current.name,
        message: redactSensitiveText(current.message),
      };
    }
    if (ArrayBuffer.isView(current)) {
      return Array.from(current as unknown as ArrayLike<number>);
    }
    if (Array.isArray(current)) {
      return current.map((item) => visit(item, depth + 1));
    }
    if (typeof current !== "object") return String(current);
    if (seen.has(current)) return "[Circular]";
    seen.add(current);
    return Object.fromEntries(
      Object.entries(current as AnyRecord).map(([childKey, childValue]) => [
        childKey,
        visit(childValue, depth + 1, childKey),
      ]),
    );
  };

  return visit(value, 0);
}

function isSensitiveKey(key: string): boolean {
  const normalized = key
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (normalized === "token" || normalized.endsWith("_token")) return true;
  return /(?:^|_)(?:authorization|api_key|access_token|refresh_token|auth_token|bearer|bearer_token|id_token|session_token|private_key|client_secret|credential|credentials|helicone_auth|password|secret|cookie)(?:_|$)/.test(
    normalized,
  );
}

function omitKeys(record: AnyRecord, keys: string[]): AnyRecord {
  const omitted = new Set(keys);
  return Object.fromEntries(
    Object.entries(record).filter(([key]) => !omitted.has(key)),
  );
}

function firstDefined(...values: unknown[]): any {
  return values.find((value) => value !== undefined && value !== null);
}

function stringValue(value: unknown): string | undefined {
  return value === undefined || value === null ? undefined : String(value);
}

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asRecord(value: unknown): AnyRecord {
  return isRecord(value) ? value : {};
}
