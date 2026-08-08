import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  buildReadableSpan,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes as TraceloopSpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  COHERE_SYSTEM,
  INSTRUMENTATION_NAME,
  MESSAGE_CONTENT_SUFFIX,
  MESSAGE_ROLE_SUFFIX,
  MESSAGE_TOOL_CALL_ID_SUFFIX,
  MESSAGE_TOOL_CALLS_SUFFIX,
  PACKAGE_VERSION,
  RESPAN_LOG_METHOD_TS_TRACING,
} from "./_constants.js";
import {
  logTypeForOperation,
  normalizeCohereAttrs,
  requestTypeForOperation,
} from "./_translator.js";
import {
  activeSpanContext,
  cohereContentToString,
  errorMessage,
  isRecord,
  normalizeRole,
  safeJson,
  setDefault,
  setIfPresent,
  type SpanAttributes,
} from "./_utils.js";

const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";

export type CohereOperation =
  | "chat"
  | "chatStream"
  | "generate"
  | "generateStream"
  | "embed"
  | "rerank";

export type CohereApiVersion = "v1" | "v2";

export interface OperationConfig {
  operation: CohereOperation;
  apiVersion: CohereApiVersion;
  streaming: boolean;
}

export interface SpanRecord {
  activeTraceId?: string;
  activeSpanId?: string;
  startTime: [number, number];
  attrs: SpanAttributes;
}

export interface StreamState {
  events: unknown[];
  textParts: string[];
  toolCallParts: unknown[];
  finalResponse?: unknown;
}

function cohereSpanName(operation: CohereOperation): string {
  if (operation === "chatStream") return "cohere.chat";
  if (operation === "generateStream") return "cohere.generate";
  return `cohere.${operation}`;
}

function modelFromRequest(request: any): unknown {
  return request?.model;
}

function setRequestParams(attrs: SpanAttributes, request: any): void {
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_REQUEST_MAX_TOKENS, request?.maxTokens);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_REQUEST_TEMPERATURE, request?.temperature);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_REQUEST_TOP_P, request?.p ?? request?.topP);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_TOP_K, request?.k);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_FREQUENCY_PENALTY, request?.frequencyPenalty);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_PRESENCE_PENALTY, request?.presencePenalty);
}

function v1ToolToFunction(tool: any): Record<string, any> | undefined {
  if (!isRecord(tool) || tool.name === undefined) return undefined;

  const properties: Record<string, any> = {};
  const required: string[] = [];
  if (isRecord(tool.parameterDefinitions)) {
    for (const [name, definition] of Object.entries(tool.parameterDefinitions)) {
      if (!isRecord(definition)) continue;
      const property: Record<string, any> = {};
      setIfPresent(property, "type", definition.type);
      setIfPresent(property, "description", definition.description);
      properties[name] = property;
      if (definition.required === true) required.push(name);
    }
  }

  const parameters =
    Object.keys(properties).length > 0
      ? { type: "object", properties, ...(required.length ? { required } : {}) }
      : undefined;

  return {
    type: "function",
    function: {
      name: String(tool.name),
      ...(tool.description ? { description: String(tool.description) } : {}),
      ...(parameters ? { parameters } : {}),
    },
  };
}

function normalizeToolDefinition(tool: any): Record<string, any> | undefined {
  if (!isRecord(tool)) return undefined;
  if (tool.type === "function" && isRecord(tool.function)) {
    return {
      type: "function",
      function: {
        name: String(tool.function.name ?? ""),
        ...(tool.function.description
          ? { description: String(tool.function.description) }
          : {}),
        ...(tool.function.parameters
          ? { parameters: tool.function.parameters }
          : {}),
      },
    };
  }
  return v1ToolToFunction(tool);
}

function normalizeToolDefinitions(tools: unknown): Record<string, any>[] {
  if (!Array.isArray(tools)) return [];
  return tools
    .map(normalizeToolDefinition)
    .filter((tool): tool is Record<string, any> => tool !== undefined);
}

function normalizeToolCall(call: any): Record<string, any> | undefined {
  if (!isRecord(call)) return undefined;

  if (isRecord(call.function)) {
    return {
      ...(call.id ? { id: String(call.id) } : {}),
      type: String(call.type ?? "function"),
      function: {
        ...(call.function.name ? { name: String(call.function.name) } : {}),
        ...(call.function.arguments !== undefined
          ? {
              arguments:
                typeof call.function.arguments === "string"
                  ? call.function.arguments
                  : safeJson(call.function.arguments),
            }
          : {}),
      },
    };
  }

  if (call.name !== undefined) {
    return {
      type: "function",
      function: {
        name: String(call.name),
        arguments:
          call.parameters !== undefined
            ? safeJson(call.parameters)
            : call.arguments !== undefined
              ? safeJson(call.arguments)
              : "{}",
      },
    };
  }

  return undefined;
}

function setIndexedMessages(
  attrs: SpanAttributes,
  prefix: string,
  messages: Record<string, any>[],
): void {
  messages.forEach((message, index) => {
    const attrPrefix = `${prefix}.${index}`;
    attrs[`${attrPrefix}.${MESSAGE_ROLE_SUFFIX}`] = normalizeRole(message.role);
    attrs[`${attrPrefix}.${MESSAGE_CONTENT_SUFFIX}`] = cohereContentToString(
      message.content,
    );
    if (message.tool_calls !== undefined) {
      attrs[`${attrPrefix}.${MESSAGE_TOOL_CALLS_SUFFIX}`] = safeJson(
        message.tool_calls,
      );
    }
    if (message.tool_call_id !== undefined) {
      attrs[`${attrPrefix}.${MESSAGE_TOOL_CALL_ID_SUFFIX}`] = String(
        message.tool_call_id,
      );
    }
  });
}

function v1ChatInputMessages(request: any): Record<string, any>[] {
  const messages: Record<string, any>[] = [];
  if (Array.isArray(request?.chatHistory)) {
    for (const message of request.chatHistory) {
      if (!isRecord(message)) continue;
      messages.push({
        role: message.role,
        content: message.message ?? message.content ?? "",
        tool_call_id: message.toolCallId ?? message.tool_call_id,
      });
    }
  }
  if (request?.message !== undefined) {
    messages.push({ role: "user", content: request.message });
  }
  return messages;
}

function v2InputMessages(request: any): Record<string, any>[] {
  if (!Array.isArray(request?.messages)) return [];
  return request.messages
    .filter(isRecord)
    .map((message) => ({
      role: message.role,
      content: message.content,
      tool_calls: message.toolCalls ?? message.tool_calls,
      tool_call_id: message.toolCallId ?? message.tool_call_id,
    }));
}

function buildInputValue(config: OperationConfig, request: any): unknown {
  if (config.operation === "chat" || config.operation === "chatStream") {
    return config.apiVersion === "v2"
      ? v2InputMessages(request)
      : v1ChatInputMessages(request);
  }
  if (config.operation === "generate" || config.operation === "generateStream") {
    return request?.prompt;
  }
  if (config.operation === "embed") {
    return request?.texts ?? request?.images ?? request;
  }
  if (config.operation === "rerank") {
    return {
      query: request?.query,
      documents: request?.documents,
      topN: request?.topN,
    };
  }
  return request;
}

function setPromptAttrs(
  attrs: SpanAttributes,
  config: OperationConfig,
  request: any,
): void {
  if (config.operation === "chat" || config.operation === "chatStream") {
    setIndexedMessages(
      attrs,
      TraceloopSpanAttributes.LLM_PROMPTS,
      config.apiVersion === "v2" ? v2InputMessages(request) : v1ChatInputMessages(request),
    );
    return;
  }

  if (
    (config.operation === "generate" || config.operation === "generateStream") &&
    request?.prompt !== undefined
  ) {
    attrs[`${TraceloopSpanAttributes.LLM_PROMPTS}.0.${MESSAGE_ROLE_SUFFIX}`] = "user";
    attrs[`${TraceloopSpanAttributes.LLM_PROMPTS}.0.${MESSAGE_CONTENT_SUFFIX}`] =
      cohereContentToString(request.prompt);
  }

  if (config.operation === "rerank" && request?.query !== undefined) {
    attrs[`${TraceloopSpanAttributes.LLM_PROMPTS}.0.${MESSAGE_ROLE_SUFFIX}`] = "user";
    attrs[`${TraceloopSpanAttributes.LLM_PROMPTS}.0.${MESSAGE_CONTENT_SUFFIX}`] =
      cohereContentToString(request.query);
  }
}

export function buildStartAttributes(
  config: OperationConfig,
  request: any,
  traceContent: boolean,
): SpanAttributes {
  const entityName = cohereSpanName(config.operation);
  const attrs: SpanAttributes = {
    [TraceloopSpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [TraceloopSpanAttributes.TRACELOOP_ENTITY_PATH]: entityName,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logTypeForOperation(config.operation),
    [TraceloopSpanAttributes.LLM_SYSTEM]: COHERE_SYSTEM,
    [TraceloopSpanAttributes.LLM_REQUEST_TYPE]: requestTypeForOperation(config.operation),
  };

  setIfPresent(attrs, TraceloopSpanAttributes.LLM_REQUEST_MODEL, modelFromRequest(request));
  setRequestParams(attrs, request);

  if (traceContent) {
    setDefault(attrs, TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT, safeJson(buildInputValue(config, request)));
    setPromptAttrs(attrs, config, request);
    const tools = normalizeToolDefinitions(request?.tools);
    if (tools.length > 0) {
      attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
    }
  }

  return normalizeCohereAttrs(attrs, config.operation);
}

function extractUsage(value: any): {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
} {
  const inputTokens =
    value?.usage?.tokens?.inputTokens ??
    value?.usage?.billedUnits?.inputTokens ??
    value?.meta?.tokens?.inputTokens ??
    value?.meta?.billedUnits?.inputTokens ??
    value?.token_count?.prompt_tokens;
  const outputTokens =
    value?.usage?.tokens?.outputTokens ??
    value?.usage?.billedUnits?.outputTokens ??
    value?.meta?.tokens?.outputTokens ??
    value?.meta?.billedUnits?.outputTokens ??
    value?.token_count?.response_tokens;
  const totalTokens =
    value?.token_count?.total_tokens ??
    (typeof inputTokens === "number" && typeof outputTokens === "number"
      ? inputTokens + outputTokens
      : undefined);

  return { inputTokens, outputTokens, totalTokens };
}

function setUsage(attrs: SpanAttributes, value: any): void {
  const usage = extractUsage(value);
  setIfPresent(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, usage.inputTokens);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.inputTokens);
  setIfPresent(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, usage.outputTokens);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.outputTokens);
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.totalTokens);
  setIfPresent(
    attrs,
    LLM_USAGE_CACHE_READ_INPUT_TOKENS,
    value?.usage?.cachedTokens ?? value?.meta?.cachedTokens,
  );
}

function setRerankUsage(attrs: SpanAttributes, value: any): void {
  const searchUnits =
    value?.usage?.billedUnits?.searchUnits ??
    value?.meta?.billedUnits?.searchUnits;
  setIfPresent(attrs, TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS, searchUnits);
}

function chatOutputMessage(config: OperationConfig, result: any): Record<string, any> {
  if (config.apiVersion === "v2") {
    const message = result?.message ?? {};
    return {
      role: message.role ?? "assistant",
      content: message.content,
      tool_calls: Array.isArray(message.toolCalls)
        ? message.toolCalls.map(normalizeToolCall).filter(Boolean)
        : undefined,
    };
  }

  return {
    role: "assistant",
    content: result?.text,
    tool_calls: Array.isArray(result?.toolCalls)
      ? result.toolCalls.map(normalizeToolCall).filter(Boolean)
      : undefined,
  };
}

function generationOutput(result: any): string {
  return cohereContentToString(result?.generations?.[0]?.text ?? result?.text ?? "");
}

function embeddingOutput(result: any): unknown {
  return result?.embeddings ?? result;
}

function setCompletionAttrs(
  attrs: SpanAttributes,
  config: OperationConfig,
  result: any,
): void {
  if (config.operation === "chat" || config.operation === "chatStream") {
    const message = chatOutputMessage(config, result);
    setIndexedMessages(attrs, TraceloopSpanAttributes.LLM_COMPLETIONS, [message]);
    attrs[TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([message]);
    return;
  }

  if (config.operation === "generate" || config.operation === "generateStream") {
    const content = generationOutput(result);
    attrs[`${TraceloopSpanAttributes.LLM_COMPLETIONS}.0.${MESSAGE_ROLE_SUFFIX}`] = "assistant";
    attrs[`${TraceloopSpanAttributes.LLM_COMPLETIONS}.0.${MESSAGE_CONTENT_SUFFIX}`] = content;
    attrs[TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(content);
    return;
  }

  if (config.operation === "embed") {
    attrs[TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(embeddingOutput(result));
    return;
  }

  if (config.operation === "rerank") {
    attrs[TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(result?.results ?? result);
  }
}

export function applySuccessAttributes(
  attrs: SpanAttributes,
  config: OperationConfig,
  result: any,
): SpanAttributes {
  if (config.operation === "rerank") {
    setRerankUsage(attrs, result);
  } else {
    setUsage(attrs, result);
  }
  setCompletionAttrs(attrs, config, result);
  return normalizeCohereAttrs(attrs, config.operation);
}

export function startSpanRecord(
  config: OperationConfig,
  request: any,
  traceContent: boolean,
): SpanRecord {
  const active = activeSpanContext();
  return {
    activeTraceId: active?.traceId,
    activeSpanId: active?.spanId,
    startTime: hrTime(),
    attrs: buildStartAttributes(config, request, traceContent),
  };
}

export function emitSpanRecord(
  config: OperationConfig,
  record: SpanRecord,
  result: any,
  error?: unknown,
): void {
  const attrs = { ...record.attrs };
  if (error === undefined) {
    applySuccessAttributes(attrs, config, result);
  } else {
    attrs.status_code = 500;
    attrs["error.message"] = errorMessage(error);
    normalizeCohereAttrs(attrs, config.operation);
  }

  const span = buildReadableSpan({
    name: cohereSpanName(config.operation),
    traceId: record.activeTraceId,
    parentId: record.activeSpanId,
    startTimeHr: record.startTime,
    endTimeHr: hrTime(),
    attributes: attrs,
    statusCode: error === undefined ? 200 : 500,
    errorMessage: error === undefined ? undefined : errorMessage(error),
    mergePropagated: true,
  }) as ReadableSpan & {
    instrumentationScope?: { name: string; version?: string };
  };

  span.instrumentationScope = {
    name: INSTRUMENTATION_NAME,
    version: PACKAGE_VERSION,
  };
  injectSpan(span);
}

export function createStreamState(): StreamState {
  return {
    events: [],
    textParts: [],
    toolCallParts: [],
  };
}

export function captureStreamEvent(state: StreamState, event: any): void {
  state.events.push(event);

  if (event?.eventType === "stream-end" && event.response !== undefined) {
    state.finalResponse = event.response;
  }
  if (event?.eventType === "text-generation" && typeof event.text === "string") {
    state.textParts.push(event.text);
  }
  if (event?.eventType === "tool-calls-generation" && event.toolCalls !== undefined) {
    state.toolCallParts.push(event.toolCalls);
  }

  if (event?.type === "content-delta") {
    const text =
      event.delta?.message?.content?.text ??
      event.delta?.message?.content ??
      event.delta?.text;
    if (typeof text === "string") state.textParts.push(text);
  }
  if (event?.type === "tool-call-delta" || event?.type === "tool-call-start") {
    state.toolCallParts.push(event.delta ?? event);
  }
  if (event?.type === "message-end") {
    state.finalResponse = event.delta ?? event;
  }
}

export function streamResultFromState(
  config: OperationConfig,
  state: StreamState,
): unknown {
  if (state.finalResponse !== undefined) {
    if (
      config.apiVersion === "v2" &&
      isRecord(state.finalResponse) &&
      state.finalResponse.message === undefined &&
      (state.textParts.length > 0 || state.toolCallParts.length > 0)
    ) {
      return {
        ...state.finalResponse,
        message: {
          role: "assistant",
          content: state.textParts.length
            ? [{ type: "text", text: state.textParts.join("") }]
            : undefined,
          toolCalls: state.toolCallParts.length ? state.toolCallParts : undefined,
        },
      };
    }
    return state.finalResponse;
  }

  if (config.operation === "generateStream") {
    return {
      generations: [{ text: state.textParts.join("") }],
    };
  }

  if (config.apiVersion === "v2") {
    return {
      message: {
        role: "assistant",
        content: state.textParts.length
          ? [{ type: "text", text: state.textParts.join("") }]
          : undefined,
        toolCalls: state.toolCallParts.length ? state.toolCallParts : undefined,
      },
    };
  }

  return {
    text: state.textParts.join(""),
    toolCalls: state.toolCallParts.length ? state.toolCallParts : undefined,
    events: state.events,
  };
}
