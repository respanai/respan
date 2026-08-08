import { context, trace, TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  RESPAN_LOG_METHOD_TS_TRACING,
  STATUS_CODE_ATTR,
  TOGETHER_SYSTEM_NAME,
} from "./_constants.js";
import {
  applyTokenUsage,
  extractToolExecutions,
  formatChatOutputMessage,
  formatInputMessages,
  formatTextCompletion,
  formatTextPrompt,
  formatTools,
  resolveErrorMessage,
  resolveModel,
  resolveStatusCode,
  safeJson,
  stringifyStructured,
  summarizeRequestBody,
  toSerializableValue,
  type ToolExecution,
} from "./_helpers.js";
import type { TogetherOperationSpec } from "./_types.js";

const GEN_AI_COMPLETION_0_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const GEN_AI_COMPLETION_0_ROLE = `${GEN_AI_COMPLETION_0_PREFIX}.role`;
const GEN_AI_COMPLETION_0_CONTENT = `${GEN_AI_COMPLETION_0_PREFIX}.content`;
const GEN_AI_COMPLETION_0_TOOL_CALLS = `${GEN_AI_COMPLETION_0_PREFIX}.tool_calls`;

const TOKEN_USAGE_KEYS = {
  inputTokens: ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  outputTokens: ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  promptTokens: ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  completionTokens: ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  totalTokens: SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
};

function buildInstrumentedReadableSpan(opts: {
  name: string;
  startTime: [number, number];
  endTime: [number, number];
  attributes: Record<string, any>;
  statusCode?: number;
  errorMessage?: string;
}): ReadableSpan {
  const activeSpanContext = trace.getSpan(context.active())?.spanContext();
  const span = buildReadableSpan({
    name: opts.name,
    traceId: activeSpanContext?.traceId,
    parentId: activeSpanContext?.spanId,
    startTimeHr: opts.startTime,
    endTimeHr: opts.endTime,
    attributes: opts.attributes,
    statusCode: opts.statusCode,
    errorMessage: opts.errorMessage,
  }) as ReadableSpan & {
    instrumentationScope?: { name: string; version?: string };
    spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
  };

  const originalSpanContext = span.spanContext.bind(span);
  span.spanContext = () => ({
    ...originalSpanContext(),
    traceFlags: activeSpanContext?.traceFlags ?? TraceFlags.SAMPLED,
  });
  span.instrumentationScope = {
    name: INSTRUMENTATION_LIBRARY_NAME,
    version: PACKAGE_VERSION,
  };
  return span;
}

function setIfDefined(attrs: Record<string, any>, key: string, value: unknown): void {
  if (value === undefined || value === null || value === "") return;
  attrs[key] = value;
}

function setRequestOptionAttrs(attrs: Record<string, any>, request: any): void {
  setIfDefined(attrs, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request?.max_tokens);
  setIfDefined(attrs, SpanAttributes.LLM_REQUEST_TEMPERATURE, request?.temperature);
  setIfDefined(attrs, SpanAttributes.LLM_REQUEST_TOP_P, request?.top_p);
  setIfDefined(attrs, SpanAttributes.LLM_FREQUENCY_PENALTY, request?.frequency_penalty);
  setIfDefined(attrs, SpanAttributes.LLM_PRESENCE_PENALTY, request?.presence_penalty);
  if (Array.isArray(request?.stop) && request.stop.length > 0) {
    attrs[SpanAttributes.LLM_CHAT_STOP_SEQUENCES] = request.stop.map(String);
  }
}

function setPromptAttrs(attrs: Record<string, any>, messages: Record<string, any>[]): void {
  messages.forEach((message, index) => {
    const prefix = `${ATTR_GEN_AI_PROMPT}.${index}`;
    setIfDefined(attrs, `${prefix}.role`, message.role);
    setIfDefined(attrs, `${prefix}.content`, stringifyStructured(message.content ?? ""));
    if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      attrs[`${prefix}.tool_calls`] = safeJson(message.tool_calls);
    }
  });
}

function buildBaseAttrs(
  spec: TogetherOperationSpec,
  request: any,
  response?: any,
): Record<string, any> {
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: spec.spanName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: spec.spanName,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: spec.logType,
    [SpanAttributes.LLM_SYSTEM]: TOGETHER_SYSTEM_NAME,
    [SpanAttributes.LLM_REQUEST_TYPE]: spec.requestType,
  };

  const model = resolveModel(request, response);
  if (model) attrs[SpanAttributes.LLM_REQUEST_MODEL] = model;
  setRequestOptionAttrs(attrs, request);

  if (spec.kind === "chat") {
    const messages = formatInputMessages(request?.messages);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(messages);
    setPromptAttrs(attrs, messages);

    const tools = formatTools(request?.tools);
    if (tools.length > 0) {
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
    }
    return attrs;
  }

  if (spec.kind === "text") {
    const promptMessages = formatTextPrompt(request?.prompt);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(promptMessages);
    setPromptAttrs(attrs, promptMessages);
    return attrs;
  }

  if (spec.kind === "embedding") {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request?.input);
    return attrs;
  }

  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(summarizeRequestBody(request));
  return attrs;
}

function setChatSuccessAttrs(attrs: Record<string, any>, response: any): void {
  const outputMessage = formatChatOutputMessage(response);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([outputMessage]);
  attrs[GEN_AI_COMPLETION_0_ROLE] = outputMessage.role ?? "assistant";
  attrs[GEN_AI_COMPLETION_0_CONTENT] = stringifyStructured(outputMessage.content ?? "");
  if (Array.isArray(outputMessage.tool_calls) && outputMessage.tool_calls.length > 0) {
    attrs[GEN_AI_COMPLETION_0_TOOL_CALLS] = safeJson(outputMessage.tool_calls);
  }
  applyTokenUsage(attrs, response?.usage, TOKEN_USAGE_KEYS);
}

function setTextSuccessAttrs(attrs: Record<string, any>, response: any): void {
  const content = formatTextCompletion(response);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([
    { role: "assistant", content },
  ]);
  attrs[GEN_AI_COMPLETION_0_ROLE] = "assistant";
  attrs[GEN_AI_COMPLETION_0_CONTENT] = content;
  applyTokenUsage(attrs, response?.usage, TOKEN_USAGE_KEYS);
}

function setEmbeddingSuccessAttrs(attrs: Record<string, any>, response: any): void {
  const vectors = Array.isArray(response?.data)
    ? response.data.map((item: any) => ({
        index: item?.index,
        embedding: item?.embedding,
      }))
    : response;
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(vectors);
}

function setGenericSuccessAttrs(attrs: Record<string, any>, response: any): void {
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(toSerializableValue(response));
  applyTokenUsage(attrs, response?.usage, TOKEN_USAGE_KEYS);
}

function applySuccessAttrs(
  attrs: Record<string, any>,
  spec: TogetherOperationSpec,
  response: any,
): void {
  if (spec.kind === "chat") {
    setChatSuccessAttrs(attrs, response);
  } else if (spec.kind === "text") {
    setTextSuccessAttrs(attrs, response);
  } else if (spec.kind === "embedding") {
    setEmbeddingSuccessAttrs(attrs, response);
  } else {
    setGenericSuccessAttrs(attrs, response);
  }
}

function emitSpan(
  spec: TogetherOperationSpec,
  attrs: Record<string, any>,
  startTime: [number, number],
  errorMessage?: string,
  statusCode?: number,
): void {
  try {
    const span = buildInstrumentedReadableSpan({
      name: spec.spanName,
      startTime,
      endTime: hrTime(),
      attributes: attrs,
      statusCode,
      errorMessage,
    });
    injectSpan(span);
  } catch {
    // Instrumentation must never break the application.
  }
}

export function emitSuccessSpan(
  spec: TogetherOperationSpec,
  request: any,
  startTime: [number, number],
  response: any,
): void {
  try {
    const attrs = buildBaseAttrs(spec, request, response);
    applySuccessAttrs(attrs, spec, response);
    emitSpan(spec, attrs, startTime);
  } catch {
    // Instrumentation must never break the application.
  }
}

export function emitErrorSpan(
  spec: TogetherOperationSpec,
  request: any,
  startTime: [number, number],
  err: unknown,
): void {
  try {
    const errorMessage = resolveErrorMessage(err);
    const statusCode = resolveStatusCode(err);
    const attrs = buildBaseAttrs(spec, request);
    attrs["error.message"] = errorMessage;
    if (statusCode !== undefined) attrs[STATUS_CODE_ATTR] = statusCode;
    emitSpan(spec, attrs, startTime, errorMessage, statusCode);
  } catch {
    // Instrumentation must never break the application.
  }
}

export function emitToolSpan(toolExecution: ToolExecution): void {
  const startTime = hrTime();
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: toolExecution.name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: toolExecution.name,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TOOL,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJson({
      name: toolExecution.name,
      arguments: toSerializableValue(toolExecution.input),
    }),
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: safeJson(toSerializableValue(toolExecution.output)),
  };

  if (toolExecution.id) attrs["tool_call_id"] = toolExecution.id;
  if (toolExecution.isError) attrs["error.message"] = stringifyStructured(toolExecution.output);

  const toolSpec: TogetherOperationSpec = {
    kind: "chat",
    method: "create",
    spanName: `${toolExecution.name}.tool`,
    logType: RespanLogType.TOOL,
    requestType: "tool",
  };
  emitSpan(
    toolSpec,
    attrs,
    startTime,
    toolExecution.isError ? stringifyStructured(toolExecution.output) : undefined,
  );
}

export function emitToolSpansFromMessages(messages: any[] | undefined): void {
  for (const toolExecution of extractToolExecutions(messages)) {
    try {
      emitToolSpan(toolExecution);
    } catch {
      // Instrumentation must never break the application.
    }
  }
}
