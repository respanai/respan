import { context, trace, TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import { LLMRequestTypeValues, SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  extractCompletionToolCalls,
  extractErrorMessage,
  extractStatusCode,
  extractToolExecutions,
  extractUsage,
  formatChatOutput,
  formatTextOutput,
  formatTools,
  INSTRUMENTATION_LIBRARY_NAME,
  LLM_USAGE_CACHE_READ_INPUT_TOKENS,
  normalizeMessages,
  PACKAGE_VERSION,
  RESPAN_LOG_METHOD_TS_TRACING,
  safeJsonString,
  setIfPresent,
  STATUS_CODE_ATTR,
  stringifyContent,
  type SpanAttributesRecord,
  type ToolExecution,
  WRITER_CHAT_ENTITY_NAME,
  WRITER_COMPLETION_ENTITY_NAME,
} from "./_helpers.js";

export type WriterOperationType = "chat" | "completion";

export interface EmitOperationOptions {
  type: WriterOperationType;
  body: Record<string, any>;
  startTime: [number, number];
  response?: unknown;
  error?: unknown;
}

function buildInstrumentedReadableSpan(opts: {
  name: string;
  startTime: [number, number];
  endTime: [number, number];
  attributes: SpanAttributesRecord;
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
    errorMessage: opts.errorMessage,
    mergePropagated: true,
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

function setRequestAttrs(attrs: SpanAttributesRecord, body: Record<string, any>): void {
  attrs[ATTR_GEN_AI_SYSTEM] = "writer";
  setIfPresent(attrs, ATTR_GEN_AI_REQUEST_MODEL, body.model);
  setIfPresent(attrs, ATTR_GEN_AI_REQUEST_MAX_TOKENS, body.max_tokens);
  setIfPresent(attrs, ATTR_GEN_AI_REQUEST_TEMPERATURE, body.temperature);
  setIfPresent(attrs, ATTR_GEN_AI_REQUEST_TOP_P, body.top_p);
}

function setPromptAttrs(attrs: SpanAttributesRecord, messages: Record<string, unknown>[]): void {
  for (const [index, message] of messages.entries()) {
    const prefix = `${ATTR_GEN_AI_PROMPT}.${index}`;
    attrs[`${prefix}.role`] = String(message.role ?? "user");
    attrs[`${prefix}.content`] = stringifyContent(message.content);
    if (message.tool_calls !== undefined) {
      attrs[`${prefix}.tool_calls`] = safeJsonString(message.tool_calls);
    }
  }
}

function setCompletionAttrs(
  attrs: SpanAttributesRecord,
  outputMessages: Record<string, unknown>[],
): void {
  const message = outputMessages[0];
  if (!message) return;
  const prefix = `${ATTR_GEN_AI_COMPLETION}.0`;
  attrs[`${prefix}.role`] = String(message.role ?? "assistant");
  attrs[`${prefix}.content`] = stringifyContent(message.content);
  if (message.tool_calls !== undefined) {
    attrs[`${prefix}.tool_calls`] = safeJsonString(message.tool_calls);
  }
}

function setUsageAttrs(attrs: SpanAttributesRecord, response: unknown): void {
  const usage = extractUsage(response);
  if (usage.promptTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = usage.promptTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = usage.promptTokens;
  }
  if (usage.completionTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = usage.completionTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = usage.completionTokens;
  }
  if (usage.totalTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage.totalTokens;
  }
  if (usage.cacheReadInputTokens !== undefined) {
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = usage.cacheReadInputTokens;
  }
}

function buildBaseAttrs(
  type: WriterOperationType,
  body: Record<string, any>,
): SpanAttributesRecord {
  const isChat = type === "chat";
  const entityName = isChat ? WRITER_CHAT_ENTITY_NAME : WRITER_COMPLETION_ENTITY_NAME;
  const attrs: SpanAttributesRecord = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityName,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: isChat ? RespanLogType.CHAT : RespanLogType.TEXT,
    [SpanAttributes.LLM_REQUEST_TYPE]: LLMRequestTypeValues.CHAT,
  };

  setRequestAttrs(attrs, body);

  if (isChat) {
    const messages = normalizeMessages(body.messages);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonString(messages);
    setPromptAttrs(attrs, messages);

    const tools = formatTools(body.tools);
    if (tools) {
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJsonString(tools);
    }
    if (body.tool_choice !== undefined) {
      attrs[`${RespanSpanAttributes.RESPAN_METADATA}.tool_choice`] =
        typeof body.tool_choice === "string" ? body.tool_choice : safeJsonString(body.tool_choice);
    }
    if (body.response_format !== undefined) {
      attrs[`${RespanSpanAttributes.RESPAN_METADATA}.response_format`] =
        typeof body.response_format === "string"
          ? body.response_format
          : safeJsonString(body.response_format);
    }
  } else {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonString([
      { role: "user", content: stringifyContent(body.prompt) },
    ]);
    attrs[`${ATTR_GEN_AI_PROMPT}.0.role`] = "user";
    attrs[`${ATTR_GEN_AI_PROMPT}.0.content`] = stringifyContent(body.prompt);
  }

  return attrs;
}

export function buildSuccessAttrs(
  type: WriterOperationType,
  body: Record<string, any>,
  response: unknown,
): SpanAttributesRecord {
  const attrs = buildBaseAttrs(type, body);

  if (type === "chat") {
    const outputMessages = formatChatOutput(response);
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonString(outputMessages);
    setCompletionAttrs(attrs, outputMessages);
  } else {
    const outputText = formatTextOutput(response);
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonString([
      { role: "assistant", content: outputText },
    ]);
    attrs[`${ATTR_GEN_AI_COMPLETION}.0.role`] = "assistant";
    attrs[`${ATTR_GEN_AI_COMPLETION}.0.content`] = outputText;
  }

  setUsageAttrs(attrs, response);
  return attrs;
}

export function buildErrorAttrs(
  type: WriterOperationType,
  body: Record<string, any>,
  err: unknown,
): SpanAttributesRecord {
  const attrs = buildBaseAttrs(type, body);
  attrs[ATTR_ERROR_MESSAGE] = extractErrorMessage(err);
  attrs[STATUS_CODE_ATTR] = extractStatusCode(err);
  return attrs;
}

function emitSpan(
  name: string,
  attrs: SpanAttributesRecord,
  startTime: [number, number],
  errorMessage?: string,
): void {
  try {
    injectSpan(
      buildInstrumentedReadableSpan({
        name,
        startTime,
        endTime: hrTime(),
        attributes: attrs,
        errorMessage,
      }),
    );
  } catch {
    // Instrumentation must not alter application behavior.
  }
}

export function emitOperationSuccess(opts: EmitOperationOptions): void {
  if (opts.response === undefined) return;
  const name = opts.type === "chat" ? WRITER_CHAT_ENTITY_NAME : WRITER_COMPLETION_ENTITY_NAME;
  emitSpan(name, buildSuccessAttrs(opts.type, opts.body, opts.response), opts.startTime);
}

export function emitOperationError(opts: EmitOperationOptions): void {
  const err = opts.error;
  const name = opts.type === "chat" ? WRITER_CHAT_ENTITY_NAME : WRITER_COMPLETION_ENTITY_NAME;
  emitSpan(
    name,
    buildErrorAttrs(opts.type, opts.body, err),
    opts.startTime,
    extractErrorMessage(err),
  );
}

export function emitToolSpan(toolExecution: ToolExecution): void {
  const startTime = hrTime();
  const attrs: SpanAttributesRecord = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: toolExecution.name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: toolExecution.name,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TOOL,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJsonString({
      id: toolExecution.id,
      name: toolExecution.name,
      arguments: toolExecution.arguments,
    }),
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: safeJsonString(toolExecution.output),
  };

  emitSpan(`${toolExecution.name}.tool`, attrs, startTime);
}

export function emitToolSpansFromMessages(messages: unknown): void {
  for (const toolExecution of extractToolExecutions(messages)) {
    try {
      emitToolSpan(toolExecution);
    } catch {
      // Instrumentation must not alter application behavior.
    }
  }
}
