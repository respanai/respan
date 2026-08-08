import { context, trace, TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ANTHROPIC_CHAT_ENTITY_NAME,
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  extractToolCalls,
  extractToolCallsFromInputMessages,
  extractToolExecutions,
  formatInputMessages,
  formatOutputMessage,
  formatTools,
  mergeToolCalls,
  safeJson,
  stringifyStructured,
  type ToolExecution,
} from "./_helpers.js";

function buildInstrumentedReadableSpan(opts: {
  name: string;
  startTime: [number, number];
  endTime: [number, number];
  attributes: Record<string, any>;
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
    mergePropagated: false,
  }) as ReadableSpan & {
    instrumentationScope?: { name: string; version?: string };
    spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
  };

  const originalSpanContext = span.spanContext.bind(span);
  const mutableSpan = span as typeof span & {
    spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
  };
  mutableSpan.spanContext = () => ({
    ...originalSpanContext(),
    traceFlags: activeSpanContext?.traceFlags ?? TraceFlags.SAMPLED,
  });
  mutableSpan.instrumentationScope = {
    name: INSTRUMENTATION_LIBRARY_NAME,
    version: PACKAGE_VERSION,
  };
  return mutableSpan;
}

function setStructuredAttr(
  attrs: Record<string, any>,
  key: string,
  value: unknown,
): void {
  attrs[key] = value;
}

function setStructuredCompatibilityAttrs(
  attrs: Record<string, any>,
  key: RespanSpanAttributes.RESPAN_SPAN_TOOLS | RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  legacyKey: "tools" | "tool_calls",
  value: unknown,
): void {
  setStructuredAttr(attrs, key, value);
  setStructuredAttr(attrs, legacyKey, value);
}

function buildBaseChatAttrs(kwargs: Record<string, any>, model?: string): Record<string, any> {
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: ANTHROPIC_CHAT_ENTITY_NAME,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: ANTHROPIC_CHAT_ENTITY_NAME,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
    [RespanSpanAttributes.LLM_SYSTEM]: "anthropic",
  };

  const resolvedModel = model ?? kwargs.model;
  if (resolvedModel) {
    attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = resolvedModel;
  }

  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
    formatInputMessages(kwargs.messages ?? [], kwargs.system),
  );

  const tools = formatTools(kwargs.tools);
  if (tools) {
    setStructuredCompatibilityAttrs(
      attrs,
      RespanSpanAttributes.RESPAN_SPAN_TOOLS,
      "tools",
      tools,
    );
  }

  return attrs;
}

function buildSuccessAttrs(kwargs: Record<string, any>, message: any): Record<string, any> {
  const attrs = buildBaseChatAttrs(kwargs, message?.model ?? kwargs.model);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([formatOutputMessage(message)]);

  if (message?.usage) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] =
      message.usage.input_tokens ?? 0;
    attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] =
      message.usage.output_tokens ?? 0;
  }

  const toolCalls = mergeToolCalls(
    extractToolCalls(message),
    extractToolCallsFromInputMessages(kwargs.messages),
  );
  if (toolCalls) {
    setStructuredCompatibilityAttrs(
      attrs,
      RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
      "tool_calls",
      toolCalls,
    );
  }

  return attrs;
}

function buildErrorAttrs(kwargs: Record<string, any>): Record<string, any> {
  const attrs = buildBaseChatAttrs(kwargs);

  const toolCalls = extractToolCallsFromInputMessages(kwargs.messages);
  if (toolCalls) {
    setStructuredCompatibilityAttrs(
      attrs,
      RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
      "tool_calls",
      toolCalls,
    );
  }

  return attrs;
}

function emitSpan(
  name: string,
  attrs: Record<string, any>,
  startTime: [number, number],
  errorMessage?: string,
): void {
  try {
    const span = buildInstrumentedReadableSpan({
      name,
      startTime,
      endTime: hrTime(),
      attributes: attrs,
      errorMessage,
    });
    injectSpan(span);
  } catch {
    // Never break the application.
  }
}

export function emitSuccessSpan(
  kwargs: Record<string, any>,
  startTime: [number, number],
  message: any,
): void {
  try {
    emitSpan(
      ANTHROPIC_CHAT_ENTITY_NAME,
      buildSuccessAttrs(kwargs, message),
      startTime,
    );
  } catch {
    // Never break the application.
  }
}

export function emitErrorSpan(
  kwargs: Record<string, any>,
  startTime: [number, number],
  err: unknown,
): void {
  try {
    const errorMessage = String(err);
    const attrs = buildErrorAttrs(kwargs);
    attrs["error.message"] = errorMessage;
    emitSpan(ANTHROPIC_CHAT_ENTITY_NAME, attrs, startTime, errorMessage);
  } catch {
    // Never break the application.
  }
}

export function emitToolSpan(toolExecution: ToolExecution): void {
  const startTime = hrTime();
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: toolExecution.name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: toolExecution.name,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.TOOL,
    [SpanAttributes.TRACELOOP_ENTITY_INPUT]: safeJson([
      { role: "tool", content: stringifyStructured(toolExecution.input) },
    ]),
    [SpanAttributes.TRACELOOP_ENTITY_OUTPUT]: safeJson({
      role: "tool",
      content: stringifyStructured(toolExecution.output),
    }),
  };

  if (toolExecution.id) {
    attrs.tool_call_id = toolExecution.id;
  }
  if (toolExecution.isError) {
    attrs["error.message"] = stringifyStructured(toolExecution.output);
  }

  emitSpan(
    `${toolExecution.name}.tool`,
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
      // Never break the application.
    }
  }
}
