import { context, trace, TraceFlags } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan, injectSpan } from "@respan/tracing";
import {
  LLMRequestTypeValues,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import {
  AWS_BEDROCK_CHAT_SPAN_NAME,
  AWS_BEDROCK_INSTRUMENTATION_PACKAGE,
  AWS_BEDROCK_SYSTEM_NAME,
  PACKAGE_VERSION,
  RESPAN_LOG_METHOD_TS_TRACING,
  STATUS_CODE_ATTR,
} from "./_constants.js";
import {
  type BedrockResponse,
  captureInvokeResponsePayload,
  parseBedrockRequest,
  parseBedrockResponse,
  parseBedrockStreamResponse,
  safeJson,
  toJsonAttr,
} from "./_translator.js";

type HrTime = [number, number];

export interface EmitBedrockSpanOptions {
  operationName: string;
  apiParams?: Record<string, unknown>;
  startTimeHr: HrTime;
  responsePayload?: unknown;
  streamEvents?: unknown[];
  errorMessage?: string;
  statusCode?: number;
}

function baseAttrs(): Record<string, unknown> {
  const attrs: Record<string, unknown> = {
    [ATTR_GEN_AI_SYSTEM]: AWS_BEDROCK_SYSTEM_NAME,
    [SpanAttributes.LLM_REQUEST_TYPE]: LLMRequestTypeValues.CHAT,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: AWS_BEDROCK_CHAT_SPAN_NAME,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: AWS_BEDROCK_CHAT_SPAN_NAME,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
  };

  const workflowName = (context.active() as any).getValue(SpanAttributes.TRACELOOP_ENTITY_NAME);
  if (typeof workflowName === "string" && workflowName) {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
  }
  return attrs;
}

function setPromptAttrs(
  attrs: Record<string, unknown>,
  messages: Record<string, unknown>[],
): void {
  messages.forEach((message, index) => {
    const prefix = `${SpanAttributes.LLM_PROMPTS}.${index}`;
    const role = message.role;
    const content = message.content;
    if (role !== undefined) {
      attrs[`${prefix}.role`] = String(role);
    }
    if (content !== undefined) {
      attrs[`${prefix}.content`] = toJsonAttr(content);
    }
    if (message.tool_calls !== undefined) {
      attrs[`${prefix}.tool_calls`] = safeJson(message.tool_calls);
    }
    if (message.tool_call_id !== undefined) {
      attrs[`${prefix}.tool_call_id`] = String(message.tool_call_id);
    }
  });
}

function setUsageAttrs(
  attrs: Record<string, unknown>,
  usage: Record<string, number>,
): void {
  const inputTokens = usage.input_tokens;
  const outputTokens = usage.output_tokens;
  const totalTokens = usage.total_tokens;

  if (inputTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = inputTokens;
  }
  if (outputTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = outputTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = outputTokens;
  }
  if (totalTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
}

function setResponseAttrs(
  attrs: Record<string, unknown>,
  response: BedrockResponse,
): void {
  attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.role`] = response.role;
  attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.content`] = response.content;
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = response.content;
  if (response.toolCalls.length > 0) {
    attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.tool_calls`] = safeJson(response.toolCalls);
  }
  setUsageAttrs(attrs, response.usage);
}

export function buildBedrockAttrs(options: {
  operationName: string;
  apiParams?: Record<string, unknown>;
  responsePayload?: unknown;
  streamEvents?: unknown[];
}): Record<string, unknown> {
  const attrs = baseAttrs();
  const request = parseBedrockRequest({
    operationName: options.operationName,
    apiParams: options.apiParams,
  });

  if (request.modelId) {
    attrs[ATTR_GEN_AI_REQUEST_MODEL] = request.modelId;
  }

  if (request.messages.length > 0) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request.messages);
    setPromptAttrs(attrs, request.messages);
  } else if (request.rawPayload !== undefined) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request.rawPayload);
  }

  if (request.tools.length > 0) {
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(request.tools);
  }

  let response: BedrockResponse | undefined;
  if (options.streamEvents !== undefined) {
    response = parseBedrockStreamResponse({
      operationName: options.operationName,
      events: options.streamEvents,
    });
  } else if (options.responsePayload !== undefined) {
    response = parseBedrockResponse({
      operationName: options.operationName,
      responsePayload: options.responsePayload,
    });
  }

  if (response) {
    setResponseAttrs(attrs, response);
  }
  return attrs;
}

function activeSpanContext(): ReturnType<ReadableSpan["spanContext"]> | undefined {
  return trace.getSpan(context.active())?.spanContext();
}

function buildInstrumentedReadableSpan(
  options: Parameters<typeof buildReadableSpan>[0],
): ReadableSpan {
  const activeContext = activeSpanContext();
  const span = buildReadableSpan(options) as ReadableSpan & {
    instrumentationScope?: {
      name: string;
      version?: string;
    };
    spanContext: () => ReturnType<ReadableSpan["spanContext"]>;
  };

  const originalSpanContext = span.spanContext.bind(span);
  span.spanContext = () => ({
    ...originalSpanContext(),
    traceFlags: activeContext?.traceFlags ?? TraceFlags.SAMPLED,
  });
  span.instrumentationScope = {
    name: AWS_BEDROCK_INSTRUMENTATION_PACKAGE,
    version: PACKAGE_VERSION,
  };
  return span;
}

export function emitBedrockSpan(options: EmitBedrockSpanOptions): void {
  try {
    const attrs = buildBedrockAttrs({
      operationName: options.operationName,
      apiParams: options.apiParams,
      responsePayload: options.responsePayload,
      streamEvents: options.streamEvents,
    });
    let statusCode = options.statusCode ?? 200;
    if (options.errorMessage) {
      attrs[ATTR_ERROR_MESSAGE] = options.errorMessage;
      statusCode = statusCode >= 400 ? statusCode : 500;
    }
    attrs[STATUS_CODE_ATTR] = statusCode;

    const activeContext = activeSpanContext();
    const span = buildInstrumentedReadableSpan({
      name: AWS_BEDROCK_CHAT_SPAN_NAME,
      traceId: activeContext?.traceId,
      parentId: activeContext?.spanId,
      startTimeHr: options.startTimeHr,
      endTimeHr: hrTime(),
      attributes: attrs,
      statusCode,
      errorMessage: options.errorMessage,
    });
    injectSpan(span);
  } catch {
    // Instrumentation must not break application code.
  }
}

export function responsePayloadForInvoke(response: unknown): unknown {
  return captureInvokeResponsePayload(response);
}
