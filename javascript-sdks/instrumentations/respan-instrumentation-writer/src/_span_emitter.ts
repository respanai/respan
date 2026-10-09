import {
  context,
  trace,
  type Context,
  SpanStatusCode,
} from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { ATTR_HTTP_RESPONSE_STATUS_CODE } from "@opentelemetry/semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  LLMRequestTypeValues,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import {
  data,
  capture,
  guard,
  refresh,
  snapshot,
  strip,
  type CaptureOptions,
  type CapturePolicy,
} from "./_privacy.js";
import {
  extractUsage,
  formatChatOutput,
  formatTextOutput,
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  safeJsonString,
  setIfPresent,
  stringifyContent,
  WRITER_CHAT_ENTITY_NAME,
  WRITER_COMPLETION_ENTITY_NAME,
  type SpanAttributesRecord,
} from "./_helpers.js";
export type WriterOperationType = "chat" | "completion";
export interface Operation {
  type: WriterOperationType;
  ctx: Context;
  startTime: [number, number];
  policy: CapturePolicy;
  body: Record<string, any>;
  handled: boolean;
  status?: number;
  span: any;
}
function base(
  type: WriterOperationType,
  body: any,
  policy: CapturePolicy,
): SpanAttributesRecord {
  const attrs: SpanAttributesRecord = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]:
      type === "chat" ? WRITER_CHAT_ENTITY_NAME : WRITER_COMPLETION_ENTITY_NAME,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]:
      type === "chat" ? RespanLogType.CHAT : RespanLogType.TEXT,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
    [SpanAttributes.LLM_SYSTEM]: "writer",
    [SpanAttributes.LLM_REQUEST_TYPE]: LLMRequestTypeValues.CHAT,
  };
  for (const [key, source] of [
    [SpanAttributes.LLM_REQUEST_MODEL, "model"],
    [SpanAttributes.LLM_REQUEST_MAX_TOKENS, "max_tokens"],
    [SpanAttributes.LLM_REQUEST_TEMPERATURE, "temperature"],
    [SpanAttributes.LLM_REQUEST_TOP_P, "top_p"],
  ])
    setIfPresent(attrs, key, body[source]);
  if (policy.inputs) {
    const messages =
      type === "chat"
        ? body.messages
        : [{ role: "user", content: body.prompt }];
    if (messages !== undefined)
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJsonString(messages);
    if (Array.isArray(messages))
      for (const [index, message] of messages.entries()) {
        if (!message || typeof message !== "object") continue;
        for (const key of [
          "role",
          "content",
          "tool_calls",
          "tool_call_id",
          "name",
        ])
          if (message[key] !== undefined)
            attrs[`${SpanAttributes.LLM_PROMPTS}.${index}.${key}`] =
              typeof message[key] === "string"
                ? message[key]
                : safeJsonString(message[key]);
      }
    if (body.tools !== undefined)
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJsonString(body.tools);
    for (const key of ["tool_choice", "response_format"])
      if (body[key] !== undefined)
        attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJsonString({
          ...JSON.parse(attrs[RespanSpanAttributes.RESPAN_METADATA] ?? "{}"),
          [key]: body[key],
        });
  }
  return attrs;
}
export function buildSuccessAttrs(
  type: WriterOperationType,
  body: any,
  response: unknown,
  policy = capture(),
): SpanAttributesRecord {
  refresh(policy);
  const request = policy.inputs
    ? (snapshot(body) ?? {})
    : {
        model: data(body, "model"),
        max_tokens: data(body, "max_tokens"),
        temperature: data(body, "temperature"),
        top_p: data(body, "top_p"),
      };
  const value = policy.outputs
    ? (snapshot(response) ?? {})
    : {
        model: data(response, "model"),
        id: data(response, "id"),
        usage: snapshot(data(response, "usage")),
      };
  const attrs = base(type, request, policy);
  setIfPresent(attrs, ATTR_GEN_AI_RESPONSE_MODEL, value.model);
  setIfPresent(attrs, ATTR_GEN_AI_RESPONSE_ID, value.id);
  if (policy.outputs && response !== undefined) {
    const messages =
      type === "chat"
        ? formatChatOutput(value)
        : [{ role: "assistant", content: formatTextOutput(value) }];
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJsonString(messages);
    for (const [index, message] of messages.entries())
      for (const key of ["role", "content", "tool_calls"])
        if (message?.[key] !== undefined)
          attrs[`${SpanAttributes.LLM_COMPLETIONS}.${index}.${key}`] =
            typeof message[key] === "string"
              ? message[key]
              : safeJsonString(message[key]);
  }
  const usage = extractUsage(value);
  for (const [key, count] of [
    [ATTR_GEN_AI_USAGE_INPUT_TOKENS, usage.promptTokens],
    [SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.promptTokens],
    [ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, usage.completionTokens],
    [SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completionTokens],
    [SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.totalTokens],
    [ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, usage.cacheReadInputTokens],
  ] as const)
    if (count !== undefined) attrs[key] = count;
  strip(attrs, policy);
  return attrs;
}
export function buildErrorAttrs(
  type: WriterOperationType,
  body: any,
  error: unknown,
  policy = capture(),
): SpanAttributesRecord {
  refresh(policy);
  const request = policy.inputs
    ? (snapshot(body) ?? {})
    : {
        model: data(body, "model"),
        max_tokens: data(body, "max_tokens"),
        temperature: data(body, "temperature"),
        top_p: data(body, "top_p"),
      };
  const attrs = base(type, request, policy);
  if (policy.inputs && policy.outputs)
    setIfPresent(attrs, ATTR_ERROR_MESSAGE, data(error, "message"));
  const status = data(error, "status");
  if (
    typeof status === "number" &&
    Number.isInteger(status) &&
    status >= 100 &&
    status <= 599
  )
    attrs[ATTR_HTTP_RESPONSE_STATUS_CODE] = status;
  strip(attrs, policy);
  return attrs;
}
export function startOperation(
  type: WriterOperationType,
  body: unknown,
  options: CaptureOptions,
  startTime: [number, number] = hrTime(),
): Operation {
  const ctx = context.active(),
    policy = capture(options, ctx);
  const span = policy.emit
    ? trace
        .getTracer(INSTRUMENTATION_LIBRARY_NAME, PACKAGE_VERSION)
        .startSpan(
          type === "chat"
            ? WRITER_CHAT_ENTITY_NAME
            : WRITER_COMPLETION_ENTITY_NAME,
          { startTime },
          ctx,
        )
    : undefined;
  policy.emit &&=
    !!span && span.isRecording() && (span.spanContext().traceFlags & 1) !== 0;
  refresh(policy);
  // Content is copied only after every gate, including the actual provider sampler. Metadata fields use data descriptors.
  const request: any = policy.inputs ? (snapshot(body) ?? {}) : {};
  request.stream = data(body, "stream");
  if (!policy.inputs)
    for (const key of ["model", "max_tokens", "temperature", "top_p"])
      request[key] = data(body, key);
  return {
    type,
    ctx,
    policy,
    body: request,
    startTime,
    span,
    handled: false,
  };
}
export function finishOperation(
  op: Operation,
  response?: unknown,
  error?: unknown,
): void {
  if (op.handled) return;
  op.handled = true;
  try {
    refresh(op.policy);
    if (!op.policy.emit) return;
    const safeResponse =
      response === undefined
        ? undefined
        : op.policy.outputs
          ? snapshot(response)
          : {
              model: data(response, "model"),
              id: data(response, "id"),
              usage: snapshot(data(response, "usage")),
            };
    const attrs =
      error === undefined
        ? buildSuccessAttrs(op.type, op.body, safeResponse, op.policy)
        : buildErrorAttrs(op.type, op.body, error, op.policy);
    if (op.status !== undefined)
      attrs[ATTR_HTTP_RESPONSE_STATUS_CODE] = op.status;
    context.with(op.ctx, () => {
      const span = op.span;
      const inheritedMetadata =
        span.attributes[RespanSpanAttributes.RESPAN_METADATA];
      const requestMetadata = attrs[RespanSpanAttributes.RESPAN_METADATA];
      if (
        typeof inheritedMetadata === "string" &&
        typeof requestMetadata === "string"
      ) {
        try {
          attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJsonString({
            ...JSON.parse(inheritedMetadata),
            ...JSON.parse(requestMetadata),
          });
        } catch {
          /* Keep the canonical request metadata if an upstream value is malformed. */
        }
      }
      span.attributes = { ...span.attributes, ...attrs };
      span.status = {
        code: error === undefined ? SpanStatusCode.OK : SpanStatusCode.ERROR,
      };
      if (
        error !== undefined &&
        op.policy.inputs &&
        op.policy.outputs &&
        typeof data(error, "message") === "string"
      )
        span.status.message = data(error, "message");
      guard(span, op.policy);
      span.end(hrTime());
    });
  } catch {
    /* Telemetry must not change the native outcome. */
  }
}
// Compatibility with the package's existing helper exports.
export interface EmitOperationOptions {
  type: WriterOperationType;
  body: Record<string, any>;
  startTime: [number, number];
  response?: unknown;
  error?: unknown;
}
export function emitOperationSuccess(opts: EmitOperationOptions): void {
  const op = startOperation(opts.type, opts.body, {}, opts.startTime);
  op.startTime = opts.startTime;
  finishOperation(op, opts.response);
}
export function emitOperationError(opts: EmitOperationOptions): void {
  const op = startOperation(opts.type, opts.body, {}, opts.startTime);
  op.startTime = opts.startTime;
  finishOperation(op, undefined, opts.error);
}
