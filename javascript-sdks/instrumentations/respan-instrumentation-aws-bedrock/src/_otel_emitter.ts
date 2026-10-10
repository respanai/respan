import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
} from "@opentelemetry/semantic-conventions/incubating";
import { ATTR_HTTP_RESPONSE_STATUS_CODE } from "@opentelemetry/semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
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
} from "./_constants.js";
import {
  captureInvokeResponsePayload,
  parseBedrockRequest,
  parseBedrockResponse,
  parseBedrockStreamResponse,
  safeJson,
  toJsonAttr,
} from "./_translator.js";
import {
  capture,
  attributesOf,
  data,
  guard,
  refresh,
  snapshot,
  type CaptureOptions,
  type CapturePolicy,
} from "./_privacy.js";

type Attrs = Record<string, any>;
export interface EmitBedrockSpanOptions {
  operationName: string;
  apiParams?: Record<string, unknown>;
  startTimeHr: [number, number];
  responsePayload?: unknown;
  streamEvents?: unknown[];
  errorMessage?: string;
  statusCode?: number;
}
function baseAttrs(model?: unknown): Attrs {
  const attrs: Attrs = {
    [ATTR_GEN_AI_SYSTEM]: AWS_BEDROCK_SYSTEM_NAME,
    [SpanAttributes.LLM_REQUEST_TYPE]: LLMRequestTypeValues.CHAT,
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: AWS_BEDROCK_CHAT_SPAN_NAME,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
  };
  if (typeof model === "string") attrs[ATTR_GEN_AI_REQUEST_MODEL] = model;
  return attrs;
}
function requestAttrs(operationName: string, apiParams?: Attrs): Attrs {
  const attrs: Attrs = {};
  const request = parseBedrockRequest({ operationName, apiParams });
  if (request.tools.length)
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(request.tools);
  const metadata = {
    bedrock_operation: operationName,
    request: request.rawPayload,
  };
  attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson(metadata);
  if (request.messages.length) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request.messages);
    request.messages.forEach((message, index) => {
      const prefix = `${SpanAttributes.LLM_PROMPTS}.${index}`;
      if (message.role !== undefined) attrs[`${prefix}.role`] = message.role;
      if (message.content !== undefined)
        attrs[`${prefix}.content`] = toJsonAttr(message.content);
      if (message.tool_calls !== undefined)
        attrs[`${prefix}.tool_calls`] = safeJson(message.tool_calls);
      if (message.tool_call_id !== undefined)
        attrs[`${prefix}.tool_call_id`] = message.tool_call_id;
    });
  } else if (request.rawPayload !== undefined)
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(request.rawPayload);
  return attrs;
}
function responseAttrs(
  operationName: string,
  payload?: unknown,
  events?: unknown[],
): Attrs {
  const attrs: Attrs = {};
  const parsed =
    events !== undefined
      ? parseBedrockStreamResponse({ operationName, events })
      : parseBedrockResponse({ operationName, responsePayload: payload });
  if (parsed.rawPayload !== undefined)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(parsed.rawPayload);
  if (parsed.hasContent) {
    attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.role`] = parsed.role;
    attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.content`] = parsed.content;
    if (parsed.toolCalls.length)
      attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.tool_calls`] = safeJson(
        parsed.toolCalls,
      );
  }
  if (parsed.embedding !== undefined) {
    attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = RespanLogType.EMBEDDING;
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = "embedding";
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(parsed.embedding);
    if (Array.isArray(parsed.embedding))
      attrs[ATTR_GEN_AI_EMBEDDINGS_DIMENSION_COUNT] = Array.isArray(
        parsed.embedding[0],
      )
        ? parsed.embedding[0].length
        : parsed.embedding.length;
  }
  const u = parsed.usage;
  if (u.input_tokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = u.input_tokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = u.input_tokens;
  }
  if (u.output_tokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = u.output_tokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = u.output_tokens;
  }
  if (u.total_tokens !== undefined)
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = u.total_tokens;
  if (u.cache_read_input_tokens !== undefined)
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] =
      u.cache_read_input_tokens;
  if (u.cache_creation_input_tokens !== undefined)
    attrs[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] =
      u.cache_creation_input_tokens;
  if (parsed.stopReason !== undefined)
    attrs[ATTR_GEN_AI_RESPONSE_FINISH_REASONS] = [parsed.stopReason];
  return attrs;
}
/** Separate unbounded indexed views from canonical payload attributes. */
function takeIndexed(attrs: Attrs): Attrs {
  const indexed: Attrs = {};
  for (const key of Object.keys(attrs)) {
    if (
      key.startsWith(`${SpanAttributes.LLM_PROMPTS}.`) ||
      key.startsWith(`${SpanAttributes.LLM_COMPLETIONS}.`)
    ) {
      indexed[key] = attrs[key];
      delete attrs[key];
    }
  }
  return indexed;
}
export function buildBedrockAttrs(options: {
  operationName: string;
  apiParams?: Attrs;
  responsePayload?: unknown;
  streamEvents?: unknown[];
}): Attrs {
  return {
    ...baseAttrs(data(options.apiParams, "modelId")),
    ...requestAttrs(options.operationName, snapshot(options.apiParams)),
    ...(options.responsePayload !== undefined ||
    options.streamEvents !== undefined
      ? responseAttrs(
          options.operationName,
          snapshot(options.responsePayload),
          snapshot(options.streamEvents),
        )
      : {}),
  };
}
export interface BedrockSession {
  ctx: Context;
  event(event: unknown): void;
  response(response: unknown): void;
  finish(response?: unknown, error?: unknown, statusCode?: number): void;
}
export function startBedrockSession(
  operationName: string,
  input: unknown,
  options: CaptureOptions,
  ctx: Context = context.active(),
  startTime?: [number, number],
): BedrockSession | undefined {
  const policy: CapturePolicy = capture(options, ctx);
  if (!policy.emit) return undefined;
  const attrs = baseAttrs(data(input, "modelId"));
  // Only identity and model enter the sampler; request payloads have not been copied.
  const span = trace
    .getTracer(AWS_BEDROCK_INSTRUMENTATION_PACKAGE, PACKAGE_VERSION)
    .startSpan(
      AWS_BEDROCK_CHAT_SPAN_NAME,
      {
        kind: SpanKind.CLIENT,
        attributes: attrs,
        ...(startTime ? { startTime } : {}),
      },
      ctx,
    );
  if (!span.isRecording()) {
    span.end();
    return undefined;
  }
  if (!(span.spanContext().traceFlags & 1)) {
    policy.inputs = policy.outputs = false;
    guard(span, policy);
    span.end();
    return undefined;
  }
  guard(span, policy);
  const callContext = trace.setSpan(ctx, span);
  const events: unknown[] = [];
  let ended = false;
  let observedStatus: number | undefined;
  let requestIndexed: Attrs = {};
  try {
    if (refresh(policy).inputs) {
      const request = requestAttrs(operationName, snapshot(input));
      requestIndexed = takeIndexed(request);
      const parentMetadata = data(
        attributesOf(policy.parent),
        RespanSpanAttributes.RESPAN_METADATA,
      );
      if (typeof parentMetadata === "string") {
        try {
          const inherited = JSON.parse(parentMetadata);
          request[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
            ...inherited,
            ...JSON.parse(request[RespanSpanAttributes.RESPAN_METADATA]),
          });
        } catch {
          /* Malformed metadata is not a request failure. */
        }
      }
      span.setAttributes(request);
    }
  } catch {
    /* Instrumentation is observational. */
  }
  return {
    ctx: callContext,
    response(response) {
      const status = data(data(response, "$metadata"), "httpStatusCode");
      if (typeof status === "number") observedStatus = status;
    },
    event(event) {
      if (refresh(policy).outputs) events.push(snapshot(event));
    },
    finish(response, error, statusCode) {
      if (ended) return;
      ended = true;
      try {
        refresh(policy);
        const status =
          statusCode ??
          data(data(response, "$metadata"), "httpStatusCode") ??
          data(data(error, "$metadata"), "httpStatusCode") ??
          observedStatus;
        if (typeof status === "number")
          span.setAttribute(ATTR_HTTP_RESPONSE_STATUS_CODE, status);
        let responseIndexed: Attrs = {};
        if (error !== undefined) {
          const message = data(error, "message");
          span.setStatus({
            code: SpanStatusCode.ERROR,
            ...(policy.inputs && policy.outputs && typeof message === "string"
              ? { message }
              : {}),
          });
          if (policy.inputs && policy.outputs && typeof message === "string")
            span.setAttribute(ATTR_ERROR_MESSAGE, message);
        } else {
          if (policy.outputs) {
            const payload =
              operationName === "InvokeModel"
                ? captureInvokeResponsePayload(snapshot(response))
                : snapshot(response);
            const canonicalResponse = responseAttrs(
              operationName,
              payload,
              operationName.endsWith("Stream") ? events : undefined,
            );
            responseIndexed = takeIndexed(canonicalResponse);
            span.setAttributes(canonicalResponse);
          }
          span.setStatus({ code: SpanStatusCode.OK });
        }
        // Attempt every indexed view last. A finite native budget may omit
        // indices, while complete input/tools/output and usage stay intact.
        refresh(policy);
        if (policy.outputs && error === undefined)
          span.setAttributes(responseIndexed);
        if (policy.inputs) span.setAttributes(requestIndexed);
      } catch {
        /* Preserve the native SDK result or error. */
      } finally {
        span.end();
      }
    },
  };
}
/** Compatibility helper: native calls use startBedrockSession before invoking AWS. */
export function emitBedrockSpan(options: EmitBedrockSpanOptions): void {
  try {
    const session = startBedrockSession(
      options.operationName,
      options.apiParams,
      {},
      context.active(),
      options.startTimeHr,
    );
    if (!session) return;
    options.streamEvents?.forEach((e) => session.event(e));
    session.finish(
      options.responsePayload,
      options.errorMessage === undefined
        ? undefined
        : new Error(options.errorMessage),
      options.statusCode,
    );
  } catch {
    /* Instrumentation must not break application code. */
  }
}
export function responsePayloadForInvoke(response: unknown): unknown {
  return captureInvokeResponsePayload(response);
}
