import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import {
  ATTR_ERROR_TYPE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_TOOL_CALL_ID,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes, RespanLogType } from "@respan/respan-sdk";
import {
  SpanAttributes,
  LLMRequestTypeValues,
} from "@traceloop/ai-semantic-conventions";
import {
  capture,
  refresh,
  guard,
  data,
  snapshot,
  attributesOf,
  type CaptureOptions,
  type CapturePolicy,
} from "./_privacy.js";
import {
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  safeJson,
  textOrJson,
  messages,
  tools,
  toolCalls,
} from "./_helpers.js";
export type Operation =
  | "messages"
  | "countTokens"
  | "batches.create"
  | "batches.results"
  | "completion"
  | "agent"
  | "tool";
export interface Session {
  ctx: Context;
  span: Span;
  policy: CapturePolicy;
  headers(response: unknown): void;
  input(value: unknown): void;
  finish(value?: unknown, error?: unknown, success?: boolean): void;
}
export function start(
  operation: Operation,
  body: unknown,
  options: CaptureOptions = {},
  ctx = context.active(),
  toolId?: unknown,
): Session | undefined {
  const policy = capture(options, ctx);
  if (!policy.emit) return;
  const llm = operation === "messages" || operation === "completion";
  const name =
    operation === "tool"
      ? (data(body, "name") ?? "tool")
      : `anthropic.${operation}`;
  const attrs: Record<string, any> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]:
      operation === "agent"
        ? RespanLogType.AGENT
        : operation === "tool"
          ? RespanLogType.TOOL
          : llm
            ? operation === "completion"
              ? RespanLogType.TEXT
              : RespanLogType.CHAT
            : RespanLogType.TASK,
  };
  if (llm) {
    attrs[ATTR_GEN_AI_SYSTEM] = "anthropic";
    attrs[SpanAttributes.LLM_REQUEST_TYPE] =
      operation === "completion"
        ? LLMRequestTypeValues.COMPLETION
        : LLMRequestTypeValues.CHAT;
    const model = data(body, "model");
    if (typeof model === "string") attrs[ATTR_GEN_AI_REQUEST_MODEL] = model;
  }
  if (operation === "tool" && typeof toolId === "string")
    attrs[ATTR_GEN_AI_TOOL_CALL_ID] = toolId;
  const span = trace
    .getTracer(INSTRUMENTATION_LIBRARY_NAME, PACKAGE_VERSION)
    .startSpan(
      String(name),
      { kind: llm ? SpanKind.CLIENT : SpanKind.INTERNAL, attributes: attrs },
      ctx,
    );
  if (!span.isRecording()) {
    span.end();
    return;
  }
  if (!(span.spanContext().traceFlags & 1))
    policy.inputs = policy.outputs = false;
  guard(span, policy);
  let ended = false;
  let promptAttributes: Record<string, any> | undefined;
  const session: Session = {
    ctx: trace.setSpan(ctx, span),
    span,
    policy,
    headers(response) {
      const status =
        response instanceof Response
          ? response.status
          : data(response, "status");
      if (typeof status === "number") {
        span.setAttribute(ATTR_HTTP_RESPONSE_STATUS_CODE, status);
      }
    },
    input(value) {
      if (!refresh(policy).inputs) return;
      const b = snapshot(value);
      if (b === undefined) return;
      if (llm) {
        const inputMessages =
          operation === "completion"
            ? [{ role: "user", content: b.prompt }]
            : messages(b);
        span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_INPUT,
          safeJson(inputMessages),
        );
        const definitions = tools(b.tools);
        if (definitions.length)
          span.setAttribute(
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            safeJson(definitions),
          );
        let inherited: Record<string, unknown> = {};
        try {
          const prior = data(
            attributesOf(policy.parent),
            RespanSpanAttributes.RESPAN_METADATA,
          );
          if (typeof prior === "string") inherited = JSON.parse(prior);
        } catch {}
        span.setAttribute(
          RespanSpanAttributes.RESPAN_METADATA,
          safeJson({
            ...inherited,
            anthropic_operation: operation,
            request: b,
          }),
        );
        promptAttributes = {};
        for (const [index, message] of inputMessages.entries()) {
          const p = `${SpanAttributes.LLM_PROMPTS}.${index}`;
          if (message.role !== undefined)
            promptAttributes[`${p}.role`] = message.role;
          promptAttributes[`${p}.content`] = textOrJson(message.content);
          if (message.tool_calls)
            promptAttributes[`${p}.tool_calls`] = safeJson(message.tool_calls);
          if (message.tool_call_id)
            promptAttributes[`${p}.tool_call_id`] = message.tool_call_id;
        }
      } else if (operation === "agent") {
        let inherited: Record<string, unknown> = {};
        try {
          const prior = data(
            attributesOf(policy.parent),
            RespanSpanAttributes.RESPAN_METADATA,
          );
          if (typeof prior === "string") inherited = JSON.parse(prior);
        } catch {}
        span.setAttribute(
          RespanSpanAttributes.RESPAN_METADATA,
          safeJson({ ...inherited, request: b, tools: b.tools }),
        );
      } else
        span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, safeJson(b));
    },
    finish(value, error, success = true) {
      if (ended) return;
      ended = true;
      try {
        refresh(policy);
        if (error !== undefined) {
          const status = data(error, "status");
          if (typeof status === "number") {
            span.setAttribute(ATTR_HTTP_RESPONSE_STATUS_CODE, status);
          }
          const type = data(error, "name");
          if (typeof type === "string")
            span.setAttribute(ATTR_ERROR_TYPE, type);
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
          if (policy.outputs && value !== undefined) {
            const output = snapshot(value);
            span.setAttribute(
              SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
              safeJson(output),
            );
            if (llm && output) {
              if (typeof output.model === "string")
                span.setAttribute(ATTR_GEN_AI_RESPONSE_MODEL, output.model);
              if (typeof output.id === "string")
                span.setAttribute(ATTR_GEN_AI_RESPONSE_ID, output.id);
              const content =
                operation === "completion" ? output.completion : output.content;
              if (content !== undefined) {
                span.setAttribute(
                  `${SpanAttributes.LLM_COMPLETIONS}.0.role`,
                  output.role ?? "assistant",
                );
                span.setAttribute(
                  `${SpanAttributes.LLM_COMPLETIONS}.0.content`,
                  textOrJson(content),
                );
                const calls = toolCalls(content);
                if (calls.length)
                  span.setAttribute(
                    `${SpanAttributes.LLM_COMPLETIONS}.0.tool_calls`,
                    safeJson(calls),
                  );
              }
              if (typeof output.stop_reason === "string")
                span.setAttribute(ATTR_GEN_AI_RESPONSE_FINISH_REASONS, [
                  output.stop_reason,
                ]);
              const u = output.usage ?? {};
              for (const [key, modern, legacy] of [
                [
                  "input_tokens",
                  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
                  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
                ],
                [
                  "output_tokens",
                  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
                  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
                ],
              ] as const)
                if (typeof u[key] === "number") {
                  span.setAttribute(modern, u[key]);
                  span.setAttribute(legacy, u[key]);
                }
              if (typeof u.total_tokens === "number")
                span.setAttribute(
                  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                  u.total_tokens,
                );
              if (typeof u.cache_read_input_tokens === "number")
                span.setAttribute(
                  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                  u.cache_read_input_tokens,
                );
              if (typeof u.cache_creation_input_tokens === "number")
                span.setAttribute(
                  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                  u.cache_creation_input_tokens,
                );
            }
          }
          if (success) span.setStatus({ code: SpanStatusCode.OK });
        }
      } catch {
      } finally {
        if (refresh(policy).inputs && promptAttributes)
          span.setAttributes(promptAttributes);
        promptAttributes = undefined;
        span.end();
      }
    },
  };
  try {
    session.input(body);
  } catch {}
  return session;
}
