import { types } from "node:util";
import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  TraceFlags,
} from "@opentelemetry/api";
import {
  ATTR_ERROR_TYPE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import { ATTR_HTTP_STATUS_CODE } from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { capture, data, guard, refresh, snapshot } from "./_privacy.js";
import { setRequest, setResult, StreamAccumulator } from "./_translator.js";
import type { AzureOpenAIInstrumentorOptions } from "./index.js";
export type Operation = "chat" | "completion" | "responses" | "embedding";
let policyID = 0;

export function observeRequest(
  original: any,
  receiver: any,
  args: any[],
  operation: Operation,
  options: AzureOpenAIInstrumentorOptions,
  legacy: boolean,
): any {
  const parent = context.active();
  const policy = capture(options, parent);
  if (!policy.emit) return original.apply(receiver, args);
  const raw = legacy ? args[2] : args[0];
  const model = legacy
    ? args[0]
    : (data(raw, "model") ?? data(receiver?._client, "deploymentName"));
  const name = `azure_openai.${operation}`;
  const span = trace
    .getTracer("@respan/instrumentation-azure-openai")
    .startSpan(
      name,
      {
        kind: SpanKind.CLIENT,
        attributes: {
          [RespanSpanAttributes.RESPAN_LOG_TYPE]:
            operation === "responses"
              ? "chat"
              : operation === "completion"
                ? "text"
                : operation,
          [SpanAttributes.LLM_SYSTEM]: "azure",
          [SpanAttributes.LLM_REQUEST_TYPE]:
            operation === "embedding" ? "embedding" : "chat",
          [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
          [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
          ...(typeof model === "string"
            ? { [SpanAttributes.LLM_REQUEST_MODEL]: model }
            : {}),
        },
      },
      parent,
    );
  const sampled =
    span.isRecording() &&
    Boolean(span.spanContext().traceFlags & TraceFlags.SAMPLED);
  if (!sampled) policy.inputs = policy.outputs = false;
  guard(span, policy);
  const active = trace.setSpan(parent, span);
  let ended = false;
  let requestIndexed: (() => void) | undefined;
  let cleanup: (() => void) | undefined;
  let removeAbort: (() => void) | undefined;
  const safely = (fn: () => void) => {
    try {
      fn();
    } catch (error) {
      try {
        options.exceptionLogger?.(
          error instanceof Error ? error : new Error("Azure telemetry failed"),
        );
      } catch {}
    }
  };
  const status = (value: unknown) => {
    if (typeof value !== "number" || !Number.isInteger(value)) return;
    span.setAttribute(ATTR_HTTP_RESPONSE_STATUS_CODE, value);
    span.setAttribute(ATTR_HTTP_STATUS_CODE, value);
  };
  const finish = (value?: any, error?: unknown) => {
    if (ended) return;
    ended = true;
    removeAbort?.();
    cleanup?.();
    safely(() => {
      refresh(policy);
      let responseIndexed: (() => void) | undefined;
      if (sampled && value !== undefined)
        responseIndexed = setResult(
          span,
          operation,
          normalize(value, legacy, policy.outputs && error === undefined),
          policy.outputs && error === undefined,
        );
      if (error !== undefined) {
        const type =
          data(error, "name") ??
          (error instanceof Error ? error.name : "Error");
        span.setAttribute(
          ATTR_ERROR_TYPE,
          typeof type === "string" ? type : "Error",
        );
        status(data(error, "status") ?? data(error, "statusCode"));
        if (policy.inputs && policy.outputs && error instanceof Error)
          span.recordException(error);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          ...(policy.inputs && policy.outputs && error instanceof Error
            ? { message: error.message }
            : {}),
        });
      }
      refresh(policy);
      if (policy.outputs && error === undefined) responseIndexed?.();
      if (policy.inputs) requestIndexed?.();
    });
    span.end();
  };
  // Sampling has run before copying any input fields or user attributes.
  safely(() => {
    refresh(policy);
    if (!sampled) return;
    let params: any = {};
    for (const key of [
      "max_tokens",
      "max_completion_tokens",
      "max_output_tokens",
      "maxTokens",
      "temperature",
      "top_p",
      "topP",
      "frequency_penalty",
      "presence_penalty",
    ]) {
      const v = data(raw, key);
      if (typeof v === "number") params[key] = v;
    }
    params.max_tokens ??= params.maxTokens;
    params.top_p ??= params.topP;
    if (policy.inputs) {
      if (legacy)
        params[
          operation === "chat"
            ? "messages"
            : operation === "completion"
              ? "prompt"
              : "input"
        ] = snapshot(args[1]);
      else
        for (const key of [
          "messages",
          "prompt",
          "input",
          "instructions",
          "tools",
          "functions",
        ]) {
          const value = data(raw, key);
          if (value !== undefined) params[key] = snapshot(value);
        }
      // Legacy schemas use camelCase function and tool fields.
      if (legacy && operation === "chat")
        params.messages = params.messages?.map(normalizeMessage);
      if (legacy)
        for (const key of ["tools", "functions"]) {
          const value = data(raw, key);
          if (value !== undefined) params[key] = snapshot(value);
        }
    }
    requestIndexed = setRequest(span, operation, params, policy.inputs);
    const extra = data(raw, "extraAttributes");
    if (extra && typeof extra === "object" && !types.isProxy(extra)) {
      // The guarded actual span filters denied keys before inspecting their values.

      for (const key of Object.keys(extra)) {
        const attr = Object.getOwnPropertyDescriptor(extra, key);
        if (!attr || !("value" in attr)) continue;
        if (
          key === RespanSpanAttributes.RESPAN_METADATA &&
          policy.inputs &&
          policy.outputs &&
          typeof attr.value === "string"
        ) {
          const inherited = (span as any).attributes?.[key];
          try {
            span.setAttribute(
              key,
              JSON.stringify({
                ...JSON.parse(inherited ?? "{}"),
                ...JSON.parse(attr.value),
              }),
            );
          } catch {
            span.setAttribute(key, attr.value);
          }
        } else span.setAttribute(key, attr.value);
      }
    }
  });
  const accumulator = new StreamAccumulator(operation);
  let received = false;
  const accumulated = () => (received ? accumulator.result() : undefined);
  const seen = new WeakSet<object>();
  const onResult = (value: any): any => {
    if (!value || typeof value[Symbol.asyncIterator] !== "function") {
      finish(value);
      return value;
    }
    if (seen.has(value)) return value;
    seen.add(value);
    const key =
      typeof value.iterator === "function" ? "iterator" : Symbol.asyncIterator;
    const iterator = value[key];
    let pending = 0;
    value[key] = function (this: any, ...iteratorArgs: any[]) {
      const source = iterator.apply(this, iteratorArgs);
      const invoke = async (method: string, input: any) => {
        pending++;
        try {
          const result = await context.with(active, () =>
            source[method](input),
          );
          if (result.done) finish(accumulated());
          else if (method === "next" || method === "throw")
            safely(() => {
              refresh(policy);
              if (sampled) {
                received = true;
                accumulator.add(
                  normalizeChunk(result.value, legacy, policy.outputs),
                );
              }
            });
          return result;
        } catch (error) {
          finish(accumulated(), error);
          throw error;
        } finally {
          pending--;
          if (method === "return") finish(accumulated());
        }
      };
      const wrapped: any = {
        next: (input: any) => invoke("next", input),
        [Symbol.asyncIterator]() {
          return this;
        },
      };
      if (source.return)
        wrapped.return = (input: any) => invoke("return", input);
      if (source.throw) wrapped.throw = (input: any) => invoke("throw", input);
      return wrapped;
    };
    const signal = value.controller?.signal;
    if (signal) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const abort = () => {
        if (!pending) finish(accumulated());
        else timer = setTimeout(() => finish(accumulated()), 0);
      };
      signal.addEventListener("abort", abort, { once: true });
      removeAbort = () => {
        signal.removeEventListener("abort", abort);
        if (timer) clearTimeout(timer);
      };
      if (signal.aborted) abort();
    }
    return value;
  };
  const callArgs = args.slice();
  const index = legacy ? 2 : 0;
  if (raw && typeof raw === "object" && Object.hasOwn(raw, "extraAttributes")) {
    const descriptors = Object.getOwnPropertyDescriptors(raw);
    delete descriptors.extraAttributes;
    callArgs[index] = Object.create(Object.getPrototypeOf(raw), descriptors);
  }
  // Legacy methods discard raw responses. Observe their real Azure pipeline instead.
  if (legacy)
    safely(() => {
      const pipeline = receiver?._client?.pipeline;
      if (typeof pipeline?.addPolicy !== "function") return;
      const policyName = `respanAzure${++policyID}`;
      pipeline.addPolicy({
        name: policyName,
        async sendRequest(request: any, next: any) {
          const belongs = trace.getSpan(context.active()) === span;
          const response = await next(request);
          if (belongs) status(response.status);
          return response;
        },
      });
      cleanup = () => pipeline.removePolicy({ name: policyName });
    });
  try {
    const result = context.with(active, () =>
      original.apply(receiver, callArgs),
    );
    if (typeof result?.parse === "function" && result.responsePromise) {
      const decorate = (promise: any): any => {
        let parsing = false;
        const parse = promise.parse;
        let parsed: any;
        promise.parse = function (this: any) {
          parsing = true;
          return (parsed ??= context
            .with(active, () => parse.call(this))
            .then(onResult, (error: any) => {
              finish(undefined, error);
              throw error;
            }));
        };
        const asResponse = promise.asResponse;
        promise.asResponse = function (this: any) {
          return asResponse.call(this).then(
            (response: any) => {
              status(response.status);
              if (!parsing) finish();
              return response;
            },
            (error: any) => {
              finish(undefined, error);
              throw error;
            },
          );
        };
        const unwrap = promise._thenUnwrap;
        if (typeof unwrap === "function")
          promise._thenUnwrap = function (this: any, ...rest: any[]) {
            return decorate(unwrap.apply(this, rest));
          };
        promise.responsePromise.then(
          (props: any) => status(props.response?.status),
          (error: any) => finish(undefined, error),
        );
        return promise;
      };
      return decorate(result);
    }
    if (result && typeof result[Symbol.asyncIterator] === "function")
      return onResult(result);
    if (result && typeof result.then === "function") {
      // Native legacy promises have no lazy parsing or APIPromise helpers.
      result.then(onResult, (error: any) => finish(undefined, error));
      return result;
    }
    return onResult(result);
  } catch (error) {
    finish(undefined, error);
    throw error;
  }
}
function normalizeMessage(message: any): any {
  if (!message) return message;
  return {
    ...message,
    tool_calls: message.tool_calls ?? message.toolCalls,
    tool_call_id: message.tool_call_id ?? message.toolCallId,
    function_call: message.function_call ?? message.functionCall,
  };
}
function usage(value: any): any {
  const result: any = {};
  for (const [snake, camel] of [
    ["prompt_tokens", "promptTokens"],
    ["completion_tokens", "completionTokens"],
    ["total_tokens", "totalTokens"],
    ["input_tokens", "inputTokens"],
    ["output_tokens", "outputTokens"],
  ]) {
    const v = data(value, snake) ?? data(value, camel);
    if (typeof v === "number") result[snake] = v;
  }
  for (const key of ["input_tokens_details", "prompt_tokens_details"]) {
    const cached = data(data(value, key), "cached_tokens");
    if (typeof cached === "number") result[key] = { cached_tokens: cached };
  }
  return result;
}
function normalize(value: any, legacy: boolean, outputs: boolean): any {
  const out: any = {
    model: data(value, "model"),
    usage: usage(data(value, "usage")),
  };
  const responseStatus = data(value, "status");
  if (responseStatus === "failed") out.status = responseStatus;
  if (!outputs) return out;
  Object.assign(out, snapshot(value));
  out.usage = usage(data(value, "usage"));
  if (legacy)
    out.choices = out.choices?.map((choice: any) => ({
      ...choice,
      message: normalizeMessage(choice.message),
    }));
  return out;
}
function normalizeChunk(value: any, legacy: boolean, outputs: boolean): any {
  const out = normalize(value, legacy, outputs);
  if (legacy && outputs)
    out.choices = out.choices?.map((choice: any) => ({
      ...choice,
      delta: normalizeMessage(choice.delta),
    }));
  if (!outputs && data(value, "response"))
    out.response = normalize(data(value, "response"), legacy, false);
  const type = data(value, "type");
  if (typeof type === "string") out.type = type;
  return out;
}
