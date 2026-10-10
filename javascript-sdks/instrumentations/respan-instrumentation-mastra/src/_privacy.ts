import {
  context,
  createContextKey,
  trace,
  type Context,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { types } from "node:util";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

// Local compatibility for clients using the historical language-model gate.
const suppressLanguageModel = createContextKey(
  "suppress_language_model_instrumentation",
);

export interface CaptureOptions {
  traceContent?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
}
export interface CapturePolicy {
  inputs: boolean;
  outputs: boolean;
  emit: boolean;
  parent?: any;
}
const policies = new WeakMap<object, CapturePolicy>();

export function data(value: unknown, key: PropertyKey): any {
  if (
    value === null ||
    (typeof value !== "object" && typeof value !== "function") ||
    types.isProxy(value)
  )
    return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}

/** Snapshot data properties only: telemetry never evaluates customer code. */
export function snapshot(value: unknown, seen = new WeakSet<object>()): any {
  if (value === null || ["string", "number", "boolean"].includes(typeof value))
    return value;
  if (
    typeof value !== "object" ||
    types.isProxy(value) ||
    seen.has(value as object)
  )
    return undefined;
  seen.add(value as object);
  const out: any = Array.isArray(value) ? [] : {};
  for (const key of Object.keys(value as object)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !("value" in descriptor)) continue;
    const copied = snapshot(descriptor.value, seen);
    if (copied !== undefined)
      Object.defineProperty(out, key, {
        value: copied,
        enumerable: true,
        configurable: true,
        writable: true,
      });
  }
  seen.delete(value as object);
  return out;
}

function denied(value: unknown): boolean {
  return (
    value === false ||
    (typeof value === "string" &&
      ["false", "0", "off", "no"].includes(value.trim().toLowerCase()))
  );
}
function ceiling(policy: CapturePolicy, other?: CapturePolicy): void {
  if (!other) return;
  policy.inputs &&= other.inputs;
  policy.outputs &&= other.outputs;
  policy.emit &&= other.emit;
}
function inspectParent(
  policy: CapturePolicy,
  parent: any,
  visited = new Set<object>(),
): void {
  if (!parent || visited.has(parent)) return;
  if (types.isProxy(parent)) {
    policy.inputs = policy.outputs = false;
    return;
  }
  visited.add(parent);
  const inherited = policies.get(parent);
  // Only invoke an accessor that this module installed itself. Foreign spans
  // remain data-only; evaluating their getters could execute customer code.
  const attrs = inherited ? parent.attributes : data(parent, "attributes");
  ceiling(policy, inherited);
  if (denied(data(attrs, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  if (inherited?.parent) inspectParent(policy, inherited.parent, visited);
}
export function refresh(policy: CapturePolicy): CapturePolicy {
  if (
    denied(process.env.RESPAN_TRACE_CONTENT) ||
    denied(process.env.TRACELOOP_TRACE_CONTENT)
  )
    policy.inputs = policy.outputs = false;
  inspectParent(policy, policy.parent);
  if (!policy.emit) policy.inputs = policy.outputs = false;
  return policy;
}
export function capture(
  options: CaptureOptions = {},
  ctx: Context = context.active(),
): CapturePolicy {
  const parent = trace.getSpan(ctx);
  const policy: CapturePolicy = {
    inputs: options.traceContent !== false && options.recordInputs !== false,
    outputs: options.traceContent !== false && options.recordOutputs !== false,
    emit:
      !isTracingSuppressed(ctx) && ctx.getValue(suppressLanguageModel) !== true,
    parent,
  };
  if (denied(ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)))
    policy.inputs = policy.outputs = false;
  if (parent && !types.isProxy(parent)) {
    const flags = parent.spanContext().traceFlags;
    if ((flags & 1) === 0) policy.emit = false;
  }
  return refresh(policy);
}
export function observeAncestor(span: any, ctx?: Context): void {
  const prior = policies.get(span);
  if (!prior) policies.set(span, capture({}, ctx));
  else if (ctx) ceiling(prior, capture({}, ctx));
  const policy = policies.get(span)!;
  if (denied(data(data(span, "attributes"), "allow_trace_content")))
    policy.inputs = policy.outputs = false;
}
const safeAttributes = new Set<string>([
  "allow_trace_content",
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  SpanAttributes.TRACELOOP_WORKFLOW_NAME,
  SpanAttributes.TRACELOOP_SPAN_KIND,
  SpanAttributes.LLM_SYSTEM,
  SpanAttributes.LLM_REQUEST_TYPE,
  SpanAttributes.LLM_REQUEST_MODEL,
  SpanAttributes.LLM_RESPONSE_MODEL,
  SpanAttributes.LLM_REQUEST_MAX_TOKENS,
  SpanAttributes.LLM_REQUEST_TEMPERATURE,
  SpanAttributes.LLM_REQUEST_TOP_P,
  SpanAttributes.LLM_TOP_K,
  SpanAttributes.LLM_FREQUENCY_PENALTY,
  SpanAttributes.LLM_PRESENCE_PENALTY,
  SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
  SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID,
  RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID,
  RespanSpanAttributes.RESPAN_THREADS_ID,
  RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
  RespanSpanAttributes.RESPAN_ENVIRONMENT,
  RespanSpanAttributes.RESPAN_PROCESSORS,
  ATTR_GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
]);
function safeValue(value: unknown): boolean {
  if (["string", "number", "boolean"].includes(typeof value)) return true;
  if (!Array.isArray(value) || types.isProxy(value)) return false;
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (key === "length") continue;
    if (
      !("value" in descriptor) ||
      !["string", "number", "boolean"].includes(typeof descriptor.value)
    )
      return false;
  }
  return true;
}
export function strip(attrs: Record<string, any>, policy: CapturePolicy): void {
  refresh(policy);
  if (denied(data(attrs, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  const privateSpan = !policy.inputs || !policy.outputs;
  for (const key of Object.keys(attrs)) {
    const descriptor = Object.getOwnPropertyDescriptor(attrs, key);
    if (
      !descriptor ||
      !("value" in descriptor) ||
      (descriptor.value &&
        typeof descriptor.value === "object" &&
        types.isProxy(descriptor.value)) ||
      (privateSpan &&
        (!safeAttributes.has(key) || !safeValue(descriptor.value)))
    )
      delete attrs[key];
  }
}
function replacementAttributes(
  value: unknown,
  policy: CapturePolicy,
): Record<string, any> {
  refresh(policy);
  if (denied(data(value, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  if (policy.inputs && policy.outputs) return snapshot(value) ?? {};
  const out: Record<string, any> = {};
  if (!value || typeof value !== "object" || types.isProxy(value)) return out;
  for (const key of safeAttributes) {
    const item = data(value, key);
    if (safeValue(item)) out[key] = snapshot(item);
  }
  return out;
}

/** Keep the original readable object safe through late processors and queueing. */
export function guard(span: ReadableSpan, policy: CapturePolicy): ReadableSpan {
  policies.set(span, policy);
  let attrs: Record<string, any> = span.attributes;
  let events = span.events;
  let status = span.status;
  Object.defineProperties(span, {
    attributes: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        return attrs;
      },
      set(value) {
        attrs = replacementAttributes(value, policy);
        strip(attrs, policy);
      },
    },
    events: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        events =
          !policy.inputs || !policy.outputs ? [] : (snapshot(events) ?? []);
        return events;
      },
      set(value) {
        strip(attrs, policy);
        events =
          policy.inputs && policy.outputs && Array.isArray(value)
            ? (snapshot(value) ?? [])
            : [];
      },
    },
    status: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        status =
          !policy.inputs || !policy.outputs
            ? { code: data(status, "code") ?? 0 }
            : (snapshot(status) ?? { code: 0 });
        return status;
      },
      set(value) {
        strip(attrs, policy);
        status =
          !policy.inputs || !policy.outputs
            ? { code: data(value, "code") ?? 0 }
            : (snapshot(value) ?? { code: 0 });
      },
    },
  });
  strip(attrs, policy);
  return span;
}
