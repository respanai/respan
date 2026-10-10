import {
  context,
  createContextKey,
  trace,
  type Context,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { types } from "node:util";

import {
  ATTR_GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { ATTR_HTTP_RESPONSE_STATUS_CODE } from "@opentelemetry/semantic-conventions";
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
const guarded = new WeakSet<object>();

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
  if (
    value &&
    (typeof value === "object" || typeof value === "function") &&
    types.isProxy(value)
  )
    return undefined;
  if (value instanceof Uint8Array && !types.isProxy(value))
    return new Uint8Array(value);
  if (value instanceof ArrayBuffer && !types.isProxy(value))
    return value.slice(0);
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
  const attrs = guarded.has(parent)
    ? parent.attributes
    : data(parent, "attributes");
  ceiling(policy, inherited);
  if (denied(data(attrs, "allow_trace_content"))) {
    policy.inputs = policy.outputs = false;
    if (!inherited)
      policies.set(parent, { inputs: false, outputs: false, emit: true });
    else inherited.inputs = inherited.outputs = false;
  }
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
  return refresh(policy);
}
export function attributesOf(
  span: object | undefined,
): Record<string, unknown> | undefined {
  if (!span) return undefined;
  return guarded.has(span)
    ? (span as any).attributes
    : data(span, "attributes");
}
const safeAttributes = new Set<string>([
  "allow_trace_content",
  ATTR_HTTP_RESPONSE_STATUS_CODE,
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
function indexedPayload(
  key: string,
  prefix: string,
  fields: string[],
): boolean {
  if (!key.startsWith(`${prefix}.`)) return false;
  const rest = key.slice(prefix.length + 1).split(".");
  return (
    rest.length === 2 && /^[0-9]+$/.test(rest[0]) && fields.includes(rest[1])
  );
}
function inputAttribute(key: string): boolean {
  return (
    key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
    key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
    indexedPayload(key, SpanAttributes.LLM_PROMPTS, [
      "role",
      "content",
      "tool_calls",
      "tool_call_id",
    ])
  );
}
function outputAttribute(key: string): boolean {
  return (
    key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
    indexedPayload(key, SpanAttributes.LLM_COMPLETIONS, [
      "role",
      "content",
      "tool_calls",
    ])
  );
}
function allowed(key: string, value: unknown, policy: CapturePolicy): boolean {
  return (
    (safeAttributes.has(key) ||
      (policy.inputs && inputAttribute(key)) ||
      (policy.outputs && outputAttribute(key))) &&
    safeValue(value)
  );
}
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
      (privateSpan && !allowed(key, descriptor.value, policy))
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
  for (const key of Object.keys(value)) {
    // Reject unknown keys before reading or copying payloads.
    if (
      !safeAttributes.has(key) &&
      !(policy.inputs && inputAttribute(key)) &&
      !(policy.outputs && outputAttribute(key))
    )
      continue;
    const item = data(value, key);
    if (allowed(key, item, policy)) out[key] = snapshot(item);
  }
  return out;
}

function statusCode(value: unknown): number {
  const code = data(value, "code");
  return code === 0 || code === 1 || code === 2 ? code : 0;
}

/** Keep the original readable object safe through late processors and queueing. */
export function guard<T extends object>(span: T, policy: CapturePolicy): T {
  const target = span as any;
  policies.set(span, policy);
  guarded.add(span);
  let attrs: Record<string, any> = target.attributes ?? {};
  let events = target.events ?? [];
  let status = target.status ?? { code: 0 };
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
            ? { code: statusCode(status) }
            : (snapshot(status) ?? { code: 0 });
        return status;
      },
      set(value) {
        strip(attrs, policy);
        status =
          !policy.inputs || !policy.outputs
            ? { code: statusCode(value) }
            : (snapshot(value) ?? { code: 0 });
      },
    },
  });
  strip(attrs, policy);
  return span;
}
