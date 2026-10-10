import {
  context,
  createContextKey,
  trace,
  type Context,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { types } from "node:util";
import type { Span } from "@opentelemetry/api";
type ReadableSpan = Span & {
  attributes: Record<string, any>;
  events: any[];
  status: any;
};
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  ATTR_ERROR_TYPE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import {
  ATTR_HTTP_STATUS_CODE,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

import {
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";

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
  ceiling(policy, inherited);
  const attrs = guarded.has(parent)
    ? parent.attributes
    : data(parent, "attributes");
  if (denied(data(attrs, "allow_trace_content"))) {
    policy.inputs = policy.outputs = false;
    if (inherited) inherited.inputs = inherited.outputs = false;
    else policies.set(parent, { inputs: false, outputs: false, emit: true });
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
  if (parent && !types.isProxy(parent)) {
    // Parent sampling is enforced by the real tracer sampler below.
    // A non-recording parent alone must not disable an AlwaysOn child sampler.
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
  ATTR_HTTP_RESPONSE_STATUS_CODE,
  ATTR_HTTP_STATUS_CODE,
  ATTR_ERROR_TYPE,
  SpanAttributes.LLM_REQUEST_MAX_TOKENS,
  SpanAttributes.LLM_REQUEST_TEMPERATURE,
  SpanAttributes.LLM_REQUEST_TOP_P,
  SpanAttributes.LLM_FREQUENCY_PENALTY,
  SpanAttributes.LLM_PRESENCE_PENALTY,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  "allow_trace_content",
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL,
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  SpanAttributes.TRACELOOP_WORKFLOW_NAME,
  SpanAttributes.LLM_REQUEST_TYPE,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
]);
function allowedKey(key: string, policy: CapturePolicy): boolean {
  if (safeAttributes.has(key)) return true;
  if (
    policy.inputs &&
    (key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
      key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
      key.startsWith(`${SpanAttributes.LLM_PROMPTS}.`))
  )
    return true;
  if (
    policy.outputs &&
    (key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
      key.startsWith(`${SpanAttributes.LLM_COMPLETIONS}.`))
  )
    return true;
  return policy.inputs && policy.outputs;
}
function safeValue(value: unknown): boolean {
  if (["string", "number", "boolean"].includes(typeof value)) return true;
  if (!Array.isArray(value) || types.isProxy(value)) return false;
  return Object.entries(Object.getOwnPropertyDescriptors(value)).every(
    ([key, d]) =>
      key === "length" ||
      ("value" in d &&
        ["string", "number", "boolean"].includes(typeof d.value)),
  );
}
export function strip(attrs: Record<string, any>, policy: CapturePolicy): void {
  refresh(policy);
  if (denied(data(attrs, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  const privateSpan = !policy.inputs || !policy.outputs;
  for (const key of Object.keys(attrs)) {
    if (privateSpan && !allowedKey(key, policy)) {
      delete attrs[key];
      continue;
    }
    const descriptor = Object.getOwnPropertyDescriptor(attrs, key);
    if (
      !descriptor ||
      !("value" in descriptor) ||
      (privateSpan && !safeValue(descriptor.value)) ||
      (descriptor.value &&
        typeof descriptor.value === "object" &&
        types.isProxy(descriptor.value))
    )
      delete attrs[key];
  }
}
function replacement(
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
    if (!allowedKey(key, policy)) continue;
    const item = data(value, key);
    if (safeValue(item)) out[key] = snapshot(item);
  }
  return out;
}
/** Keep the original readable object safe through late processors and queueing. */
export function guard(span: any, policy: CapturePolicy): any {
  policies.set(span, policy);
  guarded.add(span);
  const setAttribute = span.setAttribute;
  const setAttributes = span.setAttributes;
  span.setAttribute = function (key: string, value: unknown) {
    refresh(policy);
    if (key === "allow_trace_content" && denied(value))
      policy.inputs = policy.outputs = false;
    if (
      !allowedKey(key, policy) ||
      ((!policy.inputs || !policy.outputs) && !safeValue(value))
    )
      return this;
    return setAttribute.call(this, key, value);
  };
  span.setAttributes = function (value: unknown) {
    return setAttributes.call(this, replacement(value, policy));
  };
  let attrs: Record<string, any> = span.attributes ?? {};
  let events = span.events ?? [];
  let status = span.status ?? { code: 0 };
  Object.defineProperties(span, {
    attributes: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        return attrs;
      },
      set(value) {
        attrs = replacement(value, policy);
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
        refresh(policy);
        events =
          !policy.inputs || !policy.outputs
            ? []
            : Array.isArray(value)
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
            ? {
                code: [0, 1, 2].includes(data(status, "code"))
                  ? data(status, "code")
                  : 0,
              }
            : (snapshot(status) ?? { code: 0 });
        return status;
      },
      set(value) {
        refresh(policy);
        status =
          !policy.inputs || !policy.outputs
            ? {
                code: [0, 1, 2].includes(data(value, "code"))
                  ? data(value, "code")
                  : 0,
              }
            : (snapshot(value) ?? { code: 0 });
      },
    },
  });
  strip(attrs, policy);
  return span;
}
