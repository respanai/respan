import {
  context,
  createContextKey,
  trace,
  type Context,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions/incubating";
import { types } from "node:util";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

const suppressLanguageModel = createContextKey(
  "suppress_language_model_instrumentation",
);
export const internalCall = createContextKey("respan.cohere.internal_call");
export interface CaptureOptions {
  traceContent?: boolean;
  recordInputs?: boolean;
  recordOutputs?: boolean;
}
export interface Policy {
  inputs: boolean;
  outputs: boolean;
  emit: boolean;
  parent?: any;
}
const policies = new WeakMap<object, Policy>();
const getters = new WeakMap<object, () => Record<string, any>>();

export function data(value: unknown, key: PropertyKey): any {
  if (
    value === null ||
    (typeof value !== "object" && typeof value !== "function") ||
    types.isProxy(value)
  )
    return undefined;
  if (key === "attributes" && getters.has(value as object))
    return getters.get(value as object)!();
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}
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
function inspect(
  policy: Policy,
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
  if (inherited) {
    policy.inputs &&= inherited.inputs;
    policy.outputs &&= inherited.outputs;
    policy.emit &&= inherited.emit;
  }
  if (denied(data(data(parent, "attributes"), "allow_trace_content"))) {
    policy.inputs = policy.outputs = false;
    if (inherited) inherited.inputs = inherited.outputs = false;
  }
  if (inherited?.parent) inspect(policy, inherited.parent, visited);
}
export function refresh(policy: Policy): Policy {
  if (
    denied(process.env.RESPAN_TRACE_CONTENT) ||
    denied(process.env.TRACELOOP_TRACE_CONTENT)
  )
    policy.inputs = policy.outputs = false;
  inspect(policy, policy.parent);
  if (!policy.emit) policy.inputs = policy.outputs = false;
  return policy;
}
export function capture(
  options: CaptureOptions,
  ctx: Context = context.active(),
): Policy {
  const parent = trace.getSpan(ctx);
  const policy: Policy = {
    inputs: options.traceContent !== false && options.recordInputs !== false,
    outputs: options.traceContent !== false && options.recordOutputs !== false,
    emit:
      !isTracingSuppressed(ctx) && ctx.getValue(suppressLanguageModel) !== true,
    parent,
  };
  if (denied(ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)))
    policy.inputs = policy.outputs = false;
  if (parent && !policies.has(parent))
    policies.set(parent, { ...policy, parent: undefined });
  return refresh(policy);
}
const safeKeys = new Set<string>([
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  SpanAttributes.LLM_SYSTEM,
  SpanAttributes.LLM_REQUEST_TYPE,
  SpanAttributes.LLM_REQUEST_MODEL,
  SpanAttributes.LLM_REQUEST_MAX_TOKENS,
  SpanAttributes.LLM_REQUEST_TEMPERATURE,
  SpanAttributes.LLM_REQUEST_TOP_P,
  SpanAttributes.LLM_TOP_K,
  SpanAttributes.LLM_FREQUENCY_PENALTY,
  SpanAttributes.LLM_PRESENCE_PENALTY,
  SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
  SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  "llm.usage.cache_read_input_tokens",
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
  "status_code",
  "allow_trace_content",
]);
function statusCode(value: unknown): number {
  return value === 0 || value === 1 || value === 2 ? value : 0;
}
function primitive(value: unknown): boolean {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}
function allowed(key: string, policy: Policy): boolean {
  if (policy.inputs && policy.outputs) return true;
  if (safeKeys.has(key)) return true;
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
  return false;
}
function clean(attrs: any, policy: Policy): Record<string, any> {
  refresh(policy);
  if (denied(data(attrs, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  if (!attrs || types.isProxy(attrs)) return {};
  for (const key of Object.keys(attrs)) {
    // Filter the key before looking at its value or traversing any nested payload.
    if (!allowed(key, policy)) {
      delete attrs[key];
      continue;
    }
    const descriptor = Object.getOwnPropertyDescriptor(attrs, key);
    if (
      !descriptor ||
      !("value" in descriptor) ||
      (safeKeys.has(key) && !primitive(descriptor.value))
    )
      delete attrs[key];
  }
  return attrs;
}
function copyAttributes(value: unknown, policy: Policy): Record<string, any> {
  refresh(policy);
  if (denied(data(value, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  if (!value || typeof value !== "object" || types.isProxy(value)) return {};
  const out: Record<string, any> = {};
  for (const key of Object.keys(value)) {
    if (!allowed(key, policy)) continue;
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !("value" in descriptor)) continue;
    if (safeKeys.has(key) && !primitive(descriptor.value)) continue;
    const copied = snapshot(descriptor.value);
    if (copied !== undefined)
      Object.defineProperty(out, key, {
        value: copied,
        enumerable: true,
        configurable: true,
        writable: true,
      });
  }
  return out;
}
/** Guard the actual RecordingSpan, including late processor writes and replacements. */
export function guard(span: any, policy: Policy): void {
  policies.set(span, policy);
  let attrs = copyAttributes(data(span, "attributes"), policy);
  let events: any[] =
    policy.inputs && policy.outputs
      ? (snapshot(data(span, "events")) ?? [])
      : [];
  let status: any =
    policy.inputs && policy.outputs
      ? (snapshot(data(span, "status")) ?? { code: 0 })
      : { code: statusCode(data(data(span, "status"), "code")) };
  const readAttrs = () => clean(attrs, policy);
  getters.set(span, readAttrs);
  Object.defineProperties(span, {
    attributes: {
      enumerable: true,
      configurable: false,
      get: readAttrs,
      set(value) {
        attrs = copyAttributes(value, policy);
      },
    },
    events: {
      enumerable: true,
      configurable: false,
      get() {
        refresh(policy);
        if (!policy.inputs || !policy.outputs) events = [];
        return events;
      },
      set(value) {
        refresh(policy);
        events = policy.inputs && policy.outputs ? (snapshot(value) ?? []) : [];
      },
    },
    status: {
      enumerable: true,
      configurable: false,
      get() {
        refresh(policy);
        if (!policy.inputs || !policy.outputs)
          status = { code: statusCode(data(status, "code")) };
        return status;
      },
      set(value) {
        refresh(policy);
        status =
          policy.inputs && policy.outputs
            ? (snapshot(value) ?? { code: 0 })
            : { code: statusCode(data(value, "code")) };
      },
    },
  });
  for (const method of [
    "setAttribute",
    "setAttributes",
    "addEvent",
    "recordException",
    "setStatus",
  ] as const) {
    const original = span[method];
    if (typeof original !== "function") continue;
    Object.defineProperty(span, method, {
      configurable: false,
      writable: false,
      value: function (...args: any[]) {
        refresh(policy);
        if (method === "setAttribute") {
          if (
            !allowed(args[0], policy) ||
            (safeKeys.has(args[0]) && !primitive(args[1]))
          )
            return this;
        }
        if (method === "setAttributes")
          args[0] = copyAttributes(args[0], policy);
        if (
          (method === "addEvent" || method === "recordException") &&
          (!policy.inputs || !policy.outputs)
        )
          return this;
        if (method === "setStatus" && (!policy.inputs || !policy.outputs))
          args[0] = { code: statusCode(data(args[0], "code")) };
        return original.apply(this, args);
      },
    });
  }
}
