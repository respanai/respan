import {
  context,
  createContextKey,
  trace,
  type Context,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { types } from "node:util";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

// Local compatibility for clients using the historical language-model gate.
const suppressLanguageModel = createContextKey(
  "suppress_language_model_instrumentation",
);
export const internalWriterCall = createContextKey(
  "respan.writer.internal_call",
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
  ceiling(policy, inherited);
  const attrs = data(parent, "attributes");
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
export function strip(attrs: Record<string, any>, policy: CapturePolicy): void {
  refresh(policy);
  if (denied(data(attrs, "allow_trace_content")))
    policy.inputs = policy.outputs = false;
  for (const key of Object.keys(attrs)) {
    const descriptor = Object.getOwnPropertyDescriptor(attrs, key);
    if (
      !descriptor ||
      !("value" in descriptor) ||
      (descriptor.value &&
        typeof descriptor.value === "object" &&
        types.isProxy(descriptor.value))
    ) {
      delete attrs[key];
      continue;
    }
    const input =
      key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
      key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
      key.startsWith(`${SpanAttributes.LLM_PROMPTS}.`) ||
      key.startsWith("gen_ai.input.") ||
      key === "gen_ai.system_instructions" ||
      key.startsWith("gen_ai.tool.");
    const output =
      key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
      key.startsWith(`${SpanAttributes.LLM_COMPLETIONS}.`) ||
      key.startsWith("gen_ai.output.");
    const shared =
      key === RespanSpanAttributes.RESPAN_METADATA ||
      key.startsWith(`${RespanSpanAttributes.RESPAN_METADATA}.`) ||
      key.startsWith("error.") ||
      key.startsWith("exception.");
    if (
      (!policy.inputs && input) ||
      (!policy.outputs && output) ||
      ((!policy.inputs || !policy.outputs) && shared)
    )
      delete attrs[key];
  }
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
        attrs = snapshot(value) ?? {};
        strip(attrs, policy);
      },
    },
    events: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        events = snapshot(events) ?? [];
        if (!policy.inputs || !policy.outputs) events.length = 0;
        return events;
      },
      set(value) {
        events = Array.isArray(value) ? (snapshot(value) ?? []) : [];
        if (!policy.inputs || !policy.outputs) events.length = 0;
      },
    },
    status: {
      enumerable: true,
      configurable: false,
      get() {
        strip(attrs, policy);
        status = snapshot(status) ?? { code: 0 };
        if ((!policy.inputs || !policy.outputs) && status.message !== undefined)
          delete status.message;
        return status;
      },
      set(value) {
        status = snapshot(value) ?? { code: 0 };
        if ((!policy.inputs || !policy.outputs) && status.message !== undefined)
          delete status.message;
      },
    },
  });
  strip(attrs, policy);
  return span;
}
