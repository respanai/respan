import { types } from "node:util";
import { createRequire } from "node:module";
import {
  context,
  createContextKey,
  SamplingDecision,
  SpanKind,
  TraceFlags,
  trace,
  type Context,
  type Span,
  type SpanContext,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

// Local compatibility for clients using the historical LM-only key.
const suppressLanguageModel = createContextKey(
  "suppress_language_model_instrumentation",
);
const observedDenials = new WeakSet<object>();

export function own(value: unknown, key: string): any {
  if (
    !value ||
    (typeof value !== "object" && typeof value !== "function") ||
    types.isProxy(value)
  )
    return undefined;
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && "value" in descriptor ? descriptor.value : undefined;
  } catch {
    return undefined;
  }
}

/** Copy only data descriptors. Never call customer getters, serializers or iterators. */
export function safeCopy(value: unknown, seen = new WeakSet<object>()): any {
  if (value === null || typeof value === "string" || typeof value === "boolean")
    return value;
  if (typeof value === "number")
    return Number.isFinite(value) ? value : undefined;
  if (typeof value === "bigint") return `${value}`;
  if (
    !value ||
    typeof value !== "object" ||
    types.isProxy(value) ||
    seen.has(value)
  )
    return undefined;
  try {
    const prototype = Object.getPrototypeOf(value);
    if (
      !Array.isArray(value) &&
      prototype !== Object.prototype &&
      prototype !== null
    )
      return undefined;
    seen.add(value);
    const output: any = Array.isArray(value) ? [] : {};
    for (const [key, descriptor] of Object.entries(
      Object.getOwnPropertyDescriptors(value),
    )) {
      if (
        !descriptor.enumerable ||
        !("value" in descriptor) ||
        key === "toJSON"
      )
        continue;
      const copied = safeCopy(descriptor.value, seen);
      if (copied !== undefined)
        Object.defineProperty(output, key, {
          value: copied,
          enumerable: true,
          configurable: true,
          writable: true,
        });
    }
    seen.delete(value);
    return output;
  } catch {
    seen.delete(value);
    return undefined;
  }
}

function denied(value: unknown): boolean {
  return (
    value === false ||
    (typeof value === "string" &&
      ["false", "0", "off", "no"].includes(value.trim().toLowerCase()))
  );
}

const safeAttributes = new Set([
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_SESSION_ID,
  RespanSpanAttributes.RESPAN_THREADS_ID,
  RespanSpanAttributes.RESPAN_TRACE_GROUP_ID,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND,
  RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL,
  RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN,
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  SpanAttributes.TRACELOOP_WORKFLOW_NAME,
  SpanAttributes.LLM_SYSTEM,
  SpanAttributes.LLM_REQUEST_TYPE,
  SpanAttributes.LLM_REQUEST_MODEL,
  SpanAttributes.LLM_RESPONSE_MODEL,
  "telemetry.sdk.name",
  "telemetry.sdk.version",
  "allow_trace_content",
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
  SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  "error.type",
]);
function safeAttribute(key: string): boolean {
  return safeAttributes.has(key);
}

/** A run owns its policy. Vetoes survive queueing, deactivation and exporter contexts. */
export class PiCapturePolicy {
  public content: boolean;
  public emit = true;
  private readonly parent?: Span;
  private admitted = false;
  public sampledContext?: SpanContext;
  public samplingAttributes: Record<string, unknown> = {};

  constructor(
    traceContent = true,
    enabled = true,
    private readonly ancestor?: PiCapturePolicy,
  ) {
    this.content = traceContent && enabled;
    this.parent = trace.getSpan(context.active());
    this.observe();
  }

  /** Apply the actual installed OTel SDK sampler once for this owned span. */
  admit(
    traceId: string,
    spanId: string,
    parent: Context,
    name: string,
    attributes: Record<string, unknown>,
  ): void {
    if (this.admitted) return;
    this.admitted = true;
    this.observe();
    try {
      const tracer = trace.getTracer("@respan/instrumentation-pi");
      if (
        types.isProxy(tracer) ||
        !nativeTracerPrototype ||
        Object.getPrototypeOf(tracer) !== nativeTracerPrototype
      ) {
        this.emit = false;
        this.content = false;
        return;
      }
      const sampler = own(tracer, "_sampler");
      const shouldSample = dataMethod(sampler, "shouldSample");
      if (!shouldSample) {
        this.emit = false;
        this.content = false;
        return;
      }
      const result = shouldSample.call(
        sampler,
        parent,
        traceId,
        name,
        SpanKind.INTERNAL,
        safeCopy(attributes) ?? {},
        [],
      );
      const decision = own(result, "decision");
      this.emit &&= decision === SamplingDecision.RECORD_AND_SAMPLED;
      this.samplingAttributes = safeCopy(own(result, "attributes")) ?? {};
      if (denied(own(this.samplingAttributes, "allow_trace_content")))
        this.content = false;
      this.sampledContext = {
        traceId,
        spanId,
        traceFlags:
          decision === SamplingDecision.RECORD_AND_SAMPLED
            ? TraceFlags.SAMPLED
            : TraceFlags.NONE,
        traceState:
          own(result, "traceState") ?? trace.getSpanContext(parent)?.traceState,
      };
    } catch {
      this.emit = false;
    }
    if (!this.emit) this.content = false;
  }

  get recording(): boolean {
    return this.admitted && this.emit;
  }

  observe(evaluateContext = true): void {
    if (this.ancestor) {
      this.ancestor.observe(evaluateContext);
      this.content &&= this.ancestor.content;
      this.emit &&= this.ancestor.emit;
    }
    if (
      denied(process.env.RESPAN_TRACE_CONTENT) ||
      denied(process.env.TRACELOOP_TRACE_CONTENT)
    )
      this.content = false;
    if (evaluateContext) {
      const active = context.active();
      if (
        isTracingSuppressed(active) ||
        active.getValue(suppressLanguageModel) === true
      )
        this.emit = false;
      if (denied(active.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT))) {
        this.content = false;
        const parent = trace.getSpan(active);
        if (parent) observedDenials.add(parent);
      }
      this.observeParent(trace.getSpan(active));
    }
    this.observeParent(this.parent);
    if (!this.emit) this.content = false;
  }

  private observeParent(parent: Span | undefined): void {
    if (!parent) return;
    if (types.isProxy(parent)) {
      this.content = false;
      this.emit = false;
      return;
    }
    try {
      const id = safeSpanContext(parent);
      if (!id) {
        this.emit = false;
        this.content = false;
        return;
      }
      if ((id.traceFlags & 1) === 0) this.emit = false;
      const attributes = own(parent, "attributes");
      if (denied(own(attributes, "allow_trace_content")))
        observedDenials.add(parent);
      if (observedDenials.has(parent)) this.content = false;
      if (!attributes && !id.isRemote) this.content = false;
    } catch {
      this.content = false;
    }
  }

  snapshot(value: unknown, fields: readonly string[] = []): any {
    this.observe();
    if (!this.admitted) this.content = false;
    if (this.content) return safeCopy(value);
    const output: Record<string, unknown> = {};
    for (const key of fields) {
      const copied = safeCopy(own(value, key));
      if (copied !== undefined) output[key] = copied;
    }
    return output;
  }

  guard(span: ReadableSpan): void {
    const state: Record<string, any> = {
      attributes: span.attributes,
      events: span.events,
      status: span.status,
    };
    const refresh = () => {
      // Exporters intentionally suppress tracing during transport. Only original
      // application observations and readable-span/ancestor vetoes apply here.
      this.observe(false);
      if (denied(own(state.attributes, "allow_trace_content")))
        this.content = false;
      if (!this.content) {
        for (const key of Object.keys(state.attributes))
          if (!safeAttribute(key)) delete state.attributes[key];
        state.events = [];
        state.status = { code: own(state.status, "code") ?? 0 };
      }
      if (!this.emit)
        state.attributes[RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN] = true;
    };
    for (const field of ["attributes", "events", "status"]) {
      Object.defineProperty(span, field, {
        enumerable: true,
        configurable: false,
        get: () => {
          refresh();
          return this.content ? state[field] : safeCopy(state[field]);
        },
        set: (value) => {
          state[field] = safeCopy(value) ?? (field === "events" ? [] : {});
          refresh();
        },
      });
    }
    refresh();
  }
}

// OTel 2.10 owns Tracer in sdk-trace-base; 2.12 moved it to sdk-trace.
// Bind to exact installed class identity and data descriptors, never an unknown
// custom provider's private fields or getters. Unknown providers fail closed.
let nativeTracerPrototype: object | undefined;
try {
  const require = createRequire(import.meta.url);
  const base = createRequire(require.resolve("@opentelemetry/sdk-trace-base"));
  let Tracer;
  try {
    Tracer = base("./Tracer.js").Tracer;
  } catch {
    Tracer = base("@opentelemetry/sdk-trace/build/src/Tracer.js").Tracer;
  }
  nativeTracerPrototype = Tracer?.prototype;
} catch {
  /* Unsupported/private ABI: fail closed. */
}
function dataMethod(value: any, key: string): Function | undefined {
  if (!value || typeof value !== "object" || types.isProxy(value))
    return undefined;
  for (let current = value; current; current = Object.getPrototypeOf(current)) {
    if (types.isProxy(current)) return undefined;
    const descriptor = Object.getOwnPropertyDescriptor(current, key);
    if (descriptor)
      return "value" in descriptor && typeof descriptor.value === "function"
        ? descriptor.value
        : undefined;
  }
  return undefined;
}

export function safeSpanContext(span: unknown): SpanContext | undefined {
  try {
    return dataMethod(span, "spanContext")?.call(span);
  } catch {
    return undefined;
  }
}
