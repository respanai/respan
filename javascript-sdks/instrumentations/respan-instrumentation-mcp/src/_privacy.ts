import { context, trace, TraceFlags, type Span } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { types } from "node:util";

export interface CapturePolicy {
  traceContent: boolean;
  recordInputs: boolean;
  recordOutputs: boolean;
}
const states = new WeakMap<object, ContentState>();
const deniedAncestors = new WeakSet<object>();
const safeKeys = new Set<string>([
  RespanSpanAttributes.RESPAN_LOG_TYPE,
  RespanSpanAttributes.RESPAN_LOG_METHOD,
  SpanAttributes.TRACELOOP_ENTITY_NAME,
  SpanAttributes.TRACELOOP_ENTITY_PATH,
  "mcp.method",
  "mcp.operation",
  "mcp.transport",
  "mcp.request.id",
  "allow_trace_content",
]);
export function own(value: unknown, key: PropertyKey): unknown {
  if (
    !value ||
    (typeof value !== "object" && typeof value !== "function") ||
    types.isProxy(value)
  )
    return undefined;
  try {
    return Object.getOwnPropertyDescriptor(value, key)?.value;
  } catch {
    return undefined;
  }
}
function primitive(value: unknown): value is string | number | boolean {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}
function denied(value: unknown): boolean {
  return (
    value === false ||
    (typeof value === "string" &&
      ["false", "0", "off", "no"].includes(value.trim().toLowerCase()))
  );
}
export function environmentDenied(): boolean {
  return (
    denied(process.env.TRACELOOP_TRACE_CONTENT) ||
    denied(process.env.RESPAN_TRACE_CONTENT)
  );
}
export function isGuarded(span: unknown): boolean {
  return !!span && typeof span === "object" && states.has(span);
}
export function suppressed(): boolean {
  return isTracingSuppressed(context.active());
}
export class ContentState {
  private admitted: boolean;
  public readonly sampled: boolean;
  private inputs: boolean;
  private outputs: boolean;
  private ended = false;
  private readonly parent?: Span;
  private readonly ancestor?: ContentState;
  private readVeto?: () => void;
  constructor(policy: CapturePolicy, span: Span) {
    this.parent = trace.getSpan(context.active());
    this.ancestor = this.parent && states.get(this.parent);
    this.inputs = policy.recordInputs;
    this.outputs = policy.recordOutputs;
    // Sampling belongs to this span; it is never an ancestor privacy veto.
    this.sampled =
      span.isRecording() &&
      Boolean(span.spanContext().traceFlags & TraceFlags.SAMPLED);
    this.admitted =
      policy.traceContent &&
      !environmentDenied() &&
      !denied(context.active().getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)) &&
      !suppressed();
    states.set(span, this);
  }
  deny(): void {
    this.admitted = false;
  }
  observeAttributes(value: unknown): void {
    if (denied(own(value, "allow_trace_content"))) this.deny();
  }
  setVetoReader(reader: () => void): void {
    this.readVeto = reader;
  }
  allowed(ambient = true): boolean {
    // This reader only examines our own stored data, never a customer getter.
    this.readVeto?.();
    if (this.ancestor) {
      if (!this.ancestor.allowed(false)) this.deny();
      this.inputs &&= this.ancestor.inputs;
      this.outputs &&= this.ancestor.outputs;
    } else if (this.parent) {
      if (denied(own(own(this.parent, "attributes"), "allow_trace_content")))
        deniedAncestors.add(this.parent);
      if (deniedAncestors.has(this.parent)) this.deny();
    }
    if (
      !this.ended &&
      ambient &&
      (environmentDenied() ||
        denied(context.active().getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)) ||
        suppressed())
    )
      this.deny();
    return this.admitted;
  }
  input(): boolean {
    return this.sampled && this.allowed() && this.inputs;
  }
  output(): boolean {
    return this.sampled && this.allowed() && this.outputs;
  }
  open(): boolean {
    return this.sampled && this.allowed() && this.inputs && this.outputs;
  }
  end(): void {
    this.allowed();
    this.ended = true;
  }
  permitsKey(key: string): boolean {
    if (safeKeys.has(key)) return true;
    if (!this.sampled || !this.allowed()) return false;
    if (key === SpanAttributes.TRACELOOP_ENTITY_INPUT) return this.inputs;
    if (key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT) return this.outputs;
    return this.inputs && this.outputs;
  }
  permits(key: string, value: unknown): boolean {
    if (key === "allow_trace_content" && denied(value)) this.deny();
    if (!this.permitsKey(key)) return false;
    return safeKeys.has(key) || !this.open() ? primitive(value) : true;
  }
}
function filterAttributes(
  value: unknown,
  state: ContentState,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  if (!value || typeof value !== "object" || types.isProxy(value))
    return result;
  state.observeAttributes(value);
  for (const key of Object.keys(value)) {
    // Reject unknown channels before inspecting their value or descriptor.
    if (!state.permitsKey(key)) continue;
    const item = own(value, key);
    if (state.permits(key, item)) result[key] = item;
  }
  return result;
}
function numericStatus(value: unknown): { code: number } {
  const code = own(value, "code");
  return { code: code === 0 || code === 1 || code === 2 ? code : 0 };
}
/** Guard the actual SDK span, including processor-facing late replacements. */
export function guardSpan(span: Span, policy: CapturePolicy): ContentState {
  const state = new ContentState(policy, span);
  const object = span as unknown as Record<string, unknown>;
  const protect = (key: string, filter: (value: unknown) => unknown) => {
    const descriptor = Object.getOwnPropertyDescriptor(object, key);
    if (descriptor && !descriptor.configurable) return;
    let value = descriptor?.value;
    if (key === "attributes")
      state.setVetoReader(() => state.observeAttributes(value));
    try {
      Object.defineProperty(object, key, {
        enumerable: descriptor?.enumerable ?? true,
        configurable: false,
        get() {
          value = filter(value);
          return value;
        },
        set(next: unknown) {
          value = filter(next);
        },
      });
    } catch {
      /* Non-extensible external spans still use guarded methods. */
    }
  };
  protect("attributes", (value) => filterAttributes(value, state));
  protect("events", (value) => (state.open() ? value : []));
  protect("status", (value) => (state.open() ? value : numericStatus(value)));
  protect("_status", (value) => (state.open() ? value : numericStatus(value)));
  const setAttribute = span.setAttribute.bind(span);
  span.setAttribute = (key, value) => {
    if (state.permits(key, value)) setAttribute(key, value);
    return span;
  };
  span.setAttributes = (attributes) => {
    for (const [key, value] of Object.entries(
      filterAttributes(attributes, state),
    ))
      span.setAttribute(key, value as never);
    return span;
  };
  const addEvent = span.addEvent.bind(span);
  span.addEvent = ((...args: unknown[]) => {
    if (state.open()) (addEvent as (...args: unknown[]) => unknown)(...args);
    return span;
  }) as Span["addEvent"];
  const recordException = span.recordException.bind(span);
  span.recordException = (...args) => {
    if (state.open()) recordException(...args);
  };
  const setStatus = span.setStatus.bind(span);
  span.setStatus = (status) => {
    setStatus(state.open() ? status : numericStatus(status));
    return span;
  };
  const end = span.end.bind(span);
  span.end = (...args) => {
    state.end();
    end(...args);
  };
  return state;
}
