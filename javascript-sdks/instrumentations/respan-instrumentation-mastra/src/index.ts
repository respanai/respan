import type {
  AnyExportedSpan,
  ObservabilityExporter,
  TracingEvent,
} from "@mastra/core/observability";
import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  admissionAttributes,
  contentAttributes,
  data,
  metadataAttributes,
  scalarAttributes,
} from "./_translator.js";
import { capture, guard, refresh, type CapturePolicy } from "./_privacy.js";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

export interface MastraInstrumentorOptions {
  excludeSpanTypes?: string[];
  traceContent?: boolean;
}
interface SpanState {
  span?: Span;
  context: Context;
  parent?: SpanState;
  policy: CapturePolicy;
  emit: boolean;
  ended: boolean;
}

/** Mastra's native event exporter, using the application's active OTEL provider. */
export class MastraInstrumentor implements ObservabilityExporter {
  public readonly name = "mastra";
  private _enabled = true;
  private readonly _excluded: Set<string>;
  private readonly _states = new Map<string, SpanState>();
  private readonly _traces = new Map<string, Set<string>>();
  private readonly _events = new WeakSet<object>();
  private readonly _traceContent: boolean;

  constructor(options: MastraInstrumentorOptions = {}) {
    this._excluded = new Set([
      "model_chunk",
      ...(options.excludeSpanTypes ?? []),
    ]);
    this._traceContent = options.traceContent !== false;
  }
  activate(): void {
    this._enabled = true;
  }
  deactivate(): void {
    this._enabled = false;
    this._close();
  }
  onTracingEvent(event: TracingEvent): void {
    // Exporter failures must never affect Mastra results or callbacks.
    try {
      this._handle(event);
    } catch {}
  }
  async exportTracingEvent(event: TracingEvent): Promise<void> {
    this.onTracingEvent(event);
  }
  async flush(): Promise<void> {}
  async shutdown(): Promise<void> {
    this.deactivate();
  }

  private _handle(event: TracingEvent): void {
    if (!this._enabled || this._events.has(event)) return;
    this._events.add(event);
    const native = event.exportedSpan;
    const key = `${native.traceId}:${native.id}`;
    if (event.type === "span_started") {
      if (this._states.has(key)) return;
      const active = context.active();
      const parent = native.parentSpanId
        ? this._states.get(`${native.traceId}:${native.parentSpanId}`)
        : undefined;
      const parentContext = parent?.context ?? active;
      const excluded = this._excluded.has(native.type);
      const policy = capture(
        { traceContent: this._traceContent },
        parentContext,
      );
      const entry = capture({ traceContent: this._traceContent }, active);
      policy.inputs &&= entry.inputs;
      policy.outputs &&= entry.outputs;
      policy.emit &&= entry.emit && (parent?.emit ?? true);
      if (parent) {
        policy.inputs &&= parent.policy.inputs;
        policy.outputs &&= parent.policy.outputs;
      }
      const state: SpanState = {
        context: parentContext,
        parent,
        policy,
        emit: policy.emit,
        ended: false,
      };
      this._states.set(key, state);
      const traceKeys = this._traces.get(native.traceId) ?? new Set<string>();
      traceKeys.add(key);
      this._traces.set(native.traceId, traceKeys);
      if (excluded || !policy.emit) return;
      const span = trace
        .getTracer("@respan/instrumentation-mastra", "0.1.0")
        .startSpan(
          native.name,
          {
            kind: SpanKind.INTERNAL,
            startTime: native.startTime,
            attributes: admissionAttributes(native),
          },
          parentContext,
        );
      state.span = span;
      state.context = trace.setSpan(parentContext, span);
      if (!span.isRecording() || (span.spanContext().traceFlags & 1) === 0) {
        policy.emit = policy.inputs = policy.outputs = false;
        state.emit = false;
        return;
      }
      guard(span as unknown as ReadableSpan, policy);
      refresh(policy);
      span.setAttributes(scalarAttributes(native));
      return;
    }
    if (event.type !== "span_ended") return;
    const state = this._states.get(key);
    // An ended-only replay has no immutable entry context or sampler decision.
    if (!state || state.ended) return;
    state.ended = true;
    const actual = state.span;
    if (!actual || !actual.isRecording()) {
      this._release(native.traceId);
      return;
    }
    if (!state.emit) {
      actual.end(native.endTime);
      this._release(native.traceId);
      return;
    }
    const attrs = (actual as Span & { attributes?: Record<string, unknown> })
      .attributes;
    for (let ancestor = state.parent; ancestor; ancestor = ancestor.parent) {
      refresh(ancestor.policy);
      state.policy.inputs &&= ancestor.policy.inputs;
      state.policy.outputs &&= ancestor.policy.outputs;
    }
    refresh(state.policy);
    // Reading the guarded actual span observes any late allow_trace_content veto.
    void (actual as unknown as ReadableSpan).attributes;
    const allowContent = state.policy.inputs && state.policy.outputs;
    actual.setAttributes(scalarAttributes(native));
    if (allowContent) {
      actual.setAttributes(
        metadataAttributes(
          native,
          data(attrs, RespanSpanAttributes.RESPAN_METADATA),
        ),
      );
      actual.setAttributes(contentAttributes(native));
    }
    if (native.errorInfo) {
      const message = allowContent
        ? data(native.errorInfo, "message")
        : undefined;
      actual.setStatus({
        code: SpanStatusCode.ERROR,
        ...(typeof message === "string" ? { message } : {}),
      });
    }
    actual.end(native.endTime);
    this._release(native.traceId);
  }
  private _release(traceId: string): void {
    const keys = this._traces.get(traceId);
    if (!keys || [...keys].some((key) => !this._states.get(key)?.ended)) return;
    for (const key of keys) this._states.delete(key);
    this._traces.delete(traceId);
  }
  private _close(): void {
    for (const state of this._states.values()) {
      if (!state.ended) {
        state.ended = true;
        state.span?.end();
      }
    }
    this._states.clear();
    this._traces.clear();
  }
}
export { MastraInstrumentor as RespanMastraExporter };
