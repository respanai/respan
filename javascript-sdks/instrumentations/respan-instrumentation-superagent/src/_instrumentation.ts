/** Native SafetyClient operations and its public provider request/response boundaries. */
import {
  context,
  createContextKey,
  trace,
  SpanStatusCode,
  type Context,
  type Span,
} from "@opentelemetry/api";
import { types } from "node:util";
import { registerSpanTransformer, WORKFLOW_NAME_KEY } from "@respan/tracing";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import type * as SafetyAgentModule from "safety-agent";
import {
  SAFETY_AGENT_MODULE_NAME,
  SUPERAGENT_INSTRUMENTATION_NAME,
  SUPPORTED_METHODS,
} from "./_constants.js";
import {
  capture,
  data,
  snapshot,
  guard,
  observeAncestor,
  refresh,
  type CaptureOptions,
  type CapturePolicy,
} from "./_privacy.js";
import {
  identity,
  modelIdentity,
  requestAttributes,
  responseAttributes,
} from "./_span_attributes.js";
import { normalizeCallInput, safeJsonStringify } from "./_serialization.js";

type Fn = (this: any, ...args: any[]) => any;
interface Patch {
  target: object;
  key: string;
  descriptor: PropertyDescriptor;
  wrapped: Fn;
}
interface ModuleRecord {
  module: typeof SafetyAgentModule;
  prototype: object;
  owners: Set<SuperagentInstrumentor>;
  patches: Patch[];
  pending: number;
}
interface ModelCall {
  span: Span;
  policy: CapturePolicy;
  ended: boolean;
}
interface Operation {
  record: ModuleRecord;
  owner: SuperagentInstrumentor;
  span: Span;
  policy: CapturePolicy;
  scope: ModelScope;
}
interface ModelScope {
  operation: Operation;
  calls: Set<ModelCall>;
}
const OPERATION = createContextKey("respan.superagent.operation");
const MODEL_SCOPE = createContextKey("respan.superagent.provider_scope");
const records = new WeakMap<object, ModuleRecord>();
const liveRecords = new Set<ModuleRecord>();
let observer: ReturnType<typeof registerSpanTransformer> | undefined;

function ensureObserver() {
  if (observer) return;
  try {
    observer = registerSpanTransformer("respan.superagent.ancestors", {
      onStart: observeAncestor,
    });
  } catch {
    /* Standalone OTEL providers still get native sampling and owned-span privacy guards. */
  }
}

export interface SuperagentInstrumentorOptions extends CaptureOptions {
  methods?: string[];
  safetyAgentModule?: typeof SafetyAgentModule;
  metadata?: Record<string, unknown>;
}
function apply(span: Span, attrs: Record<string, any>) {
  for (const [key, value] of Object.entries(attrs)) {
    if (key === RespanSpanAttributes.RESPAN_METADATA) {
      let prior: any = {},
        next: any = {};
      try {
        prior = JSON.parse((span as any).attributes?.[key] ?? "{}");
      } catch {}
      try {
        next = JSON.parse(value);
      } catch {}
      span.setAttribute(key, safeJsonStringify({ ...prior, ...next }));
    } else span.setAttribute(key, value);
  }
}
function error(span: Span, policy: CapturePolicy, reason: unknown) {
  refresh(policy);
  const message =
    policy.inputs && policy.outputs ? data(reason, "message") : undefined;
  span.setStatus({
    code: SpanStatusCode.ERROR,
    ...(typeof message === "string" ? { message } : {}),
  });
  if (typeof message === "string")
    span.recordException({ name: "Error", message });
}
function endCall(call: ModelCall, reason?: unknown) {
  if (call.ended) return;
  call.ended = true;
  if (reason !== undefined) error(call.span, call.policy, reason);
  call.span.end();
}
function finishScope(
  scope: ModelScope,
  reason?: unknown,
  failedAttempt = false,
) {
  for (const call of scope.calls) {
    if (failedAttempt) call.span.setStatus({ code: SpanStatusCode.ERROR });
    endCall(call, reason);
  }
  scope.calls.clear();
}
function observe(
  result: unknown,
  done: (value: any, reason?: unknown) => void,
): unknown {
  if (types.isPromise(result)) {
    Promise.prototype.then.call(
      result,
      (value: any) => {
        try {
          done(value);
        } catch {}
      },
      (reason: any) => {
        try {
          done(undefined, reason);
        } catch {}
      },
    );
  } else done(result);
  return result;
}
function patch(
  record: ModuleRecord,
  target: object,
  key: string,
  factory: (original: Fn) => Fn,
) {
  if (record.patches.some((p) => p.target === target && p.key === key)) return;
  const descriptor = Object.getOwnPropertyDescriptor(target, key);
  if (
    !descriptor ||
    typeof descriptor.value !== "function" ||
    descriptor.configurable === false
  )
    return;
  const wrapped = factory(descriptor.value);
  Object.defineProperty(target, key, { ...descriptor, value: wrapped });
  record.patches.push({ target, key, descriptor, wrapped });
}
function cleanup(record: ModuleRecord) {
  if (record.owners.size || record.pending) return;
  for (const p of record.patches.reverse()) {
    if (Object.getOwnPropertyDescriptor(p.target, p.key)?.value === p.wrapped)
      Object.defineProperty(p.target, p.key, p.descriptor);
  }
  record.patches.length = 0;
  records.delete(record.prototype);
  liveRecords.delete(record);
  if (!liveRecords.size) {
    observer?.unregister();
    observer = undefined;
  }
}
function providers(record: ModuleRecord) {
  const registry = data(record.module, "providers");
  if (!registry || types.isProxy(registry)) return;
  for (const name of Object.keys(registry)) {
    const provider = data(registry, name);
    if (!provider || types.isProxy(provider)) continue;
    patch(
      record,
      provider,
      "transformRequest",
      (original) =>
        function (this: any, ...args: any[]) {
          const scope = context.active().getValue(MODEL_SCOPE) as
            ModelScope | undefined;
          if (!scope || scope.operation.record !== record)
            return original.apply(this, args);
          // A native fallback starts a new actual request. Prior failed attempts have no fabricated response.
          finishScope(scope, undefined, true);
          const policy = capture(scope.operation.owner.options);
          policy.inputs &&= refresh(scope.operation.policy).inputs;
          policy.outputs &&= scope.operation.policy.outputs;
          if (!policy.emit) return original.apply(this, args);
          const span = trace
            .getTracer(SUPERAGENT_INSTRUMENTATION_NAME)
            .startSpan("superagent.model", {
              attributes: modelIdentity(name, args[0]),
            });
          if (
            !span.isRecording() ||
            (span.spanContext().traceFlags & 1) === 0
          ) {
            span.end();
            return original.apply(this, args);
          }
          guard(span as any, policy);
          const call: ModelCall = { span, policy, ended: false };
          scope.calls.add(call);
          try {
            const body = original.apply(this, args);
            if (refresh(policy).inputs)
              apply(span, requestAttributes(body, args[1]));
            return body;
          } catch (reason) {
            endCall(call, reason);
            throw reason;
          }
        },
    );
    patch(
      record,
      provider,
      "transformResponse",
      (original) =>
        function (this: any, ...args: any[]) {
          const scope = context.active().getValue(MODEL_SCOPE) as
            ModelScope | undefined;
          const call =
            scope?.operation.record === record
              ? [...scope.calls].find((c) => !c.ended)
              : undefined;
          if (!call) return original.apply(this, args);
          try {
            const result = original.apply(this, args);
            const policy = refresh(call.policy);
            apply(
              call.span,
              responseAttributes(args[0], result, policy.outputs),
            );
            call.span.setStatus({ code: SpanStatusCode.OK });
            endCall(call);
            return result;
          } catch (reason) {
            endCall(call, reason);
            throw reason;
          }
        },
    );
  }
}
function internal(record: ModuleRecord, key: string) {
  patch(
    record,
    record.prototype,
    key,
    (original) =>
      function (this: any, ...args: any[]) {
        const operation = context.active().getValue(OPERATION) as
          Operation | undefined;
        if (!operation || operation.record !== record)
          return original.apply(this, args);
        const scope: ModelScope = { operation, calls: new Set() };
        try {
          const result = context.with(
            context.active().setValue(MODEL_SCOPE, scope),
            () => original.apply(this, args),
          );
          return observe(result, (_value, reason) =>
            finishScope(scope, reason),
          );
        } catch (reason) {
          finishScope(scope, reason);
          throw reason;
        }
      },
  );
}
function operation(record: ModuleRecord, method: string) {
  patch(
    record,
    record.prototype,
    method,
    (original) =>
      function (this: any, ...args: any[]) {
        const owner = [...record.owners].find(
          (o) => o.isActive() && o.methods.includes(method),
        );
        if (!owner) return original.apply(this, args);
        const ctx = context.active();
        const policy = capture(owner.options, ctx);
        if (!policy.emit)
          return context.with(
            ctx.deleteValue(OPERATION).deleteValue(MODEL_SCOPE),
            () => original.apply(this, args),
          );
        const span = trace
          .getTracer(SUPERAGENT_INSTRUMENTATION_NAME)
          .startSpan(
            `superagent.${method}`,
            { attributes: identity(method) },
            ctx,
          );
        if (!span.isRecording() || (span.spanContext().traceFlags & 1) === 0) {
          span.end();
          return context.with(
            ctx.deleteValue(OPERATION).deleteValue(MODEL_SCOPE),
            () => original.apply(this, args),
          );
        }
        guard(span as any, policy);
        const state = {} as Operation;
        Object.assign(state, {
          record,
          owner,
          span,
          policy,
          scope: { operation: state, calls: new Set() },
        });
        const active = trace
          .setSpan(ctx, span)
          .setValue(OPERATION, state)
          .setValue(MODEL_SCOPE, state.scope);
        record.pending++;
        try {
          if (refresh(policy).inputs) {
            span.setAttribute(
              SpanAttributes.TRACELOOP_ENTITY_INPUT,
              safeJsonStringify(normalizeCallInput(method, args)),
            );
            const workflow = ctx.getValue(WORKFLOW_NAME_KEY);
            if (typeof workflow === "string")
              span.setAttribute(
                SpanAttributes.TRACELOOP_WORKFLOW_NAME,
                workflow,
              );
            apply(span, {
              [RespanSpanAttributes.RESPAN_METADATA]: safeJsonStringify({
                ...snapshot(owner.options.metadata),
                integration: "superagent",
                superagent_method: method,
              }),
            });
          }
        } catch {
          /* Payload preparation must never replace a native SDK call. */
        }
        let settled = false;
        const done = (value: any, reason?: unknown) => {
          if (settled) return;
          settled = true;
          finishScope(state.scope, reason);
          if (reason !== undefined) error(span, policy, reason);
          else {
            if (refresh(policy).outputs && value !== undefined)
              span.setAttribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                safeJsonStringify(value),
              );
            if (policy.inputs && policy.outputs && method === "guard") {
              const classification = data(value, "classification");
              if (typeof classification === "string")
                apply(span, {
                  [RespanSpanAttributes.RESPAN_METADATA]: safeJsonStringify({
                    superagent_classification: classification,
                    guardrail_name: "superagent.guard",
                    triggered: classification === "block",
                  }),
                });
            }
            span.setStatus({ code: SpanStatusCode.OK });
          }
          span.end();
          record.pending--;
          cleanup(record);
        };
        try {
          providers(record);
          return observe(
            context.with(active, () => original.apply(this, args)),
            done,
          );
        } catch (reason) {
          done(undefined, reason);
          throw reason;
        }
      },
  );
}
export class SuperagentInstrumentor {
  public readonly name = SUPERAGENT_INSTRUMENTATION_NAME;
  public readonly options: SuperagentInstrumentorOptions;
  public readonly methods: string[];
  private _isInstrumented = false;
  private _pending?: Promise<void>;
  private _generation = 0;
  private _record?: ModuleRecord;
  constructor(options: SuperagentInstrumentorOptions = {}) {
    this.options = {};
    for (const key of [
      "traceContent",
      "recordInputs",
      "recordOutputs",
      "metadata",
      "safetyAgentModule",
      "methods",
    ] as const) {
      const value = data(options, key);
      if (value !== undefined) (this.options as any)[key] = value;
    }
    const requested = data(options, "methods");
    this.methods =
      Array.isArray(requested) && !types.isProxy(requested)
        ? snapshot(requested)
        : [...SUPPORTED_METHODS];
  }
  isActive(): boolean {
    return this._isInstrumented;
  }
  activate(): Promise<void> {
    if (this._isInstrumented) return Promise.resolve();
    if (this._pending) return this._pending;
    const generation = ++this._generation;
    const pending = Promise.resolve()
      .then(async () => {
        let module: typeof SafetyAgentModule;
        try {
          module =
            this.options.safetyAgentModule ??
            (await import(SAFETY_AGENT_MODULE_NAME));
        } catch {
          return;
        }
        if (generation !== this._generation) return;
        const prototype = data(data(module, "SafetyClient"), "prototype");
        if (!prototype) return;
        let record = records.get(prototype);
        if (!record) {
          record = {
            module,
            prototype,
            owners: new Set(),
            patches: [],
            pending: 0,
          };
          records.set(prototype, record);
          liveRecords.add(record);
        }
        ensureObserver();
        this._record = record;
        record.owners.add(this);
        this._isInstrumented = true;
        for (const method of this.methods) operation(record, method);
        internal(record, "guardSingleText");
        internal(record, "guardImage");
        providers(record);
      })
      .finally(() => {
        if (this._pending === pending) this._pending = undefined;
      });
    this._pending = pending;
    return pending;
  }
  deactivate(): void {
    this._generation++;
    this._pending = undefined;
    this._isInstrumented = false;
    const record = this._record;
    this._record = undefined;
    if (record) {
      record.owners.delete(this);
      cleanup(record);
    }
  }
}
