import { types } from "node:util";
import {
  context,
  createContextKey,
  trace,
  type Context,
  type Span,
  type SpanOptions,
  type Tracer,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_RESPONSE_ID,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

type Event = Record<string, any>;
type Subscriber = (event: Event, ctx?: unknown) => void;
type Disposer = () => void | Promise<void>;

export interface FlueRuntimeModule {
  observe?: (subscriber: Subscriber) => Disposer;
  instrument?: (instrumentation: any) => Disposer;
}

export interface FlueInstrumentorOptions {
  runtimeModule?: FlueRuntimeModule;
  workflowName?: string;
  traceContent?: boolean;
}

interface RecordState {
  span: Span;
  native: Span;
  event: Event;
  denied: boolean;
  parent?: RecordState;
  parentSpan?: Span;
  ended: boolean;
  readable?: { attributes: Event; events: any[]; status: Event };
}

interface Registration {
  runtime: FlueRuntimeModule;
  owner: FlueInstrumentor;
  owners: Set<FlueInstrumentor>;
  traceContent?: boolean;
  workflowName?: string;
}
const registrations = new WeakMap<FlueRuntimeModule, Registration>();

// These are accepted privacy signals, not exported telemetry aliases.
const CONTENT_FLAGS = [
  "allow_trace_content",
  "trace_content",
  "traceloop.trace_content",
  "respan.trace_content",
];
const LM_SUPPRESSION = createContextKey(
  "suppress_language_model_instrumentation",
);
const recordsBySpan = new WeakMap<object, RecordState>();
const recordsById = new Map<string, RecordState>();
const VERSION = "0.1.0";

/** Reuses Flue's official lifecycle and interceptor with canonical, complete content. */
export class FlueInstrumentor {
  readonly name = "flue";
  private active = false;
  private activating?: Promise<void>;
  private dispose?: Disposer;
  private delegate?: {
    observe: Subscriber;
    interceptor?: any;
    dispose?: Disposer;
  };
  private readonly records = new Set<RecordState>();
  private currentEvent?: Event;
  private registration?: Registration;

  constructor(private readonly options: FlueInstrumentorOptions = {}) {}

  activate(): Promise<void> {
    if (this.activating) return this.activating;
    if (this.active) return Promise.resolve();
    this.active = true;
    this.activating = this.install()
      .catch((error) => {
        this.active = false;
        throw error;
      })
      .finally(() => {
        this.activating = undefined;
      });
    return this.activating;
  }

  private async install(): Promise<void> {
    const runtime: FlueRuntimeModule =
      this.options.runtimeModule ??
      ((await import("@flue/runtime")) as unknown as FlueRuntimeModule);
    const upstream: any = await import("@flue/opentelemetry");
    if (!this.active) return;
    const existing = registrations.get(runtime);
    if (existing) {
      if (
        existing.traceContent !== (this.options.traceContent !== false) ||
        existing.workflowName !== this.options.workflowName
      )
        throw new Error(
          "Flue instrumentors sharing a runtime must use the same content and workflow options.",
        );
      existing.owners.add(this);
      this.registration = existing;
      return;
    }
    const registration: Registration = {
      runtime,
      owner: this,
      owners: new Set([this]),
      traceContent: this.options.traceContent !== false,
      workflowName: this.options.workflowName,
    };
    const tracer: Tracer = {
      startSpan: (name, options, parent) =>
        this.startSpan(name, options, parent),
      startActiveSpan: (..._args: any[]): any => {
        throw new Error("Flue adapter must use startSpan");
      },
    };
    if (runtime.instrument && upstream.createOpenTelemetryInstrumentation) {
      const native = upstream.createOpenTelemetryInstrumentation({
        tracer,
        content: false,
      });
      this.delegate = native;
      this.dispose = runtime.instrument({
        ...native,
        observe: (event: Event, ctx: unknown) => this.handleEvent(event, ctx),
      });
    } else if (runtime.observe && upstream.createOpenTelemetryObserver) {
      this.delegate = {
        observe: upstream.createOpenTelemetryObserver({
          tracer,
          exportContent: () => undefined,
          resolveRootContext: () => context.active(),
        }),
      };
      this.dispose = runtime.observe((event: Event, ctx: unknown) =>
        this.handleEvent(event, ctx),
      );
    } else {
      this.active = false;
      throw new Error(
        "Install matching released @flue/runtime and @flue/opentelemetry versions (beta.1 or 2.2.2).",
      );
    }
    this.registration = registration;
    registrations.set(runtime, registration);
  }

  async deactivate(): Promise<void> {
    this.active = false;
    await this.activating;
    const registration = this.registration;
    if (registration) {
      registration.owners.delete(this);
      if (registration.owner !== this) this.registration = undefined;
      if (registration.owners.size) return;
      registrations.delete(registration.runtime);
      await registration.owner.stopRegistration();
      registration.owner.registration = undefined;
      return;
    }
    await this.stopRegistration();
  }

  private async stopRegistration(): Promise<void> {
    const dispose = this.dispose;
    this.dispose = undefined;
    await dispose?.();
    for (const record of this.records) {
      if (!record.ended) record.span.end();
      recordsById.delete(spanKey(record.native));
    }
    this.records.clear();
    this.delegate = undefined;
  }

  isActive(): boolean {
    return this.active;
  }

  handleEvent(event: Event, ctx?: unknown): void {
    if ((!this.active && !this.registration?.owners.size) || !this.delegate)
      return;
    try {
      // Policy is evaluated before copying/converting caller-owned content.
      const skeleton = eventSkeleton(event);
      const ambient = context.active();
      const terminal = [
        "turn",
        "tool",
        "task",
        "operation",
        "compaction",
        "run_end",
      ].includes(skeleton.type);
      const lateSuppression = suppressed(ambient);
      if (lateSuppression && !terminal) return;
      this.currentEvent = skeleton;
      const matching = [...this.records].filter(
        (record) => !record.ended && matches(record.event, skeleton),
      );
      if (lateSuppression) for (const record of matching) record.denied = true;
      for (const record of this.records) {
        if (
          !record.ended &&
          record.event.instanceId === skeleton.instanceId &&
          record.event.runId === skeleton.runId &&
          (record.event.operationId === undefined ||
            record.event.operationId === skeleton.operationId)
        )
          this.updatePolicy(record, ambient);
      }
      // Start upstream spans before content capture so the SDK sampler decides first.
      if (!terminal) this.delegate.observe(adapterEvent(skeleton), ctx);
      const targets = terminal
        ? matching
        : [...this.records].filter(
            (record) => !record.ended && record.event === skeleton,
          );
      const allowed =
        contentAllowed(this.options.traceContent, ambient) &&
        targets.some((record) => record.native.isRecording()) &&
        targets.every((record) => !record.denied);
      const snapshot = allowed ? (safeCopy(event) as Event) : skeleton;
      for (const record of targets) this.mapEvent(record, snapshot ?? skeleton);
      // The adapter receives structural fields only; its content budget never
      // shortens canonical payloads or performs caller-content conversion.
      if (terminal) this.delegate.observe(adapterEvent(skeleton), ctx);
    } catch {
      // Telemetry must not affect native results, callback errors, or cancellation.
    } finally {
      this.currentEvent = undefined;
    }
  }

  private startSpan(
    name: string,
    options: SpanOptions = {},
    supplied: Context = context.active(),
  ): Span {
    const initial = this.currentEvent ?? {};
    const ambient = context.active();
    const native = trace
      .getTracer("@respan/instrumentation-flue", VERSION)
      .startSpan(name, options, supplied);
    const parentSpan = trace.getSpan(supplied);
    const parent = parentSpan && recordsBySpan.get(parentSpan);
    const record: RecordState = {
      native,
      span: native,
      event: initial,
      ended: false,
      parent,
      parentSpan,
      denied:
        !contentAllowed(this.options.traceContent, ambient) ||
        !contentAllowed(this.options.traceContent, supplied) ||
        !ancestorAllows(parentSpan) ||
        flagsDeny(options.attributes),
    };
    const proxy = new Proxy(native, {
      get: (target, key) => {
        if (key === "end")
          return (time?: any) => {
            if (record.ended) return;
            this.updatePolicy(record, context.active());
            if (record.denied) scrub(target);
            stripVendorAttributes(target);
            record.ended = true;
            this.guardReadable(record);
            target.end(time);
            this.records.delete(record);
            recordsById.delete(spanKey(target));
          };
        if (key === "setAttribute" || key === "setAttributes")
          return (...args: any[]) => {
            const attrs =
              key === "setAttribute" ? { [args[0]]: args[1] } : args[0];
            if (flagsDeny(attrs)) record.denied = true;
            this.updatePolicy(record, context.active());
            (target as any)[key](...args);
            if (record.denied) scrub(target);
            return proxy;
          };
        if (
          key === "addEvent" ||
          key === "recordException" ||
          key === "setStatus"
        )
          return (...args: any[]) => {
            this.updatePolicy(record, context.active());
            if (record.denied) {
              if (key === "setStatus") target.setStatus({ code: args[0].code });
              return proxy;
            }
            (target as any)[key](...args);
            return proxy;
          };
        const value = Reflect.get(target, key, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
    record.span = proxy;
    recordsBySpan.set(proxy, record);
    recordsBySpan.set(native, record);
    recordsById.set(spanKey(native), record);
    this.records.add(record);
    const logType = classify(initial, name);
    const entity =
      initial.toolName ??
      initial.agentName ??
      initial.agent ??
      (logType === RespanLogType.CHAT ? "llm" : logType);
    native.setAttributes({
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: entity,
      [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
      ...(this.options.workflowName
        ? {
            [SpanAttributes.TRACELOOP_WORKFLOW_NAME]: this.options.workflowName,
          }
        : {}),
      ...identityAttributes(initial),
    });
    return proxy;
  }

  private updatePolicy(record: RecordState, ambient: Context): void {
    if (
      !contentAllowed(this.options.traceContent, ambient) ||
      !ancestorAllows(trace.getSpan(ambient)) ||
      flagsDeny(readSpanAttributes(record.native)) ||
      record.parent?.denied ||
      !ancestorAllows(record.parentSpan)
    )
      record.denied = true;
    for (let parent = record.parent; parent; parent = parent.parent) {
      if (parent.denied || flagsDeny(readSpanAttributes(parent.native)))
        record.denied = true;
    }
    if (record.denied) scrub(record.native);
  }

  private guardReadable(record: RecordState): void {
    // ReadableSpan buffers stay protected through processor mutation, batching,
    // shutdown and reactivation. The closure lives with the span, not registration.
    const native = record.native;
    const state = (record.readable = {
      attributes: own(native, "attributes") ?? {},
      events: own(native, "events") ?? [],
      status: own(native, "status") ?? {},
    });
    const refresh = () => {
      if (flagsDeny(state.attributes) || !ancestorAllows(record.parentSpan))
        record.denied = true;
      if (record.denied) {
        for (const key of Object.keys(state.attributes))
          if (contentKey(key)) delete state.attributes[key];
        state.events.length = 0;
        delete state.status.message;
      }
    };
    for (const field of ["attributes", "events", "status"] as const) {
      Object.defineProperty(native, field, {
        enumerable: true,
        configurable: false,
        get() {
          refresh();
          if (!record.denied) return state[field];
          if (field === "events") return [];
          return { ...state[field] };
        },
        set(value) {
          // Later processors may replace buffers, but never the original veto.
          const copied = safeCopy(value);
          (state as any)[field] = copied ?? (field === "events" ? [] : {});
          refresh();
        },
      });
    }
  }

  private mapEvent(record: RecordState, event: Event): void {
    const span = record.span;
    this.updatePolicy(record, context.active());
    if (!record.native.isRecording()) return;
    if (event.type === "turn_request") {
      const request = event.request ?? event;
      if (typeof (request.requestedModel ?? request.model) === "string")
        span.setAttribute(
          SpanAttributes.LLM_REQUEST_MODEL,
          request.requestedModel ?? request.model,
        );
      if (typeof (request.providerName ?? request.provider) === "string")
        span.setAttribute(
          SpanAttributes.LLM_SYSTEM,
          request.providerName ?? request.provider,
        );
      span.setAttribute(SpanAttributes.LLM_REQUEST_TYPE, "chat");
      if (!record.denied && request.input) {
        span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_INPUT,
          json(request.input),
        );
        promptAttributes(span, request.input);
      }
    } else if (event.type === "turn") {
      const response = event.response ?? event;
      usageAttributes(span, response.usage);
      if (typeof response.responseId === "string")
        span.setAttribute(ATTR_GEN_AI_RESPONSE_ID, response.responseId);
      if (!record.denied && response.output !== undefined) {
        span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
          json(response.output),
        );
        completionAttributes(span, response.output);
      }
    } else if (event.type === "tool_start") {
      if (typeof event.toolCallId === "string")
        span.setAttribute(ATTR_GEN_AI_TOOL_CALL_ID, event.toolCallId);
      if (!record.denied)
        span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_INPUT,
          json({ name: event.toolName, arguments: event.args }),
        );
    } else if (event.type === "tool") {
      if (!record.denied) {
        const result = Object.hasOwn(event, "effectiveResult")
          ? event.effectiveResult
          : event.result;
        if (result !== undefined)
          span.setAttribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json(result),
          );
      }
    } else if (!record.denied) {
      const input = event.agentInput ?? event.prompt ?? event.payload;
      const output = event.agentOutput ?? event.result;
      if (input !== undefined)
        span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json(input));
      if (output !== undefined)
        span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json(output));
    }
    if (event.isError && !record.denied) {
      const error = (event.response ?? event).error ?? event.errorInfo;
      if (error !== undefined) {
        const value = safeCopy(error) as Event;
        if (value?.message !== undefined)
          span.setStatus({ code: 2, message: value.message });
        if (typeof value?.message === "string")
          span.recordException({
            message: value.message,
            ...(typeof value.name === "string" ? { name: value.name } : {}),
            ...(typeof value.stack === "string" ? { stack: value.stack } : {}),
          });
      }
    }
  }
}

export { FlueInstrumentor as RespanFlueObserver };

function own(value: any, key: string): any {
  if (
    value &&
    (typeof value === "object" || typeof value === "function") &&
    types.isProxy(value)
  )
    return undefined;
  if (!value || (typeof value !== "object" && typeof value !== "function"))
    return undefined;
  try {
    return Object.getOwnPropertyDescriptor(value, key)?.value;
  } catch {
    return undefined;
  }
}
function flagsDeny(value: any): boolean {
  return CONTENT_FLAGS.some((key) => own(value, key) === false);
}
function suppressed(ctx: Context): boolean {
  return isTracingSuppressed(ctx) || ctx.getValue(LM_SUPPRESSION) === true;
}
function contentAllowed(option: boolean | undefined, ctx: Context): boolean {
  return (
    option !== false &&
    process.env.RESPAN_TRACE_CONTENT !== "false" &&
    process.env.TRACELOOP_TRACE_CONTENT !== "false" &&
    ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT) !== false
  );
}
function readSpanAttributes(span: Span): Event | undefined {
  return (
    recordsBySpan.get(span)?.readable?.attributes ?? own(span, "attributes")
  );
}
function spanKey(span: Span): string {
  const id = span.spanContext();
  return `${id.traceId}:${id.spanId}`;
}
function ancestorAllows(span: Span | undefined): boolean {
  if (!span) return true;
  const record = recordsBySpan.get(span) ?? recordsById.get(spanKey(span));
  if (record)
    return (
      !record.denied &&
      !flagsDeny(readSpanAttributes(record.native)) &&
      (!record.parentSpan || ancestorAllows(record.parentSpan))
    );
  const attrs = own(span, "attributes");
  if (attrs) return !flagsDeny(attrs);
  return span.spanContext().isRemote === true;
}
function safeCopy(value: any, seen = new WeakSet<object>()): any {
  if (value === null || ["string", "boolean"].includes(typeof value))
    return value;
  if (typeof value === "number")
    return Number.isFinite(value) ? value : undefined;
  if (typeof value === "bigint") return `${value}`;
  if (typeof value !== "object" || !value || seen.has(value)) return undefined;
  if (types.isProxy(value)) return undefined;
  const proto = Object.getPrototypeOf(value);
  if (
    !Array.isArray(value) &&
    proto !== Object.prototype &&
    proto !== null &&
    !(value instanceof Error)
  )
    return undefined;
  seen.add(value);
  const output: any = Array.isArray(value) ? [] : {};
  for (const [key, descriptor] of Object.entries(
    Object.getOwnPropertyDescriptors(value),
  )) {
    if (
      !descriptor.enumerable &&
      !(value instanceof Error && ["name", "message", "stack"].includes(key))
    )
      continue;
    if (!Object.hasOwn(descriptor, "value") || key === "toJSON") continue;
    const item = safeCopy(descriptor.value, seen);
    if (item !== undefined)
      Object.defineProperty(output, key, {
        value: item,
        enumerable: true,
        configurable: true,
        writable: true,
      });
  }
  seen.delete(value);
  return output;
}
function json(value: unknown): string {
  return JSON.stringify(value) ?? "null";
}
function eventSkeleton(event: Event): Event {
  const output: Event = {};
  for (const key of [
    "type",
    "v",
    "timestamp",
    "eventIndex",
    "runId",
    "instanceId",
    "dispatchId",
    "submissionId",
    "harness",
    "session",
    "parentSession",
    "conversationId",
    "operationId",
    "operationKind",
    "turnId",
    "taskId",
    "toolCallId",
    "toolName",
    "agentName",
    "agent",
    "purpose",
    "origin",
    "durationMs",
    "isError",
    "startedAt",
    "workflowName",
    "outcome",
  ]) {
    const value = own(event, key);
    if (["string", "number", "boolean"].includes(typeof value))
      output[key] = value;
  }
  for (const key of ["request", "response"]) {
    const input = own(event, key);
    if (input) {
      output[key] = {};
      for (const field of [
        "providerId",
        "providerName",
        "requestedModel",
        "api",
        "responseId",
        "responseModel",
        "finishReason",
        "providerFinishReason",
        "maxTokens",
        "temperature",
        "reasoningLevel",
        "serverAddress",
        "serverPort",
      ]) {
        const value = own(input, field);
        if (["string", "number", "boolean"].includes(typeof value))
          output[key][field] = value;
      }
      if (key === "request") output[key].input = { messages: [] };
      if (key === "response") output[key].usage = safeCopy(own(input, "usage"));
    }
  }
  for (const key of ["provider", "model", "api", "reasoning", "stopReason"]) {
    const value = own(event, key);
    if (typeof value === "string") output[key] = value;
  }
  output.usage = safeCopy(own(event, "usage"));
  return output;
}
function adapterEvent(event: Event): Event {
  const result = eventSkeleton(event);
  if (event.type === "turn_request" && !result.request)
    result.input = { messages: [] };
  if (result.response) result.response.output = undefined;
  // Error type is structural; messages, stacks, and arbitrary details stay behind our gate.
  for (const key of ["errorInfo", "error"]) {
    const type = own(own(event, key), "type");
    if (typeof type === "string") result[key] = { type };
  }
  if (result.response) {
    const type = own(own(event.response, "error"), "type");
    if (typeof type === "string") result.response.error = { type };
  }
  return result;
}
function matches(start: Event, event: Event): boolean {
  for (const key of [
    "runId",
    "instanceId",
    "harness",
    "session",
    "operationId",
    "taskId",
    "turnId",
    "toolCallId",
  ]) {
    if (start[key] !== undefined && event[key] !== start[key]) return false;
  }
  if (["turn_request", "turn"].includes(event.type))
    return start.type === "turn_request";
  if (["tool_start", "tool"].includes(event.type))
    return start.type === "tool_start";
  if (["operation_start", "operation"].includes(event.type))
    return start.type === "operation_start";
  if (["task_start", "task"].includes(event.type))
    return start.type === "task_start";
  if (["compaction_start", "compaction"].includes(event.type))
    return start.type === "compaction_start";
  return (
    event.type === "run_end" && ["run_start", "run_resume"].includes(start.type)
  );
}
function classify(event: Event, name: string): RespanLogType {
  if (event.type === "turn_request") return RespanLogType.CHAT;
  if (event.type === "tool_start") return RespanLogType.TOOL;
  if (["run_start", "run_resume"].includes(event.type))
    return RespanLogType.WORKFLOW;
  if (
    ["prompt", "skill"].includes(event.operationKind) ||
    event.type === "task_start" ||
    name.startsWith("invoke_agent")
  )
    return RespanLogType.AGENT;
  return RespanLogType.TASK;
}
function identityAttributes(event: Event): Record<string, string | number> {
  const result: Record<string, string | number> = {};
  for (const key of [
    "runId",
    "instanceId",
    "dispatchId",
    "submissionId",
    "harness",
    "session",
    "parentSession",
    "conversationId",
    "operationId",
    "turnId",
    "taskId",
    "toolCallId",
    "eventIndex",
  ]) {
    if (event[key] !== undefined)
      result[`${RespanSpanAttributes.RESPAN_METADATA}.flue_${key}`] =
        event[key];
  }
  return result;
}
function promptAttributes(span: Span, input: Event): void {
  let index = 0;
  if (input.systemPrompt !== undefined) {
    span.setAttribute(`${SpanAttributes.LLM_PROMPTS}.${index}.role`, "system");
    span.setAttribute(
      `${SpanAttributes.LLM_PROMPTS}.${index++}.content`,
      content(input.systemPrompt),
    );
  }
  for (const message of input.messages ?? []) {
    const prefix = `${SpanAttributes.LLM_PROMPTS}.${index++}`;
    span.setAttribute(
      `${prefix}.role`,
      message.role === "toolResult" ? "tool" : message.role,
    );
    span.setAttribute(
      `${prefix}.content`,
      content(
        message.role === "assistant" && Array.isArray(message.content)
          ? message.content.filter((block: Event) => block.type !== "toolCall")
          : message.content,
      ),
    );
    const calls = toolCalls(message);
    if (calls.length) span.setAttribute(`${prefix}.tool_calls`, json(calls));
    if (typeof message.toolCallId === "string")
      span.setAttribute(`${prefix}.tool_call_id`, message.toolCallId);
  }
  if (input.tools !== undefined)
    span.setAttribute(SpanAttributes.LLM_REQUEST_FUNCTIONS, json(input.tools));
}
function completionAttributes(span: Span, message: Event): void {
  const prefix = `${SpanAttributes.LLM_COMPLETIONS}.0`;
  span.setAttribute(`${prefix}.role`, message.role ?? "assistant");
  const blocks = Array.isArray(message.content)
    ? message.content.filter((block: Event) => block.type !== "toolCall")
    : message.content;
  span.setAttribute(`${prefix}.content`, content(blocks));
  const calls = toolCalls(message);
  if (calls.length) span.setAttribute(`${prefix}.tool_calls`, json(calls));
}
function content(value: any): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value) && value.every((block) => block?.type === "text"))
    return value.map((block) => block.text ?? "").join("");
  return json(value);
}
function toolCalls(message: Event): any[] {
  return Array.isArray(message.content)
    ? message.content
        .filter((block: Event) => block?.type === "toolCall")
        .map((block: Event) => ({
          id: block.id,
          type: "function",
          function: {
            name: block.name,
            arguments:
              typeof block.arguments === "string"
                ? block.arguments
                : json(block.arguments),
          },
        }))
    : [];
}
function usageAttributes(span: Span, usage: Event | undefined): void {
  if (!usage) return;
  for (const [source, keys] of [
    [
      "input",
      [ATTR_GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_USAGE_PROMPT_TOKENS],
    ],
    [
      "output",
      [
        ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
      ],
    ],
    ["totalTokens", [SpanAttributes.LLM_USAGE_TOTAL_TOKENS]],
    ["cacheRead", [ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]],
  ] as const)
    if (typeof usage[source] === "number" && Number.isFinite(usage[source]))
      for (const key of keys) span.setAttribute(key, usage[source]);
}
function contentKey(key: string): boolean {
  return (
    key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
    key === RespanSpanAttributes.RESPAN_METADATA ||
    (key.startsWith(`${RespanSpanAttributes.RESPAN_METADATA}.`) &&
      !key.startsWith(`${RespanSpanAttributes.RESPAN_METADATA}.flue_`)) ||
    /(?:^gen_ai\.(?:prompt|completion|input|output|system_instructions|tool\.(?:call\.(?:arguments|result)|definitions|description))|^traceloop\.entity\.(?:input|output)$|exception|error\.message|(?:^|[._])(?:prompt|input|output|arguments|result|description|message|stack)(?:[._]|$))/.test(
      key,
    )
  );
}
function scrub(span: Span): void {
  const attributes = own(span, "attributes");
  if (attributes)
    for (const key of Object.keys(attributes))
      if (contentKey(key)) delete attributes[key];
  const events = own(span, "events");
  if (Array.isArray(events)) events.length = 0;
  const status = own(span, "status");
  if (status && typeof status === "object") delete status.message;
}
function stripVendorAttributes(span: Span): void {
  const attributes = own(span, "attributes");
  if (attributes)
    for (const key of Object.keys(attributes))
      if (
        key.startsWith("flue.") ||
        key === "gen_ai.tool.call.arguments" ||
        key === "gen_ai.tool.call.result"
      )
        delete attributes[key];
}
