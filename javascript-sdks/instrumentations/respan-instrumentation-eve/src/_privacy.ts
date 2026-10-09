import {
  context,
  createContextKey,
  trace,
  type Context,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { types } from "node:util";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { ATTR_EXCEPTION_TYPE } from "@opentelemetry/semantic-conventions";
import {
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import {
  EVE_AGENT_CONTENT_INPUT,
  EVE_AGENT_CONTENT_OUTPUT,
  EVE_AGENT_RUN_ID,
  EVE_MEMORY_RECORDS,
} from "./constants/eve.js";
import {
  AI_TOOL_CALL_ARGS,
  AI_TOOL_CALL_RESULT,
  AI_TOOL_CALL,
  AI_TOOL_CALLS,
  AI_SETTINGS_CONTEXT_PREFIX,
} from "./_translator/shared.js";

// Compatibility with JS clients that use the historical LM suppression key.
// OTel's public isTracingSuppressed remains the general tracing gate.
const suppressLanguageModel = createContextKey(
  "suppress_language_model_instrumentation",
);

interface CapturePolicy {
  inputs: boolean;
  outputs: boolean;
  emit: boolean;
  parent?: string;
}

function denied(value: unknown): boolean {
  return (
    value === false ||
    (typeof value === "string" &&
      ["false", "0", "no", "off"].includes(value.trim().toLowerCase()))
  );
}

function ownData(value: object | undefined, key: string): any {
  if (!value || types.isProxy(value)) return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}

function environmentAllowsContent(): boolean {
  return (
    !denied(process.env.RESPAN_TRACE_CONTENT) &&
    !denied(process.env.TRACELOOP_TRACE_CONTENT)
  );
}

function identity(span: any): string | undefined {
  const own = span.spanContext?.();
  return own?.traceId && own?.spanId
    ? `${own.traceId}:${own.spanId}`
    : undefined;
}

function parentIdentity(span: any): string | undefined {
  const parent = span.parentSpanContext;
  const traceId = parent?.traceId ?? span.spanContext?.().traceId;
  const spanId = parent?.spanId ?? span.parentSpanId;
  return traceId && spanId ? `${traceId}:${spanId}` : undefined;
}

function merge(target: CapturePolicy, other: CapturePolicy | undefined): void {
  if (other) {
    target.inputs &&= other.inputs;
    target.outputs &&= other.outputs;
    target.emit &&= other.emit;
  }
}

/** Retain only immutable veto bits and span IDs; never retain customer data. */
export class EveCapturePolicy {
  private readonly _spans = new WeakMap<object, CapturePolicy>();
  private readonly _ancestors = new Map<string, CapturePolicy>();

  constructor(
    private readonly _options: {
      traceContent?: boolean;
      recordInputs?: boolean;
      recordOutputs?: boolean;
    } = {},
  ) {}

  capture(
    span: any,
    parentContext?: Context,
    evaluateContext = true,
  ): CapturePolicy {
    let policy =
      this._spans.get(span) ?? this._ancestors.get(identity(span) ?? "");
    if (!policy) {
      policy = {
        inputs:
          this._options.traceContent !== false &&
          this._options.recordInputs !== false,
        outputs:
          this._options.traceContent !== false &&
          this._options.recordOutputs !== false,
        emit: true,
        parent: parentIdentity(span),
      };
      this._spans.set(span, policy);
      const key = identity(span);
      if (key) this._ancestors.set(key, policy);
    }
    // OTel deliberately suppresses tracing while calling an exporter. That
    // transport context is unrelated to the original application's decision.
    for (const ctx of evaluateContext
      ? [parentContext, context.active()]
      : []) {
      if (!ctx) continue;
      if (
        isTracingSuppressed(ctx) ||
        ctx.getValue(suppressLanguageModel) === true
      )
        policy.emit = false;
      if (denied(ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)))
        policy.inputs = policy.outputs = false;
      const parent = trace.getSpan(ctx) as any;
      if (parent) {
        if (types.isProxy(parent)) {
          policy.inputs = policy.outputs = false;
          continue;
        }
        const parentAttributes = ownData(parent, "attributes");
        const retained =
          this._spans.get(parent) ??
          this._ancestors.get(identity(parent) ?? "");
        merge(policy, retained);
        // Eve's durable native runtime uses non-recording, reserved parent IDs
        // before materializing the actual activation span. Its own typed
        // session identity identifies that managed path. Unknown local parents
        // outside it have no observable capture policy and fail closed.
        const managedEveParent =
          span.instrumentationScope?.name === "eve.agent" &&
          typeof span.attributes?.[EVE_AGENT_RUN_ID] === "string";
        if (
          !retained &&
          !parentAttributes &&
          parent.spanContext?.().isRemote !== true &&
          !managedEveParent
        )
          policy.inputs = policy.outputs = false;
        this.applyAttributes(policy, parentAttributes);
        const parentFlags = parent.spanContext?.().traceFlags;
        if (parentFlags !== undefined && (parentFlags & 1) === 0)
          policy.emit = false;
      }
    }
    if (!environmentAllowsContent()) policy.inputs = policy.outputs = false;
    const flags = span.spanContext?.().traceFlags;
    if (flags !== undefined && (flags & 1) === 0) policy.emit = false;
    this.applyAttributes(policy, span.attributes);
    let ancestor = policy.parent;
    const visited = new Set<string>();
    while (ancestor && !visited.has(ancestor)) {
      visited.add(ancestor);
      const prior = this._ancestors.get(ancestor);
      merge(policy, prior);
      ancestor = prior?.parent;
    }
    if (!policy.emit) policy.inputs = policy.outputs = false;
    return policy;
  }

  private applyAttributes(
    policy: CapturePolicy,
    attrs: Record<string, unknown> | undefined,
  ): void {
    if (!attrs) return;
    if (denied(ownData(attrs, "allow_trace_content")))
      policy.inputs = policy.outputs = false;
    if (denied(ownData(attrs, EVE_AGENT_CONTENT_INPUT))) policy.inputs = false;
    if (denied(ownData(attrs, EVE_AGENT_CONTENT_OUTPUT)))
      policy.outputs = false;
  }

  stripAttributes(attrs: Record<string, unknown>, policy: CapturePolicy): void {
    if (!policy.emit)
      attrs[RespanSpanAttributes.RESPAN_INTERNAL_DROP_SPAN] = true;
    for (const key of Object.keys(attrs)) {
      const input =
        key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
        key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
        key.startsWith(`${SpanAttributes.LLM_PROMPTS}.`) ||
        key.startsWith("gen_ai.input.") ||
        key === ATTR_GEN_AI_SYSTEM_INSTRUCTIONS ||
        key === ATTR_GEN_AI_TOOL_DEFINITIONS ||
        key === ATTR_GEN_AI_TOOL_DESCRIPTION ||
        key === ATTR_GEN_AI_TOOL_CALL_ARGUMENTS ||
        key === AI_TOOL_CALL_ARGS ||
        /^(ai\.(prompt|value|values|documents)|agent\.(channel\.delivery\.input|session\.title|approval\.request))/.test(
          key,
        );
      const output =
        key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
        key.startsWith(`${SpanAttributes.LLM_COMPLETIONS}.`) ||
        key.startsWith("gen_ai.output.") ||
        key === ATTR_GEN_AI_TOOL_CALL_RESULT ||
        key === EVE_MEMORY_RECORDS ||
        key === AI_TOOL_CALL_RESULT ||
        /^(ai\.(response|embedding|embeddings|ranking)|agent\.approval\.response)/.test(
          key,
        );
      const shared =
        key.startsWith(AI_SETTINGS_CONTEXT_PREFIX) ||
        key === RespanSpanAttributes.RESPAN_METADATA ||
        key.startsWith(`${RespanSpanAttributes.RESPAN_METADATA}.`) ||
        key.startsWith("exception.") ||
        key.startsWith("error.") ||
        key === AI_TOOL_CALL ||
        key === AI_TOOL_CALLS;
      if (
        (!policy.inputs && input) ||
        (!policy.outputs && output) ||
        ((!policy.inputs || !policy.outputs) && shared)
      )
        delete attrs[key];
    }
  }

  prepare(span: ReadableSpan): ReadableSpan {
    if (!this._spans.has(span) && !this._ancestors.has(identity(span) ?? ""))
      return span;
    const policy = this.capture(span, undefined, false);
    if (policy.inputs && policy.outputs && policy.emit) return span;
    const attributes = { ...span.attributes };
    this.stripAttributes(attributes, policy);
    // Also clear the live SDK buffers, so another later processor cannot
    // recover payloads by retaining the original ReadableSpan.
    this.stripAttributes(span.attributes as Record<string, unknown>, policy);
    if (!policy.inputs || !policy.outputs) {
      for (const event of span.events) {
        if (event.attributes)
          for (const key of Object.keys(event.attributes)) {
            if (key !== ATTR_EXCEPTION_TYPE) delete event.attributes[key];
          }
      }
      span.events.length = 0;
      if (span.status.message !== undefined) delete span.status.message;
    }
    const clone = Object.create(Object.getPrototypeOf(span));
    Object.assign(clone, span);
    Object.defineProperties(clone, {
      attributes: { configurable: true, enumerable: true, value: attributes },
      events: {
        configurable: true,
        enumerable: true,
        value: policy.inputs && policy.outputs ? span.events : [],
      },
      status: {
        configurable: true,
        enumerable: true,
        value:
          policy.inputs && policy.outputs
            ? span.status
            : { code: span.status.code },
      },
    });
    return clone as ReadableSpan;
  }

  shouldExport(span: ReadableSpan): boolean {
    return (
      (this._spans.has(span) || this._ancestors.has(identity(span) ?? "")) &&
      this.capture(span, undefined, false).emit
    );
  }

  associate(original: ReadableSpan, clone: ReadableSpan): void {
    const policy =
      this._spans.get(original) ??
      this._ancestors.get(identity(original) ?? "");
    if (policy) {
      this._spans.set(clone, policy);
      const key = identity(clone);
      if (key) this._ancestors.set(key, policy);
    }
  }

  clear(): void {
    this._ancestors.clear();
  }
}
