import { context, trace, type Context, type Span } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import { types } from "node:util";
import {
  ATTR_EXCEPTION_MESSAGE,
  ATTR_EXCEPTION_STACKTRACE,
  ATTR_EXCEPTION_TYPE,
  ATTR_ERROR_TYPE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import {
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_HTTP_STATUS_CODE,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  STRANDS_AGENT_TOOLS_ATTR,
  STRANDS_TOOL_JSON_SCHEMA_ATTR,
  STRANDS_SYSTEM_PROMPT_ATTR,
  STRANDS_AGENT_INPUT_ATTR,
  STRANDS_TOOL_STATUS_ATTR,
  STRANDS_EVENT_START_TIME_ATTR,
  STRANDS_EVENT_END_TIME_ATTR,
  STRANDS_CYCLE_ID_ATTR,
  STRANDS_SPAN_NAME_ATTR,
  STRANDS_TRACE_CONTENT_ATTR,
} from "./_constants.js";

function denied(value: unknown): boolean {
  return value === false || value === 0 || value === "false" || value === "0";
}

export function contentAllowed(
  parent: Context = context.active(),
  configured = true,
): boolean {
  return (
    configured &&
    !denied(process.env.RESPAN_TRACE_CONTENT) &&
    !denied(process.env.TRACELOOP_TRACE_CONTENT) &&
    !denied(parent.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)) &&
    !isTracingSuppressed(parent)
  );
}

interface Policy {
  denied: boolean;
  ancestors: object[];
  restore?: () => void;
}

function spanVeto(span: object): boolean {
  const attrs = dataProperty(span, "attributes");
  if (denied(dataProperty(attrs, STRANDS_TRACE_CONTENT_ATTR))) return true;
  const events = dataProperty(span, "events");
  if (!Array.isArray(events) || types.isProxy(events)) return false;
  return Object.keys(Object.getOwnPropertyDescriptors(events)).some((key) =>
    denied(
      dataProperty(
        dataProperty(dataProperty(events, key), "attributes"),
        STRANDS_TRACE_CONTENT_ATTR,
      ),
    ),
  );
}

/** An observed veto survives a child scope that enables content again. */
export class ContentPolicy {
  private readonly gates = new WeakMap<object, Policy>();
  constructor(private readonly configured: boolean | (() => boolean) = true) {}
  private enabled(): boolean {
    return typeof this.configured === "function"
      ? this.configured()
      : this.configured;
  }

  onStart(span: Span, parent: Context): void {
    const ancestor = trace.getSpan(parent);
    const inherited = ancestor ? this.gates.get(ancestor) : undefined;
    const policy: Policy = {
      denied:
        !span.isRecording() ||
        (span.spanContext().traceFlags & 1) === 0 ||
        !contentAllowed(parent, this.enabled()) ||
        inherited?.denied === true ||
        (!!ancestor && spanVeto(ancestor)),
      ancestors: ancestor ? [ancestor, ...(inherited?.ancestors ?? [])] : [],
    };
    this.gates.set(span, policy);
    if (policy.denied) return;
    const originalDescriptor = Object.getOwnPropertyDescriptor(
      span,
      "setAttribute",
    );
    const original = span.setAttribute;
    const wrapped: Span["setAttribute"] = function (this: Span, key, value) {
      if (key === STRANDS_TRACE_CONTENT_ATTR && denied(value))
        policy.denied = true;
      return original.call(this, key, value);
    };
    try {
      span.setAttribute = wrapped;
      policy.restore = () => {
        if (span.setAttribute !== wrapped) return;
        if (originalDescriptor)
          Object.defineProperty(span, "setAttribute", originalDescriptor);
        else delete (span as any).setAttribute;
      };
    } catch {
      /* A read-only foreign span can still expose a final veto. */
    }
  }

  finish(span: ReadableSpan): void {
    this.gates.get(span)?.restore?.();
  }

  allowed(span: ReadableSpan): boolean {
    const policy = this.gates.get(span);
    const veto =
      (span.spanContext().traceFlags & 1) === 0 ||
      policy?.denied === true ||
      !contentAllowed(undefined, this.enabled()) ||
      spanVeto(span) ||
      policy?.ancestors.some(
        (ancestor) => this.gates.get(ancestor)?.denied || spanVeto(ancestor),
      );
    if (policy && veto) policy.denied = true;
    return !veto;
  }
}

function structuralKey(key: string): boolean {
  return (
    [
      SpanAttributes.TRACELOOP_ENTITY_NAME,
      SpanAttributes.TRACELOOP_ENTITY_PATH,
      SpanAttributes.TRACELOOP_WORKFLOW_NAME,
      SpanAttributes.TRACELOOP_SPAN_KIND,
      RespanSpanAttributes.RESPAN_LOG_TYPE,
      RespanSpanAttributes.RESPAN_LOG_METHOD,
      ATTR_GEN_AI_SYSTEM,
      ATTR_GEN_AI_PROVIDER_NAME,
      ATTR_GEN_AI_OPERATION_NAME,
      ATTR_GEN_AI_REQUEST_MODEL,
      ATTR_GEN_AI_AGENT_NAME,
      ATTR_GEN_AI_AGENT_ID,
      ATTR_GEN_AI_TOOL_NAME,
      ATTR_GEN_AI_TOOL_CALL_ID,
      STRANDS_TOOL_STATUS_ATTR,
      STRANDS_EVENT_START_TIME_ATTR,
      STRANDS_EVENT_END_TIME_ATTR,
      STRANDS_CYCLE_ID_ATTR,
      STRANDS_SPAN_NAME_ATTR,
      ATTR_HTTP_RESPONSE_STATUS_CODE,
      ATTR_HTTP_STATUS_CODE,
      ATTR_ERROR_TYPE,
    ].includes(key) ||
    key.startsWith("gen_ai.usage.") ||
    key.startsWith("llm.usage.") ||
    key.startsWith("respan.trace.") ||
    key.startsWith("respan.threads.")
  );
}

function contentKey(key: string): boolean {
  return (
    key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
    key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
    key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
    key === STRANDS_SYSTEM_PROMPT_ATTR ||
    key === STRANDS_AGENT_INPUT_ATTR ||
    key === STRANDS_AGENT_TOOLS_ATTR ||
    key === ATTR_GEN_AI_SYSTEM_INSTRUCTIONS ||
    key === ATTR_GEN_AI_TOOL_DEFINITIONS ||
    key === STRANDS_TOOL_JSON_SCHEMA_ATTR ||
    key === ATTR_GEN_AI_TOOL_DESCRIPTION ||
    key === ATTR_GEN_AI_TOOL_CALL_ARGUMENTS ||
    key === ATTR_GEN_AI_TOOL_CALL_RESULT ||
    key === ATTR_GEN_AI_INPUT_MESSAGES ||
    key === ATTR_GEN_AI_OUTPUT_MESSAGES ||
    key.startsWith(`${SpanAttributes.LLM_PROMPTS}.`) ||
    key.startsWith(`${SpanAttributes.LLM_COMPLETIONS}.`) ||
    key.startsWith(RespanSpanAttributes.RESPAN_METADATA) ||
    key.startsWith(`${SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.`) ||
    key === ATTR_EXCEPTION_MESSAGE ||
    key === ATTR_EXCEPTION_STACKTRACE ||
    key === "error.message" ||
    key === "error.stack"
  );
}

export function dataProperty(source: unknown, key: string): any {
  if (!source || typeof source !== "object" || types.isProxy(source))
    return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(source, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}

/** Read native OTel data properties only; never invoke payload getters or toJSON. */
export function copyData(value: unknown, seen = new WeakSet<object>()): any {
  if (value === null || typeof value === "string" || typeof value === "boolean")
    return value;
  if (typeof value === "number")
    return Number.isFinite(value) ? value : undefined;
  if (
    !value ||
    typeof value !== "object" ||
    types.isProxy(value) ||
    seen.has(value)
  )
    return undefined;
  const proto = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && proto !== Object.prototype && proto !== null)
    return undefined;
  seen.add(value);
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const output: any = Array.isArray(value) ? [] : {};
  for (const key of Object.keys(descriptors)) {
    if (
      key === "length" ||
      !descriptors[key].enumerable ||
      !("value" in descriptors[key])
    )
      continue;
    const copied = copyData(descriptors[key].value, seen);
    if (copied !== undefined)
      Object.defineProperty(output, key, {
        value: copied,
        enumerable: true,
        writable: true,
        configurable: true,
      });
  }
  seen.delete(value);
  return output;
}

function attributes(
  source: unknown,
  capture: boolean,
  event = false,
): Record<string, any> {
  if (!source || typeof source !== "object" || types.isProxy(source)) return {};
  const output: Record<string, any> = {};
  const descriptors = Object.getOwnPropertyDescriptors(source);
  for (const key of Object.keys(descriptors)) {
    if (
      !capture &&
      (contentKey(key) ||
        (event
          ? ![
              ATTR_EXCEPTION_TYPE,
              ATTR_ERROR_TYPE,
              ATTR_HTTP_RESPONSE_STATUS_CODE,
              "finish_reason",
            ].includes(key)
          : !structuralKey(key)))
    )
      continue;
    const descriptor = descriptors[key];
    if (!("value" in descriptor)) continue;
    const copied = copyData(descriptor.value);
    if (copied !== undefined)
      Object.defineProperty(output, key, {
        value: copied,
        enumerable: true,
        writable: true,
        configurable: true,
      });
  }
  return output;
}

/** Export-only snapshot: the SDK and other span processors keep the original. */
export function snapshotSpan(
  span: ReadableSpan,
  capture: boolean,
): ReadableSpan {
  const clone = Object.create(Object.getPrototypeOf(span));
  for (const key of Object.keys(Object.getOwnPropertyDescriptors(span))) {
    if (["attributes", "_attributes", "events", "status"].includes(key))
      continue;
    const descriptor = Object.getOwnPropertyDescriptor(span, key)!;
    if ("value" in descriptor) Object.defineProperty(clone, key, descriptor);
  }
  const attrs = attributes(span.attributes, capture);
  const events: Array<Record<string, any>> = [];
  const sourceEvents = span.events;
  if (Array.isArray(sourceEvents) && !types.isProxy(sourceEvents)) {
    for (const key of Object.keys(
      Object.getOwnPropertyDescriptors(sourceEvents),
    )) {
      const event = dataProperty(sourceEvents, key);
      const name = dataProperty(event, "name");
      if (typeof name !== "string") continue;
      events.push({
        name,
        time: copyData(dataProperty(event, "time")) ?? [0, 0],
        attributes: attributes(
          dataProperty(event, "attributes"),
          capture,
          true,
        ),
        droppedAttributesCount: dataProperty(event, "droppedAttributesCount"),
      });
    }
  }
  const sourceStatus = span.status;
  const code = dataProperty(sourceStatus, "code");
  const message = capture ? dataProperty(sourceStatus, "message") : undefined;

  Object.defineProperties(clone, {
    attributes: {
      value: attrs,
      enumerable: true,
      writable: true,
      configurable: true,
    },
    _attributes: {
      value: attrs,
      enumerable: false,
      writable: true,
      configurable: true,
    },
    events: {
      value: events,
      enumerable: true,
      writable: true,
      configurable: true,
    },
    status: {
      value: {
        code: typeof code === "number" ? code : 0,
        ...(typeof message === "string" ? { message } : {}),
      },
      enumerable: true,
      writable: true,
      configurable: true,
    },
  });
  return clone;
}
