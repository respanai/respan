import {
  ACTIVE_CONTENT_POLICIES,
  type GoogleADKInstrumentationOptions,
} from "./_config.js";
import {
  GOOGLE_ADK_SCOPE_NAME,
  GOOGLE_ADK_LOG_METHOD_TS_TRACING,
  ADK_PREFIX,
  ADK_LLM_REQUEST,
  ADK_LLM_RESPONSE,
  ADK_TOOL_CALL_ARGS,
  ADK_TOOL_RESPONSE,
  ADK_WORKFLOW_NAME,
  ADK_NODE_PATH,
  ADK_NODE_RUN_ID,
  ADK_NODE_ATTEMPT,
  ADK_NODE_STATUS,
  ADK_NODE_INTERRUPT_COUNT,
} from "./_constants.js";
import { types as utilTypes } from "node:util";
import { context, trace, TraceFlags } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import type { Context } from "@opentelemetry/api";
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_AGENT_DESCRIPTION,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_CONVERSATION_ID,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_TOOL_TYPE,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  CONTEXT_KEY_ALLOW_TRACE_CONTENT,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";

const OFF_CONTRACT_ALIASES = new Set([
  "model",
  "prompt_tokens",
  "completion_tokens",
  "total_request_tokens",
  "tools",
  "tool_calls",
  "span_tools",
  "has_tool_calls",
  "parallel_tool_calls",
  RespanSpanAttributes.RESPAN_SPAN_TOOLS,
  RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
]);

type Attributes = Record<string, any>;
type SpanSetAttribute = (key: string, value: unknown) => Span;

interface GoogleADKResponseCapture {
  readonly accumulator: GoogleADKResponseAccumulator;
  readonly originalSetAttribute: SpanSetAttribute;
  readonly wrappedSetAttribute: SpanSetAttribute;
}

interface GoogleADKResponseAccumulator {
  // Keep only assembled state: ADK may write a growing cumulative response on
  // every chunk, so retaining raw frames would grow quadratically.
  response?: Attributes;
  content?: GoogleADKContentAccumulator;
}

interface GoogleADKContentAccumulator {
  readonly attributes: Attributes;
  readonly parts: Attributes[];
  readonly partIndexes: Map<string, number>;
}

const GOOGLE_ADK_RESPONSE_CAPTURES = new WeakMap<
  object,
  GoogleADKResponseCapture
>();

interface CapturePolicy {
  denied: boolean;
  readonly suppressed: boolean;
  readonly ancestors: object[];
}
const CAPTURE_POLICIES = new WeakMap<object, CapturePolicy>();

export class GoogleADKTranslator implements SpanProcessor {
  constructor(
    private readonly options: Readonly<GoogleADKInstrumentationOptions> = {},
    private readonly exportOnly = false,
  ) {}

  onStart(span: Span, parentContext: Context): void {
    if (typeof span.isRecording === "function" && !span.isRecording()) return;
    const parent = trace.getSpan(parentContext);
    const parentPolicy = parent ? CAPTURE_POLICIES.get(parent) : undefined;
    const policy: CapturePolicy = {
      suppressed:
        isTracingSuppressed(parentContext) ||
        (typeof span.spanContext === "function" &&
          (span.spanContext().traceFlags & TraceFlags.SAMPLED) === 0) ||
        parentPolicy?.suppressed === true,
      denied:
        this.options.traceContent === false ||
        [...ACTIVE_CONTENT_POLICIES].some(
          (options) => options.traceContent === false,
        ) ||
        contentDenied(parentContext) ||
        parentPolicy?.denied === true,
      ancestors: parent ? [parent, ...(parentPolicy?.ancestors ?? [])] : [],
    };
    CAPTURE_POLICIES.set(span, policy);
    if (policy.suppressed) return;
    const writableSpan = span as any;
    if (!isGoogleADKSpan(writableSpan)) {
      return;
    }
    const spanName = String(writableSpan.name ?? "");
    const logType = resolveLogTypeFromName(spanName);
    if (logType === undefined) {
      return;
    }

    if (!policy.denied)
      installResponseCapture(writableSpan, logType === RespanLogType.CHAT);

    if (this.exportOnly) return;

    writableSpan.setAttribute(
      RespanSpanAttributes.RESPAN_LOG_METHOD,
      GOOGLE_ADK_LOG_METHOD_TS_TRACING,
    );
    writableSpan.setAttribute(RespanSpanAttributes.RESPAN_LOG_TYPE, logType);

    const entityName = resolveEntityNameFromName(spanName, logType);
    writableSpan.setAttribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
    writableSpan.setAttribute(
      SpanAttributes.TRACELOOP_ENTITY_PATH,
      logType === RespanLogType.WORKFLOW ? "" : entityName,
    );
  }

  onEnd(span: ReadableSpan): void {
    if (this.exportOnly) {
      restoreResponseCapture(span, false);
      return;
    }
    try {
      if (CAPTURE_POLICIES.get(span)?.suppressed && isGoogleADKSpan(span)) {
        const attrs = getAttributes(span);
        if (attrs) {
          cleanupAttrs(attrs);
          redactContent(attrs);
          delete attrs[RespanSpanAttributes.RESPAN_LOG_TYPE];
          delete attrs[SpanAttributes.TRACELOOP_ENTITY_PATH];
          delete attrs[SpanAttributes.TRACELOOP_ENTITY_NAME];
        }
      } else translateGoogleADKSpan(span, !spanContentDenied(span));
    } finally {
      restoreResponseCapture(span);
    }
  }

  prepareForExport(span: ReadableSpan): ReadableSpan {
    if (!isGoogleADKSpan(span)) return span;
    const policy = CAPTURE_POLICIES.get(span);
    const contentAllowed = !spanContentDenied(span);
    const descriptors: PropertyDescriptorMap =
      Object.getOwnPropertyDescriptors(span);
    descriptors.attributes = {
      value: copyAttributes(getAttributes(span) ?? {}, !contentAllowed),
      enumerable: true,
      configurable: true,
      writable: true,
    };
    if (!contentAllowed) {
      descriptors.events = {
        value: span.events.map((event) => ({
          name: Object.getOwnPropertyDescriptor(event, "name")?.value,
          time: Object.getOwnPropertyDescriptor(event, "time")?.value,
          attributes: {},
        })),
        enumerable: true,
        configurable: true,
        writable: true,
      };
      descriptors.status = {
        value: {
          code: Object.getOwnPropertyDescriptor(span.status, "code")?.value,
        },
        enumerable: true,
        configurable: true,
        writable: true,
      };
    }
    const clone = Object.create(
      Object.getPrototypeOf(span),
      descriptors,
    ) as ReadableSpan;
    if (policy) CAPTURE_POLICIES.set(clone, policy);
    const capture = GOOGLE_ADK_RESPONSE_CAPTURES.get(span);
    if (capture) GOOGLE_ADK_RESPONSE_CAPTURES.set(clone, capture);
    try {
      if (policy?.suppressed) {
        const attrs = getAttributes(clone)!;
        cleanupAttrs(attrs);
        redactContent(attrs);
        delete attrs[RespanSpanAttributes.RESPAN_LOG_TYPE];
        delete attrs[SpanAttributes.TRACELOOP_ENTITY_PATH];
        delete attrs[SpanAttributes.TRACELOOP_ENTITY_NAME];
      } else translateGoogleADKSpan(clone, contentAllowed);
      return clone;
    } finally {
      GOOGLE_ADK_RESPONSE_CAPTURES.delete(span);
      GOOGLE_ADK_RESPONSE_CAPTURES.delete(clone);
    }
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }
}

export function isGoogleADKSpan(span: ReadableSpan): boolean {
  const attrs = getAttributes(span);
  if (!attrs) {
    return false;
  }

  if (getInstrumentationScopeName(span) === GOOGLE_ADK_SCOPE_NAME) {
    return true;
  }

  if (
    Object.getOwnPropertyDescriptor(attrs, SpanAttributes.LLM_SYSTEM)?.value ===
    GOOGLE_ADK_SCOPE_NAME
  ) {
    return true;
  }

  return Object.keys(attrs).some((key) => key.startsWith(ADK_PREFIX));
}

export function translateGoogleADKSpan(
  span: ReadableSpan,
  captureContent = true,
): void {
  const originalAttrs = getAttributes(span);
  if (!originalAttrs || !isGoogleADKSpan(span)) return;
  captureContent = captureContent && !spanContentDenied(span);
  const attrs = copyAttributes(originalAttrs, !captureContent);
  Object.defineProperty(span, "attributes", {
    ...Object.getOwnPropertyDescriptor(span, "attributes"),
    value: attrs,
  });

  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] =
    GOOGLE_ADK_LOG_METHOD_TS_TRACING;

  const logType = resolveLogType(span, attrs);
  attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = logType;

  const entityName = resolveEntityName(span, attrs, logType);
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
  setDefault(
    attrs,
    SpanAttributes.TRACELOOP_ENTITY_PATH,
    logType === RespanLogType.WORKFLOW ? "" : entityName,
  );

  if (logType === RespanLogType.CHAT) {
    normalizeChatSpan(
      attrs,
      captureContent ? getCapturedResponse(span) : undefined,
      captureContent,
    );
  } else if (logType === RespanLogType.TOOL) {
    if (captureContent) normalizeToolSpan(attrs, entityName);
  } else if (logType === RespanLogType.AGENT) {
    normalizeAgentSpan(attrs, entityName);
  } else if (logType === RespanLogType.WORKFLOW) {
    normalizeWorkflowSpan(attrs, entityName);
  }

  normalizeNodeAttributes(attrs);

  cleanupAttrs(attrs);
  if (logType !== RespanLogType.TOOL) delete attrs[ATTR_GEN_AI_TOOL_CALL_ID];
  if (!captureContent) redactContent(attrs);
  restoreResponseCapture(span);
}

function resolveLogTypeFromName(spanName: string): RespanLogType | undefined {
  const normalizedName = spanName.toLowerCase();
  if (normalizedName === "execute_tool (merged)") return RespanLogType.TASK;
  if (normalizedName === "call_llm") {
    return RespanLogType.CHAT;
  }
  if (normalizedName.startsWith("execute_tool")) {
    return RespanLogType.TOOL;
  }
  if (normalizedName.startsWith("invoke_agent")) {
    return RespanLogType.AGENT;
  }
  if (normalizedName.startsWith("execute_node")) {
    return RespanLogType.TASK;
  }
  if (
    normalizedName === "invocation" ||
    normalizedName.startsWith("invoke_workflow")
  ) {
    return RespanLogType.WORKFLOW;
  }
  return undefined;
}

function resolveEntityNameFromName(
  spanName: string,
  logType: RespanLogType,
): string {
  if (logType === RespanLogType.CHAT) {
    return "google_adk.call_llm";
  }
  if (logType === RespanLogType.WORKFLOW && spanName === "invocation") {
    return "google_adk.invocation";
  }

  const [, ...rest] = spanName.split(/\s+/);
  return rest.join(" ") || spanName || "google_adk.task";
}

function getAttributes(span: ReadableSpan): Attributes | undefined {
  if (utilTypes.isProxy(span)) return undefined;
  const attrs = Object.getOwnPropertyDescriptor(span, "attributes")?.value;
  return attrs && typeof attrs === "object" && !utilTypes.isProxy(attrs)
    ? attrs
    : undefined;
}

function denied(value: unknown): boolean {
  return value === false || value === 0 || value === "false" || value === "0";
}

function contentDenied(ctx: Context): boolean {
  return (
    isTracingSuppressed(ctx) ||
    denied(ctx.getValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT)) ||
    denied(process.env.RESPAN_TRACE_CONTENT) ||
    denied(process.env.TRACELOOP_TRACE_CONTENT) ||
    denied(process.env.ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS)
  );
}

function hasSpanVeto(span: object): boolean {
  const attributes = Object.getOwnPropertyDescriptor(span, "attributes")?.value;
  if (
    attributes &&
    denied(
      Object.getOwnPropertyDescriptor(attributes, "allow_trace_content")?.value,
    )
  )
    return true;
  const events = Object.getOwnPropertyDescriptor(span, "events")?.value;
  return (
    Array.isArray(events) &&
    events.some((event) => {
      const attrs = Object.getOwnPropertyDescriptor(event, "attributes")?.value;
      return (
        attrs &&
        denied(
          Object.getOwnPropertyDescriptor(attrs, "allow_trace_content")?.value,
        )
      );
    })
  );
}

function spanContentDenied(span: object): boolean {
  const policy = CAPTURE_POLICIES.get(span);
  const veto =
    policy?.denied === true ||
    contentDenied(context.active()) ||
    hasSpanVeto(span) ||
    policy?.ancestors.some(
      (ancestor) =>
        CAPTURE_POLICIES.get(ancestor)?.denied || hasSpanVeto(ancestor),
    ) === true;
  if (policy && veto) policy.denied = true;
  return veto;
}

function isContentAttribute(key: string): boolean {
  return (
    key.startsWith(SpanAttributes.LLM_PROMPTS + ".") ||
    key.startsWith(SpanAttributes.LLM_COMPLETIONS + ".") ||
    key === SpanAttributes.TRACELOOP_ENTITY_INPUT ||
    key === SpanAttributes.TRACELOOP_ENTITY_OUTPUT ||
    key === SpanAttributes.LLM_REQUEST_FUNCTIONS ||
    key === ADK_LLM_REQUEST ||
    key === ADK_LLM_RESPONSE ||
    key === ADK_TOOL_CALL_ARGS ||
    key === ADK_TOOL_RESPONSE ||
    key === "allow_trace_content" ||
    key.startsWith(ADK_PREFIX) ||
    key.startsWith(RespanSpanAttributes.RESPAN_METADATA) ||
    key === ATTR_GEN_AI_AGENT_DESCRIPTION ||
    key === ATTR_GEN_AI_TOOL_DESCRIPTION
  );
}

function structuralAttribute(key: string): boolean {
  return (
    [
      SpanAttributes.TRACELOOP_ENTITY_NAME,
      SpanAttributes.TRACELOOP_ENTITY_PATH,
      SpanAttributes.TRACELOOP_WORKFLOW_NAME,
      SpanAttributes.TRACELOOP_SPAN_KIND,
      SpanAttributes.LLM_SYSTEM,
      SpanAttributes.LLM_REQUEST_MODEL,
      SpanAttributes.LLM_RESPONSE_MODEL,
      SpanAttributes.LLM_REQUEST_TYPE,
      SpanAttributes.LLM_REQUEST_TOP_P,
      SpanAttributes.LLM_REQUEST_MAX_TOKENS,
      SpanAttributes.LLM_REQUEST_TEMPERATURE,
      RespanSpanAttributes.RESPAN_LOG_TYPE,
      RespanSpanAttributes.RESPAN_LOG_METHOD,
      ATTR_GEN_AI_AGENT_NAME,
      ATTR_GEN_AI_CONVERSATION_ID,
      ATTR_GEN_AI_OPERATION_NAME,
      ATTR_GEN_AI_TOOL_NAME,
      ATTR_GEN_AI_TOOL_CALL_ID,
      ATTR_GEN_AI_TOOL_TYPE,
      ADK_WORKFLOW_NAME,
      ADK_NODE_PATH,
      ADK_NODE_RUN_ID,
      ADK_NODE_ATTEMPT,
      ADK_NODE_STATUS,
      ADK_NODE_INTERRUPT_COUNT,
    ].includes(key) ||
    key.startsWith("gen_ai.usage.") ||
    key.startsWith("llm.usage.") ||
    key.startsWith("respan.trace.") ||
    key.startsWith("respan.threads.")
  );
}

function copyAttributes(
  attrs: Attributes,
  excludeContent: boolean,
): Attributes {
  const result: Attributes = {};
  for (const key of Object.getOwnPropertyNames(attrs)) {
    if (
      excludeContent &&
      (isContentAttribute(key) || !structuralAttribute(key))
    )
      continue;
    const descriptor = Object.getOwnPropertyDescriptor(attrs, key);
    if (!descriptor || !("value" in descriptor)) continue;
    const value = descriptor.value;
    if (
      value === null ||
      ["string", "number", "boolean"].includes(typeof value)
    ) {
      Object.defineProperty(result, key, {
        value,
        enumerable: true,
        configurable: true,
        writable: true,
      });
    } else if (Array.isArray(value) && !utilTypes.isProxy(value)) {
      const values: Array<string | number | boolean> = [];
      let valid = true;
      for (let index = 0; index < value.length; index += 1) {
        const item = Object.getOwnPropertyDescriptor(value, String(index));
        if (
          !item ||
          !("value" in item) ||
          !["string", "number", "boolean"].includes(typeof item.value)
        ) {
          valid = false;
          break;
        }
        values.push(item.value);
      }
      if (valid)
        Object.defineProperty(result, key, {
          value: values,
          enumerable: true,
          configurable: true,
          writable: true,
        });
    }
  }
  return result;
}

function redactContent(attrs: Attributes): void {
  for (const key of Object.keys(attrs))
    if (isContentAttribute(key)) delete attrs[key];
}

function getInstrumentationScopeName(span: ReadableSpan): string {
  return (
    Object.getOwnPropertyDescriptor(
      Object.getOwnPropertyDescriptor(span, "instrumentationScope")?.value ??
        {},
      "name",
    )?.value ?? ""
  );
}

function resolveLogType(span: ReadableSpan, attrs: Attributes): RespanLogType {
  const operation = String(
    attrs[ATTR_GEN_AI_OPERATION_NAME] ?? "",
  ).toLowerCase();
  const spanName = span.name.toLowerCase();

  if (
    attrs[ATTR_GEN_AI_TOOL_NAME] === "(merged tools)" ||
    spanName === "execute_tool (merged)"
  )
    return RespanLogType.TASK;

  if (
    attrs[SpanAttributes.LLM_SYSTEM] === GOOGLE_ADK_SCOPE_NAME ||
    spanName === "call_llm"
  ) {
    return RespanLogType.CHAT;
  }
  if (operation === "execute_tool" || spanName.startsWith("execute_tool")) {
    return RespanLogType.TOOL;
  }
  if (operation === "invoke_agent" || spanName.startsWith("invoke_agent")) {
    return RespanLogType.AGENT;
  }
  if (
    operation === "invoke_workflow" ||
    spanName === "invocation" ||
    spanName.startsWith("invoke_workflow")
  ) {
    return RespanLogType.WORKFLOW;
  }
  return RespanLogType.TASK;
}

function resolveEntityName(
  span: ReadableSpan,
  attrs: Attributes,
  logType: RespanLogType,
): string {
  if (logType === RespanLogType.AGENT && attrs[ATTR_GEN_AI_AGENT_NAME]) {
    return String(attrs[ATTR_GEN_AI_AGENT_NAME]);
  }
  if (logType === RespanLogType.TOOL && attrs[ATTR_GEN_AI_TOOL_NAME]) {
    return String(attrs[ATTR_GEN_AI_TOOL_NAME]);
  }
  if (logType === RespanLogType.CHAT) {
    return "google_adk.call_llm";
  }
  if (logType === RespanLogType.WORKFLOW) {
    return typeof attrs[ADK_WORKFLOW_NAME] === "string"
      ? attrs[ADK_WORKFLOW_NAME]
      : resolveEntityNameFromName(span.name, logType);
  }
  if (span.name.startsWith("execute_node")) {
    return resolveEntityNameFromName(span.name, logType);
  }
  return span.name || "google_adk.task";
}

function installResponseCapture(span: Span, captureResponse = true): void {
  const writableSpan = span as any;
  if (
    GOOGLE_ADK_RESPONSE_CAPTURES.has(writableSpan) ||
    typeof writableSpan.setAttribute !== "function"
  ) {
    return;
  }

  const accumulator: GoogleADKResponseAccumulator = {};
  const originalSetAttribute = writableSpan.setAttribute as SpanSetAttribute;
  const wrappedSetAttribute: SpanSetAttribute = function (
    this: Span,
    key: string,
    value: unknown,
  ): Span {
    if (key === "allow_trace_content" && denied(value)) {
      const policy = CAPTURE_POLICIES.get(this);
      if (policy) policy.denied = true;
    }
    if (
      captureResponse &&
      key === ADK_LLM_RESPONSE &&
      typeof value === "string" &&
      !spanContentDenied(this)
    ) {
      accumulateResponse(accumulator, value);
    }
    return originalSetAttribute.call(this, key, value);
  };

  try {
    writableSpan.setAttribute = wrappedSetAttribute;
    GOOGLE_ADK_RESPONSE_CAPTURES.set(writableSpan, {
      accumulator,
      originalSetAttribute,
      wrappedSetAttribute,
    });
  } catch {
    // Some third-party Span implementations may not allow method wrapping.
    // Their final scalar response still follows the existing unary path.
  }
}

function getCapturedResponse(span: ReadableSpan): Attributes | undefined {
  const accumulator = GOOGLE_ADK_RESPONSE_CAPTURES.get(
    span as object,
  )?.accumulator;
  return accumulator ? materializeResponse(accumulator) : undefined;
}

function restoreResponseCapture(span: ReadableSpan, discard = true): void {
  const writableSpan = span as any;
  const capture = GOOGLE_ADK_RESPONSE_CAPTURES.get(writableSpan);
  if (!capture) {
    return;
  }

  try {
    if (writableSpan.setAttribute === capture.wrappedSetAttribute) {
      writableSpan.setAttribute = capture.originalSetAttribute;
    }
  } catch {
    // The response data has already been normalized; an ended span with a
    // non-writable method does not need further mutation.
  } finally {
    if (discard) GOOGLE_ADK_RESPONSE_CAPTURES.delete(writableSpan);
  }
}

function normalizeChatSpan(
  attrs: Attributes,
  capturedResponse?: Attributes,
  captureContent = true,
): void {
  attrs[SpanAttributes.LLM_SYSTEM] = "google";
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;

  const request = captureContent
    ? parseJson(attrs[ADK_LLM_REQUEST])
    : undefined;
  if (isRecord(request)) {
    if (request.model !== undefined) {
      setDefault(attrs, SpanAttributes.LLM_REQUEST_MODEL, request.model);
    }

    const config = isRecord(request.config) ? request.config : undefined;
    const systemInstruction = firstDefined(
      config?.systemInstruction,
      config?.system_instruction,
    );
    if (isRecord(systemInstruction) || Array.isArray(systemInstruction)) {
      addContentMessage(attrs, SpanAttributes.LLM_PROMPTS, 0, {
        role: "system",
        parts: Array.isArray(systemInstruction)
          ? systemInstruction
          : systemInstruction.parts,
      });
    }
    if (typeof systemInstruction === "string") {
      setMessage(attrs, SpanAttributes.LLM_PROMPTS, 0, {
        role: "system",
        content: systemInstruction,
      });
    }

    const tools = extractTools(config);
    if (tools.length > 0) {
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
    }

    const contents = Array.isArray(request.contents) ? request.contents : [];
    const startIndex = attrs[`${SpanAttributes.LLM_PROMPTS}.0.content`] ? 1 : 0;
    let promptIndex = startIndex;
    for (const content of contents) {
      promptIndex += addContentMessage(
        attrs,
        SpanAttributes.LLM_PROMPTS,
        promptIndex,
        content,
      );
    }
  }

  const response = captureContent
    ? (capturedResponse ?? parseJson(attrs[ADK_LLM_RESPONSE]))
    : undefined;
  if (!captureContent) {
    if (typeof attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] === "number")
      attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] =
        attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS];
    if (typeof attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] === "number")
      attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] =
        attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS];
  }
  if (isRecord(response)) {
    addContentMessage(
      attrs,
      SpanAttributes.LLM_COMPLETIONS,
      0,
      response.content,
    );

    const usage = isRecord(response.usageMetadata)
      ? response.usageMetadata
      : isRecord(response.usage_metadata)
        ? response.usage_metadata
        : undefined;
    const inputTokens = numberValue(
      firstDefined(usage?.promptTokenCount, usage?.prompt_token_count),
    );
    const outputTokens = numberValue(
      firstDefined(usage?.candidatesTokenCount, usage?.candidates_token_count),
    );
    const thoughtsTokens = numberValue(
      firstDefined(usage?.thoughtsTokenCount, usage?.thoughts_token_count),
    );
    const totalTokens = numberValue(
      firstDefined(usage?.totalTokenCount, usage?.total_token_count),
    );
    const normalizedOutputTokens =
      outputTokens === undefined
        ? undefined
        : outputTokens + (thoughtsTokens ?? 0);

    if (inputTokens !== undefined) {
      attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = inputTokens;
      attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = inputTokens;
    }
    if (normalizedOutputTokens !== undefined) {
      attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = normalizedOutputTokens;
      attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] =
        normalizedOutputTokens;
    }
    if (totalTokens !== undefined) {
      attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
    }

    const finishReason = firstDefined(
      response.finishReason,
      response.finish_reason,
    );
    if (finishReason !== undefined) {
      attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.finish_reason`] =
        String(finishReason).toLowerCase();
    }
  }
}

function accumulateResponse(
  accumulator: GoogleADKResponseAccumulator,
  value: unknown,
): void {
  const response = parseJson(value);
  if (!isRecord(response)) {
    return;
  }

  accumulator.response ??= {};
  for (const [key, responseValue] of Object.entries(response)) {
    if (key !== "content") {
      accumulator.response[key] = responseValue;
    }
  }

  if (isRecord(response.content)) {
    accumulator.content ??= {
      attributes: {},
      parts: [],
      partIndexes: new Map<string, number>(),
    };
    accumulateResponseContent(
      accumulator.content,
      response.content,
      response.partial === true,
    );
  }
}

function materializeResponse(
  accumulator: GoogleADKResponseAccumulator,
): Attributes | undefined {
  if (!accumulator.response && !accumulator.content) {
    return undefined;
  }

  const response = { ...(accumulator.response ?? {}) };
  if (accumulator.content) {
    response.content = {
      ...accumulator.content.attributes,
      ...(accumulator.content.parts.length > 0
        ? { parts: accumulator.content.parts }
        : {}),
    };
  }
  return response;
}

function accumulateResponseContent(
  accumulator: GoogleADKContentAccumulator,
  content: Attributes,
  isPartialResponse: boolean,
): void {
  for (const [key, contentValue] of Object.entries(content)) {
    if (key !== "parts") {
      accumulator.attributes[key] = contentValue;
    }
  }
  if (!Array.isArray(content.parts)) {
    return;
  }

  const functionCallSequences = new Map<string, number>();
  for (const part of content.parts) {
    if (!isRecord(part)) {
      continue;
    }

    if (isRecord(part.functionCall)) {
      const name = String(part.functionCall.name ?? "");
      const sequence = functionCallSequences.get(name) ?? 0;
      functionCallSequences.set(name, sequence + 1);
      accumulateFunctionCallPart(accumulator, part, name, sequence);
      continue;
    }

    if (typeof part.text === "string") {
      accumulateTextPart(accumulator, part, isPartialResponse);
      continue;
    }

    const identity = `part:${safeJson(part)}`;
    if (!accumulator.partIndexes.has(identity)) {
      accumulator.partIndexes.set(identity, accumulator.parts.length);
      accumulator.parts.push({ ...part });
    }
  }
}

function accumulateTextPart(
  accumulator: GoogleADKContentAccumulator,
  part: Attributes,
  isPartialResponse: boolean,
): void {
  const textShape = { ...part };
  delete textShape.text;
  const identity = `text:${safeJson(textShape)}`;
  const existingIndex = accumulator.partIndexes.get(identity);
  if (existingIndex === undefined) {
    accumulator.partIndexes.set(identity, accumulator.parts.length);
    accumulator.parts.push({ ...part });
    return;
  }

  const existingPart = accumulator.parts[existingIndex];
  existingPart.text = mergeStreamText(
    String(existingPart.text ?? ""),
    part.text,
    !isPartialResponse,
  );
}

function accumulateFunctionCallPart(
  accumulator: GoogleADKContentAccumulator,
  part: Attributes,
  name: string,
  sequence: number,
): void {
  const functionCall = part.functionCall as Attributes;
  const sequenceIdentity = `functionCall:name:${name}:sequence:${sequence}`;
  const idIdentity =
    functionCall.id === undefined
      ? undefined
      : `functionCall:id:${String(functionCall.id)}`;
  const existingIndex =
    (idIdentity === undefined
      ? undefined
      : accumulator.partIndexes.get(idIdentity)) ??
    accumulator.partIndexes.get(sequenceIdentity);

  if (existingIndex === undefined) {
    const nextIndex = accumulator.parts.length;
    accumulator.parts.push(mergeFunctionCallPart({}, part));
    accumulator.partIndexes.set(sequenceIdentity, nextIndex);
    if (idIdentity !== undefined) {
      accumulator.partIndexes.set(idIdentity, nextIndex);
    }
    return;
  }

  accumulator.parts[existingIndex] = mergeFunctionCallPart(
    accumulator.parts[existingIndex],
    part,
  );
  accumulator.partIndexes.set(sequenceIdentity, existingIndex);
  if (idIdentity !== undefined) {
    accumulator.partIndexes.set(idIdentity, existingIndex);
  }
}

function mergeStreamText(
  current: string,
  next: string,
  deduplicateEqualValue = false,
): string {
  if (!current) {
    return next;
  }
  if (!next) {
    return current;
  }

  // Some ADK providers emit deltas while others emit a cumulative terminal
  // response. Replacing a prefix-complete value keeps each chunk exactly once.
  if (
    next.startsWith(current) &&
    (next.length > current.length || deduplicateEqualValue)
  ) {
    return next;
  }
  return current + next;
}

function mergeFunctionCallPart(
  current: Attributes,
  next: Attributes,
): Attributes {
  const currentCall = isRecord(current.functionCall)
    ? current.functionCall
    : {};
  const nextCall = isRecord(next.functionCall) ? next.functionCall : {};
  const hasCurrentArgs = isRecord(currentCall.args);
  const args = hasCurrentArgs ? currentCall.args : {};

  if (Array.isArray(nextCall.partialArgs)) {
    for (const partialArg of nextCall.partialArgs) {
      applyPartialArg(args, partialArg);
    }
  }
  if (isRecord(nextCall.args)) {
    mergeRecordInPlace(args, nextCall.args);
  }

  const functionCall = { ...currentCall, ...nextCall };
  delete functionCall.partialArgs;
  delete functionCall.willContinue;
  if (
    hasCurrentArgs ||
    Array.isArray(nextCall.partialArgs) ||
    isRecord(nextCall.args)
  ) {
    functionCall.args = args;
  }

  return {
    ...current,
    ...next,
    functionCall,
  };
}

function applyPartialArg(args: Attributes, value: unknown): void {
  if (!isRecord(value) || typeof value.jsonPath !== "string") {
    return;
  }

  const path = parseJsonPath(value.jsonPath);
  const partialValue = partialArgValue(value);
  if (!path || path.length === 0 || !partialValue.present) {
    return;
  }

  let target: any = args;
  for (let index = 0; index < path.length - 1; index += 1) {
    const key = path[index];
    const nextKey = path[index + 1];
    const expectedContainer = typeof nextKey === "number" ? [] : {};
    const existing = Object.getOwnPropertyDescriptor(target, key)?.value;
    if (
      (Array.isArray(expectedContainer) && !Array.isArray(existing)) ||
      (!Array.isArray(expectedContainer) && !isRecord(existing))
    ) {
      Object.defineProperty(target, key, {
        value: expectedContainer,
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
    target = target[key];
  }

  const key = path[path.length - 1];
  const existing = Object.getOwnPropertyDescriptor(target, key)?.value;
  Object.defineProperty(target, key, {
    value:
      typeof existing === "string" && typeof partialValue.value === "string"
        ? existing + partialValue.value
        : partialValue.value,
    enumerable: true,
    configurable: true,
    writable: true,
  });
}

function partialArgValue(value: Attributes): {
  present: boolean;
  value?: unknown;
} {
  for (const key of ["stringValue", "numberValue", "boolValue", "nullValue"]) {
    if (Object.prototype.hasOwnProperty.call(value, key)) {
      return {
        present: true,
        value: key === "nullValue" ? null : value[key],
      };
    }
  }
  return { present: false };
}

function parseJsonPath(path: string): Array<string | number> | undefined {
  if (!path.startsWith("$")) {
    return undefined;
  }

  const segments: Array<string | number> = [];
  let index = 1;
  while (index < path.length) {
    if (path[index] === ".") {
      const start = index + 1;
      index = start;
      while (
        index < path.length &&
        path[index] !== "." &&
        path[index] !== "["
      ) {
        index += 1;
      }
      if (index === start) {
        return undefined;
      }
      segments.push(path.slice(start, index));
      continue;
    }

    if (path[index] === "[") {
      const end = path.indexOf("]", index + 1);
      if (end === -1) {
        return undefined;
      }
      const selector = path.slice(index + 1, end).trim();
      if (/^\d+$/.test(selector)) {
        segments.push(Number(selector));
      } else if (
        (selector.startsWith('"') && selector.endsWith('"')) ||
        (selector.startsWith("'") && selector.endsWith("'"))
      ) {
        const quote = selector[0];
        const property = selector
          .slice(1, -1)
          .replace(new RegExp(`\\\\${quote}`, "g"), quote)
          .replace(/\\\\\\\\/g, "\\");
        segments.push(property);
      } else {
        return undefined;
      }
      index = end + 1;
      continue;
    }

    return undefined;
  }
  return segments;
}

function mergeRecordInPlace(target: Attributes, source: Attributes): void {
  for (const [key, value] of Object.entries(source)) {
    const existing = Object.getOwnPropertyDescriptor(target, key)?.value;
    if (isRecord(value) && isRecord(existing)) {
      mergeRecordInPlace(existing, value);
    } else if (isRecord(value)) {
      Object.defineProperty(target, key, {
        value: { ...value },
        enumerable: true,
        configurable: true,
        writable: true,
      });
    } else if (Array.isArray(value)) {
      Object.defineProperty(target, key, {
        value: [...value],
        enumerable: true,
        configurable: true,
        writable: true,
      });
    } else {
      Object.defineProperty(target, key, {
        value,
        enumerable: true,
        configurable: true,
        writable: true,
      });
    }
  }
}

function normalizeToolSpan(attrs: Attributes, entityName: string): void {
  if (
    attrs[ATTR_GEN_AI_TOOL_CALL_ID] === "<not specified>" ||
    attrs[ATTR_GEN_AI_TOOL_CALL_ID] === ""
  )
    delete attrs[ATTR_GEN_AI_TOOL_CALL_ID];
  const rawArgs = parseJson(attrs[ADK_TOOL_CALL_ARGS]);
  const args = rawArgs === "N/A" ? undefined : rawArgs;
  const input = {
    name: entityName,
    arguments: args ?? {},
  };
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(input);

  const rawResponse = parseJson(attrs[ADK_TOOL_RESPONSE]);
  if (rawResponse !== undefined && rawResponse !== "<not specified>") {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(rawResponse);
  }

  if (attrs[ATTR_GEN_AI_TOOL_TYPE]) {
    setMetadata(
      attrs,
      "google_adk_tool_type",
      String(attrs[ATTR_GEN_AI_TOOL_TYPE]),
    );
  }
  if (attrs[ATTR_GEN_AI_TOOL_DESCRIPTION]) {
    setMetadata(
      attrs,
      "google_adk_tool_description",
      String(attrs[ATTR_GEN_AI_TOOL_DESCRIPTION]),
    );
  }
}

function normalizeAgentSpan(attrs: Attributes, entityName: string): void {
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = entityName;

  if (attrs[ATTR_GEN_AI_AGENT_DESCRIPTION]) {
    setMetadata(
      attrs,
      "google_adk_agent_description",
      String(attrs[ATTR_GEN_AI_AGENT_DESCRIPTION]),
    );
  }
  if (attrs[ATTR_GEN_AI_CONVERSATION_ID]) {
    setMetadata(
      attrs,
      "google_adk_conversation_id",
      String(attrs[ATTR_GEN_AI_CONVERSATION_ID]),
    );
  }
}

function normalizeWorkflowSpan(attrs: Attributes, entityName: string): void {
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
  setDefault(attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, "");
}

function normalizeNodeAttributes(attrs: Attributes): void {
  const nodePath = attrs[ADK_NODE_PATH];
  if (typeof nodePath === "string" && nodePath) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = nodePath;
  }
  for (const [key, name] of [
    [ADK_NODE_RUN_ID, "google_adk_node_run_id"],
    [ADK_NODE_ATTEMPT, "google_adk_node_attempt"],
    [ADK_NODE_STATUS, "google_adk_node_status"],
    [ADK_NODE_INTERRUPT_COUNT, "google_adk_node_interrupt_count"],
  ]) {
    if (attrs[key] !== undefined) {
      setMetadata(attrs, name, attrs[key]);
    }
  }
}

function addContentMessage(
  attrs: Attributes,
  prefix: string,
  index: number,
  value: unknown,
): number {
  if (!isRecord(value)) return 0;
  const role = normalizeRole(value.role);
  const { text, toolCalls, toolResponses } = extractContentParts(value.parts);
  let count = 0;
  if (text !== undefined || toolCalls !== undefined || !toolResponses?.length) {
    setMessage(attrs, prefix, index, { role, content: text, toolCalls });
    count += 1;
  }
  for (const response of toolResponses ?? []) {
    const targetIndex = index + count;
    setMessage(attrs, prefix, targetIndex, {
      role: "tool",
      content: safeJson(
        Object.prototype.hasOwnProperty.call(response, "response")
          ? response.response
          : response,
      ),
    });
    if (typeof response.id === "string")
      attrs[`${prefix}.${targetIndex}.tool_call_id`] = response.id;
    if (typeof response.name === "string")
      attrs[`${prefix}.${targetIndex}.name`] = response.name;
    count += 1;
  }
  return count;
}

function setMessage(
  attrs: Attributes,
  prefix: string,
  index: number,
  message: {
    role?: string;
    content?: string;
    toolCalls?: Array<Record<string, unknown>>;
  },
): void {
  const target = `${prefix}.${index}`;
  if (message.role) {
    attrs[`${target}.role`] = message.role;
  }
  if (message.content !== undefined) {
    attrs[`${target}.content`] = message.content;
  }
  if (message.toolCalls && message.toolCalls.length > 0) {
    attrs[`${target}.tool_calls`] = safeJson(message.toolCalls);
  }
}

function extractContentParts(parts: unknown): {
  text?: string;
  toolCalls?: Array<Record<string, unknown>>;
  toolResponses?: Attributes[];
} {
  if (!Array.isArray(parts)) {
    return {};
  }

  const textParts: string[] = [];
  const contentParts: Attributes[] = [];
  let hasMultimodalContent = false;
  const toolCalls: Array<Record<string, unknown>> = [];
  const toolResponses: Attributes[] = [];

  for (const part of parts) {
    if (!isRecord(part)) {
      continue;
    }

    if (typeof part.text === "string") {
      textParts.push(part.text);
      contentParts.push(part);
    } else if (
      !isRecord(part.functionCall) &&
      !isRecord(part.functionResponse)
    ) {
      contentParts.push(part);
      hasMultimodalContent = true;
    }

    if (isRecord(part.functionCall)) {
      const functionCall = part.functionCall;
      const functionPayload: Record<string, unknown> = {};
      if (functionCall.name !== undefined) {
        functionPayload.name = functionCall.name;
      }
      if (functionCall.args !== undefined) {
        functionPayload.arguments = safeJson(functionCall.args);
      }

      const toolCall: Record<string, unknown> = {
        type: "function",
        function: functionPayload,
      };
      if (functionCall.id !== undefined) {
        toolCall.id = functionCall.id;
      }
      toolCalls.push(toolCall);
    }

    if (isRecord(part.functionResponse)) {
      toolResponses.push(part.functionResponse);
    }
  }

  return {
    text: hasMultimodalContent
      ? safeJson(contentParts)
      : textParts.length > 0
        ? textParts.join("\n")
        : undefined,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    toolResponses: toolResponses.length > 0 ? toolResponses : undefined,
  };
}

function extractTools(
  config: Attributes | undefined,
): Array<Record<string, unknown>> {
  if (!config || !Array.isArray(config.tools)) {
    return [];
  }

  const tools: Array<Record<string, unknown>> = [];
  for (const tool of config.tools) {
    if (!isRecord(tool)) {
      continue;
    }
    const declarations = firstDefined(
      tool.functionDeclarations,
      tool.function_declarations,
    );
    if (!Array.isArray(declarations)) {
      continue;
    }
    for (const declaration of declarations) {
      if (isRecord(declaration)) {
        tools.push(declaration);
      }
    }
  }
  return tools;
}

function cleanupAttrs(attrs: Attributes): void {
  if (
    attrs[ATTR_GEN_AI_TOOL_CALL_ID] === "<not specified>" ||
    attrs[ATTR_GEN_AI_TOOL_CALL_ID] === ""
  )
    delete attrs[ATTR_GEN_AI_TOOL_CALL_ID];
  for (const key of Object.keys(attrs)) {
    if (
      key.startsWith(ADK_PREFIX) ||
      key === ADK_WORKFLOW_NAME ||
      key === ADK_NODE_PATH ||
      key === ADK_NODE_RUN_ID ||
      key === ADK_NODE_ATTEMPT ||
      key === ADK_NODE_STATUS ||
      key === ADK_NODE_INTERRUPT_COUNT ||
      key === ATTR_GEN_AI_OPERATION_NAME ||
      key === ATTR_GEN_AI_AGENT_DESCRIPTION ||
      key === ATTR_GEN_AI_AGENT_NAME ||
      key === ATTR_GEN_AI_CONVERSATION_ID ||
      key === ATTR_GEN_AI_TOOL_DESCRIPTION ||
      key === ATTR_GEN_AI_TOOL_NAME ||
      key === ATTR_GEN_AI_TOOL_TYPE ||
      OFF_CONTRACT_ALIASES.has(key)
    ) {
      delete attrs[key];
    }
  }
}

function normalizeRole(role: unknown): string {
  if (role === "model") {
    return "assistant";
  }
  if (typeof role === "string" && role) {
    return role;
  }
  return "user";
}

function parseJson(value: unknown): unknown {
  if (typeof value !== "string") {
    return undefined;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function isRecord(value: unknown): value is Attributes {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function setDefault(attrs: Attributes, key: string, value: unknown): void {
  if (attrs[key] === undefined && value !== undefined) {
    attrs[key] = value;
  }
}

function firstDefined<T>(...values: Array<T | undefined>): T | undefined {
  for (const value of values) {
    if (value !== undefined) {
      return value;
    }
  }
  return undefined;
}

function setMetadata(attrs: Attributes, name: string, value: unknown): void {
  const existing = parseJson(attrs[RespanSpanAttributes.RESPAN_METADATA]);
  attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
    ...(isRecord(existing) ? existing : {}),
    [name]: value,
  });
}
