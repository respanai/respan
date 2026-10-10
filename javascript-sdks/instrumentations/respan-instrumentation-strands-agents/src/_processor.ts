import type { Context, Span } from "@opentelemetry/api";
import type {
  ReadableSpan,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  EVENT_GEN_AI_CHOICE,
  EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS,
  EVENT_GEN_AI_SYSTEM_MESSAGE,
  EVENT_GEN_AI_TOOL_MESSAGE,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  STRANDS_AGENT_TOOLS_ATTR,
  STRANDS_EVENT_END_TIME_ATTR,
  STRANDS_EVENT_MESSAGE_PREFIX,
  STRANDS_EVENT_MESSAGE_SUFFIX,
  STRANDS_EVENT_START_TIME_ATTR,
  STRANDS_OPERATION_CHAT,
  STRANDS_OPERATION_EXECUTE_AGENT_LOOP_CYCLE,
  STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE,
  STRANDS_OPERATION_EXECUTE_NODE,
  STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT,
  STRANDS_OPERATION_EXECUTE_TOOL,
  STRANDS_OPERATION_INVOKE_AGENT,
  STRANDS_OPERATION_INVOKE_GRAPH,
  STRANDS_OPERATION_INVOKE_PREFIX,
  STRANDS_OPERATION_INVOKE_SWARM,
  STRANDS_RAW_ATTR_PREFIXES_TO_STRIP,
  STRANDS_STRUCTURED_OUTPUT_TOOL_NAME,
  STRANDS_SYSTEM_NAME,
  STRANDS_SYSTEM_PROMPT_ATTR,
  STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
  STRANDS_TOOL_JSON_SCHEMA_ATTR,
  STRANDS_TOOL_STATUS_ATTR,
  STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR,
  STRANDS_USAGE_TOTAL_TOKENS_ATTR,
} from "./_constants.js";

import {
  ContentPolicy,
  contentAllowed,
  copyData,
  snapshotSpan,
  dataProperty,
} from "./_privacy.js";

type SpanAttributesRecord = Record<string, any>;
type SpanEventRecord = {
  name?: string;
  attributes?: Record<string, any>;
};

type StructuredOutputCandidate = {
  output: unknown;
};

type StrandsTraceState = {
  parents: Map<string, string | undefined>;
  structuredOutputs: Map<string, StructuredOutputCandidate>;
};

const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const GEN_AI_PROMPT_PREFIX = `${ATTR_GEN_AI_PROMPT}.`;
const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.`;

const STRANDS_RAW_ATTRS_TO_STRIP = new Set([
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_AGENT_ID,
  STRANDS_AGENT_TOOLS_ATTR,
  STRANDS_SYSTEM_PROMPT_ATTR,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  STRANDS_TOOL_STATUS_ATTR,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  STRANDS_TOOL_JSON_SCHEMA_ATTR,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  STRANDS_EVENT_START_TIME_ATTR,
  STRANDS_EVENT_END_TIME_ATTR,
  STRANDS_USAGE_TOTAL_TOKENS_ATTR,
  STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR,
]);

const STRANDS_NON_LLM_ATTRS_TO_STRIP = new Set([
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  STRANDS_USAGE_TOTAL_TOKENS_ATTR,
  STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR,
  SpanAttributes.LLM_REQUEST_TYPE,
  SpanAttributes.LLM_REQUEST_FUNCTIONS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
]);

const OFF_CONTRACT_ALIAS_ATTRS = new Set([
  ...STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
  RespanSpanAttributes.RESPAN_SPAN_TOOLS,
  RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
]);

export class StrandsAgentsSpanProcessor implements SpanProcessor {
  private readonly _traceStates = new Map<string, StrandsTraceState>();
  private readonly _policy: ContentPolicy;
  private readonly _prepared = new WeakMap<object, ReadableSpan>();

  constructor(options: { traceContent?: boolean | (() => boolean) } = {}) {
    this._policy = new ContentPolicy(options.traceContent ?? true);
  }

  onStart(span: Span, parentContext: Context): void {
    this._policy.onStart(span, parentContext);
  }

  prepareForExport(span: ReadableSpan): ReadableSpan {
    const prepared = this._prepared.get(span);
    this._prepared.delete(span);
    return prepared ? snapshotSpan(prepared, this._policy.allowed(span)) : span;
  }

  onEnd(span: ReadableSpan): void {
    const original = span;
    this._policy.finish(span);
    if (!isStrandsAgentsSpan(span, span.attributes)) return;
    span = translatedStrandsAgentsSpan(span, this._policy.allowed(span));
    this._prepared.set(original, span);
    const traceId = span.spanContext().traceId;
    const spanId = span.spanContext().spanId;
    const parentSpanId = span.parentSpanContext?.spanId;
    const state = this._traceState(traceId);
    state.parents.set(spanId, parentSpanId);

    const attrs = (span as any).attributes as SpanAttributesRecord | undefined;
    if (
      attrs?.[RespanSpanAttributes.RESPAN_LOG_TYPE] === RespanLogType.TOOL &&
      attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] ===
        STRANDS_STRUCTURED_OUTPUT_TOOL_NAME
    ) {
      const output = normalizeStructuredOutput(
        attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT],
      );
      if (output !== undefined) {
        state.structuredOutputs.set(spanId, { output });
      }
    }

    if (attrs?.[RespanSpanAttributes.RESPAN_LOG_TYPE] === RespanLogType.AGENT) {
      this._recoverStructuredAgentOutput(attrs, spanId, state);
      this._clearAgentState(spanId, state);
    }

    if (
      attrs?.[RespanSpanAttributes.RESPAN_LOG_TYPE] === RespanLogType.WORKFLOW
    )
      this._clearAgentState(spanId, state);
    if (
      !parentSpanId ||
      (state.parents.size === 0 && state.structuredOutputs.size === 0)
    ) {
      this._traceStates.delete(traceId);
    }
  }

  shutdown(): Promise<void> {
    this._traceStates.clear();
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  private _traceState(traceId: string): StrandsTraceState {
    let state = this._traceStates.get(traceId);
    if (!state) {
      state = {
        parents: new Map(),
        structuredOutputs: new Map(),
      };
      this._traceStates.set(traceId, state);
    }
    return state;
  }

  private _recoverStructuredAgentOutput(
    attrs: SpanAttributesRecord,
    agentSpanId: string,
    state: StrandsTraceState,
  ): void {
    if (
      hasMeaningfulAgentOutput(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    ) {
      return;
    }

    let recovered: unknown;
    for (const [toolSpanId, candidate] of state.structuredOutputs) {
      if (isDescendantSpan(toolSpanId, agentSpanId, state.parents)) {
        recovered = candidate.output;
      }
    }
    if (recovered === undefined) {
      return;
    }

    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson([
      { role: "assistant", content: recovered },
    ]);
  }

  private _clearAgentState(
    agentSpanId: string,
    state: StrandsTraceState,
  ): void {
    for (const toolSpanId of state.structuredOutputs.keys()) {
      if (isDescendantSpan(toolSpanId, agentSpanId, state.parents)) {
        state.structuredOutputs.delete(toolSpanId);
      }
    }
    for (const spanId of state.parents.keys()) {
      if (
        spanId === agentSpanId ||
        isDescendantSpan(spanId, agentSpanId, state.parents)
      ) {
        state.parents.delete(spanId);
      }
    }
  }
}

function isDescendantSpan(
  spanId: string,
  ancestorSpanId: string,
  parents: Map<string, string | undefined>,
): boolean {
  const visited = new Set<string>();
  let current = parents.get(spanId);
  while (current && !visited.has(current)) {
    if (current === ancestorSpanId) {
      return true;
    }
    visited.add(current);
    current = parents.get(current);
  }
  return false;
}

function normalizeStructuredOutput(value: unknown): unknown {
  let parsed = safeJsonLoads(value);
  if (Array.isArray(parsed) && parsed.length === 1) {
    parsed = parsed[0];
  }
  if (isRecord(parsed) && "json" in parsed) {
    return toSerializableValue(parsed.json);
  }
  return toSerializableValue(parsed);
}

function hasMeaningfulAgentOutput(value: unknown): boolean {
  const parsed = safeJsonLoads(value);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    return !isEmptyValue(parsed);
  }
  return parsed.some((message) => {
    if (!isRecord(message)) {
      return !isEmptyValue(message);
    }
    return !isEmptyValue(message.content);
  });
}

/** Retained in-place helper; the instrumentor uses a private export copy. */
export function enrichStrandsAgentsSpan(span: ReadableSpan): void {
  const translated = translatedStrandsAgentsSpan(span, contentAllowed());
  if (translated !== span) replaceSpanAttributes(span, translated.attributes);
}

function translatedStrandsAgentsSpan(
  span: ReadableSpan,
  captureContent: boolean,
): ReadableSpan {
  const originalAttrs = (span as any).attributes as
    SpanAttributesRecord | undefined;
  if (!originalAttrs || !isStrandsAgentsSpan(span, originalAttrs)) return span;
  span = snapshotSpan(
    span,
    captureContent && (span.spanContext().traceFlags & 1) !== 0,
  );
  const attrs = { ...span.attributes };

  const logType = extractLogType(span, attrs);
  if (!logType) {
    return span;
  }

  switch (logType) {
    case RespanLogType.WORKFLOW:
      enrichWorkflowSpan(span, attrs);
      break;
    case RespanLogType.AGENT:
      enrichAgentSpan(span, attrs);
      break;
    case RespanLogType.TASK:
      enrichTaskSpan(span, attrs);
      break;
    case RespanLogType.CHAT:
      enrichChatSpan(span, attrs);
      break;
    case RespanLogType.TOOL:
      enrichToolSpan(span, attrs);
      break;
    default:
      return span;
  }

  replaceSpanAttributes(span, stripRawAttrs(attrs, logType));
  return span;
}

function isStrandsAgentsSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): boolean {
  const scope = span.instrumentationScope?.name;
  const service = process.env.OTEL_SERVICE_NAME || STRANDS_SYSTEM_NAME;
  return (
    dataProperty(attrs, ATTR_GEN_AI_SYSTEM) === STRANDS_SYSTEM_NAME ||
    dataProperty(attrs, ATTR_GEN_AI_PROVIDER_NAME) === STRANDS_SYSTEM_NAME ||
    (scope === service &&
      dataProperty(attrs, STRANDS_EVENT_START_TIME_ATTR) !== undefined &&
      (KNOWN_STRANDS_OPERATIONS.has(
        dataProperty(attrs, ATTR_GEN_AI_OPERATION_NAME),
      ) ||
        isStrandsTaskSpanName(span.name) ||
        span.name.startsWith("execute_node")))
  );
}

const KNOWN_STRANDS_OPERATIONS = new Set([
  STRANDS_OPERATION_INVOKE_AGENT,
  STRANDS_OPERATION_CHAT,
  STRANDS_OPERATION_EXECUTE_TOOL,
  STRANDS_OPERATION_EXECUTE_AGENT_LOOP_CYCLE,
  STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE,
  STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT,
  STRANDS_OPERATION_EXECUTE_NODE,
  STRANDS_OPERATION_INVOKE_GRAPH,
  STRANDS_OPERATION_INVOKE_SWARM,
]);

function extractLogType(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): RespanLogType | undefined {
  if (typeof attrs[ATTR_GEN_AI_TOOL_NAME] === "string") {
    return RespanLogType.TOOL;
  }

  const operationName = attrs[ATTR_GEN_AI_OPERATION_NAME];
  switch (operationName) {
    case STRANDS_OPERATION_INVOKE_GRAPH:
    case STRANDS_OPERATION_INVOKE_SWARM:
      return RespanLogType.WORKFLOW;
    case STRANDS_OPERATION_INVOKE_AGENT:
      return RespanLogType.AGENT;
    case STRANDS_OPERATION_CHAT:
      return RespanLogType.CHAT;
    case STRANDS_OPERATION_EXECUTE_TOOL:
      return RespanLogType.TOOL;
    case STRANDS_OPERATION_EXECUTE_AGENT_LOOP_CYCLE:
    case STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE:
    case STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT:
    case STRANDS_OPERATION_EXECUTE_NODE:
      return RespanLogType.TASK;
    default:
      break;
  }

  if (span.name.startsWith(`${STRANDS_OPERATION_EXECUTE_TOOL} `)) {
    return RespanLogType.TOOL;
  }
  if (span.name.startsWith(`${STRANDS_OPERATION_INVOKE_AGENT} `)) {
    return RespanLogType.AGENT;
  }
  if (span.name.startsWith(`${STRANDS_OPERATION_INVOKE_GRAPH} `)) {
    return RespanLogType.WORKFLOW;
  }
  if (span.name.startsWith(`${STRANDS_OPERATION_INVOKE_SWARM} `)) {
    return RespanLogType.WORKFLOW;
  }
  if (isStrandsTaskSpanName(span.name)) {
    return RespanLogType.TASK;
  }
  if (
    typeof operationName === "string" &&
    operationName.startsWith(STRANDS_OPERATION_INVOKE_PREFIX)
  ) {
    return RespanLogType.AGENT;
  }
  return undefined;
}

function enrichWorkflowSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const entityName = extractWorkflowName(span, attrs);
  const workflowName = existingWorkflowName(attrs);
  setCommonAttrs(attrs, {
    logType: RespanLogType.WORKFLOW,
    entityName,
    entityPath: "",
  });
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName ?? entityName;
  setInputOutputAttrs(attrs, {
    inputMessages: extractInputMessages(span, attrs),
    outputMessages: extractOutputMessages(span, attrs),
  });
}

function enrichAgentSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const agentName = extractAgentName(span, attrs);
  const workflowName = existingWorkflowName(attrs);
  setCommonAttrs(attrs, {
    logType: RespanLogType.AGENT,
    entityName: agentName,
    entityPath: agentName,
  });
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName ?? agentName;
  setInputOutputAttrs(attrs, {
    inputMessages: extractInputMessages(span, attrs),
    outputMessages: extractOutputMessages(span, attrs),
  });

  const observedNames = safeJsonLoads(attrs[STRANDS_AGENT_TOOLS_ATTR]);
  const observedDefinitions = safeJsonLoads(
    attrs[ATTR_GEN_AI_TOOL_DEFINITIONS] ??
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS],
  );
  if (observedNames !== undefined || observedDefinitions !== undefined) {
    const inherited = safeJsonLoads(
      attrs[RespanSpanAttributes.RESPAN_METADATA],
    );
    attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
      ...(isRecord(inherited) ? inherited : {}),
      ...(observedNames !== undefined
        ? { strands_agent_tools: observedNames }
        : {}),
      ...(observedDefinitions !== undefined
        ? { strands_tool_definitions: observedDefinitions }
        : {}),
    });
  }
}

function enrichTaskSpan(span: ReadableSpan, attrs: SpanAttributesRecord): void {
  const operationName = attrs[ATTR_GEN_AI_OPERATION_NAME];
  const entityName =
    typeof operationName === "string" && operationName
      ? operationName
      : span.name;
  setCommonAttrs(attrs, {
    logType: RespanLogType.TASK,
    entityName,
    entityPath: entityName,
  });
  setInputOutputAttrs(attrs, {
    inputMessages: extractInputMessages(span, attrs),
    outputMessages: extractOutputMessages(span, attrs),
  });
}

function enrichChatSpan(span: ReadableSpan, attrs: SpanAttributesRecord): void {
  setCommonAttrs(attrs, {
    logType: RespanLogType.CHAT,
    entityName: STRANDS_OPERATION_CHAT,
    entityPath: STRANDS_OPERATION_CHAT,
  });
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  const provider =
    attrs[ATTR_GEN_AI_PROVIDER_NAME] ?? attrs[ATTR_GEN_AI_SYSTEM];
  attrs[ATTR_GEN_AI_SYSTEM] =
    typeof provider === "string" && provider !== STRANDS_SYSTEM_NAME
      ? provider
      : STRANDS_SYSTEM_NAME;
  const definitions = extractToolDefinitions(attrs);
  if (definitions?.length)
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(definitions);

  const inputMessages = extractInputMessages(span, attrs);
  const outputMessages = extractOutputMessages(span, attrs);
  setInputOutputAttrs(attrs, { inputMessages, outputMessages });
  if (inputMessages?.length) {
    setIndexedMessages(attrs, GEN_AI_PROMPT_PREFIX, inputMessages);
  }
  if (outputMessages?.length) {
    setIndexedMessages(attrs, GEN_AI_COMPLETION_PREFIX, outputMessages);
  }
  setUsageAttrs(attrs);
}

function enrichToolSpan(span: ReadableSpan, attrs: SpanAttributesRecord): void {
  const toolName = extractToolName(span, attrs);
  setCommonAttrs(attrs, {
    logType: RespanLogType.TOOL,
    entityName: toolName,
    entityPath: toolName,
  });

  const toolArguments =
    attrs[ATTR_GEN_AI_TOOL_CALL_ARGUMENTS] ??
    extractToolEventPayload(span, EVENT_GEN_AI_TOOL_MESSAGE, "content") ??
    modernToolPayload(span, "input");
  const toolResult =
    attrs[ATTR_GEN_AI_TOOL_CALL_RESULT] ??
    extractToolEventPayload(span, EVENT_GEN_AI_CHOICE, "message") ??
    modernToolPayload(span, "output");

  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({
    name: toolName,
    ...(typeof attrs[ATTR_GEN_AI_TOOL_CALL_ID] === "string"
      ? { id: attrs[ATTR_GEN_AI_TOOL_CALL_ID] }
      : {}),
    ...(toolArguments !== undefined
      ? { arguments: toSerializableValue(safeJsonLoads(toolArguments)) }
      : {}),
  });

  if (toolResult !== undefined) {
    const output = contentForMessage(safeJsonLoads(toolResult));
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(output);
  }
}

function modernToolPayload(
  span: ReadableSpan,
  direction: "input" | "output",
): unknown {
  const messages = operationDetailMessages(
    span,
    direction === "input"
      ? ATTR_GEN_AI_INPUT_MESSAGES
      : ATTR_GEN_AI_OUTPUT_MESSAGES,
    "tool",
  );
  for (const message of messages ?? []) {
    for (const block of Array.isArray(message.content) ? message.content : []) {
      if (direction === "input" && isRecord(block.toolUse))
        return block.toolUse.input;
      if (direction === "output" && isRecord(block.toolResult))
        return block.toolResult.content;
    }
  }
  return undefined;
}

function setCommonAttrs(
  attrs: SpanAttributesRecord,
  options: { logType: RespanLogType; entityName: string; entityPath: string },
): void {
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = options.logType;
  attrs[SpanAttributes.TRACELOOP_ENTITY_NAME] = options.entityName;
  attrs[SpanAttributes.TRACELOOP_ENTITY_PATH] = options.entityPath;
  delete attrs[SpanAttributes.TRACELOOP_SPAN_KIND];
}

function setInputOutputAttrs(
  attrs: SpanAttributesRecord,
  options: {
    inputMessages?: Array<Record<string, any>>;
    outputMessages?: Array<Record<string, any>>;
  },
): void {
  if (options.inputMessages?.length) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
      options.inputMessages,
    );
  }
  if (options.outputMessages?.length) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(
      options.outputMessages,
    );
  }
}

function existingWorkflowName(attrs: SpanAttributesRecord): string | undefined {
  const workflowName = attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME];
  return typeof workflowName === "string" && workflowName
    ? workflowName
    : undefined;
}

function isStrandsTaskSpanName(spanName: string): boolean {
  return (
    spanName === STRANDS_OPERATION_EXECUTE_AGENT_LOOP_CYCLE ||
    spanName.startsWith(`${STRANDS_OPERATION_EXECUTE_AGENT_LOOP_CYCLE} `) ||
    spanName === STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE ||
    spanName.startsWith(`${STRANDS_OPERATION_EXECUTE_EVENT_LOOP_CYCLE} `) ||
    spanName === STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT ||
    spanName.startsWith(`${STRANDS_OPERATION_EXECUTE_STRUCTURED_OUTPUT} `)
  );
}

function extractWorkflowName(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): string {
  const operationName = attrs[ATTR_GEN_AI_OPERATION_NAME];
  const orchestratorId = attrs[ATTR_GEN_AI_AGENT_ID];
  if (typeof orchestratorId === "string" && orchestratorId) {
    const type =
      operationName === STRANDS_OPERATION_INVOKE_SWARM ? "swarm" : "graph";
    return `${type}:${orchestratorId}`;
  }
  if (typeof operationName === "string") {
    return spanSuffixName(span.name, operationName, operationName);
  }
  return span.name || STRANDS_SYSTEM_NAME;
}

function extractAgentName(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): string {
  const agentName = attrs[ATTR_GEN_AI_AGENT_NAME];
  if (typeof agentName === "string" && agentName) {
    return agentName;
  }
  return spanSuffixName(
    span.name,
    STRANDS_OPERATION_INVOKE_AGENT,
    STRANDS_SYSTEM_NAME,
  );
}

function extractToolName(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): string {
  const toolName = attrs[ATTR_GEN_AI_TOOL_NAME];
  if (typeof toolName === "string" && toolName) {
    return toolName;
  }
  return spanSuffixName(
    span.name,
    STRANDS_OPERATION_EXECUTE_TOOL,
    STRANDS_OPERATION_EXECUTE_TOOL,
  );
}

function spanSuffixName(
  spanName: string,
  prefix: string,
  fallback: string,
): string {
  if (spanName.startsWith(`${prefix} `)) {
    const suffix = spanName.slice(prefix.length + 1).trim();
    if (suffix) {
      return suffix;
    }
  }
  return fallback;
}

function extractInputMessages(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): Array<Record<string, any>> | undefined {
  const attrMessages = normalizeMessages(
    attrs[ATTR_GEN_AI_INPUT_MESSAGES],
    "user",
  );
  const operationMessages = operationDetailMessages(
    span,
    ATTR_GEN_AI_INPUT_MESSAGES,
    "user",
  );
  const legacyMessages = legacyInputMessages(span);
  const messages = attrMessages ?? operationMessages ?? legacyMessages ?? [];
  const instructions =
    attrs[ATTR_GEN_AI_SYSTEM_INSTRUCTIONS] ??
    attrs[STRANDS_SYSTEM_PROMPT_ATTR] ??
    getEvents(span).find(
      ([name, eventAttrs]) =>
        name === EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS &&
        ATTR_GEN_AI_SYSTEM_INSTRUCTIONS in eventAttrs,
    )?.[1][ATTR_GEN_AI_SYSTEM_INSTRUCTIONS];
  if (instructions !== undefined) {
    const parsed = safeJsonLoads(instructions);
    const content =
      Array.isArray(parsed) &&
      parsed.some((part) => isRecord(part) && "type" in part)
        ? partsToContent(parsed)
        : parsed;
    messages.unshift({ role: "system", content: contentForMessage(content) });
  }
  return messages.length ? messages : undefined;
}

function extractOutputMessages(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
  defaultRole = "assistant",
): Array<Record<string, any>> | undefined {
  const attrMessages = normalizeMessages(
    attrs[ATTR_GEN_AI_OUTPUT_MESSAGES],
    defaultRole,
  );
  if (attrMessages?.length) {
    return attrMessages;
  }

  const operationMessages = operationDetailMessages(
    span,
    ATTR_GEN_AI_OUTPUT_MESSAGES,
    defaultRole,
  );
  if (operationMessages?.length) {
    return operationMessages;
  }

  const legacyMessages = legacyOutputMessages(span, defaultRole);
  return legacyMessages?.length ? legacyMessages : undefined;
}

function operationDetailMessages(
  span: ReadableSpan,
  attrName: string,
  defaultRole: string,
): Array<Record<string, any>> | undefined {
  const messages: Array<Record<string, any>> = [];
  for (const [eventName, eventAttrs] of getEvents(span)) {
    if (eventName !== EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS) {
      continue;
    }
    const normalized = normalizeMessages(eventAttrs[attrName], defaultRole);
    if (normalized?.length) {
      messages.push(...normalized);
    }
  }
  return messages.length ? messages : undefined;
}

function legacyInputMessages(
  span: ReadableSpan,
): Array<Record<string, any>> | undefined {
  const messages: Array<Record<string, any>> = [];
  for (const [eventName, eventAttrs] of getEvents(span)) {
    let normalized: Record<string, any> | undefined;
    if (eventName === EVENT_GEN_AI_SYSTEM_MESSAGE) {
      normalized = normalizeMessage(
        { role: "system", content: safeJsonLoads(eventAttrs.content) },
        "system",
      );
    } else if (
      eventName.startsWith(STRANDS_EVENT_MESSAGE_PREFIX) &&
      eventName.endsWith(STRANDS_EVENT_MESSAGE_SUFFIX)
    ) {
      const role = eventName.slice(
        STRANDS_EVENT_MESSAGE_PREFIX.length,
        -STRANDS_EVENT_MESSAGE_SUFFIX.length,
      );
      normalized = normalizeMessage(
        {
          role: eventAttrs.role ?? role,
          content: safeJsonLoads(eventAttrs.content),
        },
        role,
      );
    }
    if (normalized) {
      messages.push(normalized);
    }
  }
  return messages.length ? messages : undefined;
}

function legacyOutputMessages(
  span: ReadableSpan,
  defaultRole: string,
): Array<Record<string, any>> | undefined {
  const messages: Array<Record<string, any>> = [];
  for (const [eventName, eventAttrs] of getEvents(span)) {
    if (eventName !== EVENT_GEN_AI_CHOICE) {
      continue;
    }
    const normalized = normalizeMessage(
      {
        role: eventAttrs.role ?? defaultRole,
        content:
          span.attributes[ATTR_GEN_AI_OPERATION_NAME] ===
          STRANDS_OPERATION_INVOKE_AGENT
            ? eventAttrs.message
            : safeJsonLoads(eventAttrs.message),
      },
      defaultRole,
    );
    if (normalized) {
      messages.push(normalized);
    }
  }
  return messages.length ? messages : undefined;
}

function normalizeMessages(
  value: unknown,
  defaultRole: string,
): Array<Record<string, any>> | undefined {
  const parsed = safeJsonLoads(value);
  if (Array.isArray(parsed)) {
    const messages = parsed
      .map((item) => normalizeMessage(item, defaultRole))
      .filter((item): item is Record<string, any> => item !== undefined);
    return messages.length ? messages : undefined;
  }
  const message = normalizeMessage(parsed, defaultRole);
  return message ? [message] : undefined;
}

function normalizeMessage(
  rawMessage: unknown,
  defaultRole: string,
): Record<string, any> | undefined {
  const parsedMessage = safeJsonLoads(rawMessage);
  if (!isRecord(parsedMessage)) {
    const content = contentForMessage(parsedMessage);
    if (isEmptyValue(content)) {
      return undefined;
    }
    return { role: defaultRole, content };
  }

  const role = parsedMessage.role ?? defaultRole;
  let content = parsedMessage.content;
  if (content === undefined && parsedMessage.parts !== undefined) {
    content = partsToContent(parsedMessage.parts);
  }
  const normalizedContent = contentForMessage(content);
  const toolResults = Array.isArray(normalizedContent)
    ? normalizedContent.filter(
        (block) => isRecord(block) && isRecord(block.toolResult),
      )
    : [];
  return {
    ...parsedMessage,
    role:
      toolResults.length &&
      Array.isArray(normalizedContent) &&
      toolResults.length === normalizedContent.length
        ? "tool"
        : typeof role === "string"
          ? role
          : defaultRole,
    content: normalizedContent,
    ...(toolResults.length === 1
      ? { tool_call_id: toolResults[0].toolResult.toolUseId }
      : {}),
    parts: undefined,
  };
}

function partsToContent(parts: unknown): unknown {
  const parsedParts = safeJsonLoads(parts);
  if (!Array.isArray(parsedParts)) {
    return toSerializableValue(parsedParts);
  }

  const contentBlocks = parsedParts.map((part) => {
    if (!isRecord(part)) {
      return toSerializableValue(part);
    }
    switch (part.type) {
      case "text":
        return { text: part.content ?? "" };
      case "tool_call":
        return {
          toolUse: {
            name: part.name ?? "",
            toolUseId: part.id ?? "",
            input: part.arguments ?? {},
          },
        };
      case "tool_call_response":
        return {
          toolResult: {
            toolUseId: part.id ?? "",
            content: part.response ?? "",
          },
        };
      default:
        return toSerializableValue(part);
    }
  });

  const text = extractTextFromContent(contentBlocks);
  return text ?? contentBlocks;
}

function contentForMessage(content: unknown): unknown {
  const parsedContent = content;
  const text = extractTextFromContent(parsedContent);
  if (text !== undefined) {
    return text;
  }
  return toSerializableValue(parsedContent);
}

function extractTextFromContent(content: unknown): string | undefined {
  const parsed = content;
  if (typeof parsed === "string") {
    return parsed;
  }
  if (isRecord(parsed)) {
    if (typeof parsed.text === "string") {
      return parsed.text;
    }
    if (parsed.type === "textBlock" && typeof parsed.text === "string") {
      return parsed.text;
    }
    return undefined;
  }
  if (!Array.isArray(parsed)) {
    return undefined;
  }

  const textParts: string[] = [];
  for (const item of parsed) {
    if (!isRecord(item) || typeof item.text !== "string") {
      return undefined;
    }
    textParts.push(item.text);
  }
  return textParts.join("\n");
}

function setIndexedMessages(
  attrs: SpanAttributesRecord,
  prefix: string,
  messages: Array<Record<string, any>>,
): void {
  messages.forEach((message, index) => {
    const indexedPrefix = `${prefix}${index}`;
    attrs[`${indexedPrefix}.role`] = String(message.role ?? "");
    const content = message.content;
    attrs[`${indexedPrefix}.content`] = messageContentAttrValue(content);
    for (const field of ["tool_call_id", "name"]) {
      if (typeof message[field] === "string")
        attrs[`${indexedPrefix}.${field}`] = message[field];
    }
    const toolCalls = toolCallsFromContent(content);
    if (toolCalls.length) {
      attrs[`${indexedPrefix}.tool_calls`] = safeJson(toolCalls);
    }
  });
}

function messageContentAttrValue(content: unknown): string {
  const text = extractTextFromContent(content);
  if (text !== undefined) {
    return text;
  }
  return jsonString(content) ?? "";
}

function toolCallsFromContent(content: unknown): Array<Record<string, any>> {
  const parsedContent = safeJsonLoads(content);
  if (!Array.isArray(parsedContent)) {
    return [];
  }

  const toolCalls: Array<Record<string, any>> = [];
  for (const item of parsedContent) {
    if (!isRecord(item)) {
      continue;
    }

    if (isRecord(item.toolUse)) {
      toolCalls.push(
        normalizeToolCall(
          item.toolUse.name,
          item.toolUse.toolUseId,
          item.toolUse.input,
        ),
      );
      continue;
    }

    if (item.type === "toolUse" || item.type === "tool_call") {
      toolCalls.push(
        normalizeToolCall(
          item.name,
          item.toolUseId ?? item.id,
          item.input ?? item.arguments,
        ),
      );
    }
  }

  return toolCalls.filter((toolCall) => toolCall.function.name);
}

function normalizeToolCall(
  name: unknown,
  toolCallId: unknown,
  args: unknown,
): Record<string, any> {
  return {
    ...(typeof toolCallId === "string" ? { id: toolCallId } : {}),
    type: "function",
    function: {
      name: typeof name === "string" ? name : "",
      arguments: jsonString(args) ?? "",
    },
  };
}

function extractToolDefinitions(
  attrs: SpanAttributesRecord,
): Array<Record<string, any>> | undefined {
  const rawToolDefinitions = safeJsonLoads(attrs[ATTR_GEN_AI_TOOL_DEFINITIONS]);
  const rawAgentTools = safeJsonLoads(attrs[STRANDS_AGENT_TOOLS_ATTR]);
  const rawTools = rawToolDefinitions ?? rawAgentTools;
  const iterableTools = normalizeToolDefinitionInput(rawTools);
  if (!iterableTools.length) {
    return undefined;
  }

  const normalized = iterableTools
    .map((toolDefinition) => normalizeToolDefinition(toolDefinition))
    .filter((tool): tool is Record<string, any> => tool !== undefined);
  return normalized.length ? normalized : undefined;
}

function normalizeToolDefinitionInput(value: unknown): unknown[] {
  const parsed = safeJsonLoads(value);
  if (Array.isArray(parsed)) {
    return parsed;
  }
  if (isRecord(parsed)) {
    return Object.entries(parsed).map(([name, definition]) => {
      if (isRecord(definition)) {
        return { name, ...definition };
      }
      return name;
    });
  }
  return parsed === undefined || parsed === null ? [] : [parsed];
}

function normalizeToolDefinition(
  toolDefinition: unknown,
): Record<string, any> | undefined {
  if (typeof toolDefinition === "string" && toolDefinition) {
    return { type: "function", function: { name: toolDefinition } };
  }
  if (!isRecord(toolDefinition)) {
    return undefined;
  }

  const toolName = toolDefinition.name;
  if (typeof toolName !== "string" || !toolName) {
    return undefined;
  }

  const functionPayload: Record<string, any> = { name: toolName };
  if (toolDefinition.description) {
    functionPayload.description = toSerializableValue(
      toolDefinition.description,
    );
  }
  const inputSchema = toolDefinition.inputSchema ?? toolDefinition.parameters;
  if (inputSchema !== undefined) {
    functionPayload.parameters = toSerializableValue(inputSchema);
  }
  return { type: "function", function: functionPayload };
}

function setUsageAttrs(attrs: SpanAttributesRecord): void {
  const promptTokens = coerceInteger(
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] ??
      attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS],
  );
  const completionTokens = coerceInteger(
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] ??
      attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS],
  );
  const totalTokens = coerceInteger(
    attrs[STRANDS_USAGE_TOTAL_TOKENS_ATTR] ??
      attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS],
  );
  const cacheCreationTokens = coerceInteger(
    attrs[STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR] ??
      attrs[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS],
  );
  const cacheReadTokens = coerceInteger(
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] ??
      attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS],
  );

  if (promptTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = promptTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = promptTokens;
  }
  if (completionTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = completionTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = completionTokens;
  }
  if (totalTokens !== undefined) {
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
  if (cacheCreationTokens !== undefined)
    attrs[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = cacheCreationTokens;
  if (cacheReadTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cacheReadTokens;
  }
}

function stripRawAttrs(
  attrs: SpanAttributesRecord,
  logType: RespanLogType,
): SpanAttributesRecord {
  const stripped: SpanAttributesRecord = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (STRANDS_RAW_ATTRS_TO_STRIP.has(key)) {
      continue;
    }
    if (OFF_CONTRACT_ALIAS_ATTRS.has(key)) {
      continue;
    }
    if (
      STRANDS_RAW_ATTR_PREFIXES_TO_STRIP.some((prefix) =>
        key.startsWith(prefix),
      )
    ) {
      continue;
    }
    if (
      logType !== RespanLogType.CHAT &&
      (STRANDS_NON_LLM_ATTRS_TO_STRIP.has(key) ||
        key.startsWith("gen_ai.usage.") ||
        key.startsWith("llm.usage."))
    ) {
      continue;
    }
    stripped[key] = value;
  }
  return stripped;
}

function replaceSpanAttributes(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const target = (span as any).attributes as SpanAttributesRecord;
  for (const key of Object.keys(target)) {
    delete target[key];
  }
  Object.assign(target, attrs);
  if ((span as any)._attributes) {
    (span as any)._attributes = target;
  }
}

function extractToolEventPayload(
  span: ReadableSpan,
  eventName: string,
  attrName: string,
): unknown {
  for (const [currentEventName, eventAttrs] of getEvents(span)) {
    if (currentEventName === eventName && attrName in eventAttrs) {
      return eventAttrs[attrName];
    }
  }
  return undefined;
}

function getEvents(span: ReadableSpan): Array<[string, Record<string, any>]> {
  const rawEvents = ((span as any).events ?? []) as SpanEventRecord[];
  const events: Array<[string, Record<string, any>]> = [];
  for (const event of rawEvents) {
    if (typeof event?.name === "string") {
      events.push([event.name, event.attributes ?? {}]);
    }
  }
  return events;
}

function safeJsonLoads(value: unknown): unknown {
  if (typeof value !== "string") {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function jsonString(value: unknown): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value === "string") {
    return value;
  }
  return safeJson(value);
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(toSerializableValue(value), (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return "null";
  }
}

function toSerializableValue(value: unknown): unknown {
  return copyData(value);
}

function coerceInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : undefined;
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isEmptyValue(value: unknown): boolean {
  return (
    value === undefined ||
    value === null ||
    value === "" ||
    (Array.isArray(value) && value.length === 0) ||
    (isRecord(value) && Object.keys(value).length === 0)
  );
}
