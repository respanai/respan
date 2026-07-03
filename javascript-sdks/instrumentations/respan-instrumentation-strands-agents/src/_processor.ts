import type { Context } from "@opentelemetry/api";
import type {
  ReadableSpan,
  Span,
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
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_CALL_ID,
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
  GEN_AI_SYSTEM_VALUE_ANTHROPIC,
  GEN_AI_SYSTEM_VALUE_AWS_BEDROCK,
  GEN_AI_SYSTEM_VALUE_GCP_GEMINI,
  GEN_AI_SYSTEM_VALUE_OPENAI,
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
  STRANDS_SYSTEM_NAME,
  STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
  STRANDS_TOOL_JSON_SCHEMA_ATTR,
  STRANDS_TOOL_STATUS_ATTR,
  STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR,
  STRANDS_USAGE_TOTAL_TOKENS_ATTR,
} from "./_constants.js";

type SpanAttributesRecord = Record<string, any>;
type SpanEventRecord = {
  name?: string;
  attributes?: Record<string, any>;
};

const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const GEN_AI_PROMPT_PREFIX = `${ATTR_GEN_AI_PROMPT}.`;
const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.`;
const LLM_USAGE_CACHE_READ_INPUT_TOKENS_ATTR =
  "llm.usage.cache_read_input_tokens";

const STRANDS_RAW_ATTRS_TO_STRIP = new Set([
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_AGENT_ID,
  STRANDS_AGENT_TOOLS_ATTR,
  ATTR_GEN_AI_TOOL_CALL_ID,
  STRANDS_TOOL_STATUS_ATTR,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  STRANDS_TOOL_JSON_SCHEMA_ATTR,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  STRANDS_EVENT_START_TIME_ATTR,
  STRANDS_EVENT_END_TIME_ATTR,
]);

const STRANDS_NON_LLM_ATTRS_TO_STRIP = new Set([
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  STRANDS_USAGE_TOTAL_TOKENS_ATTR,
  STRANDS_USAGE_CACHE_WRITE_INPUT_TOKENS_ATTR,
  SpanAttributes.LLM_REQUEST_TYPE,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
  LLM_USAGE_CACHE_READ_INPUT_TOKENS_ATTR,
]);

const OFF_CONTRACT_ALIAS_ATTRS = new Set([
  ...STRANDS_TOP_LEVEL_ALIAS_ATTRS_TO_STRIP,
  RespanSpanAttributes.RESPAN_SPAN_TOOLS,
  RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
  RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
]);

export class StrandsAgentsSpanProcessor implements SpanProcessor {
  onStart(_span: Span, _parentContext: Context): void {
    // Translation happens on ended spans so message/usage events are complete.
  }

  onEnd(span: ReadableSpan): void {
    enrichStrandsAgentsSpan(span);
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}

export function enrichStrandsAgentsSpan(span: ReadableSpan): void {
  const originalAttrs = (span as any).attributes as SpanAttributesRecord | undefined;
  if (!originalAttrs) {
    return;
  }

  const attrs = { ...originalAttrs };
  if (!isStrandsAgentsSpan(span, attrs)) {
    return;
  }

  const logType = extractLogType(span, attrs);
  if (!logType) {
    return;
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
      return;
  }

  replaceSpanAttributes(span, stripRawAttrs(attrs, logType));
}

function isStrandsAgentsSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): boolean {
  const operationName = attrs[ATTR_GEN_AI_OPERATION_NAME];
  return (
    attrs[ATTR_GEN_AI_SYSTEM] === STRANDS_SYSTEM_NAME ||
    attrs[ATTR_GEN_AI_PROVIDER_NAME] === STRANDS_SYSTEM_NAME ||
    KNOWN_STRANDS_OPERATIONS.has(String(operationName)) ||
    typeof attrs[ATTR_GEN_AI_AGENT_NAME] === "string" ||
    typeof attrs[ATTR_GEN_AI_TOOL_NAME] === "string" ||
    span.name.startsWith(`${STRANDS_OPERATION_INVOKE_AGENT} `) ||
    span.name.startsWith(`${STRANDS_OPERATION_EXECUTE_TOOL} `) ||
    span.name.startsWith(`${STRANDS_OPERATION_INVOKE_GRAPH} `) ||
    span.name.startsWith(`${STRANDS_OPERATION_INVOKE_SWARM} `) ||
    isStrandsTaskSpanName(span.name)
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

  const toolDefinitions = extractToolDefinitions(attrs);
  if (toolDefinitions?.length) {
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(toolDefinitions);
  }
}

function enrichTaskSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
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

function enrichChatSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  setCommonAttrs(attrs, {
    logType: RespanLogType.CHAT,
    entityName: STRANDS_OPERATION_CHAT,
    entityPath: STRANDS_OPERATION_CHAT,
  });
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[ATTR_GEN_AI_SYSTEM] = inferGenAISystem(attrs[ATTR_GEN_AI_REQUEST_MODEL]);

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

function enrichToolSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const toolName = extractToolName(span, attrs);
  setCommonAttrs(attrs, {
    logType: RespanLogType.TOOL,
    entityName: toolName,
    entityPath: toolName,
  });

  const toolArguments = extractToolEventPayload(
    span,
    EVENT_GEN_AI_TOOL_MESSAGE,
    "content",
  );
  const toolResult = extractToolEventPayload(
    span,
    EVENT_GEN_AI_CHOICE,
    "message",
  );

  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson({
    name: toolName,
    id: attrs[ATTR_GEN_AI_TOOL_CALL_ID] ?? "",
    arguments: toSerializableValue(safeJsonLoads(toolArguments)),
  });

  if (toolResult !== undefined) {
    const output = contentForMessage(toolResult);
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = jsonString(output);
  }
}

function setCommonAttrs(
  attrs: SpanAttributesRecord,
  options: { logType: RespanLogType; entityName: string; entityPath: string },
): void {
  attrs[RespanSpanAttributes.RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[RespanSpanAttributes.RESPAN_LOG_TYPE] = options.logType;
  attrs[ATTR_GEN_AI_SYSTEM] = STRANDS_SYSTEM_NAME;
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
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(options.inputMessages);
  }
  if (options.outputMessages?.length) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(options.outputMessages);
  }
}

function existingWorkflowName(attrs: SpanAttributesRecord): string | undefined {
  const workflowName = attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME];
  return typeof workflowName === "string" && workflowName ? workflowName : undefined;
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
  const attrMessages = normalizeMessages(attrs[ATTR_GEN_AI_INPUT_MESSAGES], "user");
  if (attrMessages?.length) {
    return attrMessages;
  }

  const operationMessages = operationDetailMessages(
    span,
    ATTR_GEN_AI_INPUT_MESSAGES,
    "user",
  );
  if (operationMessages?.length) {
    return operationMessages;
  }

  const legacyMessages = legacyInputMessages(span);
  const systemInstructions = attrs[ATTR_GEN_AI_SYSTEM_INSTRUCTIONS];
  if (systemInstructions !== undefined) {
    const systemMessage = normalizeMessage(
      { role: "system", content: systemInstructions },
      "system",
    );
    if (systemMessage) {
      return [systemMessage, ...(legacyMessages ?? [])];
    }
  }
  return legacyMessages?.length ? legacyMessages : undefined;
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

function legacyInputMessages(span: ReadableSpan): Array<Record<string, any>> | undefined {
  const messages: Array<Record<string, any>> = [];
  for (const [eventName, eventAttrs] of getEvents(span)) {
    let normalized: Record<string, any> | undefined;
    if (eventName === EVENT_GEN_AI_SYSTEM_MESSAGE) {
      normalized = normalizeMessage(
        { role: "system", content: eventAttrs.content },
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
        { role: eventAttrs.role ?? role, content: eventAttrs.content },
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
      { role: eventAttrs.role ?? defaultRole, content: eventAttrs.message },
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
  return {
    role: String(role),
    content: contentForMessage(content),
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
  const parsedContent = safeJsonLoads(content);
  const text = extractTextFromContent(parsedContent);
  if (text !== undefined) {
    return text;
  }
  return toSerializableValue(parsedContent);
}

function extractTextFromContent(content: unknown): string | undefined {
  const parsed = safeJsonLoads(content);
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
    id: String(toolCallId ?? ""),
    type: "function",
    function: {
      name: String(name ?? ""),
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
    functionPayload.description = toSerializableValue(toolDefinition.description);
  }
  const inputSchema = toolDefinition.inputSchema ?? toolDefinition.parameters;
  if (inputSchema !== undefined) {
    functionPayload.parameters = toSerializableValue(inputSchema);
  }
  return { type: "function", function: functionPayload };
}

function setUsageAttrs(attrs: SpanAttributesRecord): void {
  const promptTokens = coerceInteger(
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] ?? attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS],
  );
  const completionTokens = coerceInteger(
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] ??
      attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS],
  );
  const totalTokens =
    coerceInteger(STRANDS_USAGE_TOTAL_TOKENS_ATTR in attrs
      ? attrs[STRANDS_USAGE_TOTAL_TOKENS_ATTR]
      : attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]) ??
    (promptTokens !== undefined || completionTokens !== undefined
      ? (promptTokens ?? 0) + (completionTokens ?? 0)
      : undefined);
  const cacheReadTokens = coerceInteger(
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] ??
      attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS_ATTR],
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
  if (cacheReadTokens !== undefined) {
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS_ATTR] = cacheReadTokens;
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
    if (STRANDS_RAW_ATTR_PREFIXES_TO_STRIP.some((prefix) => key.startsWith(prefix))) {
      continue;
    }
    if (logType !== RespanLogType.CHAT && STRANDS_NON_LLM_ATTRS_TO_STRIP.has(key)) {
      continue;
    }
    stripped[key] = value;
  }
  return stripped;
}

function replaceSpanAttributes(span: ReadableSpan, attrs: SpanAttributesRecord): void {
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

function inferGenAISystem(model: unknown): string {
  const modelText = String(model ?? "").toLowerCase();
  if (
    modelText.startsWith("gpt-") ||
    modelText.startsWith("o") ||
    modelText.includes("openai")
  ) {
    return GEN_AI_SYSTEM_VALUE_OPENAI;
  }
  if (modelText.includes("claude") || modelText.includes("anthropic")) {
    return GEN_AI_SYSTEM_VALUE_ANTHROPIC;
  }
  if (modelText.includes("gemini") || modelText.includes("google")) {
    return GEN_AI_SYSTEM_VALUE_GCP_GEMINI;
  }
  if (
    modelText.includes("bedrock") ||
    modelText.startsWith("global.") ||
    modelText.startsWith("us.") ||
    modelText.startsWith("eu.")
  ) {
    return GEN_AI_SYSTEM_VALUE_AWS_BEDROCK;
  }
  return STRANDS_SYSTEM_NAME;
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
  if (value === undefined || value === null) {
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
    return String(value);
  }
}

function toSerializableValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value.map((item) => toSerializableValue(item));
  }
  if (isRecord(value)) {
    if (typeof value.toJSON === "function") {
      try {
        return toSerializableValue(value.toJSON());
      } catch {
        // Fall through to structural copy.
      }
    }
    const normalized: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      normalized[key] = toSerializableValue(item);
    }
    return normalized;
  }
  return String(value);
}

function coerceInteger(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
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
