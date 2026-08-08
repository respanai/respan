import type { Context } from "@opentelemetry/api";
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { RespanLogType } from "@respan/respan-sdk";
import {
  GEN_AI_COMPLETION_PREFIX,
  GEN_AI_PROMPT_PREFIX,
  GEN_AI_PROVIDER_NAME,
  GEN_AI_REQUEST_MODEL,
  GEN_AI_SYSTEM,
  GEN_AI_USAGE_COMPLETION_TOKENS,
  GEN_AI_USAGE_INPUT_TOKENS,
  GEN_AI_USAGE_OUTPUT_TOKENS,
  GEN_AI_USAGE_PROMPT_TOKENS,
  LLM_REQUEST_FUNCTIONS,
  LLM_REQUEST_TYPE,
  LLM_USAGE_CACHE_READ_INPUT_TOKENS,
  LLM_USAGE_TOTAL_TOKENS,
  OFF_CONTRACT_ALIAS_ATTRS,
  OI_AGENT_NAME,
  OI_INPUT_VALUE,
  OI_LLM_INVOCATION_PARAMETERS,
  OI_LLM_MODEL_NAME,
  OI_LLM_PROVIDER,
  OI_LLM_SYSTEM,
  OI_LLM_TOKEN_COUNT_CACHE_READ,
  OI_LLM_TOKEN_COUNT_COMPLETION,
  OI_LLM_TOKEN_COUNT_PROMPT,
  OI_LLM_TOKEN_COUNT_TOTAL,
  OI_LLM_TOOLS,
  OI_OUTPUT_VALUE,
  OI_RAW_EXACT_ATTRS,
  OI_RAW_PREFIXES,
  OI_SPAN_KIND,
  OTEL_NOISE_EXACT_ATTRS,
  OTEL_NOISE_PREFIXES,
  PYDANTIC_AI_AGENT_NAME,
  PYDANTIC_AI_FINAL_RESULT,
  PYDANTIC_AI_INPUT_MESSAGES,
  PYDANTIC_AI_LEGACY_AGENT_NAME,
  PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS,
  PYDANTIC_AI_LEGACY_TOOL_RESULT,
  PYDANTIC_AI_MODEL_NAME,
  PYDANTIC_AI_OPERATION_NAME,
  PYDANTIC_AI_OUTPUT_MESSAGES,
  PYDANTIC_AI_RAW_EXACT_ATTRS,
  PYDANTIC_AI_REQUEST_PARAMETERS,
  PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME,
  PYDANTIC_AI_SCOPE_MARKERS,
  PYDANTIC_AI_TOOL_CALL_ARGUMENTS,
  PYDANTIC_AI_TOOL_CALL_RESULT,
  PYDANTIC_AI_TOOL_DEFINITIONS,
  PYDANTIC_AI_TOOL_NAME,
  PYDANTIC_AI_TOOLS,
  PYDANTIC_AI_USAGE_TOTAL_TOKENS,
  RESPAN_LOG_METHOD,
  RESPAN_LOG_TYPE,
  TRACELOOP_ENTITY_INPUT,
  TRACELOOP_ENTITY_NAME,
  TRACELOOP_ENTITY_OUTPUT,
  TRACELOOP_ENTITY_PATH,
  TRACELOOP_SPAN_KIND,
  TRACELOOP_WORKFLOW_NAME,
} from "./_constants.js";

type SpanAttributesRecord = Record<string, any>;

export interface PydanticAISpanProcessorOptions {
  includeNativeSpans?: boolean;
  includeOpenInferenceSpans?: boolean;
}

const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";

const PydanticAIOperationToLogType: Record<string, RespanLogType> = {
  chat: RespanLogType.CHAT,
  response: RespanLogType.CHAT,
  embedding: RespanLogType.EMBEDDING,
  speech: RespanLogType.TEXT,
  transcription: RespanLogType.TEXT,
};

const OIKindToLogType: Record<string, RespanLogType> = {
  LLM: RespanLogType.CHAT,
  EMBEDDING: RespanLogType.EMBEDDING,
  TOOL: RespanLogType.TOOL,
  AGENT: RespanLogType.AGENT,
  CHAIN: RespanLogType.WORKFLOW,
  RETRIEVER: RespanLogType.TASK,
  RERANKER: RespanLogType.TASK,
  GUARDRAIL: RespanLogType.GUARDRAIL,
  EVALUATOR: RespanLogType.TASK,
  PROMPT: RespanLogType.TASK,
  UNKNOWN: RespanLogType.TASK,
};

const LLM_LOG_TYPES = new Set<RespanLogType>([
  RespanLogType.CHAT,
  RespanLogType.TEXT,
  RespanLogType.EMBEDDING,
]);

const LLM_CANONICAL_EXACT_ATTRS = new Set([
  GEN_AI_SYSTEM,
  GEN_AI_PROVIDER_NAME,
  GEN_AI_REQUEST_MODEL,
  GEN_AI_USAGE_INPUT_TOKENS,
  GEN_AI_USAGE_OUTPUT_TOKENS,
  GEN_AI_USAGE_PROMPT_TOKENS,
  GEN_AI_USAGE_COMPLETION_TOKENS,
  PYDANTIC_AI_USAGE_TOTAL_TOKENS,
  LLM_REQUEST_TYPE,
  LLM_REQUEST_FUNCTIONS,
  LLM_USAGE_TOTAL_TOKENS,
  LLM_USAGE_CACHE_READ_INPUT_TOKENS,
]);

const LLM_CANONICAL_PREFIXES = [
  `${GEN_AI_PROMPT_PREFIX}.`,
  `${GEN_AI_COMPLETION_PREFIX}.`,
];

export class PydanticAISpanProcessor implements SpanProcessor {
  private readonly _options: Required<PydanticAISpanProcessorOptions>;

  constructor(options: PydanticAISpanProcessorOptions = {}) {
    this._options = {
      includeNativeSpans: options.includeNativeSpans ?? true,
      includeOpenInferenceSpans: options.includeOpenInferenceSpans ?? true,
    };
  }

  onStart(_span: Span, _parentContext: Context): void {
    // Translation happens on ended spans so output and usage are complete.
  }

  onEnd(span: ReadableSpan): void {
    enrichPydanticAISpan(span, this._options);
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}

export function enrichPydanticAISpan(
  span: ReadableSpan,
  options: PydanticAISpanProcessorOptions = {},
): void {
  const attrs = getMutableAttributes(span);
  if (!attrs) {
    return;
  }

  const resolvedOptions = {
    includeNativeSpans: options.includeNativeSpans ?? true,
    includeOpenInferenceSpans: options.includeOpenInferenceSpans ?? true,
  };

  const scopeName = getInstrumentationScopeName(span);

  if (
    resolvedOptions.includeOpenInferenceSpans &&
    isPydanticAIOpenInferenceSpan(span, attrs, scopeName)
  ) {
    const logType = enrichOpenInferenceSpan(span, attrs);
    if (logType) {
      replaceSpanAttributes(span, stripRawAttrs(attrs, logType, "openinference"));
    }
    return;
  }

  if (
    resolvedOptions.includeNativeSpans &&
    isPydanticAINativeSpan(span, attrs, scopeName)
  ) {
    const logType = enrichNativePydanticAISpan(span, attrs);
    if (logType) {
      replaceSpanAttributes(span, stripRawAttrs(attrs, logType, "native"));
    }
    return;
  }
}

export function isPydanticAISpan(span: ReadableSpan): boolean {
  const attrs = getMutableAttributes(span);
  const scopeName = getInstrumentationScopeName(span);
  return Boolean(
    attrs &&
      (isPydanticAINativeSpan(span, attrs, scopeName) ||
        isPydanticAIOpenInferenceSpan(span, attrs, scopeName)),
  );
}

export function isPydanticAINativeSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
  scopeName = getInstrumentationScopeName(span),
): boolean {
  if (scopeLooksLikePydanticAI(scopeName)) {
    return true;
  }

  return Boolean(
    PYDANTIC_AI_REQUEST_PARAMETERS in attrs ||
      PYDANTIC_AI_TOOL_DEFINITIONS in attrs ||
      PYDANTIC_AI_INPUT_MESSAGES in attrs ||
      PYDANTIC_AI_OUTPUT_MESSAGES in attrs ||
      PYDANTIC_AI_AGENT_NAME in attrs ||
      PYDANTIC_AI_LEGACY_AGENT_NAME in attrs ||
      PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS in attrs ||
      PYDANTIC_AI_LEGACY_TOOL_RESULT in attrs ||
      PYDANTIC_AI_FINAL_RESULT in attrs ||
      (span.name === PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME &&
        PYDANTIC_AI_TOOLS in attrs),
  );
}

export function isPydanticAIOpenInferenceSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
  scopeName = getInstrumentationScopeName(span),
): boolean {
  return Boolean(
    attrs[OI_SPAN_KIND] !== undefined &&
      (scopeLooksLikePydanticAI(scopeName) ||
        isPydanticAINativeSpan(span, attrs, scopeName)),
  );
}

function enrichNativePydanticAISpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): RespanLogType | undefined {
  const logType = extractNativeLogType(span, attrs);
  if (!logType) {
    return undefined;
  }

  switch (logType) {
    case RespanLogType.CHAT:
      enrichNativeChatSpan(span, attrs);
      break;
    case RespanLogType.EMBEDDING:
    case RespanLogType.TEXT:
      enrichNativeModelSpan(span, attrs, logType);
      break;
    case RespanLogType.TOOL:
      enrichNativeToolSpan(span, attrs);
      break;
    case RespanLogType.AGENT:
      enrichNativeAgentSpan(span, attrs);
      break;
    case RespanLogType.TASK:
      enrichNativeTaskSpan(span, attrs);
      break;
    default:
      return undefined;
  }

  delete attrs[TRACELOOP_SPAN_KIND];
  return logType;
}

function enrichOpenInferenceSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): RespanLogType | undefined {
  const oiKind = String(attrs[OI_SPAN_KIND] ?? "").toUpperCase();
  const logType = OIKindToLogType[oiKind] ?? RespanLogType.TASK;
  const entityName = stringValue(attrs[OI_AGENT_NAME]) ?? span.name;
  const entityPath = logType === RespanLogType.WORKFLOW ? "" : entityName;

  setCommonAttrs(attrs, logType, entityName, entityPath);

  if (attrs[OI_INPUT_VALUE] !== undefined) {
    attrs[TRACELOOP_ENTITY_INPUT] = jsonString(attrs[OI_INPUT_VALUE]);
  }
  if (attrs[OI_OUTPUT_VALUE] !== undefined) {
    attrs[TRACELOOP_ENTITY_OUTPUT] = jsonString(attrs[OI_OUTPUT_VALUE]);
  }

  const model = stringValue(attrs[OI_LLM_MODEL_NAME]);
  if (model) {
    attrs[GEN_AI_REQUEST_MODEL] = model;
  }
  const provider = stringValue(attrs[OI_LLM_PROVIDER] ?? attrs[OI_LLM_SYSTEM]);
  if (provider) {
    attrs[GEN_AI_SYSTEM] = normalizeProvider(provider);
    attrs[GEN_AI_PROVIDER_NAME] = normalizeProvider(provider);
  }

  setUsageAttrs(attrs, {
    promptTokens: coerceInteger(attrs[OI_LLM_TOKEN_COUNT_PROMPT]),
    completionTokens: coerceInteger(attrs[OI_LLM_TOKEN_COUNT_COMPLETION]),
    totalTokens: coerceInteger(attrs[OI_LLM_TOKEN_COUNT_TOTAL]),
    cacheReadTokens: coerceInteger(attrs[OI_LLM_TOKEN_COUNT_CACHE_READ]),
  });

  if (logType === RespanLogType.CHAT || logType === RespanLogType.EMBEDDING) {
    attrs[LLM_REQUEST_TYPE] =
      logType === RespanLogType.EMBEDDING
        ? RespanLogType.EMBEDDING
        : RespanLogType.CHAT;
  }

  if (logType === RespanLogType.CHAT) {
    oiMessagesToCanonical(attrs, "llm.input_messages.", GEN_AI_PROMPT_PREFIX);
    oiMessagesToCanonical(
      attrs,
      "llm.output_messages.",
      GEN_AI_COMPLETION_PREFIX,
    );
    mapOpenInferenceInvocationParameters(attrs);
    if (attrs[OI_LLM_TOOLS] !== undefined) {
      attrs[LLM_REQUEST_FUNCTIONS] = jsonString(parseMaybeJson(attrs[OI_LLM_TOOLS]));
    }
  }

  if (logType === RespanLogType.WORKFLOW) {
    attrs[TRACELOOP_WORKFLOW_NAME] = entityName;
  }

  delete attrs[TRACELOOP_SPAN_KIND];
  return logType;
}

function extractNativeLogType(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): RespanLogType | undefined {
  if (
    PYDANTIC_AI_TOOL_NAME in attrs ||
    PYDANTIC_AI_TOOL_CALL_ARGUMENTS in attrs ||
    PYDANTIC_AI_TOOL_CALL_RESULT in attrs ||
    PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS in attrs ||
    PYDANTIC_AI_LEGACY_TOOL_RESULT in attrs
  ) {
    return RespanLogType.TOOL;
  }

  const operationName = stringValue(attrs[PYDANTIC_AI_OPERATION_NAME]);
  if (operationName) {
    const mapped = PydanticAIOperationToLogType[operationName];
    if (mapped) {
      return mapped;
    }
  }

  if (
    PYDANTIC_AI_AGENT_NAME in attrs ||
    PYDANTIC_AI_LEGACY_AGENT_NAME in attrs
  ) {
    return RespanLogType.AGENT;
  }

  if (span.name === PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME) {
    return RespanLogType.TASK;
  }

  if (
    PYDANTIC_AI_INPUT_MESSAGES in attrs ||
    PYDANTIC_AI_OUTPUT_MESSAGES in attrs
  ) {
    return RespanLogType.CHAT;
  }

  return undefined;
}

function enrichNativeChatSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const entityName = span.name || "pydantic_ai.chat";
  setCommonAttrs(attrs, RespanLogType.CHAT, entityName, entityName);
  attrs[LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  setModelAndProviderAttrs(attrs);

  const inputMessages = extractMessages(attrs, PYDANTIC_AI_INPUT_MESSAGES);
  const outputMessages = extractMessages(attrs, PYDANTIC_AI_OUTPUT_MESSAGES);
  if (inputMessages?.length) {
    attrs[TRACELOOP_ENTITY_INPUT] = safeJson(inputMessages);
    setIndexedMessages(attrs, GEN_AI_PROMPT_PREFIX, inputMessages, "user");
  }
  if (outputMessages?.length) {
    attrs[TRACELOOP_ENTITY_OUTPUT] = safeJson(outputMessages);
    setIndexedMessages(
      attrs,
      GEN_AI_COMPLETION_PREFIX,
      outputMessages,
      "assistant",
    );
  }

  const tools = extractTools(attrs);
  if (tools?.length) {
    attrs[LLM_REQUEST_FUNCTIONS] = safeJson(tools);
  }

  setUsageAttrs(attrs, extractNativeUsage(attrs));
}

function enrichNativeModelSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
  logType: RespanLogType,
): void {
  const entityName = span.name || `pydantic_ai.${logType}`;
  setCommonAttrs(attrs, logType, entityName, entityName);
  attrs[LLM_REQUEST_TYPE] = logType;
  setModelAndProviderAttrs(attrs);
  setUsageAttrs(attrs, extractNativeUsage(attrs));
}

function enrichNativeToolSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const toolName =
    stringValue(attrs[PYDANTIC_AI_TOOL_NAME]) ??
    toolNameFromSpanName(span.name) ??
    "pydantic_ai.tool";
  setCommonAttrs(attrs, RespanLogType.TOOL, toolName, toolName);

  const rawArguments =
    attrs[PYDANTIC_AI_TOOL_CALL_ARGUMENTS] ??
    attrs[PYDANTIC_AI_LEGACY_TOOL_ARGUMENTS];
  attrs[TRACELOOP_ENTITY_INPUT] = safeJson({
    name: toolName,
    arguments: toSerializableValue(parseMaybeJson(rawArguments ?? {})),
  });

  const rawResult =
    attrs[PYDANTIC_AI_TOOL_CALL_RESULT] ??
    attrs[PYDANTIC_AI_LEGACY_TOOL_RESULT];
  if (rawResult !== undefined) {
    attrs[TRACELOOP_ENTITY_OUTPUT] = jsonString(parseMaybeJson(rawResult));
  }
}

function enrichNativeAgentSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const agentName =
    stringValue(attrs[PYDANTIC_AI_AGENT_NAME]) ??
    stringValue(attrs[PYDANTIC_AI_LEGACY_AGENT_NAME]) ??
    (span.name || "pydantic_ai.agent");
  setCommonAttrs(attrs, RespanLogType.AGENT, agentName, agentName);
  attrs[TRACELOOP_WORKFLOW_NAME] = agentName;

  const inputMessages = extractMessages(attrs, PYDANTIC_AI_INPUT_MESSAGES);
  if (inputMessages?.length) {
    attrs[TRACELOOP_ENTITY_INPUT] = safeJson(inputMessages);
  }

  const finalResult = attrs[PYDANTIC_AI_FINAL_RESULT];
  const outputMessages = extractMessages(attrs, PYDANTIC_AI_OUTPUT_MESSAGES);
  if (finalResult !== undefined) {
    attrs[TRACELOOP_ENTITY_OUTPUT] = jsonString(parseMaybeJson(finalResult));
  } else if (outputMessages?.length) {
    attrs[TRACELOOP_ENTITY_OUTPUT] = safeJson(outputMessages);
  }

  const tools = extractTools(attrs);
  if (tools?.length) {
    attrs[LLM_REQUEST_FUNCTIONS] = safeJson(tools);
  }
}

function enrichNativeTaskSpan(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const taskName =
    span.name === PYDANTIC_AI_RUNNING_TOOLS_SPAN_NAME
      ? "running_tools"
      : span.name || "pydantic_ai.task";
  setCommonAttrs(attrs, RespanLogType.TASK, taskName, taskName);

  const toolNames = extractToolNames(attrs);
  if (toolNames?.length) {
    attrs[TRACELOOP_ENTITY_INPUT] = safeJson({ tools: toolNames });
  }
}

function setCommonAttrs(
  attrs: SpanAttributesRecord,
  logType: RespanLogType,
  entityName: string,
  entityPath: string,
): void {
  attrs[RESPAN_LOG_METHOD] = RESPAN_LOG_METHOD_TS_TRACING;
  attrs[RESPAN_LOG_TYPE] = logType;
  attrs[TRACELOOP_ENTITY_NAME] = entityName;
  attrs[TRACELOOP_ENTITY_PATH] = entityPath;
}

function setModelAndProviderAttrs(attrs: SpanAttributesRecord): void {
  const model =
    stringValue(attrs[GEN_AI_REQUEST_MODEL]) ??
    stringValue(attrs[PYDANTIC_AI_MODEL_NAME]) ??
    stringValue(extractRequestParameters(attrs)?.model);
  if (model) {
    attrs[GEN_AI_REQUEST_MODEL] = model;
  }

  const provider = stringValue(attrs[GEN_AI_PROVIDER_NAME] ?? attrs[GEN_AI_SYSTEM]);
  if (provider) {
    attrs[GEN_AI_SYSTEM] = normalizeProvider(provider);
    attrs[GEN_AI_PROVIDER_NAME] = normalizeProvider(provider);
  }
}

function extractNativeUsage(attrs: SpanAttributesRecord): {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  cacheReadTokens?: number;
} {
  const promptTokens = coerceInteger(
    attrs[GEN_AI_USAGE_INPUT_TOKENS] ?? attrs[GEN_AI_USAGE_PROMPT_TOKENS],
  );
  const completionTokens = coerceInteger(
    attrs[GEN_AI_USAGE_OUTPUT_TOKENS] ??
      attrs[GEN_AI_USAGE_COMPLETION_TOKENS],
  );
  return {
    promptTokens,
    completionTokens,
    totalTokens: coerceInteger(
      attrs[PYDANTIC_AI_USAGE_TOTAL_TOKENS] ?? attrs[LLM_USAGE_TOTAL_TOKENS],
    ),
    cacheReadTokens: coerceInteger(attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS]),
  };
}

function setUsageAttrs(
  attrs: SpanAttributesRecord,
  usage: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
    cacheReadTokens?: number;
  },
): void {
  if (usage.promptTokens !== undefined) {
    attrs[GEN_AI_USAGE_INPUT_TOKENS] = usage.promptTokens;
    attrs[GEN_AI_USAGE_PROMPT_TOKENS] = usage.promptTokens;
  }
  if (usage.completionTokens !== undefined) {
    attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = usage.completionTokens;
    attrs[GEN_AI_USAGE_COMPLETION_TOKENS] = usage.completionTokens;
  }
  const totalTokens =
    usage.totalTokens ??
    (usage.promptTokens !== undefined || usage.completionTokens !== undefined
      ? (usage.promptTokens ?? 0) + (usage.completionTokens ?? 0)
      : undefined);
  if (totalTokens !== undefined) {
    attrs[LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
  if (usage.cacheReadTokens !== undefined) {
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = usage.cacheReadTokens;
  }
}

function extractMessages(
  attrs: SpanAttributesRecord,
  key: string,
): Array<Record<string, any>> | undefined {
  const rawMessages = parseMaybeJson(attrs[key]);
  if (!Array.isArray(rawMessages)) {
    return undefined;
  }

  const messages = rawMessages
    .map((message) => normalizeMessage(message))
    .filter((message): message is Record<string, any> => message !== undefined);
  return messages.length ? messages : undefined;
}

function normalizeMessage(value: unknown): Record<string, any> | undefined {
  const parsed = parseMaybeJson(value);
  if (!isRecord(parsed)) {
    const content = contentToValue(parsed);
    return content === undefined ? undefined : { role: "user", content };
  }

  const role = stringValue(parsed.role) ?? "user";
  const content =
    parsed.content !== undefined
      ? contentToValue(parsed.content)
      : parsed.parts !== undefined
        ? contentToValue(parsed.parts)
        : undefined;
  const message: Record<string, any> = {
    role,
    content: content ?? "",
  };

  const toolCalls = normalizeToolCalls(parsed.tool_calls ?? parsed.toolCalls);
  if (toolCalls.length) {
    message.tool_calls = toolCalls;
  }
  return message;
}

function setIndexedMessages(
  attrs: SpanAttributesRecord,
  prefix: string,
  messages: Array<Record<string, any>>,
  defaultRole: string,
): void {
  messages.forEach((message, index) => {
    const base = `${prefix}.${index}`;
    attrs[`${base}.role`] = stringValue(message.role) ?? defaultRole;
    attrs[`${base}.content`] = messageContentAttrValue(message.content);
    if (message.tool_calls !== undefined) {
      attrs[`${base}.tool_calls`] = safeJson(message.tool_calls);
    }
  });
}

function extractTools(attrs: SpanAttributesRecord): Array<Record<string, any>> | undefined {
  const directTools = normalizeToolDefinitions(
    parseMaybeJson(attrs[PYDANTIC_AI_TOOL_DEFINITIONS]),
  );
  if (directTools.length) {
    return directTools;
  }

  const params = extractRequestParameters(attrs);
  const paramTools = normalizeToolDefinitions([
    ...(Array.isArray(params?.function_tools) ? params.function_tools : []),
    ...(Array.isArray(params?.output_tools) ? params.output_tools : []),
  ]);
  return paramTools.length ? paramTools : undefined;
}

function extractToolNames(attrs: SpanAttributesRecord): string[] | undefined {
  const tools = parseMaybeJson(attrs[PYDANTIC_AI_TOOLS]);
  if (!Array.isArray(tools)) {
    return undefined;
  }
  const names = tools
    .map((tool) => stringValue(tool))
    .filter((tool): tool is string => Boolean(tool));
  return names.length ? names : undefined;
}

function normalizeToolDefinitions(value: unknown): Array<Record<string, any>> {
  const parsed = parseMaybeJson(value);
  const items = Array.isArray(parsed)
    ? parsed
    : parsed === undefined || parsed === null
      ? []
      : [parsed];
  const tools: Array<Record<string, any>> = [];

  for (const item of items) {
    const parsedItem = parseMaybeJson(item);
    if (typeof parsedItem === "string" && parsedItem) {
      tools.push({ type: "function", function: { name: parsedItem } });
      continue;
    }

    if (!isRecord(parsedItem)) {
      continue;
    }

    const existingFunction = isRecord(parsedItem.function)
      ? parsedItem.function
      : undefined;
    const name = stringValue(existingFunction?.name ?? parsedItem.name);
    if (!name) {
      continue;
    }

    const functionPayload: Record<string, any> = { name };
    const description = existingFunction?.description ?? parsedItem.description;
    if (description !== undefined) {
      functionPayload.description = String(description);
    }
    const parameters =
      existingFunction?.parameters ??
      parsedItem.parameters ??
      parsedItem.parameters_json_schema ??
      parsedItem.input_schema;
    if (parameters !== undefined) {
      functionPayload.parameters = toSerializableValue(parameters);
    }
    const strict = existingFunction?.strict ?? parsedItem.strict;
    if (strict !== undefined) {
      functionPayload.strict = Boolean(strict);
    }

    tools.push({
      type: stringValue(parsedItem.type) ?? "function",
      function: functionPayload,
    });
  }

  return tools;
}

function normalizeToolCalls(value: unknown): Array<Record<string, any>> {
  const parsed = parseMaybeJson(value);
  const calls = Array.isArray(parsed)
    ? parsed
    : parsed === undefined || parsed === null
      ? []
      : [parsed];

  return calls.flatMap((call, index) => {
    const parsedCall = parseMaybeJson(call);
    if (!isRecord(parsedCall)) {
      return [];
    }
    const rawFunction = isRecord(parsedCall.function)
      ? parsedCall.function
      : {};
    const name = stringValue(
      rawFunction.name ?? parsedCall.name ?? parsedCall.toolName,
    );
    if (!name) {
      return [];
    }
    const args =
      rawFunction.arguments ??
      parsedCall.arguments ??
      parsedCall.args ??
      parsedCall.input ??
      {};
    return [
      {
        id: stringValue(parsedCall.id) ?? `call_${index}`,
        type: "function",
        function: {
          name,
          arguments: typeof args === "string" ? args : safeJson(args),
        },
      },
    ];
  });
}

function extractRequestParameters(
  attrs: SpanAttributesRecord,
): Record<string, any> | undefined {
  const parsed = parseMaybeJson(attrs[PYDANTIC_AI_REQUEST_PARAMETERS]);
  return isRecord(parsed) ? parsed : undefined;
}

function mapOpenInferenceInvocationParameters(attrs: SpanAttributesRecord): void {
  const params = parseMaybeJson(attrs[OI_LLM_INVOCATION_PARAMETERS]);
  if (!isRecord(params)) {
    return;
  }
  if (params.model !== undefined && attrs[GEN_AI_REQUEST_MODEL] === undefined) {
    attrs[GEN_AI_REQUEST_MODEL] = stringValue(params.model);
  }
}

function oiMessagesToCanonical(
  attrs: SpanAttributesRecord,
  oiPrefix: string,
  genAiPrefix: string,
): void {
  const buckets = new Map<number, Map<string, any>>();

  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith(oiPrefix)) {
      continue;
    }
    const rest = key.slice(oiPrefix.length);
    const dotIndex = rest.indexOf(".");
    const indexText = dotIndex === -1 ? rest : rest.slice(0, dotIndex);
    if (!/^\d+$/.test(indexText)) {
      continue;
    }

    const index = Number.parseInt(indexText, 10);
    const field = dotIndex === -1 ? "" : rest.slice(dotIndex + 1);
    const bucket = buckets.get(index) ?? new Map<string, any>();
    bucket.set(field, value);
    buckets.set(index, bucket);
  }

  for (const index of [...buckets.keys()].sort((left, right) => left - right)) {
    const bucket = buckets.get(index)!;
    const base = `${genAiPrefix}.${index}`;
    const role = bucket.get("message.role");
    if (role !== undefined) {
      attrs[`${base}.role`] = String(role);
    }

    const content =
      bucket.get("message.content") ??
      contentBlocksFromOpenInferenceBucket(bucket);
    if (content !== undefined) {
      attrs[`${base}.content`] = messageContentAttrValue(content);
    }

    const toolCalls = toolCallsFromOpenInferenceBucket(bucket);
    if (toolCalls.length) {
      attrs[`${base}.tool_calls`] = safeJson(toolCalls);
    }
  }
}

function contentBlocksFromOpenInferenceBucket(
  bucket: Map<string, any>,
): unknown {
  const contentBlocks = new Map<number, Map<string, any>>();
  for (const [field, value] of bucket) {
    if (!field.startsWith("message.contents.")) {
      continue;
    }
    const rest = field.slice("message.contents.".length);
    const dotIndex = rest.indexOf(".");
    if (dotIndex === -1) {
      continue;
    }
    const indexText = rest.slice(0, dotIndex);
    if (!/^\d+$/.test(indexText)) {
      continue;
    }
    const blockIndex = Number.parseInt(indexText, 10);
    const blockField = rest
      .slice(dotIndex + 1)
      .replace(/^message_content\./, "");
    const block = contentBlocks.get(blockIndex) ?? new Map<string, any>();
    block.set(blockField, value);
    contentBlocks.set(blockIndex, block);
  }

  if (!contentBlocks.size) {
    return undefined;
  }

  const ordered = [...contentBlocks.keys()]
    .sort((left, right) => left - right)
    .map((index) => Object.fromEntries(contentBlocks.get(index)!));
  const text = ordered
    .map((block) => stringValue(block.text))
    .filter((block): block is string => Boolean(block));
  return text.length === ordered.length ? text.join("\n") : ordered;
}

function toolCallsFromOpenInferenceBucket(
  bucket: Map<string, any>,
): Array<Record<string, any>> {
  const toolCallBuckets = new Map<number, Record<string, any>>();
  for (const [field, value] of bucket) {
    if (!field.startsWith("message.tool_calls.")) {
      continue;
    }
    const rest = field.slice("message.tool_calls.".length);
    const dotIndex = rest.indexOf(".");
    if (dotIndex === -1) {
      continue;
    }
    const indexText = rest.slice(0, dotIndex);
    if (!/^\d+$/.test(indexText)) {
      continue;
    }

    const index = Number.parseInt(indexText, 10);
    const outputField = rest
      .slice(dotIndex + 1)
      .replace(/^tool_call\./, "");
    const current = toolCallBuckets.get(index) ?? {};
    current[outputField] = value;
    toolCallBuckets.set(index, current);
  }

  return [...toolCallBuckets.keys()]
    .sort((left, right) => left - right)
    .map((index) => toolCallBuckets.get(index)!)
    .map((call) => {
      if (call.function?.name) {
        return call;
      }
      return {
        id: stringValue(call.id) ?? "",
        type: stringValue(call.type) ?? "function",
        function: {
          name: stringValue(call["function.name"] ?? call.name) ?? "",
          arguments:
            typeof call["function.arguments"] === "string"
              ? call["function.arguments"]
              : safeJson(call["function.arguments"] ?? call.arguments ?? {}),
        },
      };
    })
    .filter((call) => call.function.name);
}

function stripRawAttrs(
  attrs: SpanAttributesRecord,
  logType: RespanLogType,
  source: "native" | "openinference",
): SpanAttributesRecord {
  const stripped: SpanAttributesRecord = {};
  const rawExact = source === "native" ? PYDANTIC_AI_RAW_EXACT_ATTRS : OI_RAW_EXACT_ATTRS;
  const rawPrefixes = source === "native" ? [] : OI_RAW_PREFIXES;

  for (const [key, value] of Object.entries(attrs)) {
    if (rawExact.has(key) || OFF_CONTRACT_ALIAS_ATTRS.has(key)) {
      continue;
    }
    if (OTEL_NOISE_EXACT_ATTRS.has(key)) {
      continue;
    }
    if (rawPrefixes.some((prefix) => key.startsWith(prefix))) {
      continue;
    }
    if (OTEL_NOISE_PREFIXES.some((prefix) => key.startsWith(prefix))) {
      continue;
    }
    if (!LLM_LOG_TYPES.has(logType) && isLlmOnlyCanonicalAttr(key)) {
      continue;
    }
    stripped[key] = value;
  }
  return stripped;
}

function isLlmOnlyCanonicalAttr(key: string): boolean {
  return (
    LLM_CANONICAL_EXACT_ATTRS.has(key) ||
    LLM_CANONICAL_PREFIXES.some((prefix) => key.startsWith(prefix))
  );
}

function getMutableAttributes(
  span: ReadableSpan,
): SpanAttributesRecord | undefined {
  const spanAny = span as any;
  if (spanAny.attributes && typeof spanAny.attributes === "object") {
    return spanAny.attributes;
  }
  if (spanAny._attributes && typeof spanAny._attributes === "object") {
    return spanAny._attributes;
  }
  return undefined;
}

function replaceSpanAttributes(
  span: ReadableSpan,
  attrs: SpanAttributesRecord,
): void {
  const spanAny = span as any;
  const target = getMutableAttributes(span);
  if (target) {
    for (const key of Object.keys(target)) {
      delete target[key];
    }
    Object.assign(target, attrs);
  }
  if (spanAny.attributes && spanAny.attributes !== target) {
    spanAny.attributes = attrs;
  }
  if (spanAny._attributes) {
    spanAny._attributes = target ?? attrs;
  }
}

function getInstrumentationScopeName(span: ReadableSpan): string {
  const spanAny = span as any;
  return String(
    spanAny.instrumentationScope?.name ??
      spanAny.instrumentationScope?.name ??
      "",
  );
}

function scopeLooksLikePydanticAI(scopeName: string): boolean {
  const normalized = scopeName.toLowerCase();
  return PYDANTIC_AI_SCOPE_MARKERS.some((marker) =>
    normalized.includes(marker),
  );
}

function toolNameFromSpanName(spanName: string): string | undefined {
  const match = spanName.match(/(?:execute[_ ]tool|tool)[: ]+(.+)$/i);
  return match?.[1]?.trim() || undefined;
}

function normalizeProvider(provider: string): string {
  return provider.toLowerCase().replace(/^@/, "").replace(/[^a-z0-9._-]/g, "_");
}

function parseMaybeJson(value: unknown): unknown {
  if (typeof value !== "string") {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function jsonString(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  return safeJson(value);
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(toSerializableValue(value));
  } catch {
    return String(value);
  }
}

function toSerializableValue(value: unknown): unknown {
  if (value === undefined || value === null) {
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
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        toSerializableValue(item),
      ]),
    );
  }
  return String(value);
}

function contentToValue(value: unknown): unknown {
  const parsed = parseMaybeJson(value);
  const text = extractText(parsed);
  return text ?? toSerializableValue(parsed);
}

function messageContentAttrValue(value: unknown): string {
  const text = extractText(value);
  return text ?? jsonString(value);
}

function extractText(value: unknown): string | undefined {
  const parsed = parseMaybeJson(value);
  if (typeof parsed === "string") {
    return parsed;
  }
  if (Array.isArray(parsed)) {
    const parts = parsed.map((item) => extractText(item));
    return parts.every((part) => part !== undefined)
      ? parts.join("\n")
      : undefined;
  }
  if (isRecord(parsed)) {
    if (typeof parsed.text === "string") {
      return parsed.text;
    }
    if (parsed.type === "text" && typeof parsed.content === "string") {
      return parsed.content;
    }
    if (typeof parsed.content === "string") {
      return parsed.content;
    }
  }
  return undefined;
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

function stringValue(value: unknown): string | undefined {
  if (typeof value === "string" && value.length) {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
