import type { Span } from "@opentelemetry/api";
import {
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
  ATTR_GEN_AI_TOOL_DEFINITIONS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_TOOL_TYPE,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  LIVEKIT_LOG_METHOD_TS_TRACING,
  LIVEKIT_LOG_TYPE_BY_SPAN_NAME,
  LIVEKIT_INTERNAL_ACTIVITY_SPAN_NAMES,
  LIVEKIT_RAW_ATTRIBUTE_PREFIX,
  LIVEKIT_SPAN_NAME_ATTRIBUTE,
  LIVEKIT_SPAN_NAMES,
  LK_AGENT_LABEL,
  LK_AGENT_NAME,
  LK_AGENT_PARENT_TURN_ID,
  LK_AGENT_TURN_ID,
  LK_AMD_CATEGORY,
  LK_AMD_DELAY,
  LK_AMD_INTERRUPT_ON_MACHINE,
  LK_AMD_IS_MACHINE,
  LK_AMD_REASON,
  LK_AMD_SPEECH_DURATION,
  LK_AMD_TRANSCRIPT,
  LK_CHAT_CTX,
  LK_END_OF_TURN_DELAY,
  LK_E2E_LATENCY,
  LK_EOU_DELAY,
  LK_EOU_DETECTION_DELAY,
  LK_EOU_FROM_CACHE,
  LK_EOU_LANGUAGE,
  LK_EOU_PROBABILITY,
  LK_EOU_SOURCE,
  LK_EOU_UNLIKELY_THRESHOLD,
  LK_FUNCTION_TOOL_ARGS,
  LK_FUNCTION_TOOL_ID,
  LK_FUNCTION_TOOL_IS_ERROR,
  LK_FUNCTION_TOOL_NAME,
  LK_FUNCTION_TOOL_OUTPUT,
  LK_FUNCTION_TOOLS,
  LK_INSTRUCTIONS,
  LK_INTERRUPTION_DETECTION_DELAY,
  LK_INTERRUPTION_PREDICTION_DURATION,
  LK_INTERRUPTION_PROBABILITY,
  LK_INTERRUPTION_TOTAL_DURATION,
  LK_IS_INTERRUPTION,
  LK_JOB_ID,
  LK_LLM_METRICS,
  LK_PARTICIPANT_ID,
  LK_PARTICIPANT_IDENTITY,
  LK_PARTICIPANT_KIND,
  LK_PROVIDER_REQUEST_IDS,
  LK_PROVIDER_TOOLS,
  LK_REALTIME_MODEL_METRICS,
  LK_RESPONSE_FUNCTION_CALLS,
  LK_RESPONSE_TEXT,
  LK_RESPONSE_TTFB,
  LK_RESPONSE_TTFT,
  LK_RETRY_COUNT,
  LK_ROOM_NAME,
  LK_SESSION_OPTIONS,
  LK_SPEECH_ID,
  LK_SPEECH_INTERRUPTED,
  LK_TOOL_SETS,
  LK_TRANSCRIPT_CONFIDENCE,
  LK_TRANSCRIPTION_DELAY,
  LK_TTS_INPUT_TEXT,
  LK_TTS_LABEL,
  LK_TTS_METRICS,
  LK_TTS_STREAMING,
  LK_USER_INPUT,
  LK_USER_TRANSCRIPT,
  OFF_CONTRACT_ALIAS_KEYS,
} from "./_constants.js";

const GEN_AI_COMPLETION_PREFIX = `${SpanAttributes.LLM_COMPLETIONS}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;
const LANGFUSE_COMPLETION_START_TIME =
  "langfuse.observation.completion_start_time";

export type MutableAttributes = Record<string, unknown>;

export interface TranslateLiveKitSpanOptions {
  attributes?: MutableAttributes;
  spanName?: string;
  /** Skip Span API writes when translating from an onEnd transformer hook. */
  writeToSpan?: boolean;
}

interface ChatContextJson {
  items?: LiveKitChatItem[];
}

interface LiveKitChatItem {
  type?: string;
  role?: string;
  content?: unknown;
  text?: string;
  name?: string;
  args?: unknown;
  callId?: string;
  call_id?: string;
  id?: string;
  output?: unknown;
  isError?: boolean;
}

export function translateLiveKitSpan(
  span: Span,
  options: TranslateLiveKitSpanOptions = {},
): void {
  const attrs = options.attributes ?? getSpanAttributes(span);
  const spanName = options.spanName ?? getSpanName(span, attrs);
  const writableSpan = options.writeToSpan === false ? undefined : span;
  normalizeNativeContent(attrs);

  if (LIVEKIT_INTERNAL_ACTIVITY_SPAN_NAMES.has(spanName)) {
    return;
  }

  if (!isLiveKitSpan(spanName, attrs)) {
    return;
  }

  const logType = resolveLogType(spanName, attrs);
  setAttr(writableSpan, attrs, LIVEKIT_SPAN_NAME_ATTRIBUTE, spanName);
  setAttr(
    writableSpan,
    attrs,
    RespanSpanAttributes.RESPAN_LOG_METHOD,
    LIVEKIT_LOG_METHOD_TS_TRACING,
  );
  setAttr(writableSpan, attrs, RespanSpanAttributes.RESPAN_LOG_TYPE, logType);

  const entityName = resolveEntityName(spanName, attrs);
  setAttr(
    writableSpan,
    attrs,
    SpanAttributes.TRACELOOP_ENTITY_NAME,
    entityName,
  );
  setAttr(
    writableSpan,
    attrs,
    SpanAttributes.TRACELOOP_ENTITY_PATH,
    spanName || entityName,
  );
  const entityInput = buildEntityInput(spanName, attrs);
  if (hasEntityValue(entityInput)) {
    setAttr(
      writableSpan,
      attrs,
      SpanAttributes.TRACELOOP_ENTITY_INPUT,
      safeJson(entityInput),
    );
  }
  const entityOutput = buildEntityOutput(spanName, attrs);
  if (hasEntityValue(entityOutput)) {
    setAttr(
      writableSpan,
      attrs,
      SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
      safeJson(entityOutput),
    );
  }

  if (logType === RespanLogType.WORKFLOW) {
    setAttr(
      writableSpan,
      attrs,
      SpanAttributes.TRACELOOP_WORKFLOW_NAME,
      entityName,
    );
  }

  if (logType === RespanLogType.CHAT) {
    addChatAttributes(writableSpan, attrs);
  } else if (logType === RespanLogType.TOOL) {
    const callId =
      attrs[ATTR_GEN_AI_TOOL_CALL_ID] ?? attrs[LK_FUNCTION_TOOL_ID];
    if (typeof callId === "string")
      setAttr(writableSpan, attrs, ATTR_GEN_AI_TOOL_CALL_ID, callId);
    for (const key of [
      ATTR_GEN_AI_TOOL_NAME,
      ATTR_GEN_AI_TOOL_TYPE,
      ATTR_GEN_AI_TOOL_DESCRIPTION,
      ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
      ATTR_GEN_AI_TOOL_CALL_RESULT,
      ATTR_GEN_AI_OPERATION_NAME,
    ])
      delete attrs[key];
  } else if (logType === RespanLogType.TEXT) {
    addTextModelAttributes(writableSpan, attrs);
  }

  addLiveKitMetadata(writableSpan, attrs);

  for (const key of [
    ATTR_GEN_AI_INPUT_MESSAGES,
    ATTR_GEN_AI_OUTPUT_MESSAGES,
    ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
    ATTR_GEN_AI_TOOL_DEFINITIONS,
  ])
    delete attrs[key];
  if (logType !== RespanLogType.CHAT && logType !== RespanLogType.TEXT) {
    for (const key of Object.keys(attrs))
      if (
        (key.startsWith("gen_ai.") || key.startsWith("llm.")) &&
        !(logType === RespanLogType.TOOL && key === ATTR_GEN_AI_TOOL_CALL_ID)
      )
        delete attrs[key];
  }

  for (const key of OFF_CONTRACT_ALIAS_KEYS) {
    delete attrs[key];
  }
  stripLiveKitRawAttributes(attrs);
}

export function isLiveKitSpan(
  spanName: string,
  attrs: MutableAttributes,
): boolean {
  if (spanName in LIVEKIT_LOG_TYPE_BY_SPAN_NAME) {
    return true;
  }
  return Object.keys(attrs).some((key) =>
    key.startsWith(LIVEKIT_RAW_ATTRIBUTE_PREFIX),
  );
}

export function getSpanAttributes(span: Span): MutableAttributes {
  const maybeAttrs = (span as unknown as { attributes?: MutableAttributes })
    .attributes;
  if (maybeAttrs && typeof maybeAttrs === "object") {
    return maybeAttrs;
  }
  return {};
}

function getSpanName(span: Span, attrs: MutableAttributes): string {
  const explicit = attrs[LIVEKIT_SPAN_NAME_ATTRIBUTE];
  if (typeof explicit === "string" && explicit) {
    return explicit;
  }
  return String((span as unknown as { name?: string }).name ?? "livekit.span");
}

function resolveLogType(
  spanName: string,
  attrs: MutableAttributes,
): RespanLogType {
  if (spanName in LIVEKIT_LOG_TYPE_BY_SPAN_NAME) {
    return LIVEKIT_LOG_TYPE_BY_SPAN_NAME[spanName]!;
  }
  if (attrs[LK_FUNCTION_TOOL_NAME] !== undefined) {
    return RespanLogType.TOOL;
  }
  if (attrs[LK_LLM_METRICS] !== undefined) {
    return RespanLogType.CHAT;
  }
  if (
    attrs[LK_TTS_METRICS] !== undefined ||
    attrs[LK_TTS_INPUT_TEXT] !== undefined
  ) {
    return RespanLogType.TEXT;
  }
  return RespanLogType.TASK;
}

function resolveEntityName(spanName: string, attrs: MutableAttributes): string {
  const agentLabel =
    stringAttr(attrs, LK_AGENT_LABEL) ?? stringAttr(attrs, LK_AGENT_NAME);

  if (spanName === LIVEKIT_SPAN_NAMES.AGENT_SESSION) {
    return agentLabel ? `${agentLabel}.session` : "livekit.agent_session";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.AGENT_TURN) {
    return agentLabel ? `${agentLabel}.turn` : "livekit.agent_turn";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.FUNCTION_TOOL) {
    return stringAttr(attrs, LK_FUNCTION_TOOL_NAME) ?? "livekit.function_tool";
  }
  if (
    spanName === LIVEKIT_SPAN_NAMES.LLM_NODE ||
    spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST
  ) {
    const provider = providerName(attrs);
    const model = stringAttr(attrs, SpanAttributes.LLM_REQUEST_MODEL);
    if (provider && model) {
      return `${provider}.${model}`;
    }
    return "livekit.llm";
  }
  if (
    spanName === LIVEKIT_SPAN_NAMES.TTS_NODE ||
    spanName === LIVEKIT_SPAN_NAMES.TTS_REQUEST
  ) {
    return stringAttr(attrs, LK_TTS_LABEL) ?? "livekit.tts";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.USER_TURN) {
    return stringAttr(attrs, LK_PARTICIPANT_IDENTITY) ?? "livekit.user_turn";
  }
  return agentLabel
    ? `${agentLabel}.${spanName}`
    : `livekit.${spanName || "span"}`;
}

function buildEntityInput(spanName: string, attrs: MutableAttributes): unknown {
  if (spanName === LIVEKIT_SPAN_NAMES.FUNCTION_TOOL) {
    return compactRecord({
      name: stringAttr(attrs, LK_FUNCTION_TOOL_NAME),
      arguments: parseMaybeJson(
        attrs[LK_FUNCTION_TOOL_ARGS] ?? attrs[ATTR_GEN_AI_TOOL_CALL_ARGUMENTS],
      ),
    });
  }

  if (
    spanName === LIVEKIT_SPAN_NAMES.LLM_NODE ||
    spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST
  ) {
    return compactRecord({
      chat_ctx: parseMaybeJson(attrs[LK_CHAT_CTX]),
      input_messages: parseMaybeJson(attrs[ATTR_GEN_AI_INPUT_MESSAGES]),
      system_instructions: parseMaybeJson(
        attrs[ATTR_GEN_AI_SYSTEM_INSTRUCTIONS],
      ),
      tool_definitions: parseMaybeJson(attrs[ATTR_GEN_AI_TOOL_DEFINITIONS]),
      function_tools: parseMaybeJson(attrs[LK_FUNCTION_TOOLS]),
      provider_tools: parseMaybeJson(attrs[LK_PROVIDER_TOOLS]),
      tool_sets: parseMaybeJson(attrs[LK_TOOL_SETS]),
    });
  }

  if (
    spanName === LIVEKIT_SPAN_NAMES.TTS_NODE ||
    spanName === LIVEKIT_SPAN_NAMES.TTS_REQUEST
  ) {
    return compactRecord({
      text: stringAttr(attrs, LK_TTS_INPUT_TEXT),
      streaming: attrs[LK_TTS_STREAMING],
      label: stringAttr(attrs, LK_TTS_LABEL),
    });
  }

  if (spanName === LIVEKIT_SPAN_NAMES.AGENT_TURN) {
    return compactRecord({
      speech_id: attrs[LK_SPEECH_ID],
      user_input: attrs[LK_USER_INPUT],
      instructions: attrs[LK_INSTRUCTIONS],
      generation_id: attrs[LK_AGENT_TURN_ID],
      parent_generation_id: attrs[LK_AGENT_PARENT_TURN_ID],
    });
  }

  return compactRecord({
    agent_label: attrs[LK_AGENT_LABEL],
    job_id: attrs[LK_JOB_ID],
    room_name: attrs[LK_ROOM_NAME],
    session_options: parseMaybeJson(attrs[LK_SESSION_OPTIONS]),
    participant_id: attrs[LK_PARTICIPANT_ID],
    participant_identity: attrs[LK_PARTICIPANT_IDENTITY],
    participant_kind: attrs[LK_PARTICIPANT_KIND],
    retry_count: attrs[LK_RETRY_COUNT],
  });
}

function buildEntityOutput(
  spanName: string,
  attrs: MutableAttributes,
): unknown {
  if (spanName === LIVEKIT_SPAN_NAMES.FUNCTION_TOOL) {
    return compactRecord({
      output: parseMaybeJson(
        attrs[LK_FUNCTION_TOOL_OUTPUT] ?? attrs[ATTR_GEN_AI_TOOL_CALL_RESULT],
      ),
      is_error: attrs[LK_FUNCTION_TOOL_IS_ERROR],
    });
  }

  if (
    spanName === LIVEKIT_SPAN_NAMES.LLM_NODE ||
    spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST
  ) {
    return compactRecord({
      text: attrs[LK_RESPONSE_TEXT],
      function_calls: parseMaybeJson(attrs[LK_RESPONSE_FUNCTION_CALLS]),
    });
  }

  if (
    spanName === LIVEKIT_SPAN_NAMES.TTS_NODE ||
    spanName === LIVEKIT_SPAN_NAMES.TTS_REQUEST
  ) {
    return undefined;
  }

  if (spanName === LIVEKIT_SPAN_NAMES.USER_TURN) {
    return compactRecord({
      transcript: attrs[LK_USER_TRANSCRIPT],
      confidence: attrs[LK_TRANSCRIPT_CONFIDENCE],
      transcription_delay: attrs[LK_TRANSCRIPTION_DELAY],
      end_of_turn_delay: attrs[LK_END_OF_TURN_DELAY],
      eou: compactRecord({
        probability: attrs[LK_EOU_PROBABILITY],
        unlikely_threshold: attrs[LK_EOU_UNLIKELY_THRESHOLD],
        delay: attrs[LK_EOU_DELAY],
        language: attrs[LK_EOU_LANGUAGE],
        source: attrs[LK_EOU_SOURCE],
        from_cache: attrs[LK_EOU_FROM_CACHE],
        detection_delay: attrs[LK_EOU_DETECTION_DELAY],
      }),
    });
  }

  return compactRecord({
    response_text: attrs[LK_RESPONSE_TEXT],
    interrupted: attrs[LK_SPEECH_INTERRUPTED],
    e2e_latency: attrs[LK_E2E_LATENCY],
    amd: compactRecord({
      category: attrs[LK_AMD_CATEGORY],
      reason: attrs[LK_AMD_REASON],
      is_machine: attrs[LK_AMD_IS_MACHINE],
      interrupt_on_machine: attrs[LK_AMD_INTERRUPT_ON_MACHINE],
      speech_duration: attrs[LK_AMD_SPEECH_DURATION],
      delay: attrs[LK_AMD_DELAY],
      transcript: attrs[LK_AMD_TRANSCRIPT],
    }),
    interruption: compactRecord({
      is_interruption: attrs[LK_IS_INTERRUPTION],
      probability: attrs[LK_INTERRUPTION_PROBABILITY],
      total_duration: attrs[LK_INTERRUPTION_TOTAL_DURATION],
      prediction_duration: attrs[LK_INTERRUPTION_PREDICTION_DURATION],
      detection_delay: attrs[LK_INTERRUPTION_DETECTION_DELAY],
    }),
    realtime_model_metrics: parseMaybeJson(attrs[LK_REALTIME_MODEL_METRICS]),
  });
}

function addChatAttributes(
  span: Span | undefined,
  attrs: MutableAttributes,
): void {
  const provider = providerName(attrs);
  const model = stringAttr(attrs, SpanAttributes.LLM_REQUEST_MODEL);

  if (provider) {
    setAttr(span, attrs, ATTR_GEN_AI_PROVIDER_NAME, provider);
    setAttr(
      span,
      attrs,
      SpanAttributes.LLM_SYSTEM,
      normalizeProvider(provider),
    );
  }
  if (model) {
    setAttr(span, attrs, SpanAttributes.LLM_REQUEST_MODEL, model);
  }
  setAttr(span, attrs, SpanAttributes.LLM_REQUEST_TYPE, "chat");

  const inputTokens = numberAttr(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS);
  const outputTokens = numberAttr(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS);
  if (inputTokens !== undefined) {
    setAttr(span, attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, inputTokens);
  }
  if (outputTokens !== undefined) {
    setAttr(
      span,
      attrs,
      SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
      outputTokens,
    );
  }

  const metrics = parseMaybeJson(attrs[LK_LLM_METRICS]) as
    Record<string, unknown> | undefined;
  const totalTokens =
    numberAttr(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS) ??
    asNumber(metrics?.totalTokens);
  if (totalTokens !== undefined) {
    setAttr(span, attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, totalTokens);
  }
  const cachedTokens = asNumber(metrics?.promptCachedTokens);
  if (cachedTokens !== undefined) {
    setAttr(
      span,
      attrs,
      ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
      cachedTokens,
    );
  }
  // llm_node spans carry lk.response.ttft in seconds; llm_request spans carry
  // lk.llm_metrics.ttftMs in milliseconds (-1 when no chunk arrived).
  const ttftMs = asNumber(metrics?.ttftMs);
  const firstChunkSeconds =
    numberAttr(attrs, LK_RESPONSE_TTFT) ??
    (ttftMs !== undefined ? ttftMs / 1000 : undefined);
  if (
    firstChunkSeconds !== undefined &&
    firstChunkSeconds > 0 &&
    attrs[ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] === undefined
  ) {
    setAttr(
      span,
      attrs,
      ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
      firstChunkSeconds,
    );
  }

  const chatCtx = parseMaybeJson(attrs[LK_CHAT_CTX]) as
    ChatContextJson | undefined;
  // Complete output and aggregate input precede optional indexed prompt fields.
  addCompletionAttributes(span, attrs);
  addToolDefinitionAttributes(span, attrs);
  addPromptAttributes(span, attrs, chatCtx);
}

function addTextModelAttributes(
  span: Span | undefined,
  attrs: MutableAttributes,
): void {
  const metrics = parseMaybeJson(attrs[LK_TTS_METRICS]) as
    Record<string, any> | undefined;
  const provider = providerName(attrs);
  const model =
    stringAttr(attrs, SpanAttributes.LLM_REQUEST_MODEL) ??
    stringValue(metrics?.metadata?.modelName);
  if (provider) {
    setAttr(span, attrs, ATTR_GEN_AI_PROVIDER_NAME, provider);
    setAttr(
      span,
      attrs,
      SpanAttributes.LLM_SYSTEM,
      normalizeProvider(provider),
    );
  }
  if (model) {
    setAttr(span, attrs, SpanAttributes.LLM_REQUEST_MODEL, model);
  }
  const inputTokens = asNumber(metrics?.inputTokens);
  const outputTokens = asNumber(metrics?.outputTokens);
  if (inputTokens !== undefined) {
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, inputTokens);
  }
  if (outputTokens !== undefined) {
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, outputTokens);
  }
}

function addLiveKitMetadata(
  span: Span | undefined,
  attrs: MutableAttributes,
): void {
  const llmMetrics = parseMaybeJson(attrs[LK_LLM_METRICS]) as
    Record<string, unknown> | undefined;
  const providerRequestIds = normalizeProviderRequestIds([
    attrs[LK_PROVIDER_REQUEST_IDS],
    llmMetrics?.requestId,
  ]);
  if (providerRequestIds.length > 0) {
    setMetadata(span, attrs, "provider_request_ids", providerRequestIds);
  }

  for (const [metadataKey, value] of [
    ["time_to_first_token_seconds", attrs[LK_RESPONSE_TTFT]],
    ["time_to_first_byte_seconds", attrs[LK_RESPONSE_TTFB]],
    ["completion_start_time", attrs[LANGFUSE_COMPLETION_START_TIME]],
    ["llm_ttft_ms", llmMetrics?.ttftMs],
    ["llm_duration_ms", llmMetrics?.durationMs],
    ["llm_tokens_per_second", llmMetrics?.tokensPerSecond],
    ["llm_cancelled", llmMetrics?.cancelled],
    ["retry_count", attrs[LK_RETRY_COUNT]],
    ["e2e_latency_seconds", attrs[LK_E2E_LATENCY]],
  ] as const) {
    if (value !== undefined && value !== null) {
      setMetadata(span, attrs, metadataKey, value);
    }
  }
}

function setMetadata(
  span: Span | undefined,
  attrs: MutableAttributes,
  key: string,
  value: unknown,
): void {
  const existing = parseMaybeJson(attrs[RespanSpanAttributes.RESPAN_METADATA]);
  const metadata =
    existing !== null &&
    typeof existing === "object" &&
    !Array.isArray(existing)
      ? { ...(existing as Record<string, unknown>) }
      : {};
  if (metadata[key] === undefined) {
    metadata[key] = value;
  }
  setAttr(
    span,
    attrs,
    RespanSpanAttributes.RESPAN_METADATA,
    safeJson(metadata),
  );
}

function normalizeProviderRequestIds(values: unknown[]): string[] {
  const requestIds = new Set<string>();
  for (const value of values) {
    const parsed = parseMaybeJson(value);
    const items = Array.isArray(parsed) ? parsed : [parsed];
    for (const item of items) {
      if (typeof item === "string" && item.length > 0) {
        requestIds.add(item);
      }
    }
  }
  return [...requestIds];
}

function addPromptAttributes(
  span: Span | undefined,
  attrs: MutableAttributes,
  chatCtx?: ChatContextJson,
): void {
  const items = Array.isArray(chatCtx?.items)
    ? chatCtx.items
    : modernPromptItems(attrs);
  let promptIndex = 0;
  for (const item of items) {
    const normalized = normalizePromptItem(item);
    if (!normalized) {
      continue;
    }
    const prefix = `${SpanAttributes.LLM_PROMPTS}.${promptIndex}`;
    setAttr(span, attrs, `${prefix}.role`, normalized.role);
    setAttr(span, attrs, `${prefix}.content`, normalized.content);
    if (normalized.tool_calls !== undefined) {
      setAttr(
        span,
        attrs,
        `${prefix}.tool_calls`,
        safeJson(normalized.tool_calls),
      );
    }
    if (normalized.tool_call_id !== undefined)
      setAttr(span, attrs, `${prefix}.tool_call_id`, normalized.tool_call_id);
    promptIndex += 1;
  }
}

function addToolDefinitionAttributes(
  span: Span | undefined,
  attrs: MutableAttributes,
): void {
  const nativeDefinitions = parseMaybeJson(attrs[ATTR_GEN_AI_TOOL_DEFINITIONS]);
  if (Array.isArray(nativeDefinitions) && nativeDefinitions.length > 0) {
    setAttr(
      span,
      attrs,
      SpanAttributes.LLM_REQUEST_FUNCTIONS,
      safeJson(
        nativeDefinitions.map((definition: any) => ({
          type: definition.type ?? "function",
          function: compactRecord({
            name: definition.name,
            description: definition.description,
            parameters: definition.parameters,
          }),
        })),
      ),
    );
    return;
  }
  const toolNames = parseMaybeJson(attrs[LK_FUNCTION_TOOLS]);
  if (!Array.isArray(toolNames) || toolNames.length === 0) {
    return;
  }

  const toolDefinitions = toolNames.map((toolName) => ({
    type: "function",
    function: {
      name: String(toolName),
    },
  }));
  setAttr(
    span,
    attrs,
    SpanAttributes.LLM_REQUEST_FUNCTIONS,
    safeJson(toolDefinitions),
  );
}

function addCompletionAttributes(
  span: Span | undefined,
  attrs: MutableAttributes,
): void {
  const responseText = stringAttr(attrs, LK_RESPONSE_TEXT);
  const rawCalls = parseMaybeJson(attrs[LK_RESPONSE_FUNCTION_CALLS]);
  const toolCalls = normalizeToolCalls(rawCalls);

  if (responseText === undefined && toolCalls.length === 0) {
    return;
  }

  setAttr(span, attrs, GEN_AI_COMPLETION_ROLE, "assistant");
  setAttr(span, attrs, GEN_AI_COMPLETION_CONTENT, responseText ?? "");
  if (toolCalls.length > 0) {
    setAttr(span, attrs, GEN_AI_COMPLETION_TOOL_CALLS, safeJson(toolCalls));
  }
}

function normalizePromptItem(item: LiveKitChatItem):
  | {
      role: string;
      content: string;
      tool_calls?: unknown[];
      tool_call_id?: string;
    }
  | undefined {
  if (item.type === "message") {
    return {
      role: item.role ?? "user",
      content: contentToText(item.content ?? item.text ?? ""),
    };
  }

  if (item.type === "function_call") {
    return {
      role: "assistant",
      content: "",
      tool_calls: normalizeToolCalls([item]),
    };
  }

  if (item.type === "function_call_output") {
    return {
      role: "tool",
      content: contentToText(item.output ?? ""),
      tool_call_id: stringValue(item.callId) ?? stringValue(item.call_id),
    };
  }

  return undefined;
}

function normalizeToolCalls(value: unknown): unknown[] {
  const calls = Array.isArray(value)
    ? value
    : value === undefined || value === null
      ? []
      : [value];
  return calls
    .filter((call) => call !== null && typeof call === "object")
    .map((call) => {
      const record =
        typeof call === "object" && call !== null
          ? (call as Record<string, unknown>)
          : {};
      const name = stringValue(record.name) ?? stringValue(record.toolName);
      const args = record.args !== undefined ? record.args : record.arguments;
      return compactRecord({
        id: stringValue(record.callId) ?? stringValue(record.call_id),
        type: "function",
        function: compactRecord({
          name,
          arguments:
            args === undefined
              ? undefined
              : typeof args === "string"
                ? args
                : safeJson(args),
        }),
      });
    });
}

function providerName(attrs: MutableAttributes): string | undefined {
  const direct = stringAttr(attrs, ATTR_GEN_AI_PROVIDER_NAME);
  if (direct) {
    return direct;
  }
  const llmMetrics = parseMaybeJson(attrs[LK_LLM_METRICS]) as
    Record<string, unknown> | undefined;
  const ttsMetrics = parseMaybeJson(attrs[LK_TTS_METRICS]) as
    Record<string, unknown> | undefined;
  return (
    stringValue(
      (llmMetrics?.metadata as Record<string, unknown> | undefined)
        ?.modelProvider,
    ) ??
    stringValue(
      (ttsMetrics?.metadata as Record<string, unknown> | undefined)
        ?.modelProvider,
    )
  );
}

function normalizeProvider(provider: string): string {
  return provider.toLowerCase().replace(/\s+/g, "_");
}

function setAttr(
  span: Span | undefined,
  attrs: MutableAttributes,
  key: string,
  value: unknown,
): void {
  if (value === undefined || value === null) {
    return;
  }
  attrs[key] = value;
  span?.setAttribute(key, value as any);
}

function stringAttr(attrs: MutableAttributes, key: string): string | undefined {
  return stringValue(attrs[key]);
}

function stringValue(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return undefined;
}

function numberAttr(attrs: MutableAttributes, key: string): number | undefined {
  return asNumber(attrs[key]);
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
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

function contentToText(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    if (value.every((part) => typeof part === "string")) return value.join("");
    return safeJson(value);
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (typeof record.text === "string") {
      return record.text;
    }
    if (typeof record.transcript === "string") {
      return record.transcript;
    }
  }
  return value === undefined || value === null ? "" : safeJson(value);
}

function compactRecord(
  record: Record<string, unknown>,
): Record<string, unknown> {
  const compacted: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value === undefined) {
      continue;
    }
    if (
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      Object.keys(value).length === 0
    ) {
      continue;
    }
    compacted[key] = value;
  }
  return compacted;
}

function hasEntityValue(value: unknown): boolean {
  if (value === undefined || value === null || value === "") {
    return false;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value).length > 0;
  }
  return true;
}

/** Remove LiveKit transport attributes after their canonical values are built. */
export function stripLiveKitRawAttributes(attrs: MutableAttributes): void {
  for (const key of Object.keys(attrs)) {
    if (
      key.startsWith(LIVEKIT_RAW_ATTRIBUTE_PREFIX) ||
      key === LIVEKIT_SPAN_NAME_ATTRIBUTE ||
      key === LANGFUSE_COMPLETION_START_TIME
    ) {
      delete attrs[key];
    }
  }
}

export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? null);
  } catch {
    return JSON.stringify(String(value));
  }
}

/** Bridge native release formats on the export copy, leaving vendor objects intact. */
function normalizeNativeContent(attrs: MutableAttributes): void {
  for (const key of [
    LK_CHAT_CTX,
    LK_USER_INPUT,
    LK_INSTRUCTIONS,
    LK_PARTICIPANT_IDENTITY,
    LK_ROOM_NAME,
    LK_RESPONSE_TEXT,
    LK_RESPONSE_FUNCTION_CALLS,
    LK_FUNCTION_TOOL_ARGS,
    LK_FUNCTION_TOOL_OUTPUT,
    LK_TTS_INPUT_TEXT,
    LK_USER_TRANSCRIPT,
    LK_AMD_TRANSCRIPT,
  ]) {
    const modern = `lk.pii.${key.slice(3)}`;
    if (attrs[key] === undefined && attrs[modern] !== undefined)
      attrs[key] = attrs[modern];
  }
  const output = parseMaybeJson(attrs[ATTR_GEN_AI_OUTPUT_MESSAGES]);
  if (Array.isArray(output)) {
    const parts = output.flatMap((message: any) =>
      Array.isArray(message.parts) ? message.parts : [],
    );
    const texts = parts
      .filter(
        (part: any) => part.type === "text" && typeof part.content === "string",
      )
      .map((part: any) => part.content);
    if (attrs[LK_RESPONSE_TEXT] === undefined && texts.length)
      attrs[LK_RESPONSE_TEXT] = texts.join("");
    if (attrs[LK_RESPONSE_FUNCTION_CALLS] === undefined) {
      const calls = parts
        .filter((part: any) => part.type === "tool_call")
        .map((part: any) => ({
          callId: part.id,
          name: part.name,
          args: part.arguments,
        }));
      if (calls.length) attrs[LK_RESPONSE_FUNCTION_CALLS] = safeJson(calls);
    }
  }
  if (
    attrs[LK_CHAT_CTX] === undefined &&
    (attrs[ATTR_GEN_AI_INPUT_MESSAGES] !== undefined ||
      attrs[ATTR_GEN_AI_SYSTEM_INSTRUCTIONS] !== undefined)
  )
    attrs[LK_CHAT_CTX] = safeJson({ items: modernPromptItems(attrs) });
}

function modernPromptItems(attrs: MutableAttributes): LiveKitChatItem[] {
  const messages = parseMaybeJson(attrs[ATTR_GEN_AI_INPUT_MESSAGES]);
  const instructions = parseMaybeJson(attrs[ATTR_GEN_AI_SYSTEM_INSTRUCTIONS]);
  const items: LiveKitChatItem[] = [];
  if (Array.isArray(instructions))
    items.push({
      type: "message",
      role: "system",
      content: instructions.map((part: any) => part.content ?? part),
    });
  if (Array.isArray(messages))
    for (const message of messages) {
      const parts = Array.isArray(message.parts) ? message.parts : [];
      const content = parts
        .filter(
          (part: any) =>
            part.type !== "tool_call" && part.type !== "tool_call_response",
        )
        .map((part: any) => (part.type === "text" ? part.content : part));
      if (content.length)
        items.push({ type: "message", role: message.role, content });
      for (const part of parts) {
        if (part.type === "tool_call")
          items.push({
            type: "function_call",
            name: part.name,
            callId: part.id,
            args: part.arguments,
          });
        if (part.type === "tool_call_response")
          items.push({
            type: "function_call_output",
            callId: part.id,
            output: part.response,
          });
      }
    }
  return items;
}
