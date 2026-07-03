import type { Span } from "@opentelemetry/api";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
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

const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;
const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";

export type MutableAttributes = Record<string, unknown>;

export interface TranslateLiveKitSpanOptions {
  attributes?: MutableAttributes;
  spanName?: string;
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

  if (LIVEKIT_INTERNAL_ACTIVITY_SPAN_NAMES.has(spanName)) {
    return;
  }

  if (!isLiveKitSpan(spanName, attrs)) {
    return;
  }

  const logType = resolveLogType(spanName, attrs);
  setAttr(span, attrs, LIVEKIT_SPAN_NAME_ATTRIBUTE, spanName);
  setAttr(span, attrs, RespanSpanAttributes.RESPAN_LOG_METHOD, LIVEKIT_LOG_METHOD_TS_TRACING);
  setAttr(span, attrs, RespanSpanAttributes.RESPAN_LOG_TYPE, logType);

  const entityName = resolveEntityName(spanName, attrs);
  setAttr(span, attrs, SpanAttributes.TRACELOOP_ENTITY_NAME, entityName);
  setAttr(span, attrs, SpanAttributes.TRACELOOP_ENTITY_PATH, spanName || entityName);
  setAttr(span, attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, safeJson(buildEntityInput(spanName, attrs)));
  setAttr(span, attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, safeJson(buildEntityOutput(spanName, attrs)));

  if (logType === RespanLogType.WORKFLOW) {
    setAttr(span, attrs, SpanAttributes.TRACELOOP_WORKFLOW_NAME, entityName);
  }

  if (logType === RespanLogType.CHAT) {
    addChatAttributes(span, attrs, spanName);
  } else if (logType === RespanLogType.TEXT) {
    addTextModelAttributes(span, attrs);
  }

  for (const key of OFF_CONTRACT_ALIAS_KEYS) {
    delete attrs[key];
  }
}

export function isLiveKitSpan(spanName: string, attrs: MutableAttributes): boolean {
  if (spanName in LIVEKIT_LOG_TYPE_BY_SPAN_NAME) {
    return true;
  }
  return Object.keys(attrs).some((key) => key.startsWith(LIVEKIT_RAW_ATTRIBUTE_PREFIX));
}

export function getSpanAttributes(span: Span): MutableAttributes {
  const maybeAttrs = (span as unknown as { attributes?: MutableAttributes }).attributes;
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

function resolveLogType(spanName: string, attrs: MutableAttributes): RespanLogType {
  if (spanName in LIVEKIT_LOG_TYPE_BY_SPAN_NAME) {
    return LIVEKIT_LOG_TYPE_BY_SPAN_NAME[spanName]!;
  }
  if (attrs[LK_FUNCTION_TOOL_NAME] !== undefined) {
    return RespanLogType.TOOL;
  }
  if (attrs[LK_CHAT_CTX] !== undefined || attrs[LK_LLM_METRICS] !== undefined) {
    return RespanLogType.CHAT;
  }
  if (attrs[LK_TTS_METRICS] !== undefined || attrs[LK_TTS_INPUT_TEXT] !== undefined) {
    return RespanLogType.TEXT;
  }
  return RespanLogType.TASK;
}

function resolveEntityName(spanName: string, attrs: MutableAttributes): string {
  const agentLabel = stringAttr(attrs, LK_AGENT_LABEL) ?? stringAttr(attrs, LK_AGENT_NAME);

  if (spanName === LIVEKIT_SPAN_NAMES.AGENT_SESSION) {
    return agentLabel ? `${agentLabel}.session` : "livekit.agent_session";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.AGENT_TURN) {
    return agentLabel ? `${agentLabel}.turn` : "livekit.agent_turn";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.FUNCTION_TOOL) {
    return stringAttr(attrs, LK_FUNCTION_TOOL_NAME) ?? "livekit.function_tool";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.LLM_NODE || spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST) {
    const provider = providerName(attrs);
    const model = stringAttr(attrs, ATTR_GEN_AI_REQUEST_MODEL);
    if (provider && model) {
      return `${provider}.${model}`;
    }
    return "livekit.llm";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.TTS_NODE || spanName === LIVEKIT_SPAN_NAMES.TTS_REQUEST) {
    return stringAttr(attrs, LK_TTS_LABEL) ?? "livekit.tts";
  }
  if (spanName === LIVEKIT_SPAN_NAMES.USER_TURN) {
    return stringAttr(attrs, LK_PARTICIPANT_IDENTITY) ?? "livekit.user_turn";
  }
  return agentLabel ? `${agentLabel}.${spanName}` : `livekit.${spanName || "span"}`;
}

function buildEntityInput(spanName: string, attrs: MutableAttributes): unknown {
  if (spanName === LIVEKIT_SPAN_NAMES.FUNCTION_TOOL) {
    return {
      name: stringAttr(attrs, LK_FUNCTION_TOOL_NAME),
      arguments: parseMaybeJson(attrs[LK_FUNCTION_TOOL_ARGS]),
      id: stringAttr(attrs, LK_FUNCTION_TOOL_ID),
    };
  }

  if (spanName === LIVEKIT_SPAN_NAMES.LLM_NODE || spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST) {
    return {
      chat_ctx: parseMaybeJson(attrs[LK_CHAT_CTX]),
      function_tools: parseMaybeJson(attrs[LK_FUNCTION_TOOLS]),
      provider_tools: parseMaybeJson(attrs[LK_PROVIDER_TOOLS]),
      tool_sets: parseMaybeJson(attrs[LK_TOOL_SETS]),
    };
  }

  if (spanName === LIVEKIT_SPAN_NAMES.TTS_NODE || spanName === LIVEKIT_SPAN_NAMES.TTS_REQUEST) {
    return {
      text: stringAttr(attrs, LK_TTS_INPUT_TEXT),
      streaming: attrs[LK_TTS_STREAMING],
      label: stringAttr(attrs, LK_TTS_LABEL),
    };
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

function buildEntityOutput(spanName: string, attrs: MutableAttributes): unknown {
  if (spanName === LIVEKIT_SPAN_NAMES.FUNCTION_TOOL) {
    return compactRecord({
      output: parseMaybeJson(attrs[LK_FUNCTION_TOOL_OUTPUT]),
      is_error: attrs[LK_FUNCTION_TOOL_IS_ERROR],
    });
  }

  if (spanName === LIVEKIT_SPAN_NAMES.LLM_NODE || spanName === LIVEKIT_SPAN_NAMES.LLM_REQUEST) {
    return compactRecord({
      text: attrs[LK_RESPONSE_TEXT],
      function_calls: parseMaybeJson(attrs[LK_RESPONSE_FUNCTION_CALLS]),
      ttft: attrs[LK_RESPONSE_TTFT],
      metrics: parseMaybeJson(attrs[LK_LLM_METRICS]),
    });
  }

  if (spanName === LIVEKIT_SPAN_NAMES.TTS_NODE || spanName === LIVEKIT_SPAN_NAMES.TTS_REQUEST) {
    return compactRecord({
      metrics: parseMaybeJson(attrs[LK_TTS_METRICS]),
      ttfb: attrs[LK_RESPONSE_TTFB],
    });
  }

  if (spanName === LIVEKIT_SPAN_NAMES.USER_TURN) {
    return compactRecord({
      transcript: attrs[LK_USER_TRANSCRIPT],
      confidence: attrs[LK_TRANSCRIPT_CONFIDENCE],
      transcription_delay: attrs[LK_TRANSCRIPTION_DELAY],
      end_of_turn_delay: attrs[LK_END_OF_TURN_DELAY],
      provider_request_ids: attrs[LK_PROVIDER_REQUEST_IDS],
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

function addChatAttributes(span: Span, attrs: MutableAttributes, spanName: string): void {
  const provider = providerName(attrs);
  const model = stringAttr(attrs, ATTR_GEN_AI_REQUEST_MODEL);

  if (provider) {
    setAttr(span, attrs, ATTR_GEN_AI_PROVIDER_NAME, provider);
    setAttr(span, attrs, ATTR_GEN_AI_SYSTEM, normalizeProvider(provider));
  }
  if (model) {
    setAttr(span, attrs, ATTR_GEN_AI_REQUEST_MODEL, model);
  }
  setAttr(span, attrs, SpanAttributes.LLM_REQUEST_TYPE, "chat");

  const inputTokens = numberAttr(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS);
  const outputTokens = numberAttr(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS);
  if (inputTokens !== undefined) {
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_PROMPT_TOKENS, inputTokens);
    setAttr(span, attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, inputTokens);
  }
  if (outputTokens !== undefined) {
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_COMPLETION_TOKENS, outputTokens);
    setAttr(span, attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, outputTokens);
  }

  const metrics = parseMaybeJson(attrs[LK_LLM_METRICS]) as Record<string, unknown> | undefined;
  const totalTokens = numberAttr(attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS)
    ?? asNumber(metrics?.totalTokens)
    ?? (inputTokens !== undefined || outputTokens !== undefined
      ? (inputTokens ?? 0) + (outputTokens ?? 0)
      : undefined);
  if (totalTokens !== undefined) {
    setAttr(span, attrs, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, totalTokens);
  }
  const cachedTokens = asNumber(metrics?.promptCachedTokens);
  if (cachedTokens !== undefined) {
    setAttr(span, attrs, LLM_USAGE_CACHE_READ_INPUT_TOKENS, cachedTokens);
  }

  const chatCtx = parseMaybeJson(attrs[LK_CHAT_CTX]) as ChatContextJson | undefined;
  addPromptAttributes(span, attrs, chatCtx);
  addToolDefinitionAttributes(span, attrs);
  addCompletionAttributes(span, attrs, spanName);
}

function addTextModelAttributes(span: Span, attrs: MutableAttributes): void {
  const provider = providerName(attrs);
  const model = stringAttr(attrs, ATTR_GEN_AI_REQUEST_MODEL);
  if (provider) {
    setAttr(span, attrs, ATTR_GEN_AI_PROVIDER_NAME, provider);
    setAttr(span, attrs, ATTR_GEN_AI_SYSTEM, normalizeProvider(provider));
  }
  if (model) {
    setAttr(span, attrs, ATTR_GEN_AI_REQUEST_MODEL, model);
  }
  const metrics = parseMaybeJson(attrs[LK_TTS_METRICS]) as Record<string, unknown> | undefined;
  const inputTokens = asNumber(metrics?.inputTokens);
  const outputTokens = asNumber(metrics?.outputTokens);
  if (inputTokens !== undefined) {
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, inputTokens);
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_PROMPT_TOKENS, inputTokens);
  }
  if (outputTokens !== undefined) {
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, outputTokens);
    setAttr(span, attrs, ATTR_GEN_AI_USAGE_COMPLETION_TOKENS, outputTokens);
  }
}

function addPromptAttributes(span: Span, attrs: MutableAttributes, chatCtx?: ChatContextJson): void {
  const items = Array.isArray(chatCtx?.items) ? chatCtx.items : [];
  let promptIndex = 0;
  for (const item of items) {
    const normalized = normalizePromptItem(item);
    if (!normalized) {
      continue;
    }
    const prefix = `${ATTR_GEN_AI_PROMPT}.${promptIndex}`;
    setAttr(span, attrs, `${prefix}.role`, normalized.role);
    setAttr(span, attrs, `${prefix}.content`, normalized.content);
    if (normalized.tool_calls !== undefined) {
      setAttr(span, attrs, `${prefix}.tool_calls`, safeJson(normalized.tool_calls));
    }
    promptIndex += 1;
  }
}

function addToolDefinitionAttributes(span: Span, attrs: MutableAttributes): void {
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
  setAttr(span, attrs, SpanAttributes.LLM_REQUEST_FUNCTIONS, safeJson(toolDefinitions));
}

function addCompletionAttributes(span: Span, attrs: MutableAttributes, spanName: string): void {
  const responseText = stringAttr(attrs, LK_RESPONSE_TEXT);
  const rawCalls = parseMaybeJson(attrs[LK_RESPONSE_FUNCTION_CALLS]);
  const toolCalls = normalizeToolCalls(rawCalls);

  if (responseText === undefined && toolCalls.length === 0 && spanName !== LIVEKIT_SPAN_NAMES.LLM_NODE) {
    return;
  }

  setAttr(span, attrs, GEN_AI_COMPLETION_ROLE, "assistant");
  setAttr(span, attrs, GEN_AI_COMPLETION_CONTENT, responseText ?? "");
  if (toolCalls.length > 0) {
    setAttr(span, attrs, GEN_AI_COMPLETION_TOOL_CALLS, safeJson(toolCalls));
  }
}

function normalizePromptItem(item: LiveKitChatItem):
  | { role: string; content: string; tool_calls?: unknown[] }
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
    };
  }

  return undefined;
}

function normalizeToolCalls(value: unknown): unknown[] {
  const calls = Array.isArray(value) ? value : value === undefined || value === null ? [] : [value];
  return calls.map((call, index) => {
    const record = typeof call === "object" && call !== null ? call as Record<string, unknown> : {};
    const name = stringValue(record.name) ?? stringValue(record.toolName) ?? "tool";
    const args = record.args ?? record.arguments ?? {};
    return {
      id: stringValue(record.callId) ?? stringValue(record.id) ?? `call_${index}`,
      type: "function",
      function: {
        name,
        arguments: typeof args === "string" ? args : safeJson(args),
      },
    };
  });
}

function providerName(attrs: MutableAttributes): string | undefined {
  const direct = stringAttr(attrs, ATTR_GEN_AI_PROVIDER_NAME);
  if (direct) {
    return direct;
  }
  const llmMetrics = parseMaybeJson(attrs[LK_LLM_METRICS]) as Record<string, unknown> | undefined;
  const ttsMetrics = parseMaybeJson(attrs[LK_TTS_METRICS]) as Record<string, unknown> | undefined;
  return stringValue((llmMetrics?.metadata as Record<string, unknown> | undefined)?.modelProvider)
    ?? stringValue((ttsMetrics?.metadata as Record<string, unknown> | undefined)?.modelProvider);
}

function normalizeProvider(provider: string): string {
  return provider.toLowerCase().replace(/\s+/g, "_");
}

function setAttr(span: Span, attrs: MutableAttributes, key: string, value: unknown): void {
  if (value === undefined || value === null) {
    return;
  }
  attrs[key] = value;
  span.setAttribute(key, value as any);
}

function stringAttr(attrs: MutableAttributes, key: string): string | undefined {
  return stringValue(attrs[key]);
}

function stringValue(value: unknown): string | undefined {
  if (typeof value === "string" && value.length > 0) {
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
    return value.map((part) => contentToText(part)).filter(Boolean).join("");
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

function compactRecord(record: Record<string, unknown>): Record<string, unknown> {
  const compacted: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value === undefined || value === null) {
      continue;
    }
    if (typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0) {
      continue;
    }
    compacted[key] = value;
  }
  return compacted;
}

export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? null);
  } catch {
    return JSON.stringify(String(value));
  }
}
