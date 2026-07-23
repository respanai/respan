import { context, trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_K,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { buildReadableSpan, getEntityPath, injectSpan, WORKFLOW_NAME_KEY } from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  ASSISTANT_ROLE,
  CANDIDATES_TOKEN_COUNT_KEY,
  GENERATION_CONFIG_KEY,
  INSTRUMENTATION_LIBRARY_NAME,
  PACKAGE_VERSION,
  PROMPT_TOKEN_COUNT_KEY,
  RESPAN_LOG_METHOD_TS_TRACING,
  SYSTEM_INSTRUCTION_KEY,
  TOOLS_KEY,
  TOTAL_TOKEN_COUNT_KEY,
  VERTEXAI_GENERATE_CONTENT_SPAN_NAME,
  VERTEXAI_SYSTEM_NAME,
} from "./_constants.js";
import {
  extractToolCalls,
  extractTools,
  extractUsage,
  formatInput,
  formatOutput,
  normalizeInputMessages,
  safeJson,
  toJsonAttr,
  type VertexAIRequestPayload,
} from "./_translator.js";

const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const LLM_USAGE_TOTAL_TOKENS = SpanAttributes.LLM_USAGE_TOTAL_TOKENS;

function maybeSetAttr(attrs: Record<string, unknown>, key: string, value: unknown): void {
  if (value === undefined || value === null || value === "") return;
  if (Array.isArray(value) && value.length === 0) return;
  attrs[key] = value as any;
}

function numberField(value: unknown, ...names: string[]): number | undefined {
  if (!value || typeof value !== "object") return undefined;
  for (const name of names) {
    const fieldValue = (value as Record<string, unknown>)[name];
    if (typeof fieldValue === "number") return fieldValue;
  }
  return undefined;
}

function generationConfigValue(payload: VertexAIRequestPayload, ...names: string[]): number | undefined {
  return numberField(payload.generationConfig, ...names);
}

function buildInstrumentedReadableSpan(opts: {
  name: string;
  startTimeIso: string;
  attributes: Record<string, unknown>;
  errorMessage?: string;
  statusCode?: number;
}): ReadableSpan {
  const activeSpanContext = trace.getSpan(context.active())?.spanContext();
  const span = buildReadableSpan({
    name: opts.name,
    traceId: activeSpanContext?.traceId,
    parentId: activeSpanContext?.spanId,
    startTimeIso: opts.startTimeIso,
    endTimeIso: new Date().toISOString(),
    attributes: opts.attributes,
    errorMessage: opts.errorMessage,
    statusCode: opts.statusCode,
  }) as ReadableSpan & {
    instrumentationLibrary?: { name: string; version?: string };
  };

  span.instrumentationLibrary = {
    name: INSTRUMENTATION_LIBRARY_NAME,
    version: PACKAGE_VERSION,
  };
  return span;
}

function baseAttrs(spanName: string): Record<string, unknown> {
  const activeContext = context.active();
  const workflowName = activeContext.getValue(WORKFLOW_NAME_KEY) as string | undefined;
  const entityPath = getEntityPath(activeContext);
  const attrs: Record<string, unknown> = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: spanName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath || spanName,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.CHAT,
    [SpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
    [ATTR_GEN_AI_SYSTEM]: VERTEXAI_SYSTEM_NAME,
  };

  if (workflowName) {
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
  }
  return attrs;
}

function setInputAttrs(attrs: Record<string, unknown>, requestPayload: VertexAIRequestPayload): void {
  const messages = normalizeInputMessages(
    requestPayload.contents,
    requestPayload.systemInstruction,
  );
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = formatInput(
    requestPayload.contents,
    requestPayload.systemInstruction,
  );

  messages.forEach((message, index) => {
    const role = message.role;
    const content = message.content;
    maybeSetAttr(attrs, `${ATTR_GEN_AI_PROMPT}.${index}.role`, role);
    if (content !== undefined) {
      attrs[`${ATTR_GEN_AI_PROMPT}.${index}.content`] = toJsonAttr(content);
    }
  });
}

function setRequestAttrs(attrs: Record<string, unknown>, requestPayload: VertexAIRequestPayload): void {
  maybeSetAttr(attrs, ATTR_GEN_AI_REQUEST_MODEL, requestPayload.model);
  maybeSetAttr(
    attrs,
    ATTR_GEN_AI_REQUEST_MAX_TOKENS,
    generationConfigValue(requestPayload, "maxOutputTokens", "max_output_tokens"),
  );
  maybeSetAttr(
    attrs,
    ATTR_GEN_AI_REQUEST_TEMPERATURE,
    generationConfigValue(requestPayload, "temperature"),
  );
  maybeSetAttr(
    attrs,
    ATTR_GEN_AI_REQUEST_TOP_P,
    generationConfigValue(requestPayload, "topP", "top_p"),
  );
  maybeSetAttr(
    attrs,
    ATTR_GEN_AI_REQUEST_TOP_K,
    generationConfigValue(requestPayload, "topK", "top_k"),
  );

  const tools = extractTools(requestPayload.tools);
  if (tools.length > 0) {
    attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(tools);
  }

  setInputAttrs(attrs, requestPayload);

  if (requestPayload.generationConfig !== undefined) {
    attrs[`vertexai.${GENERATION_CONFIG_KEY}`] = safeJson(requestPayload.generationConfig);
  }
  if (requestPayload.toolConfig !== undefined) {
    attrs["vertexai.toolConfig"] = safeJson(requestPayload.toolConfig);
  }
  if (requestPayload.systemInstruction !== undefined) {
    attrs[`vertexai.${SYSTEM_INSTRUCTION_KEY}`] = safeJson(requestPayload.systemInstruction);
  }
  if (requestPayload.tools !== undefined) {
    attrs[`vertexai.${TOOLS_KEY}`] = safeJson(requestPayload.tools);
  }
}

function setOutputAttrs(attrs: Record<string, unknown>, responseOrChunks: unknown): void {
  const output = formatOutput(responseOrChunks);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output;
  attrs[`${GEN_AI_COMPLETION_PREFIX}.role`] = ASSISTANT_ROLE;
  attrs[`${GEN_AI_COMPLETION_PREFIX}.content`] = output;

  const toolCalls = extractToolCalls(responseOrChunks);
  if (toolCalls.length > 0) {
    attrs[`${GEN_AI_COMPLETION_PREFIX}.tool_calls`] = safeJson(toolCalls);
  }

  const usage = extractUsage(responseOrChunks);
  const promptTokens = usage[PROMPT_TOKEN_COUNT_KEY];
  const completionTokens = usage[CANDIDATES_TOKEN_COUNT_KEY];
  const totalTokens = usage[TOTAL_TOKEN_COUNT_KEY];

  if (promptTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = promptTokens;
    attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = promptTokens;
  }
  if (completionTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = completionTokens;
    attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = completionTokens;
  }
  if (totalTokens !== undefined) {
    attrs[LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  }
}

export function buildGenerateContentAttrs(opts: {
  requestPayload: VertexAIRequestPayload;
  responseOrChunks?: unknown;
  spanName?: string;
}): Record<string, unknown> {
  const attrs = baseAttrs(opts.spanName ?? VERTEXAI_GENERATE_CONTENT_SPAN_NAME);
  setRequestAttrs(attrs, opts.requestPayload);
  if (opts.responseOrChunks !== undefined) {
    setOutputAttrs(attrs, opts.responseOrChunks);
  }
  return attrs;
}

export function emitGenerateContentSpan(opts: {
  requestPayload: VertexAIRequestPayload;
  startTimeIso: string;
  responseOrChunks?: unknown;
  spanName?: string;
  errorMessage?: string;
  statusCode?: number;
}): void {
  try {
    const spanName = opts.spanName ?? VERTEXAI_GENERATE_CONTENT_SPAN_NAME;
    const attrs = buildGenerateContentAttrs({
      requestPayload: opts.requestPayload,
      responseOrChunks: opts.responseOrChunks,
      spanName,
    });
    if (opts.errorMessage) {
      attrs["error.message"] = opts.errorMessage;
      attrs["status_code"] = opts.statusCode ?? 500;
    }

    injectSpan(buildInstrumentedReadableSpan({
      name: spanName,
      startTimeIso: opts.startTimeIso,
      attributes: attrs,
      errorMessage: opts.errorMessage,
      statusCode: opts.statusCode,
    }));
  } catch {
    // Instrumentation must never break application code.
  }
}
