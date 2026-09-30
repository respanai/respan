import { RespanSpanAttributes } from "@respan/respan-sdk";
import {
  LLMRequestTypeValues,
  SpanAttributes,
} from "@traceloop/ai-semantic-conventions";
import {
  EXA_METADATA_NAMESPACE,
  EXA_SYSTEM,
  LOG_TYPE_BY_FAMILY,
  METADATA_CITATIONS,
  METADATA_COST_TOTAL_USD,
  METADATA_LANGUAGE,
  METADATA_OPERATION,
  METADATA_REQUEST_ID,
  METADATA_RESEARCH_LEGACY,
  METADATA_RESOLVED_SEARCH_TYPE,
  METADATA_RESULT_COUNT,
  METADATA_STREAM,
  METADATA_STREAM_COMPLETED,
  type OperationConfig,
  type OperationFamily,
} from "./_constants.js";
import { isRecord, safeJson, toSerializable, valueAt } from "./_serialization.js";

export const CANONICAL_ATTRS = {
  entityInput: SpanAttributes.TRACELOOP_ENTITY_INPUT,
  entityName: SpanAttributes.TRACELOOP_ENTITY_NAME,
  entityOutput: SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
  entityPath: SpanAttributes.TRACELOOP_ENTITY_PATH,
  llmRequestType: SpanAttributes.LLM_REQUEST_TYPE,
};

export function resolveFamily(
  config: OperationConfig,
  streaming: boolean,
): OperationFamily {
  return streaming && config.streamFamily ? config.streamFamily : config.family;
}

export function buildStartAttributes(params: {
  config: OperationConfig;
  input: Record<string, unknown>;
  captureContent: boolean;
  streaming: boolean;
  hasParent: boolean;
}): Record<string, string | number | boolean> {
  const family = resolveFamily(params.config, params.streaming);
  const attrs: Record<string, string | number | boolean> = {
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: "ts_tracing",
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: LOG_TYPE_BY_FAMILY[family],
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: params.config.entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: params.hasParent
      ? params.config.entityName
      : "",
    [RespanSpanAttributes.RESPAN_METADATA]: safeJson({
      [EXA_METADATA_NAMESPACE]: baseMetadata(params.config, params.streaming),
    }),
  };

  if (params.captureContent) {
    const payload =
      family === "tool"
        ? { name: params.config.entityName, arguments: params.input }
        : params.input;
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(payload);
  }
  if (family === "chat") {
    setChatRequestAttributes(
      attrs,
      params.input,
      params.captureContent,
    );
  }
  return attrs;
}

export function buildSuccessAttributes(params: {
  config: OperationConfig;
  input: Record<string, unknown>;
  result: unknown;
  captureContent: boolean;
  streaming: boolean;
  streamCompleted?: boolean;
}): Record<string, string | number | boolean> {
  const attrs: Record<string, string | number | boolean> = {};
  const result = toSerializable(params.result);
  const metadata = baseMetadata(params.config, params.streaming);
  if (params.streaming) {
    metadata[METADATA_STREAM_COMPLETED] = params.streamCompleted ?? true;
  }
  if (params.captureContent) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(result);
  }

  const family = resolveFamily(params.config, params.streaming);
  if (family === "chat") {
    if (params.captureContent) {
      const answer = answerText(result);
      if (answer) {
        attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.role`] = "assistant";
        attrs[`${SpanAttributes.LLM_COMPLETIONS}.0.content`] = answer;
      }
    }
    if (isRecord(result) && result.model !== undefined && result.model !== null) {
      attrs[SpanAttributes.LLM_REQUEST_MODEL] = String(result.model);
    }
  }

  if (isRecord(result)) {
    const results = result.results;
    if (Array.isArray(results)) metadata[METADATA_RESULT_COUNT] = results.length;
    const requestId = result.requestId ?? result.request_id;
    if (requestId !== undefined) metadata[METADATA_REQUEST_ID] = String(requestId);
    const resolvedType = result.resolvedSearchType ?? result.resolved_search_type;
    if (resolvedType !== undefined) {
      metadata[METADATA_RESOLVED_SEARCH_TYPE] = String(resolvedType);
    }
    const cost = result.costDollars ?? result.cost_dollars;
    if (isRecord(cost) && typeof cost.total === "number") {
      metadata[METADATA_COST_TOTAL_USD] = cost.total;
    }
    if (params.captureContent && Array.isArray(result.citations)) {
      metadata[METADATA_CITATIONS] = toSerializable(result.citations);
    }
  }
  attrs[RespanSpanAttributes.RESPAN_METADATA] = safeJson({
    [EXA_METADATA_NAMESPACE]: metadata,
  });
  return attrs;
}

export function streamResult(chunks: unknown[]): Record<string, unknown> {
  const content: string[] = [];
  const citations: unknown[] = [];
  for (const chunk of chunks) {
    const part = valueAt(chunk, "content");
    if (part !== undefined && part !== null) content.push(String(part));
    const chunkCitations = valueAt(chunk, "citations");
    if (Array.isArray(chunkCitations)) citations.push(...chunkCitations);
  }
  return {
    content: content.join(""),
    citations: toSerializable(citations),
    chunks: toSerializable(chunks),
  };
}

function setChatRequestAttributes(
  attrs: Record<string, string | number | boolean>,
  input: Record<string, unknown>,
  captureContent: boolean,
): void {
  attrs[SpanAttributes.LLM_SYSTEM] = EXA_SYSTEM;
  if (input.model !== undefined && input.model !== null) {
    attrs[SpanAttributes.LLM_REQUEST_MODEL] = String(input.model);
  }
  attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT;
  if (!captureContent) return;
  let index = 0;
  if (input.systemPrompt) {
    attrs[`${SpanAttributes.LLM_PROMPTS}.${index}.role`] = "system";
    attrs[`${SpanAttributes.LLM_PROMPTS}.${index}.content`] = String(
      input.systemPrompt,
    );
    index += 1;
  }
  if (input.query !== undefined) {
    attrs[`${SpanAttributes.LLM_PROMPTS}.${index}.role`] = "user";
    attrs[`${SpanAttributes.LLM_PROMPTS}.${index}.content`] = String(input.query);
  }
}

function answerText(value: unknown): string | undefined {
  if (!isRecord(value)) return undefined;
  const answer = value.answer ?? value.content ?? value.output;
  if (answer === undefined || answer === null) return undefined;
  return typeof answer === "string" ? answer : safeJson(answer);
}

function baseMetadata(
  config: OperationConfig,
  streaming: boolean,
): Record<string, unknown> {
  const metadata: Record<string, unknown> = {
    [METADATA_OPERATION]: config.operation,
    [METADATA_LANGUAGE]: "typescript",
    [METADATA_STREAM]: streaming,
  };
  if (config.legacyResearch) metadata[METADATA_RESEARCH_LEGACY] = true;
  return metadata;
}
