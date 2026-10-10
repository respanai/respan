import type { AnyExportedSpan } from "@mastra/core/observability";
import type { Attributes } from "@opentelemetry/api";
import { isProxy } from "node:util/types";
import { ATTR_ERROR_TYPE } from "@opentelemetry/semantic-conventions";
import {
  ATTR_GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

/** Read data properties without invoking payload getters or proxy traps. */
export function data(value: unknown, key: PropertyKey): any {
  if (!value || typeof value !== "object" || isProxy(value)) return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor && "value" in descriptor ? descriptor.value : undefined;
}

function copy(value: unknown, seen = new WeakSet<object>()): any {
  if (value === null || typeof value === "string" || typeof value === "boolean")
    return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "bigint") return BigInt.prototype.toString.call(value);
  if (!value || typeof value !== "object" || isProxy(value) || seen.has(value))
    return undefined;
  if (value instanceof Date) return Date.prototype.toISOString.call(value);
  seen.add(value);
  try {
    if (Array.isArray(value))
      return Array.from(
        { length: value.length },
        (_, i) => copy(data(value, String(i)), seen) ?? null,
      );
    const result: Record<string, unknown> = {};
    for (const [key, descriptor] of Object.entries(
      Object.getOwnPropertyDescriptors(value),
    )) {
      if (descriptor.enumerable && "value" in descriptor) {
        const item = copy(descriptor.value, seen);
        if (item !== undefined)
          Object.defineProperty(result, key, { value: item, enumerable: true });
      }
    }
    return result;
  } finally {
    seen.delete(value);
  }
}

function json(value: unknown): string | undefined {
  try {
    return JSON.stringify(copy(value));
  } catch {
    return undefined;
  }
}
function setJson(attrs: Attributes, key: string, value: unknown): void {
  const serialized = json(value);
  if (serialized !== undefined) attrs[key] = serialized;
}
function numeric(attrs: Attributes, key: string, value: unknown): void {
  if (typeof value === "number" && Number.isFinite(value)) attrs[key] = value;
}

export function logType(type: string): string {
  if (type === "agent_run") return RespanLogType.AGENT;
  if (["model_generation", "model_step", "model_inference"].includes(type))
    return RespanLogType.CHAT;
  if (type === "rag_embedding") return RespanLogType.EMBEDDING;
  if (
    [
      "tool_call",
      "mcp_tool_call",
      "client_tool_call",
      "provider_tool_call",
    ].includes(type)
  )
    return RespanLogType.TOOL;
  if (type === "workflow_run") return RespanLogType.WORKFLOW;
  if (["scorer_run", "scorer_step"].includes(type))
    return RespanLogType.GUARDRAIL;
  return RespanLogType.TASK;
}

export function admissionAttributes(span: AnyExportedSpan): Attributes {
  const attrs: Attributes = {
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType(span.type),
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: span.entityName || span.name,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
  };
  if (span.isRootSpan || ["agent_run", "workflow_run"].includes(span.type))
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] =
      span.entityName || span.name;
  if (
    logType(span.type) !== RespanLogType.CHAT &&
    span.type !== "rag_embedding"
  )
    return attrs;
  const raw = data(span, "attributes");
  const model = data(raw, "model");
  const provider = data(raw, "provider");
  if (typeof model === "string" && model)
    attrs[SpanAttributes.LLM_REQUEST_MODEL] = model;
  if (typeof provider === "string" && provider)
    attrs[SpanAttributes.LLM_SYSTEM] = provider;
  if (
    logType(span.type) === RespanLogType.CHAT ||
    span.type === "rag_embedding"
  )
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = logType(span.type);
  return attrs;
}

export function scalarAttributes(span: AnyExportedSpan): Attributes {
  const attrs = admissionAttributes(span);
  if (
    logType(span.type) !== RespanLogType.CHAT &&
    span.type !== "rag_embedding"
  )
    return attrs;
  const raw = data(span, "attributes");
  const responseModel = data(raw, "responseModel");
  if (typeof responseModel === "string" && responseModel)
    attrs[SpanAttributes.LLM_RESPONSE_MODEL] = responseModel;
  const params = data(raw, "parameters");
  numeric(
    attrs,
    SpanAttributes.LLM_REQUEST_TEMPERATURE,
    data(params, "temperature"),
  );
  numeric(
    attrs,
    SpanAttributes.LLM_REQUEST_MAX_TOKENS,
    data(params, "maxOutputTokens"),
  );
  numeric(attrs, SpanAttributes.LLM_REQUEST_TOP_P, data(params, "topP"));
  numeric(attrs, SpanAttributes.LLM_TOP_K, data(params, "topK"));
  numeric(
    attrs,
    SpanAttributes.LLM_FREQUENCY_PENALTY,
    data(params, "frequencyPenalty"),
  );
  numeric(
    attrs,
    SpanAttributes.LLM_PRESENCE_PENALTY,
    data(params, "presencePenalty"),
  );
  numeric(
    attrs,
    ATTR_GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    data(raw, "dimensions"),
  );
  const usage = data(raw, "usage") ?? data(raw, "internalUsage");
  const input = data(usage, "inputTokens") ?? data(usage, "promptTokens");
  const output = data(usage, "outputTokens") ?? data(usage, "completionTokens");
  numeric(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, input);
  numeric(attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input);
  numeric(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, output);
  numeric(attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output);
  numeric(
    attrs,
    SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    data(usage, "totalTokens"),
  );
  numeric(
    attrs,
    ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    data(data(usage, "inputDetails"), "cacheRead"),
  );
  numeric(
    attrs,
    ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    data(data(usage, "inputDetails"), "cacheWrite"),
  );
  const finish = data(raw, "finishReason");
  if (typeof finish === "string")
    attrs[ATTR_GEN_AI_RESPONSE_FINISH_REASONS] = [finish];
  const responseId = data(raw, "responseId");
  if (typeof responseId === "string")
    attrs[ATTR_GEN_AI_RESPONSE_ID] = responseId;
  const errorName = data(span.errorInfo, "name");
  if (typeof errorName === "string") attrs[ATTR_ERROR_TYPE] = errorName;
  return attrs;
}

function toolCalls(value: unknown): any[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((call) => {
    const fn = data(call, "function");
    const name =
      data(fn, "name") ?? data(call, "toolName") ?? data(call, "name");
    if (typeof name !== "string") return [];
    const args =
      data(fn, "arguments") ??
      data(call, "args") ??
      data(call, "arguments") ??
      data(call, "input");
    const id = data(call, "toolCallId") ?? data(call, "id");
    return [
      {
        ...(typeof id === "string" ? { id } : {}),
        type: "function",
        function: {
          name,
          ...(args !== undefined
            ? { arguments: typeof args === "string" ? args : json(args) }
            : {}),
        },
      },
    ];
  });
}
function messages(value: unknown): any[] {
  if (typeof value === "string") return [{ role: "user", content: value }];
  const input = data(value, "messages") ?? value;
  if (!Array.isArray(input)) return [];
  return input.flatMap((item) => {
    const role = data(item, "role");
    if (typeof role !== "string") return [];
    const content = data(item, "content");
    const calls = toolCalls(
      data(item, "tool_calls") ?? data(item, "toolCalls"),
    );
    if (Array.isArray(content))
      for (const part of content) {
        if (data(part, "type") === "tool-call")
          calls.push(...toolCalls([part]));
      }
    return [
      { ...copy(item), role, ...(calls.length ? { tool_calls: calls } : {}) },
    ];
  });
}

export function contentAttributes(span: AnyExportedSpan): Attributes {
  const attrs: Attributes = {};
  // Keep complete native envelopes, including scalar and null results.
  if (logType(span.type) === RespanLogType.TOOL) {
    if (span.input !== undefined)
      setJson(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, {
        name: span.entityName || span.name,
        arguments: span.input,
      });
    const toolCallId = data(span.attributes, "toolCallId");
    if (typeof toolCallId === "string")
      attrs[ATTR_GEN_AI_TOOL_CALL_ID] = toolCallId;
  } else setJson(attrs, SpanAttributes.TRACELOOP_ENTITY_INPUT, span.input);
  setJson(attrs, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, span.output);
  if (logType(span.type) === RespanLogType.CHAT) {
    messages(span.input).forEach((message, index) => {
      const prefix = `${SpanAttributes.LLM_PROMPTS}.${index}`;
      attrs[`${prefix}.role`] = message.role;
      if (message.content !== undefined)
        attrs[`${prefix}.content`] =
          typeof message.content === "string"
            ? message.content
            : (json(message.content) ?? "");
      if (message.tool_calls)
        setJson(attrs, `${prefix}.tool_calls`, message.tool_calls);
    });
    if (span.output !== undefined) {
      const prefix = `${SpanAttributes.LLM_COMPLETIONS}.0`;
      attrs[`${prefix}.role`] = "assistant";
      const content =
        data(span.output, "text") ??
        data(span.output, "content") ??
        span.output;
      attrs[`${prefix}.content`] =
        typeof content === "string" ? content : (json(content) ?? "");
      const calls = toolCalls(
        data(span.output, "toolCalls") ?? data(span.output, "tool_calls"),
      );
      if (calls.length) setJson(attrs, `${prefix}.tool_calls`, calls);
    }
    const raw = span.attributes;
    const definitions = data(raw, "tools");
    const available = data(raw, "availableTools");
    const tools = Array.isArray(definitions) ? definitions : available;
    if (Array.isArray(tools)) {
      const normalized = tools.flatMap((tool) => {
        if (typeof tool === "string")
          return [{ type: "function", function: { name: tool } }];
        const name = data(tool, "name") ?? data(tool, "id");
        if (typeof name !== "string") return [];
        const fn: Record<string, unknown> = { name };
        const description = data(tool, "description");
        const parameters =
          data(tool, "parameters") ?? data(tool, "inputSchema");
        if (description !== undefined) fn.description = copy(description);
        if (parameters !== undefined) fn.parameters = copy(parameters);
        return [{ type: data(tool, "type") ?? "function", function: fn }];
      });
      setJson(attrs, SpanAttributes.LLM_REQUEST_FUNCTIONS, normalized);
    }
  }
  return attrs;
}

export function metadataAttributes(
  span: AnyExportedSpan,
  existing: unknown,
): Attributes {
  let inherited: Record<string, unknown> = {};
  if (typeof existing === "string")
    try {
      const parsed = JSON.parse(existing);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
        inherited = parsed;
    } catch {}
  const metadata = copy(span.metadata);
  const merged = {
    ...inherited,
    ...(metadata && typeof metadata === "object" ? metadata : {}),
    mastra_span_type: span.type,
    mastra_span_id: span.id,
    mastra_trace_id: span.traceId,
  };
  const externalParent =
    data(span, "externalParentSpanId") ??
    (span.isRootSpan ? span.parentSpanId : undefined);
  if (typeof externalParent === "string")
    merged.mastra_external_parent_span_id = externalParent;
  if (span.entityId) merged.mastra_entity_id = span.entityId;
  if (span.entityName) merged.mastra_entity_name = span.entityName;
  if (span.entityType) merged.mastra_entity_type = span.entityType;
  if (span.tags) merged.mastra_tags = copy(span.tags);
  const attrs: Attributes = {
    [RespanSpanAttributes.RESPAN_METADATA]: JSON.stringify(merged),
  };
  const request = data(span, "requestContext");
  const customer =
    data(request, "customer_identifier") ?? data(request, "userId");
  const thread =
    data(request, "thread_identifier") ?? data(request, "threadId");
  if (typeof customer === "string" || typeof customer === "number")
    attrs[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID] = String(customer);
  if (typeof thread === "string" || typeof thread === "number")
    attrs[RespanSpanAttributes.RESPAN_THREADS_ID] = String(thread);
  return attrs;
}
