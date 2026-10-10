import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  TraceFlags,
  type Context,
  type Span,
} from "@opentelemetry/api";
import {
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
  ATTR_ERROR_TYPE,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  INSTRUMENTATION_NAME,
  PACKAGE_VERSION,
  COHERE_SYSTEM,
  RESPAN_LOG_METHOD_TS_TRACING,
} from "./_constants.js";
import { logTypeForOperation, requestTypeForOperation } from "./_translator.js";
import { data, guard, refresh, snapshot, type Policy } from "./_privacy.js";
import {
  cohereContentToString,
  isRecord,
  normalizeRole,
  type SpanAttributes as Attributes,
} from "./_utils.js";
export type CohereOperation =
  | "chat"
  | "chatStream"
  | "generate"
  | "generateStream"
  | "embed"
  | "rerank"
  | "parse";
export type CohereApiVersion = "v1" | "v2";
export interface OperationConfig {
  operation: CohereOperation;
  apiVersion: CohereApiVersion;
  streaming: boolean;
}
export interface SpanRecord {
  span: Span;
  context: Context;
  policy: Policy;
  attrs: Attributes;
  ended: boolean;
  admitted: boolean;
  prompts: Attributes;
}
export interface StreamState {
  events: unknown[];
  textParts: string[];
  toolCallParts: unknown[];
  finalResponse?: any;
  content?: Record<number, any>;
  tools?: Record<number, any>;
  usage?: any;
}
function json(value: unknown): string {
  return JSON.stringify(value) ?? "";
}
function put(attrs: Attributes, key: string, value: unknown): void {
  if (value !== undefined && value !== null) attrs[key] = value;
}
function number(attrs: Attributes, key: string, value: unknown): void {
  if (typeof value === "number" && Number.isFinite(value)) attrs[key] = value;
}
function spanName(operation: CohereOperation): string {
  return `cohere.${operation.replace("Stream", "")}`;
}
function requestParams(attrs: Attributes, request: any): void {
  for (const [field, key] of [
    ["maxTokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS],
    ["temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE],
    ["p", SpanAttributes.LLM_REQUEST_TOP_P],
    ["k", SpanAttributes.LLM_TOP_K],
    ["frequencyPenalty", SpanAttributes.LLM_FREQUENCY_PENALTY],
    ["presencePenalty", SpanAttributes.LLM_PRESENCE_PENALTY],
  ])
    number(attrs, key, data(request, field));
}
function toolDefinition(tool: any): any {
  if (!isRecord(tool)) return tool;
  if (tool.type === "function") return tool;
  if (tool.name === undefined) return tool;
  const properties: any = {},
    required: string[] = [];
  for (const [key, value] of Object.entries(tool.parameterDefinitions ?? {})) {
    properties[key] = value;
    if ((value as any)?.required === true) required.push(key);
  }
  return {
    type: "function",
    function: {
      name: tool.name,
      ...(tool.description !== undefined
        ? { description: tool.description }
        : {}),
      parameters: {
        type: "object",
        properties,
        ...(required.length ? { required } : {}),
      },
    },
  };
}
function toolCall(call: any): any {
  if (!isRecord(call)) return call;
  if (call.function !== undefined) return call;
  return {
    ...(call.id !== undefined ? { id: call.id } : {}),
    type: "function",
    function: {
      name: call.name,
      arguments: json(call.parameters ?? call.arguments ?? {}),
    },
  };
}
function messages(config: OperationConfig, request: any): any[] {
  if (config.apiVersion === "v2") return request?.messages ?? [];
  const history = (request?.chatHistory ?? []).map((m: any) => ({
    ...m,
    role: normalizeRole(m.role),
    content: m.message ?? m.content,
    ...(m.toolCalls !== undefined ? { toolCalls: m.toolCalls } : {}),
  }));
  if (request?.preamble !== undefined)
    history.unshift({ role: "system", content: request.preamble });
  if (request?.message !== undefined)
    history.push({ role: "user", content: request.message });
  if (request?.toolResults !== undefined)
    history.push({ role: "tool", content: request.toolResults });
  return history;
}
function indexed(attrs: Attributes, prefix: string, values: any[]): void {
  values.forEach((message, index) => {
    if (!isRecord(message)) return;
    put(attrs, `${prefix}.${index}.role`, normalizeRole(message.role));
    if (message.content !== undefined)
      attrs[`${prefix}.${index}.content`] = cohereContentToString(
        message.content,
      );
    const calls = message.toolCalls ?? message.tool_calls;
    if (calls !== undefined)
      attrs[`${prefix}.${index}.tool_calls`] = json(
        Array.isArray(calls) ? calls.map(toolCall) : calls,
      );
    put(
      attrs,
      `${prefix}.${index}.tool_call_id`,
      message.toolCallId ?? message.tool_call_id,
    );
  });
}
function input(config: OperationConfig, request: any): unknown {
  if (config.operation.startsWith("chat")) return messages(config, request);
  if (config.operation.startsWith("generate")) return request?.prompt;
  if (config.operation === "embed")
    return request?.inputs ?? request?.texts ?? request?.images;
  return request;
}
export function buildStartAttributes(
  config: OperationConfig,
  request: any,
  traceContent: boolean,
): Attributes {
  if (config.operation === "parse" || config.operation === "rerank")
    return {
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: spanName(config.operation),
      [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
      [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
      [RespanSpanAttributes.RESPAN_LOG_TYPE]:
        config.operation === "parse" ? "tool" : "task",
      ...(traceContent
        ? {
            [SpanAttributes.TRACELOOP_ENTITY_INPUT]: json(
              config.operation === "parse"
                ? { name: "cohere.parse", arguments: snapshot(request) }
                : snapshot(request),
            ),
          }
        : {}),
    };
  const attrs: Attributes = {
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: spanName(config.operation),
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: "",
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logTypeForOperation(
      config.operation,
    ),
    [SpanAttributes.LLM_SYSTEM]: COHERE_SYSTEM,
    [SpanAttributes.LLM_REQUEST_TYPE]: requestTypeForOperation(
      config.operation,
    ),
  };
  const model = data(request, "model");
  if (typeof model === "string")
    attrs[SpanAttributes.LLM_REQUEST_MODEL] = model;
  requestParams(attrs, request);
  if (traceContent) {
    const copied = snapshot(request);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = json(input(config, copied));
    if (config.operation.startsWith("chat"))
      indexed(attrs, SpanAttributes.LLM_PROMPTS, messages(config, copied));
    if (config.operation.startsWith("generate") && copied?.prompt !== undefined)
      indexed(attrs, SpanAttributes.LLM_PROMPTS, [
        { role: "user", content: copied.prompt },
      ]);
    if (copied?.tools !== undefined)
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = json(
        copied.tools.map(toolDefinition),
      );
  }
  return attrs;
}
function usageAttributes(attrs: Attributes, value: any): void {
  const usage = data(value, "usage") ?? data(value, "meta");
  const tokens = data(usage, "tokens") ?? data(usage, "billedUnits");
  const old = data(value, "token_count");
  const input =
    data(tokens, "inputTokens") ??
    data(tokens, "input_tokens") ??
    data(old, "prompt_tokens");
  const output =
    data(tokens, "outputTokens") ??
    data(tokens, "output_tokens") ??
    data(old, "response_tokens");
  number(attrs, ATTR_GEN_AI_USAGE_INPUT_TOKENS, input);
  number(attrs, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input);
  number(attrs, ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, output);
  number(attrs, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output);
  number(
    attrs,
    SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    data(tokens, "totalTokens") ??
      data(tokens, "total_tokens") ??
      data(old, "total_tokens"),
  );
  number(
    attrs,
    "llm.usage.cache_read_input_tokens",
    data(usage, "cachedTokens"),
  );
}
function outputAttributes(
  attrs: Attributes,
  config: OperationConfig,
  result: any,
): void {
  if (config.operation === "parse") {
    if (result !== undefined)
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json(result);
    return;
  }
  if (config.operation.startsWith("chat")) {
    const message =
      config.apiVersion === "v2"
        ? result?.message
        : {
            role: "assistant",
            content: result?.text,
            ...(result?.toolCalls !== undefined
              ? { toolCalls: result.toolCalls }
              : {}),
          };
    if (message !== undefined) {
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json([message]);
      indexed(attrs, SpanAttributes.LLM_COMPLETIONS, [message]);
    }
  } else if (config.operation.startsWith("generate")) {
    const generations =
      result?.generations ??
      (result?.text !== undefined ? [{ text: result.text }] : undefined);
    if (generations !== undefined) {
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json(generations);
      indexed(
        attrs,
        SpanAttributes.LLM_COMPLETIONS,
        generations.map((g: any) => ({ role: "assistant", content: g.text })),
      );
    }
  } else if (config.operation === "embed") {
    if (result?.embeddings !== undefined)
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json(result.embeddings);
  } else if (result?.results !== undefined)
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = json(result.results);
}
export function applySuccessAttributes(
  attrs: Attributes,
  config: OperationConfig,
  result: any,
): Attributes {
  if (config.operation !== "rerank" && config.operation !== "parse")
    usageAttributes(attrs, result);
  outputAttributes(attrs, config, snapshot(result));
  return attrs;
}
function metadata(record: SpanRecord, value: Record<string, unknown>): void {
  let inherited: any = {};
  const existing = (record.span as any).attributes?.[
    RespanSpanAttributes.RESPAN_METADATA
  ];
  if (typeof existing === "string") {
    try {
      inherited = JSON.parse(existing);
    } catch {
      /* Ignore malformed inherited metadata. */
    }
  }
  record.span.setAttribute(
    RespanSpanAttributes.RESPAN_METADATA,
    json({ ...(isRecord(inherited) ? inherited : {}), ...value }),
  );
}
export function startSpanRecord(
  config: OperationConfig,
  request: any,
  policy: Policy,
  parentContext: Context = context.active(),
): SpanRecord {
  const attrs = buildStartAttributes(config, request, false);
  const span = trace
    .getTracer(INSTRUMENTATION_NAME, PACKAGE_VERSION)
    .startSpan(
      spanName(config.operation),
      { kind: SpanKind.CLIENT, attributes: attrs },
      parentContext,
    );
  const admitted =
    span.isRecording() &&
    (span.spanContext().traceFlags & TraceFlags.SAMPLED) !== 0;
  policy.emit &&= admitted;
  refresh(policy);
  guard(span, policy);
  const record: SpanRecord = {
    span,
    policy,
    attrs,
    ended: false,
    admitted,
    prompts: {},
    context: trace.setSpan(parentContext, span),
  };
  if (admitted && policy.inputs) {
    const captured = buildStartAttributes(config, request, true);
    // Canonical payloads and usage must survive the standard OTel attribute budget.
    // Indexed projections are attempted only after all canonical fields at completion.
    for (const key of Object.keys(captured)) {
      if (key.startsWith(`${SpanAttributes.LLM_PROMPTS}.`)) {
        record.prompts[key] = captured[key];
        delete captured[key];
      }
    }
    span.setAttributes(captured);
    if (policy.outputs && config.operation !== "parse")
      metadata(record, { cohere_request: snapshot(request) });
  }
  return record;
}
export function emitSpanRecord(
  config: OperationConfig,
  record: SpanRecord,
  result: any,
  error?: unknown,
): void {
  if (record.ended) return;
  record.ended = true;
  try {
    refresh(record.policy);
    if (record.admitted) {
      const attrs: Attributes = {};
      if (error === undefined) {
        if (config.operation !== "rerank" && config.operation !== "parse")
          usageAttributes(attrs, result);
        if (record.policy.outputs)
          outputAttributes(attrs, config, snapshot(result));
        if (
          record.policy.inputs &&
          record.policy.outputs &&
          result !== undefined &&
          config.operation !== "parse"
        )
          metadata(record, { cohere_response: snapshot(result) });
      } else {
        const status = data(error, "statusCode") ?? data(error, "status");
        if (
          typeof status === "number" &&
          Number.isInteger(status) &&
          status >= 100 &&
          status <= 599
        )
          number(attrs, ATTR_HTTP_RESPONSE_STATUS_CODE, status);
        if (record.policy.inputs && record.policy.outputs) {
          const message = data(error, "message");
          const name = data(error, "name");
          if (typeof name === "string") attrs[ATTR_ERROR_TYPE] = name;
          record.span.setStatus({
            code: SpanStatusCode.ERROR,
            ...(typeof message === "string" ? { message } : {}),
          });
          if (typeof message === "string")
            record.span.addEvent("exception", { "exception.message": message });
        } else record.span.setStatus({ code: SpanStatusCode.ERROR });
      }
      record.span.setAttributes(attrs);
      if (record.policy.inputs) record.span.setAttributes(record.prompts);
    }
  } catch {
    /* The SDK result or error always wins. */
  } finally {
    record.span.end();
  }
}
export function createStreamState(): StreamState {
  return {
    events: [],
    textParts: [],
    toolCallParts: [],
    content: {},
    tools: {},
  };
}
export function captureStreamEvent(
  state: StreamState,
  event: any,
  policy?: Policy,
): void {
  if (policy) {
    refresh(policy);
    if (!policy.emit) return;
  }
  const type = data(event, "type") ?? data(event, "eventType");
  const delta = data(event, "delta");
  if (type === "message-end")
    state.usage = { usage: snapshotUsage(data(delta, "usage")) };
  if (type === "stream-end")
    state.usage = {
      meta: snapshotUsage(data(data(event, "response"), "meta")),
    };
  if (policy && !policy.outputs) return;
  const copied = snapshot(event);
  if (!copied) return;
  state.events.push(copied);
  if (type === "stream-end" && copied.response !== undefined)
    state.finalResponse = copied.response;
  if (type === "text-generation" && typeof copied.text === "string")
    state.textParts.push(copied.text);
  if (type === "tool-calls-generation" && copied.toolCalls !== undefined)
    state.toolCallParts = copied.toolCalls;
  const index = copied.index ?? 0;
  if (type === "content-start")
    state.content![index] = copied.delta?.message?.content ?? {};
  if (type === "content-delta") {
    const content = copied.delta?.message?.content;
    const text = content?.text ?? copied.delta?.text;
    if (typeof text === "string") {
      state.textParts.push(text);
      const item = (state.content![index] ??= { type: "text", text: "" });
      item.text = (item.text ?? "") + text;
    }
  }
  if (type === "tool-call-start")
    state.tools![index] =
      copied.delta?.message?.toolCalls ??
      copied.delta?.message?.tool_calls ??
      {};
  if (type === "tool-call-delta") {
    const call =
      copied.delta?.message?.toolCalls ?? copied.delta?.message?.tool_calls;
    if (call) {
      const stored = (state.tools![index] ??= {});
      if (call.id !== undefined) stored.id = call.id;
      if (call.type !== undefined) stored.type = call.type;
      stored.function ??= {};
      if (call.function?.name !== undefined)
        stored.function.name =
          (stored.function.name ?? "") + call.function.name;
      if (call.function?.arguments !== undefined)
        stored.function.arguments =
          (stored.function.arguments ?? "") + call.function.arguments;
    }
  }
  if (type === "message-end") state.finalResponse = copied.delta;
}
function snapshotUsage(usage: any): any {
  if (!usage) return undefined;
  const out: any = {};
  for (const lane of ["tokens", "billedUnits"]) {
    const source = data(usage, lane);
    const target: any = {};
    for (const field of [
      "inputTokens",
      "outputTokens",
      "totalTokens",
      "searchUnits",
    ]) {
      const value = data(source, field);
      if (typeof value === "number") target[field] = value;
    }
    if (Object.keys(target).length) out[lane] = target;
  }
  const cached = data(usage, "cachedTokens");
  if (typeof cached === "number") out.cachedTokens = cached;
  return out;
}
export function streamResultFromState(
  config: OperationConfig,
  state: StreamState,
): unknown {
  if (config.operation === "generateStream")
    return (
      state.finalResponse ?? {
        generations: state.textParts.length
          ? [{ text: state.textParts.join("") }]
          : [],
        ...state.usage,
      }
    );
  if (config.apiVersion === "v1")
    return (
      state.finalResponse ?? {
        text: state.textParts.join(""),
        toolCalls: state.toolCallParts,
        ...state.usage,
      }
    );
  const content = Object.values(state.content ?? {}),
    tools = Object.values(state.tools ?? {});
  return {
    ...state.finalResponse,
    ...state.usage,
    message: {
      role: "assistant",
      ...(content.length ? { content } : {}),
      ...(tools.length ? { toolCalls: tools } : {}),
    },
  };
}
