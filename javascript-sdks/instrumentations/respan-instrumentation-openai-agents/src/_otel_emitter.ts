/**
 * Emit OpenAI Agents SDK spans as OTEL ReadableSpan objects.
 *
 * Each per-type emitter converts an OpenAI Agents SDK Trace/Span into a
 * ReadableSpan with traceloop.* and gen_ai.* attributes, then injects it
 * into the OTEL pipeline via the active TracerProvider's span processor.
 */

import { trace, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { hrTime, hrTimeDuration } from "@opentelemetry/core";
import { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import type { Span, Trace } from "@openai/agents";

const PACKAGE_VERSION = "1.0.3";

function safeJson(obj: any): string {
  try {
    return JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value,
    );
  } catch {
    return String(obj);
  }
}

function toSerializableValue(value: any): any {
  if (value === null || value === undefined) return undefined;
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
  try {
    return JSON.parse(
      JSON.stringify(value, (_key, innerValue) =>
        typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
      ),
    );
  } catch {
    // Fall back to recursive structural cloning below.
  }
  if (Array.isArray(value)) {
    return value.map((item) => toSerializableValue(item));
  }
  if (typeof value === "object") {
    if (typeof value.toJSON === "function") {
      try {
        return toSerializableValue(value.toJSON());
      } catch {
        // Ignore and continue to shallow recursive copy.
      }
    }
    const normalized: Record<string, unknown> = {};
    Object.entries(value as Record<string, unknown>).forEach(([key, itemValue]) => {
      normalized[key] = toSerializableValue(itemValue);
    });
    return normalized;
  }
  return String(value);
}

function stringifyStructured(value: any): string {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  return safeJson(serialized);
}

function contentBlocksToText(contentBlocks: any): string {
  const serialized = toSerializableValue(contentBlocks);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  if (!Array.isArray(serialized)) {
    return stringifyStructured(serialized);
  }

  const textParts: string[] = [];
  for (const block of serialized) {
    if (typeof block === "string") {
      textParts.push(block);
      continue;
    }
    if (!block || typeof block !== "object" || Array.isArray(block)) {
      textParts.push(String(block ?? ""));
      continue;
    }

    const blockType = (block as any).type ?? "";
    if (blockType === "input_image") {
      textParts.push("[image]");
      continue;
    }
    if (blockType === "input_file") {
      textParts.push("[file]");
      continue;
    }
    if (typeof (block as any).text === "string") {
      textParts.push((block as any).text);
      continue;
    }
    textParts.push(stringifyStructured(block));
  }

  return textParts.join("\n");
}

function normalizeMessageContent(content: any): string {
  if (content === undefined || content === null) {
    return "";
  }
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return contentBlocksToText(content);
  }
  return stringifyStructured(content);
}

function stringifyToolResult(value: any): string {
  const serialized = toSerializableValue(value);
  if (serialized === undefined || serialized === null) {
    return "";
  }
  if (typeof serialized === "string") {
    return serialized;
  }
  if (Array.isArray(serialized)) {
    return contentBlocksToText(serialized);
  }
  if (typeof serialized === "object") {
    const blockType = (serialized as any).type ?? "";
    if (
      (blockType === "text" ||
        blockType === "output_text" ||
        blockType === "input_text") &&
      typeof (serialized as any).text === "string"
    ) {
      return (serialized as any).text;
    }
    if (
      (serialized as any).output !== undefined ||
      (serialized as any).result !== undefined
    ) {
      return stringifyToolResult(
        (serialized as any).output ?? (serialized as any).result,
      );
    }
  }
  return stringifyStructured(serialized);
}

function setJsonStructuredAttr(
  attrs: Record<string, any>,
  key: string,
  value: any,
): void {
  if (value === undefined || value === null) return;
  if (value === "") return;
  if (Array.isArray(value) && value.length === 0) return;
  attrs[key] = safeJson(value);
}

function normalizeToolCall(rawToolCall: any): Record<string, any> | null {
  const toolCall = toSerializableValue(rawToolCall);
  if (!toolCall || typeof toolCall !== "object" || Array.isArray(toolCall)) {
    return null;
  }

  const functionName =
    (toolCall as any).name ??
    (toolCall as any).function?.name ??
    "";
  const functionArguments =
    (toolCall as any).arguments ??
    (toolCall as any).function?.arguments ??
    "";

  if (!functionName && !(toolCall as any).function && (toolCall as any).type !== "function_call") {
    return null;
  }

  return {
    id:
      (toolCall as any).call_id ??
      (toolCall as any).callId ??
      (toolCall as any).tool_call_id ??
      (toolCall as any).id ??
      "",
    type: "function",
    function: {
      name: functionName,
      arguments: stringifyStructured(functionArguments),
    },
  };
}

function extractToolCalls(output: any): Record<string, any>[] {
  const serialized = toSerializableValue(output);
  if (serialized === undefined || serialized === null) return [];

  const items = Array.isArray(serialized) ? serialized : [serialized];
  const result: Record<string, any>[] = [];

  for (const item of items) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }

    const itemType = (item as any).type ?? "";
    if (itemType === "function_call" || itemType === "function") {
      const toolCall = normalizeToolCall(item);
      if (toolCall) result.push(toolCall);
      continue;
    }

    if (Array.isArray((item as any).tool_calls)) {
      for (const toolCall of (item as any).tool_calls) {
        const normalized = normalizeToolCall(toolCall);
        if (normalized) result.push(normalized);
      }
    }
  }

  return result;
}

function extractTools(tools: any): Record<string, any>[] {
  const serialized = toSerializableValue(tools);
  if (!Array.isArray(serialized)) return [];

  const result: Record<string, any>[] = [];
  for (const tool of serialized) {
    if (!tool || typeof tool !== "object" || Array.isArray(tool)) {
      continue;
    }

    const toolType = (tool as any).type ?? "";
    if (toolType === "function") {
      const func: Record<string, any> = {
        name: (tool as any).name ?? (tool as any).function?.name ?? "",
      };
      const description =
        (tool as any).description ??
        (tool as any).function?.description;
      if (description) func.description = description;

      const parameters =
        (tool as any).parameters ??
        (tool as any).function?.parameters;
      if (parameters !== undefined) func.parameters = parameters;

      result.push({ type: "function", function: func });
      continue;
    }

    result.push(tool as Record<string, any>);
  }

  return result;
}

function normalizeChatMessage(rawMessage: any): Record<string, any> {
  const message = toSerializableValue(rawMessage);
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return { role: "user", content: stringifyStructured(rawMessage) };
  }

  const normalized: Record<string, any> = {
    role: (message as any).role ?? "user",
    content: normalizeMessageContent((message as any).content),
  };

  if (Array.isArray((message as any).tool_calls)) {
    const toolCalls = extractToolCalls((message as any).tool_calls);
    if (toolCalls.length) {
      normalized.tool_calls = toolCalls;
      if (!normalized.content) normalized.content = "";
    }
  }

  if ((message as any).tool_call_id) {
    normalized.tool_call_id = (message as any).tool_call_id;
  }

  return normalized;
}

function responsesApiItemToMessage(rawItem: any): Record<string, any> | null {
  const item = toSerializableValue(rawItem);
  if (!item || typeof item !== "object" || Array.isArray(item)) {
    return null;
  }

  const itemType = (item as any).type ?? "";
  if (itemType === "message") {
    return {
      role: (item as any).role ?? "user",
      content: normalizeMessageContent((item as any).content),
    };
  }

  if (itemType === "function_call") {
    const toolCall = normalizeToolCall(item);
    if (!toolCall) return null;
    return {
      role: "assistant",
      content: "",
      tool_calls: [toolCall],
    };
  }

  if (itemType === "function_call_output" || itemType === "function_call_result") {
    return {
      role: "tool",
      content: stringifyToolResult(
        (item as any).output ?? (item as any).result ?? "",
      ),
      tool_call_id:
        (item as any).call_id ??
        (item as any).callId ??
        (item as any).tool_call_id ??
        "",
    };
  }

  if ((item as any).role) {
    return normalizeChatMessage(item);
  }

  return null;
}

function parseISOToHrTime(iso: string | undefined): [number, number] | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime();
  if (!Number.isFinite(ms)) return null;

  const secs = Math.floor(ms / 1000);
  const nanos = (ms % 1000) * 1_000_000;
  return [secs, nanos];
}

function hashStringToHexId(s: string, length: number): string {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  }
  const hex = Math.abs(hash).toString(16).padStart(8, "0");
  return (hex + hex + hex + hex).slice(0, length);
}

function ensureTraceId(id: string): string {
  if (/^[0-9a-f]{32}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 32);
}

function ensureSpanId(id: string): string {
  if (/^[0-9a-f]{16}$/i.test(id)) return id.toLowerCase();
  return hashStringToHexId(id, 16);
}

function resolveSpanTimes(item: Span<any>): {
  startTimeHr: [number, number] | null;
  endTimeHr: [number, number] | null;
} {
  const serialized = toSerializableValue(item) as Record<string, any> | undefined;
  return {
    startTimeHr: parseISOToHrTime(serialized?.started_at),
    endTimeHr: parseISOToHrTime(serialized?.ended_at),
  };
}

function baseAttrs(
  entityName: string,
  entityPath: string,
  logType: string,
): Record<string, any> {
  return {
    // Leave traceloop.span.kind unset for injected OpenAI Agents spans.
    // In the JS tracing pipeline, that attribute is treated as a root-span
    // marker and would flatten the parent/child tree.
    [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
    [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
  };
}

interface BuildSpanOptions {
  name: string;
  traceId: string;
  spanId: string;
  parentId?: string;
  startTimeHr?: [number, number] | null;
  endTimeHr?: [number, number] | null;
  attributes: Record<string, any>;
  statusCode?: number;
  errorMessage?: string;
}

function buildReadableSpan(opts: BuildSpanOptions): ReadableSpan {
  const startTime = opts.startTimeHr ?? hrTime();
  const endTime = opts.endTimeHr ?? startTime;

  const traceId = ensureTraceId(opts.traceId);
  const spanId = ensureSpanId(opts.spanId);
  const parentSpanId = opts.parentId ? ensureSpanId(opts.parentId) : undefined;

  const status =
    opts.statusCode && opts.statusCode >= 400
      ? { code: SpanStatusCode.ERROR, message: opts.errorMessage ?? "" }
      : { code: SpanStatusCode.OK, message: "" };

  return {
    name: opts.name,
    kind: SpanKind.INTERNAL,
    spanContext: () => ({
      traceId,
      spanId,
      traceFlags: 1,
      isRemote: false,
    }),
    parentSpanId,
    startTime,
    endTime,
    duration: hrTimeDuration(startTime, endTime),
    status,
    attributes: opts.attributes,
    links: [],
    events: [],
    resource: { attributes: {} } as any,
    instrumentationLibrary: {
      name: "@respan/instrumentation-openai-agents",
      version: PACKAGE_VERSION,
    },
    ended: true,
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } as unknown as ReadableSpan;
}

function injectSpan(span: ReadableSpan): void {
  const tp = trace.getTracerProvider() as any;
  const processor =
    tp?.activeSpanProcessor ??
    tp?._delegate?.activeSpanProcessor ??
    tp?._delegate?._tracerProvider?.activeSpanProcessor;
  if (processor && typeof processor.onEnd === "function") {
    processor.onEnd(span);
  }
}

function formatInputMessages(input: any): any[] | null {
  const serialized = toSerializableValue(input);
  if (serialized === undefined || serialized === null) return null;

  if (Array.isArray(serialized)) {
    const hasResponsesApiItems = serialized.some(
      (item) =>
        item &&
        typeof item === "object" &&
        !Array.isArray(item) &&
        "type" in item,
    );

    if (hasResponsesApiItems) {
      const messages: Record<string, any>[] = [];
      for (const item of serialized) {
        if (!item || typeof item !== "object" || Array.isArray(item)) {
          continue;
        }
        if ("type" in item) {
          const message = responsesApiItemToMessage(item);
          if (message) messages.push(message);
        } else if ("role" in item) {
          messages.push(normalizeChatMessage(item));
        }
      }
      return messages.length ? messages : serialized;
    }

    if (
      serialized.length > 0 &&
      serialized.every(
        (item) =>
          item &&
          typeof item === "object" &&
          !Array.isArray(item) &&
          "role" in item,
      )
    ) {
      return serialized.map((item) => normalizeChatMessage(item));
    }

    return serialized.map((item) => ({
      role: "user",
      content: stringifyStructured(item),
    }));
  }

  if (typeof serialized === "string") {
    return [{ role: "user", content: serialized }];
  }

  if (typeof serialized === "object") {
    return [{ role: "user", content: safeJson(serialized) }];
  }

  return [{ role: "user", content: String(serialized) }];
}

function formatOutput(output: any): string {
  const serialized = toSerializableValue(output);
  if (serialized === undefined || serialized === null) return "";

  if (typeof serialized === "string") {
    return serialized;
  }

  if (typeof serialized === "object" && !Array.isArray(serialized)) {
    if ((serialized as any).content === undefined) {
      return safeJson(serialized);
    }
    return normalizeMessageContent((serialized as any).content);
  }

  if (Array.isArray(serialized)) {
    const textParts: string[] = [];
    for (const item of serialized) {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        textParts.push(stringifyStructured(item));
        continue;
      }

      const itemType = (item as any).type ?? "";
      if (
        itemType === "function_call" ||
        itemType === "function_call_output" ||
        itemType === "function_call_result"
      ) {
        continue;
      }
      if (
        itemType === "output_text" ||
        itemType === "text" ||
        itemType === "input_text"
      ) {
        textParts.push((item as any).text ?? "");
        continue;
      }
      if (itemType === "message") {
        textParts.push(normalizeMessageContent((item as any).content));
        continue;
      }
      if ("content" in item) {
        textParts.push(normalizeMessageContent((item as any).content));
        continue;
      }
      textParts.push(stringifyStructured(item));
    }
    return textParts.filter(Boolean).join("\n");
  }

  return stringifyStructured(serialized);
}

function isTrace(item: any): item is Trace {
  return "traceId" in item && "name" in item && !("spanId" in item);
}

function isSpan(item: any): item is Span<any> {
  return "spanId" in item && "spanData" in item;
}

function emitTrace(traceObj: Trace): void {
  const traceName = traceObj.name || "trace";
  const attrs = baseAttrs(traceName, "", RespanLogType.WORKFLOW);
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = traceName;

  const span = buildReadableSpan({
    name: `${traceName}.workflow`,
    traceId: traceObj.traceId,
    spanId: traceObj.traceId,
    attributes: attrs,
  });
  injectSpan(span);
}

function emitAgent(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = data.name || "agent";

  const attrs = baseAttrs(name, name, RespanLogType.AGENT);
  attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = name;
  attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] = name;
  setJsonStructuredAttr(
    attrs,
    RespanSpanAttributes.RESPAN_SPAN_TOOLS,
    extractTools(data.tools),
  );
  setJsonStructuredAttr(
    attrs,
    RespanSpanAttributes.RESPAN_SPAN_HANDOFFS,
    toSerializableValue(data.handoffs),
  );

  const span = buildReadableSpan({
    name: `${name}.agent`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitResponse(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);

  const attrs = baseAttrs("chat", "chat", RespanLogType.CHAT);
  attrs[RespanSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
  attrs[RespanSpanAttributes.LLM_SYSTEM] = "openai";

  const inputMsgs = formatInputMessages(data._input);
  if (inputMsgs) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);
  }

  const resp = toSerializableValue(data._response);
  if (resp && typeof resp === "object" && !Array.isArray(resp)) {
    if ((resp as any).model) {
      attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = (resp as any).model;
    }

    if ("output" in resp) {
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = formatOutput((resp as any).output);
      setJsonStructuredAttr(
        attrs,
        RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
        extractToolCalls((resp as any).output),
      );
    }

    if ("tools" in resp) {
      setJsonStructuredAttr(
        attrs,
        RespanSpanAttributes.RESPAN_SPAN_TOOLS,
        extractTools((resp as any).tools),
      );
    }

    if ((resp as any).usage) {
      attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] =
        (resp as any).usage.input_tokens ?? 0;
      attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] =
        (resp as any).usage.output_tokens ?? 0;
    }
  }

  const span = buildReadableSpan({
    name: "openai.chat",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitFunction(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = data.name || "function";

  const attrs = baseAttrs(name, name, RespanLogType.TOOL);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson([
    { role: "tool", content: stringifyStructured(data.input) },
  ]);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson({
    role: "tool",
    content: stringifyToolResult(data.output),
  });

  const span = buildReadableSpan({
    name: `${name}.tool`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitGeneration(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);

  const attrs = baseAttrs("chat", "chat", RespanLogType.CHAT);
  attrs[RespanSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;

  if (data.model) attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = data.model;

  const inputMsgs = formatInputMessages(data.input);
  if (inputMsgs) {
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(inputMsgs);
  }

  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = formatOutput(data.output);
  setJsonStructuredAttr(
    attrs,
    RespanSpanAttributes.RESPAN_SPAN_TOOL_CALLS,
    extractToolCalls(data.output),
  );

  if (data.usage) {
    attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] =
      data.usage.prompt_tokens ?? data.usage.input_tokens ?? 0;
    attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] =
      data.usage.completion_tokens ?? data.usage.output_tokens ?? 0;
  }

  const span = buildReadableSpan({
    name: "openai.chat",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitHandoff(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);

  const fromAgent = data.from_agent || "";
  const toAgent = data.to_agent || "";

  const attrs = baseAttrs("handoff", "handoff", RespanLogType.HANDOFF);
  attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(fromAgent);
  attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(toAgent);
  attrs[RespanSpanAttributes.RESPAN_METADATA_FROM_AGENT] = fromAgent;
  attrs[RespanSpanAttributes.RESPAN_METADATA_TO_AGENT] = toAgent;

  const span = buildReadableSpan({
    name: "handoff.task",
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitGuardrail(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = `guardrail:${data.name}`;

  const attrs = baseAttrs(name, name, RespanLogType.GUARDRAIL);
  attrs[RespanSpanAttributes.RESPAN_METADATA_GUARDRAIL_NAME] = data.name;
  attrs[RespanSpanAttributes.RESPAN_METADATA_TRIGGERED] = String(data.triggered);

  const span = buildReadableSpan({
    name: `${name}.task`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

function emitCustom(item: Span<any>): void {
  const data = item.spanData as any;
  const { startTimeHr, endTimeHr } = resolveSpanTimes(item);
  const name = data.name || data.data?.name || "custom";

  const attrs = baseAttrs(name, name, RespanLogType.CUSTOM);
  const customData = data.data || {};
  for (const [key, value] of Object.entries(customData)) {
    if (key === "model") {
      attrs[RespanSpanAttributes.GEN_AI_REQUEST_MODEL] = value;
    } else if (key === "prompt_tokens") {
      attrs[RespanSpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = value;
    } else if (key === "completion_tokens") {
      attrs[RespanSpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = value;
    } else if (key === "name") {
      continue;
    } else if (key === "input") {
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(value);
    } else if (key === "output") {
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(value);
    } else {
      attrs[`respan.metadata.${key}`] = String(value);
    }
  }

  const span = buildReadableSpan({
    name: `${name}.task`,
    traceId: item.traceId,
    spanId: item.spanId,
    parentId: item.parentId || item.traceId,
    startTimeHr,
    endTimeHr,
    attributes: attrs,
    statusCode: item.error ? 400 : 200,
    errorMessage: item.error ? String(item.error) : undefined,
  });
  injectSpan(span);
}

export function emitSdkItem(item: Trace | Span<any>): void {
  if (isTrace(item)) {
    emitTrace(item);
    return;
  }

  if (!isSpan(item)) return;

  const spanData = item.spanData as any;
  const type = spanData?.type;

  try {
    if (type === "response") emitResponse(item);
    else if (type === "function") emitFunction(item);
    else if (type === "generation") emitGeneration(item);
    else if (type === "agent") emitAgent(item);
    else if (type === "handoff") emitHandoff(item);
    else if (type === "custom") emitCustom(item);
    else if (typeof spanData?.triggered === "boolean") emitGuardrail(item);
    else if (spanData?.name && spanData?.data) emitCustom(item);
    else {
      console.debug(`[Respan] Unknown OpenAI Agents span data type: ${type}`);
    }
  } catch (error) {
    console.error(`[Respan] Error emitting OpenAI Agents span:`, error);
  }
}
