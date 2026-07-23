import {
  ATTR_ERROR_MESSAGE,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_USAGE_COMPLETION_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_PROMPT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes as TraceloopSpanAttributes } from "@traceloop/ai-semantic-conventions";
import {
  MESSAGE_CONTENT_SUFFIX,
  MESSAGE_ROLE_SUFFIX,
  MESSAGE_TOOL_CALL_ID_SUFFIX,
  MESSAGE_TOOL_CALLS_SUFFIX,
  RESPAN_LOG_METHOD_TS_TRACING,
} from "./_constants.js";
import {
  activeSpanContext,
  emitReadableSpan,
  extractResponseMessage,
  extractResponseModel,
  extractResponseText,
  extractUsage,
  formatMessage,
  formatMessages,
  formatToolCall,
  inferGenAISystem,
  randomHex,
  safeJson,
  type HrTime,
  type SpanAttributes,
} from "./_helpers.js";

export interface SpanRecord {
  id: string;
  name: string;
  logType: string;
  spanId: string;
  traceId?: string;
  parentId?: string;
  startTime: HrTime;
  input?: unknown;
}

export interface EmitterOptions {
  workflowName?: string;
}

function baseAttrs(params: {
  name: string;
  logType: string;
  input?: unknown;
  output?: unknown;
  workflowName?: string;
}): SpanAttributes {
  const attrs: SpanAttributes = {
    [TraceloopSpanAttributes.TRACELOOP_ENTITY_NAME]: params.name,
    [TraceloopSpanAttributes.TRACELOOP_ENTITY_PATH]: params.name,
    [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
    [RespanSpanAttributes.RESPAN_LOG_TYPE]: params.logType,
  };
  if (params.input !== undefined) {
    attrs[TraceloopSpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(params.input);
  }
  if (params.output !== undefined) {
    attrs[TraceloopSpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(params.output);
  }
  if (params.workflowName) {
    attrs[TraceloopSpanAttributes.TRACELOOP_WORKFLOW_NAME] = params.workflowName;
  }
  return attrs;
}

function setIndexedMessages(
  attrs: SpanAttributes,
  messages: Record<string, unknown>[],
): void {
  messages.forEach((message, index) => {
    const prefix = `${ATTR_GEN_AI_PROMPT}.${index}`;
    attrs[`${prefix}.${MESSAGE_ROLE_SUFFIX}`] = String(message.role ?? "user");
    attrs[`${prefix}.${MESSAGE_CONTENT_SUFFIX}`] = String(message.content ?? "");
    if (message.tool_calls !== undefined) {
      attrs[`${prefix}.${MESSAGE_TOOL_CALLS_SUFFIX}`] = safeJson(message.tool_calls);
    }
    if (message.tool_call_id !== undefined) {
      attrs[`${prefix}.${MESSAGE_TOOL_CALL_ID_SUFFIX}`] = String(message.tool_call_id);
    }
  });
}

function setCompletionMessage(
  attrs: SpanAttributes,
  message: Record<string, unknown>,
): void {
  const prefix = `${ATTR_GEN_AI_COMPLETION}.0`;
  attrs[`${prefix}.${MESSAGE_ROLE_SUFFIX}`] = String(message.role ?? "assistant");
  attrs[`${prefix}.${MESSAGE_CONTENT_SUFFIX}`] = String(message.content ?? "");
  if (message.tool_calls !== undefined) {
    attrs[`${prefix}.${MESSAGE_TOOL_CALLS_SUFFIX}`] = safeJson(message.tool_calls);
  }
}

export class LlamaIndexSpanEmitter {
  private readonly options: EmitterOptions;
  private readonly records = new Map<string, SpanRecord>();
  private readonly stack: string[] = [];
  private readonly pendingToolCalls = new Map<
    string,
    { startTime: HrTime; traceId?: string; parentId?: string; input: unknown }
  >();

  constructor(options: EmitterOptions = {}) {
    this.options = options;
  }

  startRecord(params: {
    id: string;
    name: string;
    logType: string;
    startTime: HrTime;
    input?: unknown;
  }): void {
    const activeContext = activeSpanContext();
    const parent = this.currentRecord();
    const traceId = activeContext?.traceId ?? parent?.traceId;
    const parentId = activeContext?.spanId ?? parent?.spanId;
    const record: SpanRecord = {
      id: params.id,
      name: params.name,
      logType: params.logType,
      spanId: randomHex(16),
      traceId,
      parentId,
      startTime: params.startTime,
      input: params.input,
    };
    this.records.set(record.id, record);
    this.stack.push(record.id);
  }

  endRecord(params: {
    id: string;
    output?: unknown;
    errorMessage?: string;
    endTime: HrTime;
  }): void {
    const record = this.records.get(params.id);
    if (!record) {
      return;
    }
    this.records.delete(params.id);
    this.removeFromStack(params.id);

    const attrs = baseAttrs({
      name: record.name,
      logType: record.logType,
      input: record.input,
      output: params.output,
      workflowName: this.options.workflowName,
    });

    emitReadableSpan({
      name: record.name,
      traceId: record.traceId,
      spanId: record.spanId,
      parentId: record.parentId,
      startTime: record.startTime,
      endTime: params.endTime,
      attributes: attrs,
      errorMessage: params.errorMessage,
    });
  }

  startLLM(params: { id: string; messages: unknown; startTime: HrTime }): void {
    this.startRecord({
      id: params.id,
      name: "llamaindex.llm",
      logType: RespanLogType.CHAT,
      startTime: params.startTime,
      input: formatMessages(params.messages),
    });
  }

  endLLM(params: {
    id: string;
    response: unknown;
    errorMessage?: string;
    endTime: HrTime;
  }): void {
    const record = this.records.get(params.id);
    if (!record) {
      return;
    }
    this.records.delete(params.id);
    this.removeFromStack(params.id);

    const inputMessages = Array.isArray(record.input)
      ? (record.input as Record<string, unknown>[])
      : [];
    const completion = extractResponseMessage(params.response);
    const model = extractResponseModel(params.response);
    const usage = extractUsage(params.response);
    const attrs = baseAttrs({
      name: record.name,
      logType: RespanLogType.CHAT,
      input: inputMessages,
      output: [completion],
      workflowName: this.options.workflowName,
    });

    attrs[TraceloopSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
    if (model) {
      attrs[ATTR_GEN_AI_REQUEST_MODEL] = model;
      attrs[ATTR_GEN_AI_SYSTEM] = inferGenAISystem(model);
    }
    setIndexedMessages(attrs, inputMessages);
    setCompletionMessage(attrs, completion);
    const completionToolCalls = completion.tool_calls;
    if (completionToolCalls !== undefined) {
      attrs[TraceloopSpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(completionToolCalls);
    }
    if (usage.inputTokens !== undefined) {
      attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = usage.inputTokens;
      attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = usage.inputTokens;
    }
    if (usage.outputTokens !== undefined) {
      attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = usage.outputTokens;
      attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = usage.outputTokens;
    }
    if (usage.totalTokens !== undefined) {
      attrs[TraceloopSpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage.totalTokens;
    }

    emitReadableSpan({
      name: record.name,
      traceId: record.traceId,
      spanId: record.spanId,
      parentId: record.parentId,
      startTime: record.startTime,
      endTime: params.endTime,
      attributes: attrs,
      errorMessage: params.errorMessage,
    });
  }

  recordToolCall(params: { toolCall: unknown; startTime: HrTime }): void {
    const formatted = formatToolCall(params.toolCall);
    const functionPayload = formatted.function as Record<string, unknown>;
    const toolId = String(
      formatted.id ?? `${functionPayload.name ?? "tool"}-${randomHex(8)}`,
    );
    const current = this.currentRecord();
    this.pendingToolCalls.set(toolId, {
      startTime: params.startTime,
      traceId: current?.traceId,
      parentId: current?.spanId,
      input: {
        name: functionPayload.name,
        arguments: functionPayload.arguments,
      },
    });
  }

  emitToolResult(params: {
    toolCall: unknown;
    toolResult: unknown;
    endTime: HrTime;
  }): void {
    const formattedCall = formatToolCall(params.toolCall);
    const functionPayload = formattedCall.function as Record<string, unknown>;
    const toolName = String(functionPayload.name ?? "unknown_tool");
    const toolId = String(formattedCall.id ?? `${toolName}-${randomHex(8)}`);
    const pending = this.pendingToolCalls.get(toolId);
    this.pendingToolCalls.delete(toolId);

    const result =
      params.toolResult && typeof params.toolResult === "object"
        ? (params.toolResult as Record<string, unknown>)
        : { output: params.toolResult };
    const isError = Boolean(result.isError);
    const attrs = baseAttrs({
      name: toolName,
      logType: RespanLogType.TOOL,
      input: pending?.input ?? {
        name: toolName,
        arguments: functionPayload.arguments,
      },
      output: result.output,
      workflowName: this.options.workflowName,
    });
    if (isError) {
      attrs[ATTR_ERROR_MESSAGE] = safeJson(result.output);
    }

    emitReadableSpan({
      name: `${toolName}.tool`,
      traceId: pending?.traceId ?? this.currentRecord()?.traceId,
      parentId: pending?.parentId ?? this.currentRecord()?.spanId,
      startTime: pending?.startTime ?? params.endTime,
      endTime: params.endTime,
      attributes: attrs,
      errorMessage: isError ? safeJson(result.output) : undefined,
    });
  }

  emitSyntheticLLMCompletion(params: {
    prompt: unknown;
    response: unknown;
    model?: string;
    startTime: HrTime;
    endTime: HrTime;
  }): void {
    const inputMessages = [
      { role: "user", content: String(params.prompt ?? "") },
    ];
    const completion = {
      role: "assistant",
      content: extractResponseText(params.response),
    };
    const attrs = baseAttrs({
      name: "llamaindex.llm",
      logType: RespanLogType.CHAT,
      input: inputMessages,
      output: [completion],
      workflowName: this.options.workflowName,
    });
    attrs[TraceloopSpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
    if (params.model) {
      attrs[ATTR_GEN_AI_REQUEST_MODEL] = params.model;
      attrs[ATTR_GEN_AI_SYSTEM] = inferGenAISystem(params.model);
    }
    setIndexedMessages(attrs, inputMessages);
    setCompletionMessage(attrs, completion);
    const current = this.currentRecord();
    emitReadableSpan({
      name: "llamaindex.llm",
      traceId: current?.traceId,
      parentId: current?.spanId,
      startTime: params.startTime,
      endTime: params.endTime,
      attributes: attrs,
    });
  }

  clear(): void {
    this.records.clear();
    this.stack.length = 0;
    this.pendingToolCalls.clear();
  }

  private currentRecord(): SpanRecord | undefined {
    for (let index = this.stack.length - 1; index >= 0; index -= 1) {
      const record = this.records.get(this.stack[index]);
      if (record) {
        return record;
      }
    }
    return undefined;
  }

  private removeFromStack(id: string): void {
    const index = this.stack.lastIndexOf(id);
    if (index >= 0) {
      this.stack.splice(index, 1);
    }
  }
}

export function formatTaskInput(detail: Record<string, unknown>): unknown {
  if ("query" in detail) return detail.query;
  if ("text" in detail) return detail.text;
  if ("documents" in detail) return detail.documents;
  if ("startStep" in detail) return detail.startStep;
  return detail;
}

export function formatTaskOutput(detail: Record<string, unknown>): unknown {
  if ("response" in detail) return detail.response;
  if ("nodes" in detail) return detail.nodes;
  if ("chunks" in detail) return detail.chunks;
  if ("endStep" in detail) return detail.endStep;
  return detail;
}

export function formatAgentStepId(detail: Record<string, any>): string | undefined {
  const step = detail.startStep ?? detail.endStep;
  if (step?.id) return String(step.id);
  return undefined;
}
