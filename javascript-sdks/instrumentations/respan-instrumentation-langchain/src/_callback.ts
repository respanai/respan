import {
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  TraceFlags,
  type Span,
} from "@opentelemetry/api";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  SpanAttributes,
  LLMRequestTypeValues,
} from "@traceloop/ai-semantic-conventions";
import {
  ATTR_ERROR_TYPE,
  ATTR_HTTP_RESPONSE_STATUS_CODE,
} from "@opentelemetry/semantic-conventions";
import {
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import {
  capture,
  data,
  guard,
  refresh,
  snapshot,
  type CapturePolicy,
} from "./_privacy.js";
import {
  LANGCHAIN_FRAMEWORK_ATTR,
  LANGCHAIN_RUN_ID_ATTR,
  LANGCHAIN_PARENT_RUN_ID_ATTR,
  detectFramework,
  extractName,
  extractModel,
  extractTools,
  extractLlmOutput,
  extractToolCallsFromMessages,
  extractUsage,
  normalizeChatMessages,
  normalizeMetadata,
  normalizeTags,
  safeJsonString,
  trimMap,
  getErrorMessage,
  isPlainRecord,
  type RunRecord,
  type SpanAttributesRecord,
  type LangChainCallbackConfig,
  type RespanCallbackHandlerOptions,
} from "./_callback_helpers.js";
export type {
  LangChainCallbackConfig,
  RespanCallbackHandlerOptions,
} from "./_callback_helpers.js";

export function getCallbackHandler(
  options: RespanCallbackHandlerOptions = {},
): RespanCallbackHandler {
  return new RespanCallbackHandler({ groupLangflowRootRuns: true, ...options });
}
function isHandler(value: unknown): boolean {
  return (
    value instanceof RespanCallbackHandler ||
    data(value, "_respanCallbackHandler") === true
  );
}
export function addRespanCallback<
  TConfig extends object = LangChainCallbackConfig,
>(
  config: TConfig = {} as TConfig,
  handler: RespanCallbackHandler = getCallbackHandler(),
): TConfig {
  const next = { ...config } as TConfig & LangChainCallbackConfig;
  const callbacks = next.callbacks;
  if (callbacks == null) next.callbacks = [handler];
  else if (Array.isArray(callbacks))
    next.callbacks = callbacks.some(isHandler)
      ? callbacks
      : [...callbacks, handler];
  else if (Array.isArray(data(callbacks, "handlers"))) {
    const handlers = data(callbacks, "handlers");
    if (!handlers.some(isHandler)) {
      const addHandler = (callbacks as any).addHandler;
      if (typeof addHandler === "function")
        addHandler.call(callbacks, handler, true);
      else handlers.push(handler);
    }
  } else
    next.callbacks = isHandler(callbacks) ? [callbacks] : [callbacks, handler];
  return next;
}
interface CompletedRun {
  span: Span;
  policy: CapturePolicy;
  path: string;
  workflowName: string;
}
export class RespanCallbackHandler {
  readonly name = "RespanCallbackHandler";
  readonly _respanCallbackHandler = true;
  readonly ignoreLLM = false;
  readonly ignoreChain = false;
  readonly ignoreAgent = false;
  readonly ignoreRetriever = false;
  readonly ignoreCustomEvent = false;
  readonly raiseError = false;
  readonly awaitHandlers = true;
  includeContent: boolean;
  includeMetadata: boolean;
  recordInputs: boolean;
  recordOutputs: boolean;
  groupLangflowRootRuns: boolean;
  maxCachedRuns: number;
  private readonly _ceilings: RespanCallbackHandlerOptions;
  private readonly _runs = new Map<string, RunRecord>();
  private readonly _completed = new Map<string, CompletedRun>();
  private _langflowRoot?: CompletedRun;
  private _active = true;
  private readonly _owners = new Set<object>();
  constructor(options: RespanCallbackHandlerOptions = {}) {
    this._ceilings = { ...options };
    this.includeContent = options.includeContent ?? true;
    this.includeMetadata = options.includeMetadata ?? true;
    this.recordInputs = options.recordInputs ?? true;
    this.recordOutputs = options.recordOutputs ?? true;
    this.groupLangflowRootRuns = options.groupLangflowRootRuns ?? false;
    this.maxCachedRuns =
      Number.isInteger(options.maxCachedRuns) && options.maxCachedRuns! >= 0
        ? options.maxCachedRuns!
        : 4096;
  }
  /** The plugin stops accepting new roots; callbacks for admitted runs still drain. */
  _setActive(active: boolean, owner: object): void {
    if (active) this._owners.add(owner);
    else this._owners.delete(owner);
    this._active = this._owners.size > 0;
  }
  private _policy(record: RunRecord): CapturePolicy {
    if (!this.includeContent)
      record.policy.inputs = record.policy.outputs = false;
    if (!this.recordInputs) record.policy.inputs = false;
    if (!this.recordOutputs) record.policy.outputs = false;
    return refresh(record.policy);
  }
  private _start(
    serialized: unknown,
    runId: unknown,
    parentRunId: unknown,
    tags: unknown,
    metadata: unknown,
    runName: unknown,
    logType: string,
    model?: string,
  ): RunRecord | undefined {
    if (typeof runId !== "string" || this._runs.has(runId)) return;
    const parentId = typeof parentRunId === "string" ? parentRunId : undefined;
    const parent =
      (parentId &&
        (this._runs.get(parentId) ?? this._completed.get(parentId))) ||
      undefined;
    if (!this._active && !(parentId && this._runs.has(parentId))) return;
    const name = extractName(serialized, logType, runName);
    const framework = detectFramework({ metadata });
    const grouped =
      !parent &&
      !trace.getSpan(context.active()) &&
      framework === "langflow" &&
      this.groupLangflowRootRuns
        ? this._langflowRoot
        : undefined;
    const ancestor = parent ?? grouped;
    const ctx = ancestor
      ? trace.setSpan(context.active(), ancestor.span)
      : context.active();
    const policy = capture(
      {
        traceContent:
          this._ceilings.includeContent !== false && this.includeContent,
        recordInputs:
          this._ceilings.recordInputs !== false && this.recordInputs,
        recordOutputs:
          this._ceilings.recordOutputs !== false && this.recordOutputs,
      },
      ctx,
    );
    if (ancestor) {
      policy.inputs &&= refresh(ancestor.policy).inputs;
      policy.outputs &&= ancestor.policy.outputs;
    }
    if (!policy.emit) return;
    const path = ancestor ? `${ancestor.path}.${name}` : "";
    const workflowName = ancestor?.workflowName ?? name;
    const attrs: SpanAttributesRecord = {
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
      [SpanAttributes.TRACELOOP_ENTITY_PATH]: path,
      [SpanAttributes.TRACELOOP_WORKFLOW_NAME]: workflowName,
      [LANGCHAIN_RUN_ID_ATTR]: runId,
      [LANGCHAIN_FRAMEWORK_ATTR]: framework,
    };
    if (parentId) attrs[LANGCHAIN_PARENT_RUN_ID_ATTR] = parentId;
    if (logType === RespanLogType.CHAT || logType === RespanLogType.TEXT) {
      attrs[SpanAttributes.LLM_REQUEST_TYPE] = LLMRequestTypeValues.CHAT;
      if (model) attrs[SpanAttributes.LLM_REQUEST_MODEL] = model;
      const provider = data(metadata, "ls_provider");
      if (typeof provider === "string")
        attrs[SpanAttributes.LLM_SYSTEM] = provider;
    }
    // The real provider decides sampling before input, metadata, messages, or tools are copied.
    const span = trace.getTracer("@respan/instrumentation-langchain").startSpan(
      name,
      {
        kind:
          logType === RespanLogType.CHAT || logType === RespanLogType.TEXT
            ? SpanKind.CLIENT
            : SpanKind.INTERNAL,
        attributes: attrs,
      },
      ctx,
    );
    policy.sampled =
      span.isRecording() &&
      !!(span.spanContext().traceFlags & TraceFlags.SAMPLED);
    guard(span, policy);
    const record: RunRecord = {
      runId,
      span,
      policy,
      name,
      path,
      workflowName,
      logType,
      model,
      tokens: [],
      indexed: {},
    };
    this._runs.set(runId, record);
    if (
      framework === "langflow" &&
      this.groupLangflowRootRuns &&
      !parent &&
      !this._langflowRoot
    )
      this._langflowRoot = { span, policy, path, workflowName };
    if (
      this._admitted(record) &&
      this._ceilings.includeMetadata !== false &&
      this.includeMetadata &&
      policy.inputs &&
      policy.outputs
    ) {
      const normalized = normalizeMetadata(metadata);
      const inherited = data(
        (span as any).attributes,
        RespanSpanAttributes.RESPAN_METADATA,
      );
      let merged: Record<string, any> = {};
      if (typeof inherited === "string") {
        try {
          const value = JSON.parse(inherited);
          if (isPlainRecord(value)) merged = value;
        } catch {}
      }
      const extra = { ...merged, ...normalized };
      const normalizedTags = normalizeTags(tags);
      if (normalizedTags?.length) extra.langchain_tags = normalizedTags;
      if (Object.keys(extra).length)
        span.setAttribute(
          RespanSpanAttributes.RESPAN_METADATA,
          JSON.stringify(extra),
        );
      const custom = data(normalized, "custom_identifier");
      if (typeof custom === "string")
        span.setAttribute(RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID, custom);
    }
    return record;
  }
  private _admitted(record: RunRecord): boolean {
    this._policy(record);
    return (
      record.span.isRecording() &&
      !!(record.span.spanContext().traceFlags & TraceFlags.SAMPLED)
    );
  }
  private _input(record: RunRecord | undefined, value: () => unknown): void {
    if (record && this._admitted(record) && record.policy.inputs) {
      const text = safeJsonString(value());
      if (text !== undefined)
        record.span.setAttribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, text);
    }
  }
  private _end(runId: unknown, output?: () => unknown, error?: unknown): void {
    if (typeof runId !== "string") return;
    const record = this._runs.get(runId);
    if (!record) return;
    this._runs.delete(runId);
    try {
      if (this._admitted(record)) {
        if (error !== undefined) {
          const httpStatus = data(error, "status");
          if (
            typeof httpStatus === "number" &&
            Number.isInteger(httpStatus) &&
            httpStatus >= 100 &&
            httpStatus <= 599
          )
            record.span.setAttribute(
              ATTR_HTTP_RESPONSE_STATUS_CODE,
              httpStatus,
            );
          const errorName = data(error, "name");
          if (typeof errorName === "string")
            record.span.setAttribute(ATTR_ERROR_TYPE, errorName);
          const message =
            record.policy.inputs && record.policy.outputs
              ? getErrorMessage(error)
              : undefined;
          record.span.setStatus({
            code: SpanStatusCode.ERROR,
            ...(message !== undefined ? { message } : {}),
          });
          if (
            record.policy.inputs &&
            record.policy.outputs &&
            error instanceof Error
          )
            record.span.recordException(error);
        } else if (record.policy.outputs && output) {
          let value = output();
          if (value === undefined && record.tokens.length)
            value = record.tokens.join("");
          const text = safeJsonString(value);
          if (text !== undefined)
            record.span.setAttribute(
              SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
              text,
            );
        }
        // Canonical input, tools, output, usage, and status take priority at OTel's default budget.
        const ordered = Object.fromEntries(
          Object.entries(record.indexed).sort(
            ([a], [b]) =>
              Number(b.startsWith(SpanAttributes.LLM_COMPLETIONS)) -
              Number(a.startsWith(SpanAttributes.LLM_COMPLETIONS)),
          ),
        );
        record.span.setAttributes(ordered);
      }
    } finally {
      record.span.end();
      this._completed.set(runId, {
        span: record.span,
        policy: record.policy,
        path: record.path,
        workflowName: record.workflowName,
      });
      trimMap(
        this._completed,
        Math.max(0, Math.trunc(this.maxCachedRuns)) || 0,
      );
    }
  }
  handleChainStart(
    serialized: unknown,
    inputs: unknown,
    runId: unknown,
    parentRunId?: unknown,
    tags?: unknown,
    metadata?: unknown,
    _runType?: unknown,
    runName?: unknown,
  ): void {
    const root = parentRunId == null;
    const logType =
      root && data(metadata, "ls_integration") === "langchain_create_agent"
        ? RespanLogType.AGENT
        : root
          ? RespanLogType.WORKFLOW
          : RespanLogType.TASK;
    const record = this._start(
      serialized,
      runId,
      parentRunId,
      tags,
      metadata,
      runName,
      logType,
    );
    this._input(record, () => inputs);
  }
  handleChainEnd(outputs: unknown, runId: unknown): void {
    this._end(runId, () => outputs);
  }
  handleChainError(error: unknown, runId: unknown): void {
    this._end(runId, undefined, error);
  }
  private _modelStart(
    serialized: unknown,
    input: unknown,
    runId: unknown,
    parentRunId: unknown,
    extra: unknown,
    tags: unknown,
    metadata: unknown,
    runName: unknown,
    chat: boolean,
  ): void {
    const model = extractModel(serialized, undefined, metadata, extra);
    const record = this._start(
      serialized,
      runId,
      parentRunId,
      tags,
      metadata,
      runName,
      chat ? RespanLogType.CHAT : RespanLogType.TEXT,
      model,
    );
    if (!record || !this._admitted(record) || !record.policy.inputs) return;
    const normalized = chat ? normalizeChatMessages(input) : snapshot(input);
    this._input(record, () => normalized);
    const tools = extractTools({ serialized, extraParams: extra });
    const toolText = safeJsonString(tools);
    if (toolText !== undefined)
      record.span.setAttribute(SpanAttributes.LLM_REQUEST_FUNCTIONS, toolText);
    const messages = chat
      ? ((normalized as Record<string, any>[][])[0] ?? [])
      : (Array.isArray(normalized) ? normalized : [normalized]).map(
          (content) => ({ role: "user", content }),
        );
    this._messages(record, SpanAttributes.LLM_PROMPTS, messages);
  }
  private _messages(
    record: RunRecord,
    prefix: string,
    messages: Record<string, any>[],
  ): void {
    messages.forEach((message, index) => {
      for (const [key, value] of Object.entries(message))
        if (value !== undefined)
          record.indexed[`${prefix}.${index}.${key}`] =
            typeof value === "object" ? JSON.stringify(value) : value;
    });
  }
  handleChatModelStart(
    serialized: unknown,
    messages: unknown,
    runId: unknown,
    parentRunId?: unknown,
    extra?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
  ): void {
    this._modelStart(
      serialized,
      messages,
      runId,
      parentRunId,
      extra,
      tags,
      metadata,
      runName,
      true,
    );
  }
  handleLLMStart(
    serialized: unknown,
    prompts: unknown,
    runId: unknown,
    parentRunId?: unknown,
    extra?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
  ): void {
    this._modelStart(
      serialized,
      prompts,
      runId,
      parentRunId,
      extra,
      tags,
      metadata,
      runName,
      false,
    );
  }
  handleLLMNewToken(token: unknown, _idx?: unknown, runId?: unknown): void {
    const record =
      typeof runId === "string" ? this._runs.get(runId) : undefined;
    if (
      record &&
      this._admitted(record) &&
      record.policy.outputs &&
      typeof token === "string"
    )
      record.tokens.push(token);
  }
  handleLLMEnd(output: unknown, runId: unknown): void {
    const record =
      typeof runId === "string" ? this._runs.get(runId) : undefined;
    if (!record) return;
    let payload: unknown;
    if (this._admitted(record)) {
      const model = extractModel(undefined, output);
      if (!record.model && model)
        record.span.setAttribute(SpanAttributes.LLM_REQUEST_MODEL, model);
      const usage = extractUsage(output);
      for (const [key, value] of [
        [SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.promptTokens],
        [ATTR_GEN_AI_USAGE_INPUT_TOKENS, usage.promptTokens],
        [SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completionTokens],
        [ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, usage.completionTokens],
        [SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.totalTokens],
        [ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, usage.cacheReadInputTokens],
      ] as const)
        if (value !== undefined) record.span.setAttribute(key, value);
      if (record.policy.outputs) {
        const extracted = extractLlmOutput(output);
        payload = extracted.outputPayload;
        this._messages(
          record,
          SpanAttributes.LLM_COMPLETIONS,
          extracted.completionMessages,
        );
        const calls = extractToolCallsFromMessages(
          extracted.completionMessages,
        );
        if (calls)
          record.span.setAttribute(
            `${SpanAttributes.LLM_COMPLETIONS}.0.tool_calls`,
            JSON.stringify(calls),
          );
      }
    }
    this._end(runId, () => payload);
  }
  handleLLMError(error: unknown, runId: unknown): void {
    this._end(runId, undefined, error);
  }
  handleToolStart(
    serialized: unknown,
    input: unknown,
    runId: unknown,
    parentRunId?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
    toolCallId?: unknown,
  ): void {
    const record = this._start(
      serialized,
      runId,
      parentRunId,
      tags,
      metadata,
      runName,
      RespanLogType.TOOL,
    );
    if (record && this._admitted(record) && typeof toolCallId === "string")
      record.span.setAttribute(ATTR_GEN_AI_TOOL_CALL_ID, toolCallId);
    this._input(record, () => {
      let args = input;
      if (typeof input === "string") {
        try {
          args = JSON.parse(input);
        } catch {}
      }
      return { name: record!.name, arguments: args };
    });
  }
  handleToolEnd(output: unknown, runId: unknown): void {
    const record =
      typeof runId === "string" ? this._runs.get(runId) : undefined;
    if (record && this._admitted(record)) {
      const id = data(output, "tool_call_id");
      if (typeof id === "string")
        record.span.setAttribute(ATTR_GEN_AI_TOOL_CALL_ID, id);
      const nativeName = data(output, "name");
      if (typeof nativeName === "string" && nativeName) {
        record.span.setAttribute(
          SpanAttributes.TRACELOOP_ENTITY_NAME,
          nativeName,
        );
        if (record.policy.inputs) {
          const input = data(
            (record.span as any).attributes,
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
          );
          if (typeof input === "string") {
            try {
              record.span.setAttribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                JSON.stringify({ ...JSON.parse(input), name: nativeName }),
              );
            } catch {}
          }
        }
      }
    }
    this._end(runId, () =>
      data(output, "tool_call_id") !== undefined
        ? data(output, "content")
        : output,
    );
  }
  handleToolError(error: unknown, runId: unknown): void {
    this._end(runId, undefined, error);
  }
  handleRetrieverStart(
    serialized: unknown,
    query: unknown,
    runId: unknown,
    parentRunId?: unknown,
    tags?: unknown,
    metadata?: unknown,
    runName?: unknown,
  ): void {
    const record = this._start(
      serialized,
      runId,
      parentRunId,
      tags,
      metadata,
      runName,
      RespanLogType.TASK,
    );
    this._input(record, () => query);
  }
  handleRetrieverEnd(documents: unknown, runId: unknown): void {
    this._end(runId, () => documents);
  }
  handleRetrieverError(error: unknown, runId: unknown): void {
    this._end(runId, undefined, error);
  }
  handleText(text: unknown, runId?: unknown): void {
    this.handleLLMNewToken(text, undefined, runId);
  }
  handleAgentAction(action: unknown, runId?: unknown): void {
    const record =
      typeof runId === "string" ? this._runs.get(runId) : undefined;
    if (
      record &&
      this._admitted(record) &&
      record.policy.inputs &&
      record.policy.outputs
    )
      record.span.addEvent("langchain.agent.action", {
        action: safeJsonString(action) ?? "",
      });
  }
  handleAgentEnd(_action: unknown, _runId?: unknown): void {
    /* The enclosing chain callback owns the actual agent boundary. */
  }
  handleCustomEvent(
    eventName: string,
    dataValue: unknown,
    runId?: unknown,
    tags?: unknown,
    metadata?: unknown,
  ): void {
    const eventId = globalThis.crypto.randomUUID();
    const record = this._start(
      undefined,
      eventId,
      runId,
      tags,
      metadata,
      eventName,
      RespanLogType.TASK,
    );
    this._input(record, () => dataValue);
    this._end(eventId, () => dataValue);
  }
}
