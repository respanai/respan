import { context, trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
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
import {
  RespanLogType,
  RespanSpanAttributes,
} from "@respan/respan-sdk";
import {
  buildReadableSpan,
  ensureTraceId,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";
import type {
  FlueContext,
  FlueEvent,
  FlueEventSubscriber,
  LlmAssistantMessage,
  LlmMessage,
  LlmTool,
  LlmToolCall,
  PromptUsage,
} from "@flue/runtime";

const PACKAGE_VERSION = "0.1.0";
const INSTRUMENTATION_NAME = "@respan/instrumentation-flue";
const FLUE_INSTRUMENTATION_NAME = "flue";
const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const STATUS_CODE_ATTR = "status_code";

const GEN_AI_COMPLETION_PREFIX = `${ATTR_GEN_AI_COMPLETION}.0`;
const GEN_AI_COMPLETION_ROLE = `${GEN_AI_COMPLETION_PREFIX}.role`;
const GEN_AI_COMPLETION_CONTENT = `${GEN_AI_COMPLETION_PREFIX}.content`;
const GEN_AI_COMPLETION_TOOL_CALLS = `${GEN_AI_COMPLETION_PREFIX}.tool_calls`;

type FlueObserve = (subscriber: FlueEventSubscriber) => () => void;

export interface FlueRuntimeModule {
  observe?: FlueObserve;
}

export interface FlueInstrumentorOptions {
  runtimeModule?: FlueRuntimeModule;
  workflowName?: string;
}

interface StartedEvent {
  timestamp: string;
  event: FlueEvent;
}

interface TurnState {
  start?: StartedEvent;
  request?: Extract<FlueEvent, { type: "turn_request" }>;
}

interface ToolState {
  start?: StartedEvent;
}

interface SpanInput {
  name: string;
  logType: RespanLogType;
  entityName: string;
  event: FlueEvent;
  attributes?: Record<string, unknown>;
  input?: unknown;
  output?: unknown;
  start?: StartedEvent;
  durationMs?: number;
  error?: unknown;
  parentId?: string;
  spanId?: string;
  statusCode?: number;
  workflowName?: string;
}

export class FlueInstrumentor {
  public readonly name = FLUE_INSTRUMENTATION_NAME;

  private readonly _runtimeModule?: FlueRuntimeModule;
  private readonly _fallbackWorkflowName?: string;
  private _unsubscribe?: () => void;
  private _isActive = false;

  private readonly _traceIds = new Map<string, string>();
  private readonly _workflowNames = new Map<string, string>();
  private readonly _runStarts = new Map<string, StartedEvent>();
  private readonly _agentStarts = new Map<string, StartedEvent>();
  private readonly _operationStarts = new Map<string, StartedEvent>();
  private readonly _taskStarts = new Map<string, StartedEvent>();
  private readonly _toolStarts = new Map<string, ToolState>();
  private readonly _turns = new Map<string, TurnState>();
  private readonly _compactionStarts = new Map<string, StartedEvent>();

  constructor(options: FlueInstrumentorOptions = {}) {
    this._runtimeModule = options.runtimeModule;
    this._fallbackWorkflowName = options.workflowName;
  }

  async activate(): Promise<void> {
    if (this._isActive) {
      return;
    }

    const runtime = this._runtimeModule ?? await this._resolveRuntimeModule();
    if (!runtime?.observe) {
      return;
    }

    this._unsubscribe = runtime.observe((event, ctx) => {
      this.handleEvent(event, ctx);
    });
    this._isActive = true;
  }

  deactivate(): void {
    this._unsubscribe?.();
    this._unsubscribe = undefined;
    this._isActive = false;
    this._traceIds.clear();
    this._workflowNames.clear();
    this._runStarts.clear();
    this._agentStarts.clear();
    this._operationStarts.clear();
    this._taskStarts.clear();
    this._toolStarts.clear();
    this._turns.clear();
    this._compactionStarts.clear();
  }

  isActive(): boolean {
    return this._isActive;
  }

  handleEvent(event: FlueEvent, _ctx?: FlueContext): void {
    switch (event.type) {
      case "run_start":
        this._runStarts.set(event.runId, { event, timestamp: event.startedAt ?? event.timestamp });
        this._workflowNames.set(this._traceKey(event), event.workflowName);
        break;
      case "run_resume":
        this._runStarts.set(event.runId, { event, timestamp: event.startedAt ?? event.timestamp });
        this._workflowNames.set(this._traceKey(event), event.workflowName);
        break;
      case "run_end":
        this._emitRunEnd(event);
        break;
      case "agent_start":
        this._agentStarts.set(this._agentKey(event), { event, timestamp: event.timestamp });
        break;
      case "agent_end":
        this._emitAgentEnd(event);
        break;
      case "operation_start":
        this._operationStarts.set(event.operationId, { event, timestamp: event.timestamp });
        break;
      case "operation":
        this._emitOperation(event);
        break;
      case "task_start":
        this._taskStarts.set(event.taskId, { event, timestamp: event.timestamp });
        break;
      case "task":
        this._emitTask(event);
        break;
      case "tool_start":
        this._toolStarts.set(this._toolKey(event), {
          start: { event, timestamp: event.timestamp },
        });
        break;
      case "tool":
        this._emitTool(event);
        break;
      case "turn_start":
        this._upsertTurn(event.turnId).start = { event, timestamp: event.timestamp };
        break;
      case "turn_request":
        this._upsertTurn(event.turnId).request = event;
        break;
      case "turn":
        this._emitTurn(event);
        break;
      case "compaction_start":
        this._compactionStarts.set(this._compactionKey(event), { event, timestamp: event.timestamp });
        break;
      case "compaction":
        this._emitCompaction(event);
        break;
      case "log":
        this._emitLog(event);
        break;
      case "submission_settled":
        this._emitSubmissionSettled(event);
        break;
      case "idle":
      case "message_start":
      case "message_end":
      case "turn_messages":
      case "text_delta":
      case "thinking_start":
      case "thinking_delta":
      case "thinking_end":
        break;
    }
  }

  private async _resolveRuntimeModule(): Promise<FlueRuntimeModule | undefined> {
    try {
      return (await import("@flue/runtime")) as unknown as FlueRuntimeModule;
    } catch {
      return undefined;
    }
  }

  private _emitRunEnd(event: Extract<FlueEvent, { type: "run_end" }>): void {
    const start = this._runStarts.get(event.runId);
    const workflowName =
      this._workflowNames.get(this._traceKey(event)) ??
      startEventValue(start, "workflowName") ??
      this._fallbackWorkflowName ??
      "flue.workflow";

    this._emitSpan({
      name: `flue.workflow.${workflowName}`,
      logType: RespanLogType.WORKFLOW,
      entityName: String(workflowName),
      event,
      input: startEventValue(start, "payload"),
      output: event.isError ? errorOutput(event.error) : event.result,
      start,
      durationMs: event.durationMs,
      error: event.error,
      spanId: this._workflowSpanId(event),
      workflowName: String(workflowName),
    });
    this._runStarts.delete(event.runId);
  }

  private _emitAgentEnd(event: Extract<FlueEvent, { type: "agent_end" }>): void {
    const start = this._agentStarts.get(this._agentKey(event));
    const entityName = this._agentName(event);
    this._emitSpan({
      name: `flue.agent.${entityName}`,
      logType: RespanLogType.AGENT,
      entityName,
      event,
      input: this._identityInput(event),
      output: { messages: event.messages },
      start,
      spanId: this._agentSpanId(event),
      parentId: this._rootParentId(event),
    });
    this._agentStarts.delete(this._agentKey(event));
  }

  private _emitOperation(event: Extract<FlueEvent, { type: "operation" }>): void {
    const start = this._operationStarts.get(event.operationId);
    const logType = event.operationKind === "shell" ? RespanLogType.TOOL : RespanLogType.TASK;
    this._emitSpan({
      name: `flue.operation.${event.operationKind}`,
      logType,
      entityName: `flue.${event.operationKind}`,
      event,
      input: {
        operationId: event.operationId,
        operationKind: event.operationKind,
      },
      output: event.isError ? errorOutput(event.error) : event.result,
      start,
      durationMs: event.durationMs,
      error: event.error,
      spanId: this._operationSpanId(event),
      parentId: this._rootParentId(event),
      attributes: {
        [metadataKey("flue_operation_kind")]: event.operationKind,
      },
    });
    this._operationStarts.delete(event.operationId);
  }

  private _emitTask(event: Extract<FlueEvent, { type: "task" }>): void {
    const start = this._taskStarts.get(event.taskId);
    const taskStart = start?.event as Extract<FlueEvent, { type: "task_start" }> | undefined;
    const entityName = taskStart?.agent ? `flue.task.${taskStart.agent}` : "flue.task";
    this._emitSpan({
      name: entityName,
      logType: RespanLogType.TASK,
      entityName,
      event,
      input: {
        prompt: taskStart?.prompt,
        agent: taskStart?.agent ?? event.agent,
        cwd: taskStart?.cwd,
      },
      output: event.isError ? errorOutput(event.result) : event.result,
      start,
      durationMs: event.durationMs,
      error: event.isError ? event.result : undefined,
      spanId: this._taskSpanId(event),
      parentId: this._operationOrRootParentId(event),
      attributes: {
        [metadataKey("flue_task_agent")]: taskStart?.agent ?? event.agent,
      },
    });
    this._taskStarts.delete(event.taskId);
  }

  private _emitTool(event: Extract<FlueEvent, { type: "tool" }>): void {
    const state = this._toolStarts.get(this._toolKey(event));
    const startEvent = state?.start?.event as Extract<FlueEvent, { type: "tool_start" }> | undefined;
    this._emitSpan({
      name: `flue.tool.${event.toolName}`,
      logType: RespanLogType.TOOL,
      entityName: event.toolName,
      event,
      input: {
        name: event.toolName,
        arguments: startEvent?.args,
      },
      output: event.isError ? errorOutput(event.result) : event.result,
      start: state?.start,
      durationMs: event.durationMs,
      error: event.isError ? event.result : undefined,
      spanId: this._toolSpanId(event),
      parentId: this._operationOrRootParentId(event),
    });
    this._toolStarts.delete(this._toolKey(event));
  }

  private _emitTurn(event: Extract<FlueEvent, { type: "turn" }>): void {
    const state = this._turns.get(event.turnId);
    const request = state?.request;
    const attributes: Record<string, unknown> = {
      [SpanAttributes.LLM_REQUEST_TYPE]: RespanLogType.CHAT,
      [metadataKey("flue_turn_purpose")]: event.purpose,
    };

    const provider = request?.provider ?? event.provider;
    const model = request?.model ?? event.model;
    if (provider) {
      attributes[ATTR_GEN_AI_SYSTEM] = normalizeProvider(provider);
    }
    if (model) {
      attributes[ATTR_GEN_AI_REQUEST_MODEL] = normalizeModel(model, provider);
    }
    if (request?.api) {
      attributes[metadataKey("flue_model_api")] = request.api;
    } else if (event.api) {
      attributes[metadataKey("flue_model_api")] = event.api;
    }
    if (request?.reasoning) {
      attributes[metadataKey("flue_reasoning")] = request.reasoning;
    }

    if (request?.input) {
      addPromptAttributes(attributes, request.input.systemPrompt, request.input.messages);
      if (request.input.tools && request.input.tools.length > 0) {
        attributes[SpanAttributes.LLM_REQUEST_FUNCTIONS] = safeJson(
          request.input.tools.map(toToolDefinition),
        );
      }
    }
    if (event.output) {
      addCompletionAttributes(attributes, event.output);
    }
    addUsageAttributes(attributes, event.usage);

    this._emitSpan({
      name: `flue.turn.${event.purpose}`,
      logType: RespanLogType.CHAT,
      entityName: `flue.${event.purpose}.turn`,
      event,
      input: request?.input,
      output: event.isError ? errorOutput(event.error) : event.output,
      start: state?.start,
      durationMs: event.durationMs,
      error: event.error,
      spanId: this._turnSpanId(event),
      parentId: this._operationOrRootParentId(event),
      attributes,
    });
    this._turns.delete(event.turnId);
  }

  private _emitCompaction(event: Extract<FlueEvent, { type: "compaction" }>): void {
    const start = this._compactionStarts.get(this._compactionKey(event));
    const startEvent = start?.event as Extract<FlueEvent, { type: "compaction_start" }> | undefined;
    const attributes: Record<string, unknown> = {
      [metadataKey("flue_compaction_messages_before")]: event.messagesBefore,
      [metadataKey("flue_compaction_messages_after")]: event.messagesAfter,
    };
    if (startEvent?.reason) {
      attributes[metadataKey("flue_compaction_reason")] = startEvent.reason;
    }
    if (startEvent?.estimatedTokens !== undefined) {
      attributes[metadataKey("flue_compaction_estimated_tokens")] = startEvent.estimatedTokens;
    }
    addUsageMetadata(attributes, event.usage);

    this._emitSpan({
      name: "flue.compaction",
      logType: RespanLogType.TASK,
      entityName: "flue.compaction",
      event,
      input: {
        reason: startEvent?.reason,
        estimatedTokens: startEvent?.estimatedTokens,
      },
      output: {
        messagesBefore: event.messagesBefore,
        messagesAfter: event.messagesAfter,
      },
      start,
      durationMs: event.durationMs,
      error: event.error,
      parentId: this._operationOrRootParentId(event),
      spanId: this._compactionSpanId(event),
      attributes,
    });
    this._compactionStarts.delete(this._compactionKey(event));
  }

  private _emitLog(event: Extract<FlueEvent, { type: "log" }>): void {
    this._emitSpan({
      name: `flue.log.${event.level}`,
      logType: RespanLogType.TASK,
      entityName: `flue.log.${event.level}`,
      event,
      input: event.attributes ?? {},
      output: { level: event.level, message: event.message },
      parentId: this._operationOrRootParentId(event),
      spanId: this._eventSpanId(event, `log:${event.eventIndex}`),
      attributes: {
        [metadataKey("flue_log_level")]: event.level,
        [metadataKey("flue_log_message")]: event.message,
      },
    });
  }

  private _emitSubmissionSettled(event: Extract<FlueEvent, { type: "submission_settled" }>): void {
    this._emitSpan({
      name: `flue.submission.${event.outcome}`,
      logType: RespanLogType.TASK,
      entityName: `flue.submission.${event.outcome}`,
      event,
      input: { submissionId: event.submissionId },
      output: {
        outcome: event.outcome,
        error: event.error,
      },
      error: event.outcome === "failed" ? event.error : undefined,
      parentId: this._rootParentId(event),
      spanId: this._eventSpanId(event, `submission:${event.submissionId}`),
      attributes: {
        [metadataKey("flue_submission_id")]: event.submissionId,
        [metadataKey("flue_submission_outcome")]: event.outcome,
      },
    });
  }

  private _emitSpan(input: SpanInput): void {
    const event = input.event;
    const errorMessage = input.error === undefined ? undefined : errorMessageFrom(input.error);
    const statusCode = input.statusCode ?? (errorMessage ? 500 : 200);
    const attrs: Record<string, unknown> = {
      [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: input.logType,
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: input.entityName,
      [SpanAttributes.TRACELOOP_ENTITY_PATH]: input.entityName,
      ...(errorMessage
        ? {
            [ATTR_ERROR_MESSAGE]: errorMessage,
            [STATUS_CODE_ATTR]: statusCode,
          }
        : {}),
      [metadataKey("flue_event_type")]: event.type,
      [metadataKey("flue_event_index")]: event.eventIndex,
      ...identityMetadata(event),
      ...input.attributes,
    };

    const workflowName =
      input.workflowName ??
      this._workflowNames.get(this._traceKey(event)) ??
      this._fallbackWorkflowName;
    if (workflowName) {
      attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = workflowName;
    }

    if (input.input !== undefined) {
      attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(input.input);
    }
    if (input.output !== undefined) {
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(input.output);
    }

    const traceId = this._resolveTraceId(event);
    const activeContext = trace.getSpan(context.active())?.spanContext();
    const parentId = input.parentId ?? (
      activeContext?.traceId === traceId ? activeContext.spanId : undefined
    );
    const endTimeIso = event.timestamp;
    const startTimeIso =
      input.start?.timestamp ??
      startTimeFromDuration(event.timestamp, input.durationMs) ??
      event.timestamp;
    const readableSpan = buildReadableSpan({
      name: input.name,
      traceId,
      spanId: input.spanId,
      parentId,
      startTimeIso,
      endTimeIso,
      attributes: sanitizeAttributes(attrs),
      statusCode,
      errorMessage,
    }) as ReadableSpan & {
      instrumentationScope?: { name: string; version?: string };
    };

    readableSpan.instrumentationScope = {
      name: INSTRUMENTATION_NAME,
      version: PACKAGE_VERSION,
    };
    injectSpan(readableSpan);
  }

  private _resolveTraceId(event: FlueEvent): string {
    const key = this._traceKey(event);
    const existing = this._traceIds.get(key);
    if (existing) {
      return existing;
    }

    const activeTraceId = trace.getSpan(context.active())?.spanContext().traceId;
    const resolved = isUsableTraceId(activeTraceId)
      ? activeTraceId
      : ensureTraceId(key);
    this._traceIds.set(key, resolved);
    return resolved;
  }

  private _upsertTurn(turnId: string): TurnState {
    const existing = this._turns.get(turnId);
    if (existing) {
      return existing;
    }
    const state: TurnState = {};
    this._turns.set(turnId, state);
    return state;
  }

  private _traceKey(event: FlueEvent): string {
    return event.runId ??
      event.instanceId ??
      event.dispatchId ??
      event.submissionId ??
      `${event.harness ?? "harness"}:${event.session ?? "session"}`;
  }

  private _agentKey(event: FlueEvent): string {
    return `${this._traceKey(event)}:${event.instanceId ?? event.harness ?? event.session ?? "agent"}`;
  }

  private _toolKey(event: Extract<FlueEvent, { type: "tool_start" | "tool" }>): string {
    return `${this._traceKey(event)}:${event.toolCallId}`;
  }

  private _compactionKey(event: FlueEvent): string {
    return `${this._traceKey(event)}:${event.operationId ?? "compaction"}`;
  }

  private _agentName(event: FlueEvent): string {
    return event.instanceId ??
      event.harness ??
      event.session ??
      event.runId ??
      "agent";
  }

  private _identityInput(event: FlueEvent): Record<string, unknown> {
    return {
      runId: event.runId,
      instanceId: event.instanceId,
      dispatchId: event.dispatchId,
      harness: event.harness,
      session: event.session,
    };
  }

  private _workflowSpanId(event: FlueEvent): string {
    return this._eventSpanId(event, "workflow");
  }

  private _agentSpanId(event: FlueEvent): string {
    return this._eventSpanId(event, `agent:${this._agentName(event)}`);
  }

  private _operationSpanId(event: Extract<FlueEvent, { operationId: string }>): string {
    return this._eventSpanId(event, `operation:${event.operationId}`);
  }

  private _taskSpanId(event: Extract<FlueEvent, { taskId: string }>): string {
    return this._eventSpanId(event, `task:${event.taskId}`);
  }

  private _toolSpanId(event: Extract<FlueEvent, { toolCallId: string }>): string {
    return this._eventSpanId(event, `tool:${event.toolCallId}`);
  }

  private _turnSpanId(event: Extract<FlueEvent, { turnId: string }>): string {
    return this._eventSpanId(event, `turn:${event.turnId}`);
  }

  private _compactionSpanId(event: FlueEvent): string {
    return this._eventSpanId(event, `compaction:${event.operationId ?? event.eventIndex}`);
  }

  private _eventSpanId(event: FlueEvent, suffix: string): string {
    return `flue:${this._traceKey(event)}:${suffix}`;
  }

  private _rootParentId(event: FlueEvent): string | undefined {
    if (event.runId) {
      return this._workflowSpanId(event);
    }
    return undefined;
  }

  private _operationOrRootParentId(event: FlueEvent): string | undefined {
    if (event.operationId) {
      return this._operationSpanId(event as Extract<FlueEvent, { operationId: string }>);
    }
    return this._rootParentId(event) ?? (
      event.instanceId ? this._agentSpanId(event) : undefined
    );
  }
}

export { FlueInstrumentor as RespanFlueObserver };

function addPromptAttributes(
  attrs: Record<string, unknown>,
  systemPrompt: string | undefined,
  messages: LlmMessage[],
): void {
  let index = 0;
  if (systemPrompt) {
    attrs[promptRoleKey(index)] = "system";
    attrs[promptContentKey(index)] = systemPrompt;
    index += 1;
  }

  for (const message of messages) {
    attrs[promptRoleKey(index)] = normalizePromptRole(message.role);
    attrs[promptContentKey(index)] = messageToContent(message);
    const toolCalls = messageToToolCalls(message);
    if (toolCalls.length > 0) {
      attrs[promptToolCallsKey(index)] = safeJson(toolCalls);
    }
    index += 1;
  }
}

function addCompletionAttributes(
  attrs: Record<string, unknown>,
  message: LlmAssistantMessage,
): void {
  attrs[GEN_AI_COMPLETION_ROLE] = "assistant";
  attrs[GEN_AI_COMPLETION_CONTENT] = assistantContent(message);
  const toolCalls = assistantToolCalls(message);
  if (toolCalls.length > 0) {
    attrs[GEN_AI_COMPLETION_TOOL_CALLS] = safeJson(toolCalls);
  }
}

function addUsageAttributes(
  attrs: Record<string, unknown>,
  usage: PromptUsage | undefined,
): void {
  if (!usage) {
    return;
  }
  attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = usage.input;
  attrs[ATTR_GEN_AI_USAGE_PROMPT_TOKENS] = usage.input;
  attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = usage.output;
  attrs[ATTR_GEN_AI_USAGE_COMPLETION_TOKENS] = usage.output;
  attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = usage.totalTokens;
  addUsageMetadata(attrs, usage);
}

function addUsageMetadata(
  attrs: Record<string, unknown>,
  usage: PromptUsage | undefined,
): void {
  if (!usage) {
    return;
  }
  attrs[metadataKey("flue_usage_cache_read_tokens")] = usage.cacheRead;
  attrs[metadataKey("flue_usage_cache_write_tokens")] = usage.cacheWrite;
  attrs[metadataKey("flue_usage_cost_total")] = usage.cost.total;
}

function promptRoleKey(index: number): string {
  return `${ATTR_GEN_AI_PROMPT}.${index}.role`;
}

function promptContentKey(index: number): string {
  return `${ATTR_GEN_AI_PROMPT}.${index}.content`;
}

function promptToolCallsKey(index: number): string {
  return `${ATTR_GEN_AI_PROMPT}.${index}.tool_calls`;
}

function normalizePromptRole(role: string): string {
  if (role === "toolResult") {
    return "tool";
  }
  return role;
}

function messageToContent(message: LlmMessage): string {
  if (message.role === "toolResult") {
    return contentBlocksToText(message.content);
  }
  if (typeof message.content === "string") {
    return message.content;
  }
  return contentBlocksToText(message.content);
}

function messageToToolCalls(message: LlmMessage): unknown[] {
  if (message.role !== "assistant") {
    return [];
  }
  return assistantToolCalls(message);
}

function assistantContent(message: LlmAssistantMessage): string {
  return contentBlocksToText(
    message.content.filter((block) => block.type !== "toolCall"),
  );
}

function assistantToolCalls(message: LlmAssistantMessage): unknown[] {
  return message.content
    .filter((block): block is LlmToolCall => block.type === "toolCall")
    .map(toOpenAIToolCall);
}

function contentBlocksToText(blocks: Array<Record<string, any>>): string {
  return blocks
    .map((block) => {
      if (block.type === "text") {
        return block.text ?? "";
      }
      if (block.type === "thinking") {
        return block.thinking ?? "";
      }
      if (block.type === "image") {
        return `[image:${block.mimeType ?? "unknown"}]`;
      }
      if (block.type === "toolCall") {
        return "";
      }
      return safeJson(block);
    })
    .filter(Boolean)
    .join("\n");
}

function toOpenAIToolCall(call: LlmToolCall): unknown {
  return {
    id: call.id,
    type: "function",
    function: {
      name: call.name,
      arguments: safeJson(call.arguments ?? {}),
    },
  };
}

function toToolDefinition(tool: LlmTool): unknown {
  return {
    name: tool.name,
    description: tool.description,
    parameters: toSerializableValue(tool.parameters),
  };
}

function normalizeProvider(provider: string): string {
  return provider.toLowerCase();
}

function normalizeModel(model: string, provider?: string): string {
  const prefix = provider ? `${provider}/` : undefined;
  if (prefix && model.startsWith(prefix)) {
    return model.slice(prefix.length);
  }
  return model;
}

function startEventValue<T extends string>(start: StartedEvent | undefined, key: T): unknown {
  return start?.event && key in start.event
    ? (start.event as Record<string, unknown>)[key]
    : undefined;
}

function startTimeFromDuration(endTimeIso: string, durationMs: number | undefined): string | undefined {
  if (durationMs === undefined) {
    return undefined;
  }
  const endMs = new Date(endTimeIso).getTime();
  if (!Number.isFinite(endMs)) {
    return undefined;
  }
  return new Date(endMs - durationMs).toISOString();
}

function errorMessageFrom(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message?: unknown }).message);
  }
  return safeJson(error);
}

function errorOutput(error: unknown): unknown {
  if (error instanceof Error) {
    return {
      error: error.name,
      message: error.message,
      stack: error.stack,
    };
  }
  if (error && typeof error === "object") {
    return toSerializableValue(error);
  }
  return {
    error: String(error),
  };
}

function identityMetadata(event: FlueEvent): Record<string, unknown> {
  return {
    [metadataKey("flue_run_id")]: event.runId,
    [metadataKey("flue_instance_id")]: event.instanceId,
    [metadataKey("flue_dispatch_id")]: event.dispatchId,
    [metadataKey("flue_submission_id")]: event.submissionId,
    [metadataKey("flue_harness")]: event.harness,
    [metadataKey("flue_session")]: event.session,
    [metadataKey("flue_parent_session")]: event.parentSession,
    [metadataKey("flue_operation_id")]: event.operationId,
    [metadataKey("flue_turn_id")]: event.turnId,
    [metadataKey("flue_task_id")]: event.taskId,
  };
}

function metadataKey(key: string): string {
  return `${RespanSpanAttributes.RESPAN_METADATA}.${key}`;
}

function sanitizeAttributes(attrs: Record<string, unknown>): Record<string, string | number | boolean | string[] | number[] | boolean[]> {
  const sanitized: Record<string, string | number | boolean | string[] | number[] | boolean[]> = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null) {
      continue;
    }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      sanitized[key] = value;
      continue;
    }
    if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
      sanitized[key] = value;
      continue;
    }
    if (Array.isArray(value) && value.every((item) => typeof item === "number")) {
      sanitized[key] = value;
      continue;
    }
    if (Array.isArray(value) && value.every((item) => typeof item === "boolean")) {
      sanitized[key] = value;
      continue;
    }
    sanitized[key] = safeJson(value);
  }
  return sanitized;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(toSerializableValue(value), (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
  } catch {
    return JSON.stringify(String(value));
  }
}

function toSerializableValue(value: unknown): unknown {
  if (value === undefined) {
    return null;
  }
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (value instanceof Error) {
    return errorOutput(value);
  }
  if (Array.isArray(value)) {
    return value.map(toSerializableValue);
  }
  if (typeof value === "object") {
    const output: Record<string, unknown> = {};
    for (const [key, innerValue] of Object.entries(value as Record<string, unknown>)) {
      if (typeof innerValue === "function" || typeof innerValue === "symbol") {
        continue;
      }
      output[key] = toSerializableValue(innerValue);
    }
    return output;
  }
  return String(value);
}

function isUsableTraceId(value: string | undefined): value is string {
  return Boolean(value && /^[0-9a-f]{32}$/i.test(value));
}
