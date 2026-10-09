/**
 * `PiSessionTracer` — the pi event → Respan span state machine.
 *
 * This is a pure translator: it never imports pi, never touches the disk and
 * keeps only the state the *current* agent run needs (the pending LLM call's
 * converted input delta, in-flight tool calls and a handful of counters).
 * One instance is one pi session. Both adapters in `index.ts` — pi extension
 * events and `AgentSession.subscribe()` events — drive the same normalized
 * handlers below.
 *
 * Trace model: by default ONE TRACE PER PI SESSION (`traceScope: "session"`):
 * every agent run (user prompt → `agent_end`) adds a root to the trace whose
 * id is derived from the pi session id, so an email chain resumed over weeks
 * and across processes stays one trace. `traceScope: "run"` gives one trace
 * per run instead. Either way each run is one root agent span displayed as
 * `agent.turn-<n>` (n = the prompt's index within the session, like
 * Braintrust's "Turn n"), with one chat span per assistant message and one tool
 * span per tool execution underneath. Every run of a pi session shares
 * `respan.threads.thread_identifier` / `respan.sessions.session_identifier` /
 * `respan.trace.trace_group_identifier` = the pi session id, so a session that
 * is resumed after a week of silence simply adds another trace to the same
 * thread. With `traceScope: "session"` every run of the session shares one
 * trace id derived from the session id (one multi-root trace per session:
 * Session > turn-1, turn-2, … > llm / tool). No span is ever held open across
 * idle time in either scope.
 */

import { createRequire } from "node:module";
import { createHash } from "node:crypto";

import {
  context,
  ROOT_CONTEXT,
  isSpanContextValid,
  SpanStatusCode,
  trace,
  type Context,
  type HrTime,
} from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { RespanLogType, RespanSpanAttributes } from "@respan/respan-sdk";
import {
  buildReadableSpan,
  ensureSpanId,
  ensureTraceId,
  injectSpan,
} from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

import { PiCapturePolicy, own, safeCopy, safeSpanContext } from "./_capture.js";

import type { PiModelLike, PiToolDefinitionLike } from "./_pi_types.js";

const packageManifest = createRequire(import.meta.url)("../package.json") as {
  version?: string;
};

/** Version of this package as published (read from package.json, so pipeline bumps are reflected). */
export const PACKAGE_VERSION: string = packageManifest.version ?? "unknown";
export const PI_INSTRUMENTATION_NAME = "@respan/instrumentation-pi";

const RESPAN_LOG_METHOD_TS_TRACING = "ts_tracing";
const DEFAULT_WORKFLOW_NAME = "pi";
const DEFAULT_AGENT_NAME = "pi";
// Unlimited by default: captured content is the customer's data, and size is a
// storage concern handled at ingest (see contribution/span-contract.md), not
// something the translator should silently drop. `maxContentChars` is opt-in.
const DEFAULT_MAX_CONTENT_CHARS = 0;
const STATUS_CODE_ATTR = "status_code";
const ERROR_MESSAGE_ATTR = "error.message";
const LLM_USAGE_CACHE_READ_INPUT_TOKENS = "llm.usage.cache_read_input_tokens";
const GEN_AI_PROMPT_PREFIX = SpanAttributes.LLM_PROMPTS;
const GEN_AI_COMPLETION_PREFIX = SpanAttributes.LLM_COMPLETIONS;
const INTERRUPTED_MESSAGE = "Interrupted before completion";
const SHUTDOWN_MESSAGE = "Session shut down before the agent run completed";
const COMPACTION_ABORTED_MESSAGE = "Compaction aborted";
const TOOL_FAILED_MESSAGE = "Tool execution failed";

export type PiPromptCapture = "delta" | "full";
export type PiTraceScope = "run" | "session";
export type PiMetadataValue = string | number | boolean;

export interface PiTracerOptions {
  /** Capture payloads unless a parent/context/environment veto disables them. Default true. */
  traceContent?: boolean;
  /** `traceloop.workflow.name` on the run span. Default `"pi"`. */
  workflowName?: string;
  /** Agent name (`respan.metadata.agent_name`); the run span itself is displayed as `agent.turn-<n>`. Default `"pi"`. */
  agentName?: string;
  /**
   * `"session"` (default): every run of a pi session shares one trace id
   * derived from the pi session id (see `sessionTraceId`), so a long-lived
   * session — an email chain resumed over weeks, from any process — is one
   * multi-root trace; each run still emits its own root turn (agent) span.
   * `"run"`: one trace per agent run (a prompt → `agent_end`), nested under
   * an active OTEL span when there is one.
   */
  traceScope?: PiTraceScope;
  /**
   * `"full"` (default) records the whole context the model saw on every chat
   * span, like the other Respan instrumentations and Braintrust's pi
   * extension. `"delta"` is an opt-in for very high-volume deployments: each
   * chat span then records only the messages appended since the previous LLM
   * call of the same run (volume linear instead of quadratic in turn count;
   * the full conversation is still reconstructable from the trace).
   */
  promptCapture?: PiPromptCapture;
  /** Capture system instructions on full-context chats; once per run in delta mode. Default true. */
  captureSystemPrompt?: boolean;
  /** Record assistant thinking blocks on chat spans. Default `true`. */
  captureReasoning?: boolean;
  /** Emit one tool span per tool execution. Default `true`. */
  captureToolSpans?: boolean;
  /** Optional per-string capture cap in characters. Default `0` = unlimited (nothing is truncated). */
  maxContentChars?: number;
  customerIdentifier?: string;
  /** Override for `respan.threads.thread_identifier`; default = pi session id. */
  threadIdentifier?: string;
  metadata?: Record<string, PiMetadataValue>;
  /** Span sink. Default: `injectSpan` from `@respan/tracing`. */
  emit?: (span: ReadableSpan) => void;
  /** Spans are emitted only while this returns `true`. Default: always. */
  enabled?: () => boolean;
  /** Called after each run (prompt) closes, e.g. to show a link to its trace. */
  onRunEnd?: (info: PiRunEndInfo) => void;
}

export interface PiGitInfo {
  repository?: string;
  branch?: string;
  commit?: string;
}

export interface PiSessionInfo {
  sessionId?: string;
  sessionFile?: string;
  cwd?: string;
  piVersion?: string;
  /** Git metadata of the working directory (recorded on the turn span). */
  git?: PiGitInfo;
}

export interface PiRunEndInfo {
  traceId: string;
  turnNumber: number;
  sessionId?: string;
}

export interface PiContextOptions {
  /** Latest Pi supplies finalized system context after the compatibility context event. */
  replacePending?: boolean;
  /** LLM request start time (defaults to now). */
  startTime?: HrTime;
}

type Attrs = Record<string, unknown>;
type RecordValue = Record<string, unknown>;

interface ToolCallRecord {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface PromptMessage {
  role: string;
  content: string;
  tool_calls?: ToolCallRecord[];
  tool_call_id?: string;
}

interface PendingLlm {
  spanId: string;
  policy: PiCapturePolicy;
  startTime: HrTime;
  messages: PromptMessage[];
  offset?: number;
  firstTokenTime?: HrTime;
  truncated: boolean;
  turnIndex?: number;
}

interface PendingTool {
  policy: PiCapturePolicy;
  spanId: string;
  startTime: HrTime;
  toolName: string;
  args: unknown;
}

interface PendingTask {
  policy: PiCapturePolicy;
  traceContext: { traceId: string; parentSpanId?: string };
  spanId: string;
  startTime: HrTime;
  input: RecordValue;
}

interface RunState {
  policy: PiCapturePolicy;
  traceId: string;
  agentSpanId: string;
  parentSpanId?: string;
  /** 1-based index of this run (user prompt) within the pi session. */
  turnNumber: number;
  startTime: HrTime;
  prompt: string;
  promptKnown: boolean;
  /** The run's own prompt message was seen (every later user message is a steer). */
  promptMessageSeen: boolean;
  /**
   * User messages pi delivered into this run while it was working
   * (`steer()` / `followUp()`), in order. Each one takes the next turn number.
   */
  steers: string[];
  /** Turn numbers of steers no LLM call has consumed yet (flagged on the next chat span). */
  unseenSteerTurns: number[];
  systemPrompt?: string;
  truncated: boolean;
  chatCount: number;
  turnCount: number;
  toolCallCount: number;
  currentTurnIndex?: number;
  lastAssistantText: string;
  lastStopReason?: string;
  lastErrorMessage?: string;
  /** Delta cursor: number of context messages already captured in this run. */
  contextCursor?: number;
  pendingLlm: PendingLlm[];
  pendingTools: Map<string, PendingTool>;
  assistantMessageStart?: HrTime;
}

interface SpanRequest {
  policy?: PiCapturePolicy;
  name: string;
  traceId: string;
  spanId: string;
  parentId?: string;
  startTime: HrTime;
  endTime: HrTime;
  attributes: Attrs;
  statusCode?: number;
  errorMessage?: string;
}

/** Tracks whether any string captured for one span had to be truncated. */
class ContentCapture {
  public truncated = false;

  constructor(private readonly maxChars: number) {}

  text(value: string): string {
    if (this.maxChars <= 0 || value.length <= this.maxChars) {
      return value;
    }
    this.truncated = true;
    const dropped = value.length - this.maxChars;
    return `${value.slice(0, this.maxChars)} …[truncated ${dropped} chars]`;
  }

  deep(value: unknown): unknown {
    if (typeof value === "string") {
      return this.text(value);
    }
    if (Array.isArray(value)) {
      return value.map((item) => this.deep(item));
    }
    if (isRecord(value)) {
      const out: RecordValue = {};
      for (const [key, item] of Object.entries(value)) {
        out[key] = this.deep(item);
      }
      return out;
    }
    return value;
  }
}

/**
 * Trace id shared by every run of a pi session in `traceScope: "session"`.
 * pi session ids are UUIDs: with the dashes removed they are already 32 hex
 * chars and are used as-is (lowercased), so the Respan trace id can be looked
 * up from the session id. Anything else is hashed with SHA-256 (first 32 hex
 * chars) — collision-safe at any session volume, unlike a short repeated hash.
 */
export function sessionTraceId(sessionId: string): string {
  const hex = sessionId.replace(/-/g, "").toLowerCase();
  if (/^[0-9a-f]{32}$/.test(hex)) {
    return hex;
  }
  return createHash("sha256").update(sessionId).digest("hex").slice(0, 32);
}

export class PiSessionTracer {
  public readonly workflowName: string;
  public readonly agentName: string;
  public readonly traceScope: PiTraceScope;
  public readonly promptCapture: PiPromptCapture;
  public readonly captureSystemPrompt: boolean;
  public readonly captureReasoning: boolean;
  public readonly captureToolSpans: boolean;
  public readonly maxContentChars: number;

  private readonly customerIdentifier?: string;
  private readonly threadIdentifier?: string;
  private readonly metadata: Record<string, PiMetadataValue>;
  private readonly emitFn: (span: ReadableSpan) => void;
  private readonly enabledFn: () => boolean;
  private readonly traceContent: boolean;
  private policy: PiCapturePolicy;
  private policyTraceContext?: { traceId: string; parentSpanId?: string };
  private agentSpanId?: string;
  private pendingToolDefinitions?: unknown;
  private readonly onRunEndFn?: (info: PiRunEndInfo) => void;

  private session: PiSessionInfo = {};
  private model?: PiModelLike;
  private thinkingLevel?: string;
  /** Number of runs (prompts) seen by this tracer; the turn number when the adapter cannot read session history. */
  private runCounter = 0;
  private toolDefinitionsJson?: string;
  private toolDefinitionsTruncated = false;

  private run?: RunState;
  private lastPrompt?: string;
  private lastPromptTruncated = false;
  private compaction?: PendingTask;
  private branchSummary?: PendingTask;

  constructor(options: PiTracerOptions = {}) {
    this.workflowName =
      nonEmptyString(options.workflowName) ?? DEFAULT_WORKFLOW_NAME;
    this.agentName = nonEmptyString(options.agentName) ?? DEFAULT_AGENT_NAME;
    this.traceScope = options.traceScope === "run" ? "run" : "session";
    this.promptCapture = options.promptCapture === "delta" ? "delta" : "full";
    this.captureSystemPrompt = options.captureSystemPrompt ?? true;
    this.captureReasoning = options.captureReasoning ?? true;
    this.captureToolSpans = options.captureToolSpans ?? true;
    this.maxContentChars = normalizeMaxChars(options.maxContentChars);
    this.customerIdentifier = nonEmptyString(options.customerIdentifier);
    this.threadIdentifier = nonEmptyString(options.threadIdentifier);
    this.metadata = normalizeMetadata(options.metadata);
    this.emitFn = options.emit ?? ((span) => void injectSpan(span));
    this.enabledFn = options.enabled ?? (() => true);
    this.traceContent = options.traceContent !== false;
    this.policy = new PiCapturePolicy(this.traceContent, this.enabledFn());
    this.onRunEndFn = options.onRunEnd;
  }

  /** Internal adapter admission: evaluate policy before inspecting event payloads. */
  prepareEvent(event: unknown): RecordValue {
    const kind = own(event, "type");

    const snapshot: RecordValue = {};
    // Payload references are admitted by the owning run/chat/tool policy below.
    // Do not traverse them just to dispatch a lifecycle event.
    for (const field of [
      ...EVENT_FIELDS,
      "prompt",
      "systemPrompt",
      "messages",
      "message",
      "assistantMessageEvent",
      "args",
      "result",
      "preparation",
      "summaryEntry",
      "compactionEntry",
      "summary",
      "error",
      "errorMessage",
      "finalError",
      "model",
    ]) {
      const value = own(event, field);
      if (value !== undefined) snapshot[field] = value;
    }
    return snapshot;
  }

  get capturesContent(): boolean {
    this.policy.observe();
    return this.policy.recording && this.policy.content;
  }

  private beginPolicy(turnNumber?: unknown): void {
    if (this.policyTraceContext && !this.run) return;
    this.policyTraceContext = this.traceContext();
    this.agentSpanId = ensureSpanId();
    this.policy = new PiCapturePolicy(this.traceContent, this.enabledFn());
    this.policy.admit(
      this.policyTraceContext.traceId,
      this.agentSpanId,
      this.samplingContext(),
      `${this.agentName}.turn-${positiveInteger(turnNumber) ?? this.runCounter + 1}.agent`,
      {
        [RespanSpanAttributes.RESPAN_LOG_TYPE]: RespanLogType.AGENT,
        [SpanAttributes.TRACELOOP_ENTITY_NAME]: this.agentName,
      },
    );
  }

  private samplingContext(): Context {
    return this.traceScope === "session" && this.session.sessionId
      ? ROOT_CONTEXT
      : context.active();
  }
  private childPolicy(
    name: string,
    spanId: string,
    logType: RespanLogType,
  ): PiCapturePolicy {
    const run = this.run;
    const policy = new PiCapturePolicy(
      this.traceContent,
      this.enabledFn(),
      run?.policy,
    );
    const ids = run ?? this.traceContext();
    const parent = run?.policy.sampledContext
      ? trace.setSpanContext(context.active(), run.policy.sampledContext)
      : this.samplingContext();
    policy.admit(ids.traceId, spanId, parent, name, {
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: name,
    });
    return policy;
  }

  // ── Session context ─────────────────────────────────────────────────────

  setSession(info: PiSessionInfo): void {
    info = safeCopy(info) ?? {};
    const sessionId = nonEmptyString(info.sessionId);
    const sessionFile = nonEmptyString(info.sessionFile);
    const cwd = nonEmptyString(info.cwd);
    const piVersion = nonEmptyString(info.piVersion);
    if (sessionId) this.session.sessionId = sessionId;
    if (sessionFile) this.session.sessionFile = sessionFile;
    if (cwd) this.session.cwd = cwd;
    if (piVersion) this.session.piVersion = piVersion;
    if (info.git) this.session.git = { ...info.git };
  }

  setModel(model: unknown): void {
    model = safeCopy(model);
    if (!isRecord(model)) {
      return;
    }
    this.model = {
      id: nonEmptyString(model.id),
      provider: nonEmptyString(model.provider),
      name: nonEmptyString(model.name),
      api: nonEmptyString(model.api),
    };
  }

  setThinkingLevel(level: unknown): void {
    const value = nonEmptyString(level);
    if (value) {
      this.thinkingLevel = value;
    }
  }

  /**
   * Tool catalog recorded as `llm.request.functions` on every chat span. It is
   * re-sent on each LLM call, so it is capped like every other captured string
   * (per description / parameter schema and as a whole) — a large MCP catalog
   * must not dwarf the delta-captured prompts.
   */
  setToolDefinitions(tools: unknown): void {
    const policy = this.run?.pendingLlm.at(-1)?.policy;
    if (!policy) {
      this.pendingToolDefinitions = tools;
      return;
    }
    tools = policy.snapshot(tools);
    if (!Array.isArray(tools)) {
      this.toolDefinitionsJson = undefined;
      this.toolDefinitionsTruncated = false;
      return;
    }
    const capture = this.capture();
    const normalized: PiToolDefinitionLike[] = [];
    for (const tool of tools) {
      if (!isRecord(tool) || typeof tool.name !== "string" || !tool.name) {
        continue;
      }
      const definition: PiToolDefinitionLike = { ...tool, name: tool.name };
      if (typeof tool.description === "string") {
        definition.description = capture.text(tool.description);
      }
      if (tool.parameters !== undefined) {
        definition.parameters = capture.deep(toSerializable(tool.parameters));
      }
      normalized.push(definition);
    }
    this.toolDefinitionsJson =
      normalized.length > 0 ? capture.text(safeJson(normalized)) : undefined;
    this.toolDefinitionsTruncated =
      this.toolDefinitionsJson !== undefined && capture.truncated;
  }

  get sessionId(): string | undefined {
    return this.session.sessionId;
  }

  get hasOpenRun(): boolean {
    return this.run !== undefined;
  }

  get hasPendingLlm(): boolean {
    return this.run !== undefined && this.run.pendingLlm.length > 0;
  }

  // ── Normalized event handlers ───────────────────────────────────────────

  onSessionStart(_event?: { reason?: string }): void {
    // Nothing to emit: runs open on before_agent_start / agent_start.
  }

  onBeforeAgentStart(event: {
    prompt?: unknown;
    systemPrompt?: unknown;
    /** 1-based number of this prompt within the session, when the adapter knows it. */
    turnNumber?: unknown;
  }): void {
    if (this.run) {
      // A run was still open (missed agent_end); close it before starting the next.
      this.closeRun();
    }
    this.beginPolicy(own(event, "turnNumber"));
    event = this.policy.snapshot(event, EVENT_FIELDS) ?? {};
    const capture = this.capture();
    const prompt = renderContent(event?.prompt, capture);
    this.lastPrompt = prompt;
    this.lastPromptTruncated = capture.truncated;
    this.openRun({
      prompt,
      promptKnown: true,
      systemPrompt: nonEmptyString(event?.systemPrompt),
      truncated: capture.truncated,
      turnNumber: positiveInteger(event?.turnNumber),
    });
  }

  onAgentStart(event: { turnNumber?: unknown } = {}): void {
    if (this.run) {
      return;
    }
    this.beginPolicy(own(event, "turnNumber"));
    event = this.policy.snapshot(event, EVENT_FIELDS) ?? {};
    this.openRun({
      prompt: "",
      promptKnown: false,
      truncated: false,
      turnNumber: positiveInteger(event?.turnNumber),
    });
  }

  /**
   * Record the messages that will be sent to the LLM. Extension mode calls
   * this on `context`; subscribe mode calls it with a `session.messages`
   * snapshot when the assistant message starts streaming.
   */
  prepareLlm(options: PiContextOptions = {}): boolean {
    const run = this.run;
    if (!run) return false;
    let pending = run.pendingLlm.at(-1);
    if (!pending) {
      const spanId = ensureSpanId();
      pending = {
        spanId,
        policy: this.childPolicy("pi.chat", spanId, RespanLogType.CHAT),
        startTime: options.startTime ?? hrTime(),
        messages: [],
        truncated: false,
        turnIndex: run.currentTurnIndex,
      };
      run.pendingLlm.push(pending);
    }
    pending.policy.observe();
    return pending.policy.recording && pending.policy.content;
  }

  onContext(messages: unknown, options: PiContextOptions = {}): void {
    const run = this.run;
    if (!run) {
      return;
    }
    const previous = options.replacePending ? run.pendingLlm.pop() : undefined;
    const spanId = previous?.spanId ?? ensureSpanId();
    const policy =
      previous?.policy ??
      this.childPolicy("pi.chat", spanId, RespanLogType.CHAT);
    messages = policy.snapshot(messages) ?? [];
    const list = Array.isArray(messages) ? messages : [];
    const total = list.length;
    let start = 0;
    if (this.promptCapture === "delta") {
      const cursor = run.contextCursor;
      if (cursor === undefined || cursor > total) {
        start = lastUserIndex(list) ?? Math.max(0, total - 1);
      } else if (cursor === total) {
        // No new messages since the previous call (e.g. a retry): keep the last one.
        start = Math.max(0, total - 1);
      } else {
        start = cursor;
      }
    }
    const capture = this.capture();
    const converted: PromptMessage[] = [];
    for (const message of list.slice(start)) {
      const prompt = convertMessage(message, capture);
      if (prompt) {
        converted.push(prompt);
      }
    }
    run.contextCursor = total;
    run.pendingLlm.push({
      spanId,
      policy,
      startTime: previous?.startTime ?? options.startTime ?? hrTime(),
      messages: converted,
      offset: start,
      truncated: capture.truncated,
      turnIndex: run.currentTurnIndex,
    });
    if (this.pendingToolDefinitions !== undefined) {
      const tools = this.pendingToolDefinitions;
      this.pendingToolDefinitions = undefined;
      this.setToolDefinitions(tools);
    }
  }

  onTurnStart(event?: { turnIndex?: unknown }): void {
    const run = this.run;
    if (!run) {
      return;
    }
    const turnIndex = integerValue(event?.turnIndex);
    run.currentTurnIndex = turnIndex ?? run.turnCount;
    run.turnCount += 1;
  }

  onTurnEnd(_event?: {
    turnIndex?: unknown;
    message?: unknown;
    toolResults?: unknown;
  }): void {
    // Turn boundaries are already reflected by the chat/tool spans.
  }

  onMessageStart(message: unknown): void {
    const run = this.run;
    if (run && own(message, "role") === "assistant") {
      run.assistantMessageStart = hrTime();
      return;
    }
    message = this.policy.snapshot(message, MESSAGE_FIELDS);
    if (!run || !isRecord(message)) {
      return;
    }
    if (message.role === "assistant") {
      run.assistantMessageStart = hrTime();
      return;
    }
    if (message.role !== "user") {
      return;
    }
    // pi emits the run's own prompt as the first user message of the run.
    // Any user message after that — or after the model already answered, for
    // a continuation run that has no prompt message — was queued with
    // `steer()` / `followUp()` and delivered into the running turn.
    if (
      run.promptMessageSeen ||
      run.chatCount > 0 ||
      run.pendingLlm.length > 0
    ) {
      this.emitSteerSpan(run, message);
      return;
    }
    run.promptMessageSeen = true;
    if (!run.promptKnown) {
      const capture = this.capture();
      const prompt = renderContent(message.content, capture);
      run.prompt = prompt;
      run.promptKnown = true;
      run.truncated = run.truncated || capture.truncated;
      this.lastPrompt = prompt;
      this.lastPromptTruncated = capture.truncated;
    }
  }

  onMessageUpdate(event: {
    message?: unknown;
    assistantMessageEvent?: unknown;
  }): void {
    event = {
      assistantMessageEvent: {
        type: own(own(event, "assistantMessageEvent"), "type"),
      },
    };
    const run = this.run;
    if (!run || run.pendingLlm.length === 0) {
      return;
    }
    const kind = isRecord(event?.assistantMessageEvent)
      ? event.assistantMessageEvent.type
      : undefined;
    if (
      kind === "text_start" ||
      kind === "text_delta" ||
      kind === "thinking_start" ||
      kind === "thinking_delta"
    ) {
      const pending = run.pendingLlm[run.pendingLlm.length - 1];
      if (pending && !pending.firstTokenTime) {
        pending.firstTokenTime = hrTime();
      }
    }
  }

  onMessageEnd(message: unknown): void {
    const run = this.run;
    if (!run || own(message, "role") !== "assistant") {
      return;
    }
    const fallbackId = ensureSpanId();
    const pending: PendingLlm = run.pendingLlm.shift() ?? {
      spanId: fallbackId,
      policy: this.childPolicy("pi.chat", fallbackId, RespanLogType.CHAT),
      startTime: run.assistantMessageStart ?? hrTime(),
      messages: [],
      truncated: false,
      turnIndex: run.currentTurnIndex,
    };
    message = pending.policy.snapshot(message, MESSAGE_FIELDS);
    run.assistantMessageStart = undefined;
    this.emitChatSpan(run, pending, message as RecordValue, hrTime());
  }

  onToolExecutionStart(event: {
    toolCallId?: unknown;
    toolName?: unknown;
    args?: unknown;
  }): void {
    const run = this.run;
    if (!run) return;
    const toolCallId =
      nonEmptyString(own(event, "toolCallId")) ?? ensureSpanId();
    if (run.pendingTools.has(toolCallId)) {
      return;
    }
    const spanId = ensureSpanId();
    const toolName = nonEmptyString(own(event, "toolName")) ?? "tool";
    const policy = this.childPolicy(
      `${toolName}.tool`,
      spanId,
      RespanLogType.TOOL,
    );
    event = policy.snapshot(event, EVENT_FIELDS) ?? {};
    run.pendingTools.set(toolCallId, {
      spanId,
      policy,
      startTime: hrTime(),
      toolName: nonEmptyString(event?.toolName) ?? "tool",
      args: event?.args,
    });
  }

  onToolExecutionEnd(event: {
    toolCallId?: unknown;
    toolName?: unknown;
    result?: unknown;
    isError?: unknown;
  }): void {
    const run = this.run;
    if (!run) return;
    const toolCallId = nonEmptyString(own(event, "toolCallId")) ?? "";
    const endTime = hrTime();
    const pending = run.pendingTools.get(toolCallId);
    run.pendingTools.delete(toolCallId);
    const spanId = pending?.spanId ?? ensureSpanId();
    const policy =
      pending?.policy ??
      this.childPolicy(
        `${nonEmptyString(own(event, "toolName")) ?? "tool"}.tool`,
        spanId,
        RespanLogType.TOOL,
      );
    event = policy.snapshot(event, EVENT_FIELDS) ?? {};
    run.toolCallCount += 1;
    this.emitToolSpan(run, {
      spanId,
      policy,
      startTime: pending?.startTime ?? endTime,
      endTime,
      toolCallId,
      toolName: nonEmptyString(event?.toolName) ?? pending?.toolName ?? "tool",
      args: pending?.args,
      result: event?.result,
      isError: event?.isError === true,
    });
  }

  onAgentEnd(event?: { messages?: unknown; willRetry?: unknown }): void {
    event = this.policy.snapshot(event, EVENT_FIELDS);
    if (event?.willRetry === true) {
      return;
    }
    this.closeRun();
  }

  /**
   * `agent_settled`: pi guarantees no automatic retry, compaction or queued
   * continuation will follow. A run left open by `agent_end { willRetry: true }`
   * whose retry never happened (aborted during its backoff) is closed here so
   * the next prompt of the session starts its own run.
   */
  onAgentSettled(): void {
    this.closeRun();
  }

  /**
   * `auto_retry_end` (subscribe mode): a failed retry cycle — cancelled during
   * its backoff or out of attempts — ends the run with pi's final error.
   */
  onAutoRetryEnd(event?: { success?: unknown; finalError?: unknown }): void {
    event = this.policy.snapshot(event, EVENT_FIELDS);
    if (event?.success !== false || !this.run) {
      return;
    }
    const finalError = nonEmptyString(event.finalError);
    this.closeRun(finalError ? { errorMessage: finalError } : undefined);
  }

  onCompactionStart(event?: {
    reason?: unknown;
    willRetry?: unknown;
    tokensBefore?: unknown;
  }): void {
    event = safeCopy(event);
    const spanId = ensureSpanId();
    const policy = this.childPolicy(
      "pi.compaction",
      spanId,
      RespanLogType.TASK,
    );
    this.compaction = {
      spanId,
      policy,
      traceContext: {
        traceId: policy.sampledContext?.traceId ?? ensureTraceId(),
      },
      startTime: hrTime(),
      input: compactRecord({
        reason: nonEmptyString(event?.reason),
        willRetry: booleanValue(event?.willRetry),
        tokensBefore: integerValue(event?.tokensBefore),
      }),
    };
  }

  onCompactionEnd(event?: {
    summary?: unknown;
    tokensBefore?: unknown;
    tokensAfter?: unknown;
    firstKeptEntryId?: unknown;
    reason?: unknown;
    willRetry?: unknown;
    error?: unknown;
    aborted?: unknown;
  }): void {
    const pending = this.compaction;
    event = (pending?.policy ?? this.policy).snapshot(event, EVENT_FIELDS);
    this.compaction = undefined;
    const endTime = hrTime();
    const capture = this.capture();
    const input =
      pending?.input ??
      compactRecord({
        reason: nonEmptyString(event?.reason),
        willRetry: booleanValue(event?.willRetry),
        tokensBefore: integerValue(event?.tokensBefore),
      });
    const summary =
      typeof event?.summary === "string"
        ? capture.text(event.summary)
        : undefined;
    const output = compactRecord({
      summary,
      tokensBefore: integerValue(event?.tokensBefore),
      tokensAfter: integerValue(event?.tokensAfter),
      firstKeptEntryId: nonEmptyString(event?.firstKeptEntryId),
    });
    const errorMessage = nonEmptyString(event?.error);
    const aborted = event?.aborted === true;
    const attrs = this.baseAttrs(
      "compaction",
      "compaction",
      RespanLogType.TASK,
    );
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(input);
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(output);
    const reason =
      nonEmptyString(event?.reason) ?? nonEmptyString(input.reason);
    if (reason) {
      attrs[metadataKey("reason")] = reason;
    }
    if (capture.truncated) {
      attrs[metadataKey("truncated")] = true;
    }
    this.emitTaskSpan("pi.compaction", pending, attrs, endTime, {
      statusCode: errorMessage || aborted ? 500 : 200,
      errorMessage:
        errorMessage ?? (aborted ? COMPACTION_ABORTED_MESSAGE : undefined),
    });
  }

  onBranchSummaryStart(event?: {
    userWantsSummary?: unknown;
    targetId?: unknown;
    oldLeafId?: unknown;
    label?: unknown;
  }): void {
    event = safeCopy(event);
    if (event?.userWantsSummary !== true) {
      return;
    }
    const spanId = ensureSpanId();
    const policy = this.childPolicy(
      "pi.branch_summary",
      spanId,
      RespanLogType.TASK,
    );
    this.branchSummary = {
      spanId,
      policy,
      traceContext: {
        traceId: policy.sampledContext?.traceId ?? ensureTraceId(),
      },
      startTime: hrTime(),
      input: compactRecord({
        targetId: nonEmptyString(event?.targetId),
        oldLeafId: nonEmptyString(event?.oldLeafId),
        label: nonEmptyString(event?.label),
      }),
    };
  }

  onBranchSummaryEnd(event?: {
    newLeafId?: unknown;
    oldLeafId?: unknown;
    summaryEntry?: unknown;
  }): void {
    const pending = this.branchSummary;
    event = (pending?.policy ?? this.policy).snapshot(event, EVENT_FIELDS);
    this.branchSummary = undefined;
    const entry = isRecord(event?.summaryEntry)
      ? event.summaryEntry
      : undefined;
    if (!pending && !entry) {
      return;
    }
    const endTime = hrTime();
    const capture = this.capture();
    const output = compactRecord({
      summary:
        typeof entry?.summary === "string"
          ? capture.text(entry.summary)
          : undefined,
      label: nonEmptyString(entry?.label),
      id: nonEmptyString(entry?.id),
      fromId: nonEmptyString(entry?.fromId),
      parentId: nonEmptyString(entry?.parentId),
      newLeafId: nonEmptyString(event?.newLeafId),
      oldLeafId: nonEmptyString(event?.oldLeafId),
    });
    const attrs = this.baseAttrs(
      "branch_summary",
      "branch_summary",
      RespanLogType.TASK,
    );
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(
      pending?.input ?? {},
    );
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(output);
    if (capture.truncated) {
      attrs[metadataKey("truncated")] = true;
    }
    this.emitTaskSpan("pi.branch_summary", pending, attrs, endTime, {
      statusCode: 200,
    });
  }

  onSessionShutdown(_event?: { reason?: string }): void {
    if (this.run) {
      this.closeRun({ errorMessage: SHUTDOWN_MESSAGE });
    }
    this.compaction = undefined;
    this.branchSummary = undefined;
  }

  // ── Run lifecycle ───────────────────────────────────────────────────────

  private openRun(init: {
    prompt: string;
    promptKnown: boolean;
    systemPrompt?: string;
    truncated: boolean;
    turnNumber?: number;
  }): void {
    const { traceId, parentSpanId } =
      this.policyTraceContext ?? this.traceContext();
    // Turn numbering: the adapter passes the number of prompts already in the
    // session when it can read the session history (so numbering survives a
    // resume in a new process); otherwise count within this tracer.
    this.runCounter =
      init.turnNumber && init.turnNumber > 0
        ? init.turnNumber
        : this.runCounter + 1;
    // Random span ids: in session scope several processes (a resumed session)
    // write into one trace, so ids derived from a per-process counter could
    // collide.
    this.run = {
      policy: this.policy,
      traceId,
      agentSpanId: this.agentSpanId ?? ensureSpanId(),
      parentSpanId,
      turnNumber: this.runCounter,
      startTime: hrTime(),
      prompt: init.prompt,
      promptKnown: init.promptKnown,
      promptMessageSeen: false,
      steers: [],
      unseenSteerTurns: [],
      systemPrompt: init.systemPrompt,
      truncated: init.truncated,
      chatCount: 0,
      turnCount: 0,
      toolCallCount: 0,
      lastAssistantText: "",
      pendingLlm: [],
      pendingTools: new Map(),
    };
  }

  private closeRun(reason?: { errorMessage: string }): void {
    const run = this.run;
    if (!run) {
      return;
    }
    this.run = undefined;
    this.policyTraceContext = undefined;
    const endTime = hrTime();

    const continuation = !run.promptKnown;
    // Prompt and final text were already truncated when captured.
    const prompt = run.promptKnown ? run.prompt : (this.lastPrompt ?? "");
    const truncated =
      run.truncated || (continuation && this.lastPromptTruncated);
    // The turn's input is everything the user fed this run: the prompt and
    // the messages steered into it while it was working.
    const input = safeJson([
      { role: "user", content: prompt },
      ...run.steers.map((content) => ({ role: "user", content })),
    ]);
    const output = run.lastAssistantText;
    const toolCallCount = run.toolCallCount + run.pendingTools.size;

    let statusCode = 200;
    let errorMessage: string | undefined;
    if (reason) {
      statusCode = 500;
      errorMessage = reason.errorMessage;
    } else if (
      run.lastStopReason === "error" ||
      run.lastStopReason === "aborted"
    ) {
      statusCode = 500;
      errorMessage =
        run.lastErrorMessage ?? `pi assistant message ${run.lastStopReason}`;
    }

    // One run-level span per prompt: the agent span is the trace root (or the
    // child of an active OTEL span) and is displayed as `agent.turn-<n>`, like
    // Braintrust's "Turn n". It is a structural span: no model on it — the
    // model lives on the chat spans underneath.
    const turnDetail = `turn-${run.turnNumber}`;
    const agentAttrs = this.baseAttrs(this.agentName, "", RespanLogType.AGENT);
    agentAttrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = input;
    agentAttrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = output;
    agentAttrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME] = this.workflowName;
    agentAttrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME] =
      this.agentName;
    agentAttrs[RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND] = "agent";
    agentAttrs[RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL] =
      turnDetail;
    setMetadata(agentAttrs, "turn_number", run.turnNumber);
    setMetadata(agentAttrs, "pi_version", this.session.piVersion);
    setMetadata(agentAttrs, "thinking_level", this.thinkingLevel);
    setMetadata(agentAttrs, "session_file", this.session.sessionFile);
    setMetadata(agentAttrs, "cwd", this.session.cwd);
    setMetadata(agentAttrs, "turn_count", run.turnCount);
    setMetadata(agentAttrs, "tool_call_count", toolCallCount);
    if (run.steers.length > 0) {
      setMetadata(agentAttrs, "steer_count", run.steers.length);
    }
    setMetadata(agentAttrs, "stop_reason", run.lastStopReason);
    setMetadata(agentAttrs, "git_repository", this.session.git?.repository);
    setMetadata(agentAttrs, "git_branch", this.session.git?.branch);
    setMetadata(agentAttrs, "git_commit", this.session.git?.commit);
    if (continuation) {
      setMetadata(agentAttrs, "continuation", true);
    }
    if (truncated) {
      agentAttrs[metadataKey("truncated")] = true;
    }

    this.emitSpan({
      name: `${this.agentName}.${turnDetail}.agent`,
      traceId: run.traceId,
      spanId: run.agentSpanId,
      parentId: run.parentSpanId,
      startTime: run.startTime,
      endTime,
      attributes: agentAttrs,
      statusCode,
      errorMessage,
    });

    // Dangling LLM calls / tool executions: closed as interrupted.
    for (const pending of run.pendingLlm) {
      this.emitChatSpan(run, pending, undefined, endTime, INTERRUPTED_MESSAGE);
    }
    for (const [toolCallId, pending] of run.pendingTools) {
      this.emitToolSpan(run, {
        spanId: pending.spanId,
        policy: pending.policy,
        startTime: pending.startTime,
        endTime,
        toolCallId,
        toolName: pending.toolName,
        args: pending.args,
        result: undefined,
        isError: true,
        errorMessage: INTERRUPTED_MESSAGE,
      });
    }
    run.pendingLlm.length = 0;
    run.pendingTools.clear();

    if (this.onRunEndFn) {
      try {
        this.onRunEndFn({
          traceId: run.traceId,
          turnNumber: run.turnNumber,
          sessionId: this.session.sessionId,
        });
      } catch {
        // Callbacks are best effort and must never affect pi.
      }
    }
  }

  // ── Span builders ───────────────────────────────────────────────────────

  private emitChatSpan(
    run: RunState,
    pending: PendingLlm,
    message: RecordValue | undefined,
    endTime: HrTime,
    interruptedMessage?: string,
  ): void {
    run.chatCount += 1;
    const capture = this.capture();
    capture.truncated = pending.truncated;

    const prompts: PromptMessage[] = [];
    if (
      this.captureSystemPrompt &&
      (this.promptCapture === "full" || run.chatCount === 1) &&
      run.systemPrompt &&
      !pending.messages.some((message) => message.role === "system")
    ) {
      prompts.push({ role: "system", content: capture.text(run.systemPrompt) });
    }
    prompts.push(...pending.messages);

    const provider = nonEmptyString(message?.provider) ?? this.model?.provider;
    const attrs = this.baseAttrs(
      "pi.response",
      "pi.response",
      RespanLogType.CHAT,
    );
    if (provider) attrs[SpanAttributes.LLM_SYSTEM] = provider.toLowerCase();
    attrs[SpanAttributes.LLM_REQUEST_TYPE] = RespanLogType.CHAT;
    const requestModel = nonEmptyString(message?.model) ?? this.model?.id;
    if (requestModel) {
      attrs[SpanAttributes.LLM_REQUEST_MODEL] = requestModel;
    }
    const responseModel = nonEmptyString(message?.responseModel);
    if (responseModel) {
      attrs[SpanAttributes.LLM_RESPONSE_MODEL] = responseModel;
    }

    prompts.forEach((prompt, index) => {
      const prefix = `${GEN_AI_PROMPT_PREFIX}.${index}`;
      attrs[`${prefix}.role`] = prompt.role;
      attrs[`${prefix}.content`] = prompt.content;
      if (prompt.tool_calls && prompt.tool_calls.length > 0) {
        attrs[`${prefix}.tool_calls`] = safeJson(prompt.tool_calls);
      }
    });
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson(prompts);

    let statusCode = 200;
    let errorMessage: string | undefined;
    if (message) {
      const text = joinTextBlocks(message.content, "");
      const reasoning = joinThinkingBlocks(message.content);
      const toolCalls = toolCallsOf(message.content, capture);
      const completionContent = capture.text(text);
      run.lastAssistantText = completionContent;
      run.truncated = run.truncated || completionContent !== text;
      const stopReason = nonEmptyString(message.stopReason);
      run.lastStopReason = stopReason;
      run.lastErrorMessage = nonEmptyString(message.errorMessage);

      attrs[`${GEN_AI_COMPLETION_PREFIX}.0.role`] = "assistant";
      attrs[`${GEN_AI_COMPLETION_PREFIX}.0.content`] = completionContent;
      if (toolCalls.length > 0) {
        attrs[`${GEN_AI_COMPLETION_PREFIX}.0.tool_calls`] = safeJson(toolCalls);
      }
      const output: RecordValue = {
        role: "assistant",
        content: completionContent,
      };
      if (this.captureReasoning && reasoning) {
        output.reasoning = capture.text(reasoning);
      }
      if (toolCalls.length > 0) {
        output.tool_calls = toolCalls;
      }
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = safeJson(output);

      addUsageAttributes(attrs, message.usage);
      setMetadata(attrs, "stop_reason", stopReason);
      setMetadata(attrs, "response_id", nonEmptyString(message.responseId));
      setMetadata(attrs, "api", nonEmptyString(message.api));
      if (stopReason === "error" || stopReason === "aborted") {
        statusCode = 500;
        errorMessage =
          run.lastErrorMessage ?? `pi assistant message ${stopReason}`;
      }
    } else {
      attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = "";
      statusCode = 500;
      errorMessage = interruptedMessage ?? INTERRUPTED_MESSAGE;
    }

    attrs[metadataKey("prompt_capture")] = this.promptCapture;
    if (pending.offset !== undefined) {
      attrs[metadataKey("prompt_message_offset")] = pending.offset;
    }
    if (this.toolDefinitionsJson) {
      attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS] = this.toolDefinitionsJson;
      capture.truncated = capture.truncated || this.toolDefinitionsTruncated;
    }
    if (pending.firstTokenTime) {
      attrs[metadataKey("time_to_first_token_ms")] = hrTimeDiffMs(
        pending.startTime,
        pending.firstTokenTime,
      );
    }
    // pi emits turn_start before the LLM call, so the turn current at message_end
    // is this response's turn (the pending value covers dangling calls).
    setMetadata(attrs, "turn_index", run.currentTurnIndex ?? pending.turnIndex);
    // The first LLM call that sees a steered message is flagged, so
    // `after_steer = true` filters exactly the calls that reacted to a steer.
    if (run.unseenSteerTurns.length > 0) {
      // A string, not a boolean: the platform's metadata filter matches text,
      // and `after_steer = true` is what people type.
      setMetadata(attrs, "after_steer", "true");
      setMetadata(attrs, "steer_turn_numbers", run.unseenSteerTurns.join(","));
      run.unseenSteerTurns = [];
    }
    setMetadata(attrs, "thinking_level", this.thinkingLevel);
    if (capture.truncated) {
      attrs[metadataKey("truncated")] = true;
    }

    this.emitSpan({
      name: "pi.chat",
      policy: pending.policy,
      traceId: run.traceId,
      spanId: pending.spanId,
      parentId: run.agentSpanId,
      startTime: pending.startTime,
      endTime,
      attributes: attrs,
      statusCode,
      errorMessage,
    });
  }

  private emitToolSpan(
    run: RunState,
    tool: {
      spanId: string;
      policy: PiCapturePolicy;
      startTime: HrTime;
      endTime: HrTime;
      toolCallId: string;
      toolName: string;
      args: unknown;
      result: unknown;
      isError: boolean;
      errorMessage?: string;
    },
  ): void {
    if (!this.captureToolSpans) {
      return;
    }
    const capture = this.capture();
    const attrs = this.baseAttrs(
      tool.toolName,
      tool.toolName,
      RespanLogType.TOOL,
    );
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = capture.text(
      safeJson({
        name: tool.toolName,
        arguments: capture.deep(toSerializable(tool.args) ?? {}),
      }),
    );
    attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] = renderToolResult(
      tool.result,
      capture,
    );
    if (tool.toolCallId) {
      attrs[ATTR_GEN_AI_TOOL_CALL_ID] = tool.toolCallId;
    }
    const skillName = detectSkillName(tool.toolName, tool.args);
    if (skillName) {
      attrs[metadataKey("skill_name")] = skillName;
    }
    let statusCode = 200;
    let errorMessage: string | undefined;
    if (tool.isError) {
      statusCode = 500;
      errorMessage =
        tool.errorMessage ??
        nonEmptyString(capture.text(extractResultText(tool.result))) ??
        TOOL_FAILED_MESSAGE;
    }
    if (capture.truncated) {
      attrs[metadataKey("truncated")] = true;
    }
    this.emitSpan({
      name: `${tool.toolName}.tool`,
      policy: tool.policy,
      traceId: run.traceId,
      spanId: tool.spanId,
      parentId: run.agentSpanId,
      startTime: tool.startTime,
      endTime: tool.endTime,
      attributes: attrs,
      statusCode,
      errorMessage,
    });
  }

  /**
   * A user message pi delivered into a run that was already working
   * (`session.steer()` / `followUp()`, or typing while the agent streams).
   * It becomes its own `steer` span under the turn, so it is visible in the
   * tree instead of only as the last input item of the next LLM call. Turn
   * numbers count the session's user messages, so a steer takes the next
   * number: a turn-1 run with three steers is followed by turn 5.
   */
  private emitSteerSpan(run: RunState, message: RecordValue): void {
    const spanId = ensureSpanId();
    const name = `${this.agentName}.turn-${run.turnNumber + run.steers.length + 1}.steer`;
    const policy = this.childPolicy(name, spanId, RespanLogType.TASK);
    message = policy.snapshot(message, MESSAGE_FIELDS) ?? {};
    const capture = this.capture();
    const content = renderContent(message.content, capture);
    run.steers.push(content);
    run.truncated = run.truncated || capture.truncated;
    // Keep the in-tracer counter aligned with the session history count.
    this.runCounter = run.turnNumber + run.steers.length;
    const turnNumber = this.runCounter;
    run.unseenSteerTurns.push(turnNumber);
    const now = hrTime();

    const attrs = this.baseAttrs("steer", "steer", RespanLogType.TASK);
    attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT] = safeJson([
      { role: "user", content },
    ]);
    // Steering is a task under the shared semantic naming contract. The
    // original name and turn metadata retain its identity in legacy views.
    attrs[RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND] = "task";
    setMetadata(attrs, "turn_number", turnNumber);
    setMetadata(attrs, "steered_into_turn", run.turnNumber);
    setMetadata(attrs, "steer_index", run.steers.length);
    setMetadata(
      attrs,
      "delivered_after",
      run.chatCount === 0
        ? "run_start"
        : run.lastStopReason === "toolUse"
          ? "tool_results"
          : "assistant_reply",
    );
    if (capture.truncated) {
      attrs[metadataKey("truncated")] = true;
    }
    this.emitSpan({
      name,
      policy,
      traceId: run.traceId,
      spanId,
      parentId: run.agentSpanId,
      startTime: now,
      endTime: now,
      attributes: attrs,
    });
  }

  private emitTaskSpan(
    name: string,
    pending: PendingTask | undefined,
    attrs: Attrs,
    endTime: HrTime,
    status: { statusCode: number; errorMessage?: string },
  ): void {
    const run = this.run;
    let traceId: string;
    let parentId: string | undefined;
    if (run) {
      traceId = run.traceId;
      parentId = run.agentSpanId;
    } else {
      // Outside a run: a root of its own trace (run scope) or another root of
      // the session trace (session scope).
      ({ traceId, parentSpanId: parentId } =
        pending?.traceContext ?? this.traceContext());
    }
    this.emitSpan({
      name,
      policy: pending?.policy,
      traceId,
      spanId: pending?.spanId ?? ensureSpanId(),
      parentId,
      startTime: pending?.startTime ?? endTime,
      endTime,
      attributes: attrs,
      statusCode: status.statusCode,
      errorMessage: status.errorMessage,
    });
  }

  /**
   * Spans produced while the instrumentor is not active yet (e.g. the first
   * LLM call of an SDK session finishing before `Respan.initialize()` resolved)
   * are kept in a bounded buffer and emitted by `drainPending()` once emission
   * is enabled. Oldest spans are dropped beyond the cap.
   */
  private readonly pendingSpans: ReadableSpan[] = [];
  private static readonly MAX_PENDING_SPANS = 2000;

  drainPending(): number {
    if (!this.enabledFn() || this.pendingSpans.length === 0) {
      return 0;
    }
    const spans = this.pendingSpans.splice(0, this.pendingSpans.length);
    for (const span of spans) {
      if (this.policy.emit) this.emitFn(span);
    }
    return spans.length;
  }

  private emitSpan(request: SpanRequest): void {
    const policy = request.policy ?? this.policy;
    policy.observe();
    if (!policy.emit) return;
    Object.assign(request.attributes, policy.samplingAttributes);
    addStatusAttributes(
      request.attributes,
      request.statusCode ?? 200,
      request.errorMessage,
    );
    const span = buildReadableSpan({
      name: request.name,
      traceId: request.traceId,
      spanId: request.spanId,
      parentId: request.parentId,
      startTimeHr: request.startTime,
      endTimeHr: request.endTime,
      attributes: request.attributes,
      errorMessage: undefined,
    }) as ReadableSpan & {
      instrumentationScope?: { name: string; version?: string };
    };
    Object.assign(span, {
      status: {
        code:
          request.errorMessage || (request.statusCode ?? 0) >= 400
            ? SpanStatusCode.ERROR
            : SpanStatusCode.UNSET,
        ...(request.errorMessage ? { message: request.errorMessage } : {}),
      },
    });
    span.instrumentationScope = {
      name: PI_INSTRUMENTATION_NAME,
      version: PACKAGE_VERSION,
    };
    // Correlation precedence: explicit tracer options (already in attrs) >
    // attributes propagated via `respan.propagateAttributes()` (merged by
    // buildReadableSpan) > the pi session id.
    const merged = span.attributes as Attrs;
    const sessionId = this.session.sessionId;
    if (
      merged[RespanSpanAttributes.RESPAN_THREADS_ID] === undefined &&
      sessionId
    ) {
      merged[RespanSpanAttributes.RESPAN_THREADS_ID] = sessionId;
    }
    canonicalMetadata(merged);
    if (policy.sampledContext)
      Object.assign(span, { spanContext: () => policy.sampledContext });
    policy.guard(span);
    if (!this.enabledFn()) {
      if (this.pendingSpans.length >= PiSessionTracer.MAX_PENDING_SPANS) {
        this.pendingSpans.shift();
      }
      this.pendingSpans.push(span);
      return;
    }
    this.emitFn(span);
  }

  private baseAttrs(
    entityName: string,
    entityPath: string,
    logType: RespanLogType,
  ): Attrs {
    const attrs: Attrs = {
      [RespanSpanAttributes.RESPAN_LOG_METHOD]: RESPAN_LOG_METHOD_TS_TRACING,
      [RespanSpanAttributes.RESPAN_LOG_TYPE]: logType,
      [SpanAttributes.TRACELOOP_ENTITY_NAME]: entityName,
      [SpanAttributes.TRACELOOP_ENTITY_PATH]: entityPath,
      "telemetry.sdk.name": PI_INSTRUMENTATION_NAME,
      "telemetry.sdk.version": PACKAGE_VERSION,
    };
    const sessionId = this.session.sessionId;
    if (sessionId) {
      attrs[RespanSpanAttributes.RESPAN_SESSION_ID] = sessionId;
      // The trace group ties the traces of one resumed session together
      // (Respan's documented use of `trace_group_identifier`).
      attrs[RespanSpanAttributes.RESPAN_TRACE_GROUP_ID] = sessionId;
    }
    // The session-id default for the thread id is applied in emitSpan() so a
    // propagated thread_identifier can take precedence over it.
    if (this.threadIdentifier) {
      attrs[RespanSpanAttributes.RESPAN_THREADS_ID] = this.threadIdentifier;
    }
    if (this.customerIdentifier) {
      attrs[RespanSpanAttributes.RESPAN_CUSTOMER_PARAMS_ID] =
        this.customerIdentifier;
    }
    for (const [key, value] of Object.entries(this.metadata)) {
      attrs[metadataKey(key)] = value;
    }
    return attrs;
  }

  private capture(): ContentCapture {
    this.policy.observe();
    return new ContentCapture(this.maxContentChars);
  }

  /**
   * Trace id / parent for a new root (a run's turn span, or a compaction /
   * branch-summary span outside a run). Run scope nests under an active OTEL
   * span when there is one; session scope derives the trace id from the pi
   * session id and always emits a root (the session id wins over any active
   * span). Falls back to run behavior while the session id is unknown.
   */
  private traceContext(): { traceId: string; parentSpanId?: string } {
    const sessionId = this.session.sessionId;
    if (this.traceScope === "session" && sessionId) {
      return { traceId: sessionTraceId(sessionId) };
    }
    const active = activeSpanContext();
    return {
      traceId: ensureTraceId(active?.traceId),
      parentSpanId: active?.spanId,
    };
  }
}

// ── Message conversion ────────────────────────────────────────────────────

function convertMessage(
  message: unknown,
  capture: ContentCapture,
): PromptMessage | null {
  if (typeof message === "string") {
    return { role: "user", content: capture.text(message) };
  }
  if (!isRecord(message)) {
    return null;
  }
  const role = nonEmptyString(message.role) ?? "user";
  switch (role) {
    case "assistant": {
      const prompt: PromptMessage = {
        role: "assistant",
        content: capture.text(joinTextBlocks(message.content, "")),
      };
      const toolCalls = toolCallsOf(message.content, capture);
      if (toolCalls.length > 0) {
        prompt.tool_calls = toolCalls;
      }
      return prompt;
    }
    case "toolResult": {
      const prompt: PromptMessage = {
        role: "tool",
        content: renderToolResultContent(message.content, capture),
      };
      const toolCallId = nonEmptyString(message.toolCallId);
      if (toolCallId) {
        prompt.tool_call_id = toolCallId;
      }
      return prompt;
    }
    case "system": {
      const sections = isRecord(message.sections) ? message.sections : {};
      const content = [
        renderContent(message.content, capture),
        ...Object.values(sections).filter((value) => typeof value === "string"),
      ]
        .filter((value) => value !== "")
        .join("\n\n");
      return { role, content: capture.text(content) };
    }
    default:
      return { role, content: renderContent(message.content, capture) };
  }
}

function renderContent(content: unknown, capture: ContentCapture): string {
  if (typeof content === "string") {
    return capture.text(content);
  }
  if (content === undefined || content === null) {
    return "";
  }
  if (!Array.isArray(content)) {
    return capture.text(safeJson(toSerializable(content)));
  }
  const parts: string[] = [];
  for (const block of content) {
    const rendered = renderBlock(block);
    if (rendered !== undefined && rendered !== "") {
      parts.push(rendered);
    }
  }
  return capture.text(parts.join("\n\n"));
}

function renderBlock(block: unknown): string | undefined {
  if (typeof block === "string") {
    return block;
  }
  if (!isRecord(block)) {
    return block === undefined || block === null ? undefined : String(block);
  }
  switch (block.type) {
    case "text":
      return typeof block.text === "string" ? block.text : "";
    case "image":
      return imagePlaceholder(block);
    case "thinking":
    case "toolCall":
      return undefined;
    default:
      return safeJson(toSerializable(block));
  }
}

function renderToolResultContent(
  content: unknown,
  capture: ContentCapture,
): string {
  if (typeof content === "string") {
    return capture.text(content);
  }
  if (!Array.isArray(content)) {
    return content === undefined || content === null
      ? ""
      : capture.text(safeJson(toSerializable(content)));
  }
  if (allTextBlocks(content)) {
    return capture.text(joinTextBlocks(content, "\n"));
  }
  return capture.text(safeJson(renderBlocks(content)));
}

function renderToolResult(result: unknown, capture: ContentCapture): string {
  if (typeof result === "string") {
    return capture.text(result);
  }
  if (result === undefined) return "";
  return capture.text(safeJson(capture.deep(toSerializable(result))));
}

function renderBlocks(blocks: unknown[]): string[] {
  const rendered: string[] = [];
  for (const block of blocks) {
    const value = renderBlock(block);
    if (value !== undefined) {
      rendered.push(value);
    }
  }
  return rendered;
}

function extractResultText(result: unknown): string {
  if (typeof result === "string") {
    return result;
  }
  if (isRecord(result) && Array.isArray(result.content)) {
    return joinTextBlocks(result.content, "\n");
  }
  if (isRecord(result) && typeof result.error === "string") {
    return result.error;
  }
  return "";
}

function allTextBlocks(blocks: unknown[]): boolean {
  return (
    blocks.length > 0 &&
    blocks.every((block) => isRecord(block) && block.type === "text")
  );
}

function joinTextBlocks(content: unknown, separator: string): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts: string[] = [];
  for (const block of content) {
    if (
      isRecord(block) &&
      block.type === "text" &&
      typeof block.text === "string"
    ) {
      parts.push(block.text);
    }
  }
  return parts.join(separator);
}

function joinThinkingBlocks(content: unknown): string {
  if (!Array.isArray(content)) {
    return "";
  }
  const parts: string[] = [];
  for (const block of content) {
    if (
      isRecord(block) &&
      block.type === "thinking" &&
      typeof block.thinking === "string"
    ) {
      parts.push(block.thinking);
    }
  }
  return parts.join("");
}

function toolCallsOf(
  content: unknown,
  capture: ContentCapture,
): ToolCallRecord[] {
  if (!Array.isArray(content)) {
    return [];
  }
  const calls: ToolCallRecord[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== "toolCall") {
      continue;
    }
    calls.push({
      id: nonEmptyString(block.id) ?? ensureSpanId(),
      type: "function",
      function: {
        name: nonEmptyString(block.name) ?? "tool",
        arguments: capture.text(
          safeJson(toSerializable(block.arguments) ?? {}),
        ),
      },
    });
  }
  return calls;
}

function imagePlaceholder(block: RecordValue): string {
  const mimeType = nonEmptyString(block.mimeType) ?? "unknown";
  return `[image:${mimeType}]`;
}

function lastUserIndex(messages: unknown[]): number | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (isRecord(message) && message.role === "user") {
      return index;
    }
  }
  return undefined;
}

function detectSkillName(toolName: string, args: unknown): string | undefined {
  if (!isRecord(args)) {
    return undefined;
  }
  if (toolName === "skill") {
    return nonEmptyString(args.name);
  }
  if (toolName !== "read") {
    return undefined;
  }
  const raw = args.path ?? args.filePath ?? args.file_path;
  if (typeof raw !== "string") {
    return undefined;
  }
  const normalized = raw.replace(/\\/g, "/").replace(/\/+$/, "");
  if (!normalized.toLowerCase().endsWith("/skill.md")) {
    return undefined;
  }
  const segments = normalized.split("/");
  return segments.length >= 2
    ? nonEmptyString(segments[segments.length - 2])
    : undefined;
}

// ── Attribute helpers ─────────────────────────────────────────────────────

function metadataKey(key: string): string {
  return `${RespanSpanAttributes.RESPAN_METADATA}.${key}`;
}

function setMetadata(attrs: Attrs, key: string, value: unknown): void {
  if (value === undefined || value === null || value === "") {
    return;
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    attrs[metadataKey(key)] = value;
    return;
  }
  attrs[metadataKey(key)] = safeJson(toSerializable(value));
}

function addStatusAttributes(
  attrs: Attrs,
  statusCode: number,
  errorMessage?: string,
): void {
  if (statusCode < 400 && !errorMessage) {
    return;
  }
  // Pi exposes an outcome, not an HTTP response status. Never invent one.
  delete attrs[STATUS_CODE_ATTR];
  if (errorMessage) {
    attrs[ERROR_MESSAGE_ATTR] = errorMessage;
  }
}

function addUsageAttributes(attrs: Attrs, usage: unknown): void {
  if (!isRecord(usage)) {
    return;
  }
  const input = integerValue(usage.input);
  const output = integerValue(usage.output);
  const cacheRead = integerValue(usage.cacheRead);
  const cacheWrite = integerValue(usage.cacheWrite);
  const promptTokens =
    input === undefined
      ? undefined
      : input + (cacheRead ?? 0) + (cacheWrite ?? 0);
  const totalTokens = integerValue(usage.totalTokens);

  if (promptTokens !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_INPUT_TOKENS] = promptTokens;
    attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] = promptTokens;
  }
  if (output !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] = output;
    attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] = output;
  }
  if (totalTokens !== undefined)
    attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] = totalTokens;
  if (cacheRead !== undefined) {
    // Canonical semconv key (read by the backend for cache-aware cost) plus the
    // contract's legacy alias — publish both, like the prompt/completion pairs.
    attrs[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cacheRead;
    attrs[LLM_USAGE_CACHE_READ_INPUT_TOKENS] = cacheRead;
  }
  if (cacheWrite !== undefined) {
    attrs[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = cacheWrite;
  }
  const reasoning = integerValue(usage.reasoning);
  if (reasoning !== undefined) {
    attrs[metadataKey("reasoning_tokens")] = reasoning;
  }
  const cost = isRecord(usage.cost) ? numberValue(usage.cost.total) : undefined;
  if (cost !== undefined) {
    attrs[metadataKey("estimated_cost_usd")] = cost;
  }
}

function activeSpanContext(): { traceId: string; spanId: string } | undefined {
  const spanContext = safeSpanContext(trace.getSpan(context.active()));
  if (!spanContext || !isSpanContextValid(spanContext)) {
    return undefined;
  }
  return { traceId: spanContext.traceId, spanId: spanContext.spanId };
}

function hrTimeDiffMs(start: HrTime, end: HrTime): number {
  const ms = (end[0] - start[0]) * 1000 + (end[1] - start[1]) / 1_000_000;
  return Math.max(0, Math.round(ms));
}

function normalizeMaxChars(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return DEFAULT_MAX_CONTENT_CHARS;
  }
  return Math.trunc(value);
}

function normalizeMetadata(metadata: unknown): Record<string, PiMetadataValue> {
  const normalized: Record<string, PiMetadataValue> = {};
  if (!isRecord(metadata)) {
    return normalized;
  }
  for (const [key, value] of Object.entries(safeCopy(metadata) ?? {})) {
    if (!key || value === undefined || value === null) {
      continue;
    }
    normalized[key] =
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
        ? value
        : safeJson(toSerializable(value));
  }
  return normalized;
}

function compactRecord(record: Record<string, unknown>): RecordValue {
  const out: RecordValue = {};
  for (const [key, value] of Object.entries(record)) {
    if (value !== undefined) {
      out[key] = value;
    }
  }
  return out;
}

// ── Value helpers ─────────────────────────────────────────────────────────

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function positiveInteger(value: unknown): number | undefined {
  const parsed = integerValue(value);
  return parsed !== undefined && parsed > 0 ? parsed : undefined;
}

function integerValue(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.trunc(value);
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function booleanValue(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function safeJson(value: unknown): string {
  try {
    const serialized = JSON.stringify(value, (_key, innerValue) =>
      typeof innerValue === "bigint" ? innerValue.toString() : innerValue,
    );
    return serialized === undefined ? "" : serialized;
  } catch {
    return "";
  }
}

function toSerializable(value: unknown): unknown {
  return safeCopy(value);
}

function isRecord(value: unknown): value is RecordValue {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

const MESSAGE_FIELDS = [
  "role",
  "api",
  "provider",
  "model",
  "responseModel",
  "responseId",
  "stopReason",
  "usage",
  "toolCallId",
  "toolName",
  "isError",
  "timestamp",
] as const;
const EVENT_FIELDS = [
  "type",
  "turnIndex",
  "turnNumber",
  "toolCallId",
  "toolName",
  "isError",
  "willRetry",
  "success",
  "reason",
  "tokensBefore",
  "tokensAfter",
  "aborted",
  "userWantsSummary",
  "targetId",
  "oldLeafId",
  "newLeafId",
  "firstKeptEntryId",
  "level",
] as const;

function canonicalMetadata(attributes: Attrs): void {
  const key = RespanSpanAttributes.RESPAN_METADATA;
  let metadata: RecordValue = {};
  const current = attributes[key];
  if (typeof current === "string") {
    try {
      const parsed = JSON.parse(current);
      if (isRecord(parsed)) metadata = parsed;
    } catch {
      /* malformed metadata is omitted */
    }
  }
  for (const field of Object.keys(attributes)) {
    if (!field.startsWith(`${key}.`)) continue;
    metadata[field.slice(key.length + 1)] = attributes[field];
    delete attributes[field];
  }
  if (Object.keys(metadata).length > 0) attributes[key] = safeJson(metadata);
}
