import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { ROOT_CONTEXT, context, trace } from "@opentelemetry/api";
import {
  ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";
import { propagateAttributes } from "@respan/tracing";

import { PiInstrumentor, PiSessionTracer, createPiExtension, sessionTraceId } from "../dist/index.js";

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);

test.before(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value() {
      return {
        activeSpanProcessor: {
          onEnd(span) {
            captureState.spans.push(span);
          },
        },
      };
    },
  });
});

test.after(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
});

// ── Fakes ─────────────────────────────────────────────────────────────────

function createFakePi() {
  const handlers = new Map();
  return {
    handlers,
    on(event, handler) {
      const list = handlers.get(event) ?? [];
      list.push(handler);
      handlers.set(event, list);
    },
    getAllTools() {
      return [
        {
          name: "bash",
          description: "Run a shell command",
          parameters: { type: "object", properties: { command: { type: "string" } } },
        },
        {
          name: "read",
          description: "Read a file",
          parameters: { type: "object", properties: { path: { type: "string" } } },
        },
      ];
    },
    async emit(event, payload = {}, ctx = undefined) {
      const results = [];
      for (const handler of handlers.get(event) ?? []) {
        results.push(await handler({ type: event, ...payload }, ctx));
      }
      return results;
    },
  };
}

function createFakeCtx(overrides = {}) {
  const sessionId = overrides.sessionId ?? "sess-123";
  return {
    cwd: "/tmp/pi-demo",
    hasUI: false,
    mode: "print",
    model: {
      id: "claude-sonnet-4-5",
      provider: "anthropic",
      name: "Claude Sonnet 4.5",
      api: "anthropic-messages",
    },
    thinkingLevel: "medium",
    sessionManager: {
      getSessionId: () => sessionId,
      getSessionFile: () => `/tmp/pi-demo/.pi/sessions/${sessionId}.jsonl`,
    },
    ui: { setStatus() {} },
    ...overrides,
  };
}

function createFakeSession(sessionId, messages = []) {
  const listeners = new Set();
  return {
    sessionId,
    sessionFile: `/tmp/pi-sdk/${sessionId}.jsonl`,
    model: { id: "gpt-5", provider: "openai", name: "GPT-5", api: "openai-responses" },
    thinkingLevel: "low",
    messages,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    emit(event) {
      for (const listener of listeners) {
        listener(event);
      }
    },
    listenerCount() {
      return listeners.size;
    },
  };
}

function usage(overrides = {}) {
  return {
    input: 5,
    output: 2,
    cacheRead: 3,
    cacheWrite: 1,
    reasoning: 1,
    totalTokens: 11,
    cost: { input: 0.002, output: 0.004, cacheRead: 0.001, cacheWrite: 0.003, total: 0.01 },
    ...overrides,
  };
}

const userMessage = { role: "user", content: "Inspect the repo", timestamp: 1 };
const assistantWithTool = {
  role: "assistant",
  content: [
    { type: "thinking", thinking: "Let me look." },
    { type: "text", text: "Listing files." },
    { type: "toolCall", id: "call-1", name: "bash", arguments: { command: "ls" } },
  ],
  api: "anthropic-messages",
  provider: "anthropic",
  model: "claude-sonnet-4-5",
  responseId: "resp-1",
  usage: usage(),
  stopReason: "toolUse",
  timestamp: 2,
};
const toolResultMessage = {
  role: "toolResult",
  toolCallId: "call-1",
  toolName: "bash",
  content: [{ type: "text", text: "README.md" }],
  isError: false,
  timestamp: 3,
};
const assistantFinal = {
  role: "assistant",
  content: [{ type: "text", text: "The repo has a README." }],
  api: "anthropic-messages",
  provider: "anthropic",
  model: "claude-sonnet-4-5",
  responseId: "resp-2",
  usage: usage({ input: 7, output: 4, cacheRead: 0, cacheWrite: 0, totalTokens: 11, reasoning: undefined }),
  stopReason: "stop",
  timestamp: 4,
};

async function replayRun(pi, ctx, { toolOutput = "README.md", shutdown = true } = {}) {
  await pi.emit("session_start", { reason: "startup" }, ctx);
  await pi.emit(
    "before_agent_start",
    { prompt: "Inspect the repo", systemPrompt: "You are pi", systemPromptOptions: {} },
    ctx,
  );
  await pi.emit("agent_start", {}, ctx);
  await pi.emit("context", { messages: [userMessage] }, ctx);
  await pi.emit("turn_start", { turnIndex: 0, timestamp: Date.now() }, ctx);
  await pi.emit(
    "message_start",
    { message: { ...assistantWithTool, content: [], stopReason: "pending" } },
    ctx,
  );
  await pi.emit(
    "message_update",
    {
      message: { ...assistantWithTool, content: [{ type: "text", text: "Listing" }], stopReason: "pending" },
      assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "Listing", partial: {} },
    },
    ctx,
  );
  await pi.emit("message_end", { message: assistantWithTool }, ctx);
  await pi.emit(
    "tool_execution_start",
    { toolCallId: "call-1", toolName: "bash", args: { command: "ls" } },
    ctx,
  );
  await pi.emit(
    "tool_execution_end",
    {
      toolCallId: "call-1",
      toolName: "bash",
      result: { content: [{ type: "text", text: toolOutput }], details: { truncated: false } },
      isError: false,
    },
    ctx,
  );
  await pi.emit("turn_end", { turnIndex: 0, message: assistantWithTool, toolResults: [toolResultMessage] }, ctx);
  await pi.emit("context", { messages: [userMessage, assistantWithTool, toolResultMessage] }, ctx);
  await pi.emit("turn_start", { turnIndex: 1, timestamp: Date.now() }, ctx);
  await pi.emit("message_end", { message: assistantFinal }, ctx);
  await pi.emit("turn_end", { turnIndex: 1, message: assistantFinal, toolResults: [] }, ctx);
  await pi.emit(
    "agent_end",
    { messages: [userMessage, assistantWithTool, toolResultMessage, assistantFinal] },
    ctx,
  );
  if (shutdown) {
    await pi.emit("session_shutdown", { reason: "quit" }, ctx);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function spansByLogType(logType, spans = captureState.spans) {
  return spans.filter((span) => span.attributes["respan.entity.log_type"] === logType);
}

function spanByLogType(logType, spans = captureState.spans) {
  return spansByLogType(logType, spans)[0];
}

function parentSpanId(span) {
  return span.parentSpanContext?.spanId;
}

function assertNoBannedAliases(span) {
  assert.equal(span.attributes["traceloop.span.kind"], undefined);
  assert.equal(span.attributes["respan.span.tools"], undefined);
  assert.equal(span.attributes["respan.span.tool_calls"], undefined);
  assert.equal(span.attributes["respan.span.handoffs"], undefined);
  assert.equal(span.attributes.tools, undefined);
  assert.equal(span.attributes.tool_calls, undefined);
  assert.equal(span.attributes.model, undefined);
  assert.equal(span.attributes.prompt_tokens, undefined);
  assert.equal(span.attributes.completion_tokens, undefined);
  assert.equal(span.attributes.total_request_tokens, undefined);
  assert.equal(span.attributes.span_tools, undefined);
  assert.equal(span.attributes.has_tool_calls, undefined);
  if (span.attributes["respan.entity.log_type"] === "tool") {
    for (const key of Object.keys(span.attributes)) {
      assert.ok(!key.startsWith("gen_ai.tool."), `tool span must not carry ${key}`);
    }
  }
}

/** Minimal synchronous context manager so context.with() is honored in-process. */
class SyncContextManager {
  constructor() {
    this.current = ROOT_CONTEXT;
  }
  active() {
    return this.current;
  }
  with(ctx, fn, thisArg, ...args) {
    const previous = this.current;
    this.current = ctx;
    try {
      return fn.call(thisArg, ...args);
    } finally {
      this.current = previous;
    }
  }
  bind(_ctx, target) {
    return target;
  }
  enable() {
    return this;
  }
  disable() {
    this.current = ROOT_CONTEXT;
    return this;
  }
}

function assertCommonContract(span) {
  assert.equal(span.attributes["respan.entity.log_method"], "ts_tracing");
  assert.equal(span.attributes["telemetry.sdk.name"], "@respan/instrumentation-pi");
  assert.equal(
    span.attributes["telemetry.sdk.version"],
    JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version,
  );
  assert.equal(span.instrumentationScope.name, "@respan/instrumentation-pi");
  assert.equal(typeof span.attributes["traceloop.entity.name"], "string");
  assert.equal(typeof span.attributes["traceloop.entity.path"], "string");
  assert.equal(span.attributes["respan.trace.trace_group_identifier"] !== undefined, true);
  assertNoBannedAliases(span);
}

// ── Tests ─────────────────────────────────────────────────────────────────

test("extension replay emits canonical agent/chat/tool spans", async () => {
  captureState.spans = [];
  // Opt-in delta capture; the system prompt is recorded by default (first chat span).
  const instrumentor = new PiInstrumentor({ promptCapture: "delta" });
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  assert.equal(instrumentor.activeSessionCount, 1);

  const ctx = createFakeCtx();
  await replayRun(pi, ctx);

  // One root agent (turn) span per run — no workflow span.
  assert.equal(captureState.spans.length, 4);
  const agentSpans = spansByLogType("agent");
  const chatSpans = spansByLogType("chat");
  const toolSpans = spansByLogType("tool");
  assert.equal(spansByLogType("workflow").length, 0);
  assert.equal(agentSpans.length, 1);
  assert.equal(chatSpans.length, 2);
  assert.equal(toolSpans.length, 1);

  const [agentSpan] = agentSpans;
  const [chat1, chat2] = chatSpans;
  const [toolSpan] = toolSpans;

  // Hierarchy: the agent span is the trace root.
  assert.equal(parentSpanId(agentSpan), undefined);
  assert.equal(parentSpanId(chat1), agentSpan.spanContext().spanId);
  assert.equal(parentSpanId(chat2), agentSpan.spanContext().spanId);
  assert.equal(parentSpanId(toolSpan), agentSpan.spanContext().spanId);
  const traceId = agentSpan.spanContext().traceId;
  for (const span of captureState.spans) {
    assert.equal(span.spanContext().traceId, traceId);
    assert.equal(span.attributes["respan.threads.thread_identifier"], "sess-123");
    assert.equal(span.attributes["respan.sessions.session_identifier"], "sess-123");
    // The trace group is the pi session id too, so the traces of a resumed session group together.
    assert.equal(span.attributes["respan.trace.trace_group_identifier"], "sess-123");
    assertCommonContract(span);
  }

  // Agent (turn) span
  assert.equal(agentSpan.name, "pi.turn-1.agent");
  assert.equal(agentSpan.attributes["traceloop.entity.name"], "pi");
  assert.equal(agentSpan.attributes["traceloop.entity.path"], "");
  assert.equal(agentSpan.attributes["traceloop.workflow.name"], "pi");
  // Naming hints: the exporter displays the span as `agent.turn-1`.
  assert.equal(agentSpan.attributes["respan.internal.span_name.kind"], "agent");
  assert.equal(agentSpan.attributes["respan.internal.span_name.detail"], "turn-1");
  assert.equal(agentSpan.attributes["respan.metadata.turn_number"], 1);
  assert.deepEqual(JSON.parse(agentSpan.attributes["traceloop.entity.input"]), [
    { role: "user", content: "Inspect the repo" },
  ]);
  assert.equal(agentSpan.attributes["traceloop.entity.output"], "The repo has a README.");
  assert.equal(agentSpan.attributes.status_code, undefined);
  assert.equal(agentSpan.attributes["respan.metadata.agent_name"], "pi");
  // Structural span: the model lives on the chat spans only.
  assert.equal(agentSpan.attributes["gen_ai.request.model"], undefined);
  assert.equal(agentSpan.attributes["respan.metadata.turn_count"], 2);
  assert.equal(agentSpan.attributes["respan.metadata.tool_call_count"], 1);
  assert.equal(agentSpan.attributes["respan.metadata.stop_reason"], "stop");
  assert.equal(agentSpan.attributes["respan.metadata.thinking_level"], "medium");
  assert.equal(agentSpan.attributes["respan.metadata.cwd"], "/tmp/pi-demo");
  assert.equal(
    agentSpan.attributes["respan.metadata.session_file"],
    "/tmp/pi-demo/.pi/sessions/sess-123.jsonl",
  );
  assert.equal(agentSpan.attributes["respan.metadata.continuation"], undefined);
  assert.ok(agentSpan.startTime[0] <= agentSpan.endTime[0]);

  // Chat 1
  assert.equal(chat1.name, "pi.chat");
  assert.equal(chat1.attributes["traceloop.entity.name"], "pi.response");
  assert.equal(chat1.attributes["gen_ai.system"], "anthropic");
  assert.equal(chat1.attributes["llm.request.type"], "chat");
  assert.equal(chat1.attributes["gen_ai.request.model"], "claude-sonnet-4-5");
  assert.equal(chat1.attributes["gen_ai.prompt.0.role"], "system");
  assert.equal(chat1.attributes["gen_ai.prompt.0.content"], "You are pi");
  assert.equal(chat1.attributes["gen_ai.prompt.1.role"], "user");
  assert.equal(chat1.attributes["gen_ai.prompt.1.content"], "Inspect the repo");
  assert.equal(chat1.attributes["gen_ai.prompt.2.role"], undefined);
  assert.equal(chat1.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(chat1.attributes["gen_ai.completion.0.content"], "Listing files.");
  assert.deepEqual(JSON.parse(chat1.attributes["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call-1",
      type: "function",
      function: { name: "bash", arguments: "{\"command\":\"ls\"}" },
    },
  ]);
  const chat1Output = JSON.parse(chat1.attributes["traceloop.entity.output"]);
  assert.equal(chat1Output.role, "assistant");
  assert.equal(chat1Output.content, "Listing files.");
  assert.equal(chat1Output.reasoning, "Let me look.");
  assert.equal(chat1Output.tool_calls.length, 1);
  assert.equal(chat1.attributes["gen_ai.usage.prompt_tokens"], 9);
  assert.equal(chat1.attributes["gen_ai.usage.input_tokens"], 9);
  assert.equal(chat1.attributes["gen_ai.usage.completion_tokens"], 2);
  assert.equal(chat1.attributes["gen_ai.usage.output_tokens"], 2);
  assert.equal(chat1.attributes["llm.usage.total_tokens"], 11);
  assert.equal(chat1.attributes["llm.usage.cache_read_input_tokens"], 3);
  // Canonical semconv constants (`gen_ai.usage.cache_read.input_tokens` /
  // `gen_ai.usage.cache_creation.input_tokens` in 1.43.0) — the keys the
  // backend reads for cache-aware cost; the `llm.usage.*` alias is kept too.
  assert.equal(chat1.attributes[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS], 3);
  assert.equal(chat1.attributes[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS], 1);
  assert.equal(chat1.attributes["respan.metadata.reasoning_tokens"], 1);
  assert.equal(chat1.attributes["respan.metadata.estimated_cost_usd"], 0.01);
  assert.equal(typeof chat1.attributes["respan.metadata.time_to_first_token_ms"], "number");
  assert.equal(chat1.attributes["respan.metadata.stop_reason"], "toolUse");
  assert.equal(chat1.attributes["respan.metadata.response_id"], "resp-1");
  assert.equal(chat1.attributes["respan.metadata.turn_index"], 0);
  assert.equal(chat1.attributes["respan.metadata.api"], "anthropic-messages");
  assert.equal(chat1.attributes["respan.metadata.prompt_capture"], "delta");
  assert.equal(chat1.attributes["respan.metadata.prompt_message_offset"], 0);
  const functions = JSON.parse(chat1.attributes["llm.request.functions"]);
  assert.equal(functions.length, 2);
  assert.deepEqual(
    functions.map((tool) => tool.name),
    ["bash", "read"],
  );
  assert.equal(chat1.attributes.status_code, undefined);

  // Chat 2: delta mode → only the messages appended since chat 1.
  assert.equal(chat2.attributes["gen_ai.prompt.0.role"], "assistant");
  assert.equal(chat2.attributes["gen_ai.prompt.0.content"], "Listing files.");
  assert.deepEqual(JSON.parse(chat2.attributes["gen_ai.prompt.0.tool_calls"]), [
    {
      id: "call-1",
      type: "function",
      function: { name: "bash", arguments: "{\"command\":\"ls\"}" },
    },
  ]);
  assert.equal(chat2.attributes["gen_ai.prompt.1.role"], "tool");
  assert.equal(chat2.attributes["gen_ai.prompt.1.content"], "README.md");
  assert.equal(chat2.attributes["gen_ai.prompt.2.role"], undefined);
  assert.equal(chat2.attributes["respan.metadata.prompt_message_offset"], 1);
  assert.equal(chat2.attributes["gen_ai.completion.0.content"], "The repo has a README.");
  assert.equal(chat2.attributes["gen_ai.completion.0.tool_calls"], undefined);
  assert.equal(chat2.attributes["gen_ai.usage.prompt_tokens"], 7);
  assert.equal(chat2.attributes[ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS], 0);
  assert.equal(chat2.attributes["llm.usage.total_tokens"], 11);
  assert.equal(chat2.attributes["respan.metadata.time_to_first_token_ms"], undefined);
  assert.equal(chat2.attributes["respan.metadata.turn_index"], 1);
  const chat2Input = JSON.parse(chat2.attributes["traceloop.entity.input"]);
  assert.equal(chat2Input.length, 2);
  assert.equal(chat2Input[1].tool_call_id, "call-1");
  assert.equal(chat2Input[0].tool_calls.length, 1);

  // Tool span
  assert.equal(toolSpan.name, "bash.tool");
  assert.equal(toolSpan.attributes["traceloop.entity.name"], "bash");
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.input"]), {
    name: "bash",
    arguments: { command: "ls" },
  });
  assert.equal(toolSpan.attributes["traceloop.entity.output"], "README.md");
  assert.equal(toolSpan.attributes["respan.metadata.tool_call_id"], "call-1");
  assert.equal(toolSpan.attributes["respan.metadata.skill_name"], undefined);
  assert.equal(toolSpan.attributes.status_code, undefined);

  // session_shutdown(quit) dropped the extension tracer.
  assert.equal(instrumentor.activeSessionCount, 0);
});

test("defaults: full context on every chat span, system prompt once per run, no truncation", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const longOutput = "x".repeat(30000);
  await replayRun(pi, createFakeCtx(), { toolOutput: longOutput });

  const [chat1, chat2] = spansByLogType("chat");
  assert.equal(chat1.attributes["respan.metadata.prompt_capture"], "full");
  // First chat span of the run carries the system prompt, then the user prompt.
  assert.equal(chat1.attributes["gen_ai.prompt.0.role"], "system");
  assert.equal(chat1.attributes["gen_ai.prompt.0.content"], "You are pi");
  assert.equal(chat1.attributes["gen_ai.prompt.1.role"], "user");
  assert.equal(chat1.attributes["gen_ai.prompt.2.role"], undefined);
  // Later chat spans record the whole context again (not repeated system prompt).
  assert.equal(chat2.attributes["respan.metadata.prompt_capture"], "full");
  assert.equal(chat2.attributes["respan.metadata.prompt_message_offset"], 0);
  assert.equal(chat2.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(chat2.attributes["gen_ai.prompt.0.content"], "Inspect the repo");
  assert.equal(chat2.attributes["gen_ai.prompt.1.role"], "assistant");
  assert.equal(chat2.attributes["gen_ai.prompt.2.role"], "tool");
  assert.equal(chat2.attributes["gen_ai.prompt.3.role"], undefined);
  assert.equal(JSON.parse(chat2.attributes["traceloop.entity.input"]).length, 3);
  // Nothing is truncated by default.
  const toolSpan = spanByLogType("tool");
  assert.equal(toolSpan.attributes["traceloop.entity.output"], longOutput);
  for (const span of captureState.spans) {
    assert.equal(span.attributes["respan.metadata.truncated"], undefined);
    assertCommonContract(span);
  }
});

test("promptCapture: delta records only the messages appended since the previous LLM call", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor({ promptCapture: "delta", captureSystemPrompt: false });
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  await replayRun(pi, createFakeCtx());

  const [chat1, chat2] = spansByLogType("chat");
  assert.equal(chat1.attributes["respan.metadata.prompt_capture"], "delta");
  assert.equal(chat1.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(chat1.attributes["gen_ai.prompt.1.role"], undefined);
  assert.equal(chat2.attributes["respan.metadata.prompt_message_offset"], 1);
  assert.equal(chat2.attributes["gen_ai.prompt.0.role"], "assistant");
  assert.equal(chat2.attributes["gen_ai.prompt.1.role"], "tool");
  assert.equal(chat2.attributes["gen_ai.prompt.2.role"], undefined);
  assert.equal(JSON.parse(chat2.attributes["traceloop.entity.input"]).length, 2);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
});

test("captured strings are truncated only when maxContentChars is set", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor({ maxContentChars: 16000 });
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const longOutput = "x".repeat(30000);
  await replayRun(pi, createFakeCtx(), { toolOutput: longOutput });

  const toolSpan = spanByLogType("tool");
  const output = toolSpan.attributes["traceloop.entity.output"];
  const suffix = " …[truncated 14000 chars]";
  assert.equal(output.length, 16000 + suffix.length);
  assert.ok(output.endsWith(suffix));
  assert.ok(output.startsWith("xxxx"));
  assert.equal(toolSpan.attributes["respan.metadata.truncated"], true);
  const chatSpan = spanByLogType("chat");
  assert.equal(chatSpan.attributes["respan.metadata.truncated"], undefined);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }

  // Custom cap applies to prompts too; 0 disables truncation.
  captureState.spans = [];
  const small = new PiInstrumentor({ maxContentChars: 10 });
  small.activate();
  const smallPi = createFakePi();
  small.extension(smallPi);
  await replayRun(smallPi, createFakeCtx());
  const agentSpan = spanByLogType("agent");
  const input = JSON.parse(agentSpan.attributes["traceloop.entity.input"]);
  assert.equal(input[0].content, "Inspect th …[truncated 6 chars]");
  assert.equal(agentSpan.attributes["respan.metadata.truncated"], true);
  const smallChat = spanByLogType("chat");
  // prompt.0 is the (10-char) system prompt; the user prompt at index 1 is truncated.
  assert.ok(smallChat.attributes["gen_ai.prompt.1.content"].endsWith("…[truncated 6 chars]"));

  captureState.spans = [];
  const unlimited = new PiInstrumentor({ maxContentChars: 0 });
  unlimited.activate();
  const unlimitedPi = createFakePi();
  unlimited.extension(unlimitedPi);
  await replayRun(unlimitedPi, createFakeCtx(), { toolOutput: longOutput });
  assert.equal(spanByLogType("tool").attributes["traceloop.entity.output"], longOutput);
  assert.equal(spanByLogType("tool").attributes["respan.metadata.truncated"], undefined);
});

test("multiple attached sessions produce independent traces", () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const s1Messages = [];
  const s2Messages = [];
  const s1 = createFakeSession("session-a", s1Messages);
  const s2 = createFakeSession("session-b", s2Messages);
  const detach1 = instrumentor.attach(s1);
  const detach2 = instrumentor.attach(s2);
  assert.equal(instrumentor.activeSessionCount, 2);

  const user1 = { role: "user", content: "Task A", timestamp: 1 };
  const user2 = { role: "user", content: "Task B", timestamp: 1 };
  const reply1 = {
    ...assistantFinal,
    provider: "openai",
    model: "gpt-5",
    api: "openai-responses",
    content: [{ type: "text", text: "Done A" }],
  };
  const reply2 = {
    ...assistantFinal,
    provider: "openai",
    model: "gpt-5",
    api: "openai-responses",
    content: [{ type: "text", text: "Done B" }],
  };

  s1.emit({ type: "agent_start" });
  s2.emit({ type: "agent_start" });
  s1.emit({ type: "turn_start" });
  s2.emit({ type: "turn_start" });
  s1.emit({ type: "message_start", message: user1 });
  s1.emit({ type: "message_end", message: user1 });
  s1Messages.push(user1);
  s2.emit({ type: "message_start", message: user2 });
  s2.emit({ type: "message_end", message: user2 });
  s2Messages.push(user2);
  s1.emit({ type: "message_start", message: { ...reply1, content: [], stopReason: "pending" } });
  s2.emit({ type: "message_start", message: { ...reply2, content: [], stopReason: "pending" } });
  s1.emit({ type: "message_end", message: reply1 });
  s1Messages.push(reply1);
  s2.emit({ type: "message_end", message: reply2 });
  s2Messages.push(reply2);
  s1.emit({ type: "turn_end", message: reply1, toolResults: [] });
  s2.emit({ type: "turn_end", message: reply2, toolResults: [] });
  s2.emit({ type: "agent_end", messages: [user2, reply2], willRetry: false });
  s1.emit({ type: "agent_end", messages: [user1, reply1], willRetry: false });

  assert.equal(captureState.spans.length, 4);
  const bySession = (id) =>
    captureState.spans.filter((span) => span.attributes["respan.threads.thread_identifier"] === id);
  const spansA = bySession("session-a");
  const spansB = bySession("session-b");
  assert.equal(spansA.length, 2);
  assert.equal(spansB.length, 2);
  const traceA = spansA[0].spanContext().traceId;
  const traceB = spansB[0].spanContext().traceId;
  assert.notEqual(traceA, traceB);
  assert.ok(spansA.every((span) => span.spanContext().traceId === traceA));
  assert.ok(spansB.every((span) => span.spanContext().traceId === traceB));
  assert.equal(parentSpanId(spanByLogType("agent", spansA)), undefined);
  assert.equal(parentSpanId(spanByLogType("agent", spansB)), undefined);
  assert.equal(spanByLogType("agent", spansA).attributes["traceloop.entity.output"], "Done A");
  assert.equal(spanByLogType("agent", spansB).attributes["traceloop.entity.output"], "Done B");
  assert.equal(spanByLogType("chat", spansA).attributes["gen_ai.prompt.0.content"], "Task A");
  assert.equal(spanByLogType("chat", spansB).attributes["gen_ai.prompt.0.content"], "Task B");
  assert.equal(spanByLogType("chat", spansB).attributes["gen_ai.system"], "openai");
  assert.equal(spanByLogType("chat", spansB).attributes["gen_ai.request.model"], "gpt-5");
  assert.equal(spanByLogType("agent", spansB).attributes["gen_ai.request.model"], undefined);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }

  detach1();
  detach2();
  assert.equal(instrumentor.activeSessionCount, 0);
  assert.equal(s1.listenerCount(), 0);
  assert.equal(s2.listenerCount(), 0);
  detach1(); // idempotent
});

test("error paths: assistant errors, tool errors, retries, and shutdown mid-run", async () => {
  // Assistant stopReason "error" → chat 500 with error.message; run reflects it.
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx();
  await pi.emit("session_start", { reason: "startup" }, ctx);
  await pi.emit("before_agent_start", { prompt: "Break", systemPrompt: "sys" }, ctx);
  await pi.emit("agent_start", {}, ctx);
  await pi.emit("context", { messages: [{ role: "user", content: "Break" }] }, ctx);
  await pi.emit("turn_start", { turnIndex: 0 }, ctx);
  const failed = {
    ...assistantFinal,
    content: [],
    stopReason: "error",
    errorMessage: "overloaded_error",
  };
  await pi.emit("message_end", { message: failed }, ctx);
  await pi.emit("tool_execution_start", { toolCallId: "call-x", toolName: "bash", args: { command: "false" } }, ctx);
  await pi.emit(
    "tool_execution_end",
    {
      toolCallId: "call-x",
      toolName: "bash",
      result: { content: [{ type: "text", text: "exit status 1" }] },
      isError: true,
    },
    ctx,
  );
  await pi.emit("agent_end", { messages: [] }, ctx);

  const chatSpan = spanByLogType("chat");
  assert.equal(chatSpan.attributes.status_code, 500);
  assert.equal(chatSpan.attributes["error.message"], "overloaded_error");
  assert.equal(chatSpan.status.code, 2);
  const toolSpan = spanByLogType("tool");
  assert.equal(toolSpan.attributes.status_code, 500);
  assert.equal(toolSpan.attributes["error.message"], "exit status 1");
  assert.equal(spanByLogType("agent").attributes.status_code, 500);
  assert.equal(spanByLogType("agent").attributes["error.message"], "overloaded_error");
  assert.equal(parentSpanId(spanByLogType("agent")), undefined);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }

  // agent_end with willRetry keeps the run open; the retried run closes it.
  captureState.spans = [];
  const session = createFakeSession("retry-session", []);
  const detach = instrumentor.attach(session);
  const prompt = { role: "user", content: "Retry me", timestamp: 1 };
  session.emit({ type: "agent_start" });
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_start", message: prompt });
  session.emit({ type: "message_end", message: prompt });
  session.messages.push(prompt);
  session.emit({ type: "message_end", message: { ...failed, stopReason: "error", errorMessage: "rate limited" } });
  session.emit({ type: "agent_end", messages: [prompt], willRetry: true });
  assert.equal(spansByLogType("agent").length, 0);
  assert.equal(spansByLogType("chat").length, 1);
  session.emit({ type: "agent_start" });
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_start", message: { ...assistantFinal, content: [], stopReason: "pending" } });
  session.emit({ type: "message_end", message: assistantFinal });
  session.messages.push(assistantFinal);
  session.emit({ type: "agent_end", messages: [prompt, assistantFinal], willRetry: false });
  assert.equal(spansByLogType("agent").length, 1);
  const retryChats = spansByLogType("chat");
  assert.equal(retryChats.length, 2);
  assert.equal(retryChats[0].attributes.status_code, 500);
  assert.equal(retryChats[1].attributes.status_code, undefined);
  const retryTrace = spanByLogType("agent").spanContext().traceId;
  assert.ok(retryChats.every((span) => span.spanContext().traceId === retryTrace));
  assert.equal(spanByLogType("agent").attributes.status_code, undefined);
  assert.equal(spanByLogType("agent").attributes["traceloop.entity.output"], "The repo has a README.");
  assert.equal(JSON.parse(spanByLogType("agent").attributes["traceloop.entity.input"])[0].content, "Retry me");
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
  detach();

  // session_shutdown while a run is open closes it as an error and flushes pending work.
  captureState.spans = [];
  const pi2 = createFakePi();
  instrumentor.extension(pi2);
  const ctx2 = createFakeCtx({ sessionId: "sess-shutdown" });
  await pi2.emit("session_start", { reason: "startup" }, ctx2);
  await pi2.emit("before_agent_start", { prompt: "Long task", systemPrompt: "sys" }, ctx2);
  await pi2.emit("agent_start", {}, ctx2);
  await pi2.emit("context", { messages: [{ role: "user", content: "Long task" }] }, ctx2);
  await pi2.emit("turn_start", { turnIndex: 0 }, ctx2);
  await pi2.emit("tool_execution_start", { toolCallId: "call-y", toolName: "bash", args: { command: "sleep 10" } }, ctx2);
  await pi2.emit("session_shutdown", { reason: "quit" }, ctx2);

  const shutdownAgent = spanByLogType("agent");
  assert.equal(parentSpanId(shutdownAgent), undefined);
  assert.equal(shutdownAgent.attributes.status_code, 500);
  assert.equal(
    shutdownAgent.attributes["error.message"],
    "Session shut down before the agent run completed",
  );
  const interruptedChat = spanByLogType("chat");
  assert.equal(interruptedChat.attributes.status_code, 500);
  assert.equal(interruptedChat.attributes["error.message"], "Interrupted before completion");
  assert.equal(interruptedChat.attributes["gen_ai.prompt.0.content"], "sys");
  assert.equal(interruptedChat.attributes["gen_ai.prompt.1.content"], "Long task");
  const interruptedTool = spanByLogType("tool");
  assert.equal(interruptedTool.attributes.status_code, 500);
  assert.equal(interruptedTool.attributes["error.message"], "Interrupted before completion");
  assert.equal(spanByLogType("agent").attributes["respan.metadata.tool_call_count"], 1);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
});

test("compaction spans: root trace outside a run, child of agent inside a run", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx({ sessionId: "sess-compact" });
  await pi.emit("session_start", { reason: "startup" }, ctx);

  await pi.emit(
    "session_before_compact",
    { preparation: { tokensBefore: 120000, firstKeptEntryId: "e9" }, reason: "manual", willRetry: false },
    ctx,
  );
  await pi.emit(
    "session_compact",
    {
      compactionEntry: {
        type: "compaction",
        id: "c1",
        summary: "We inspected the repo.",
        firstKeptEntryId: "e9",
        tokensBefore: 120000,
      },
      reason: "manual",
      willRetry: false,
      fromExtension: false,
    },
    ctx,
  );
  assert.equal(captureState.spans.length, 1);
  const standalone = captureState.spans[0];
  assert.equal(standalone.name, "pi.compaction");
  assert.equal(standalone.attributes["respan.entity.log_type"], "task");
  assert.equal(standalone.attributes["traceloop.entity.name"], "compaction");
  assert.equal(parentSpanId(standalone), undefined);
  assert.equal(standalone.attributes["respan.sessions.session_identifier"], "sess-compact");
  assert.equal(standalone.attributes["respan.metadata.reason"], "manual");
  assert.deepEqual(JSON.parse(standalone.attributes["traceloop.entity.input"]), {
    reason: "manual",
    willRetry: false,
    tokensBefore: 120000,
  });
  const standaloneOutput = JSON.parse(standalone.attributes["traceloop.entity.output"]);
  assert.equal(standaloneOutput.summary, "We inspected the repo.");
  assert.equal(standaloneOutput.tokensBefore, 120000);
  assert.equal(standaloneOutput.firstKeptEntryId, "e9");
  assert.equal(standalone.attributes.status_code, undefined);

  captureState.spans = [];
  await pi.emit("before_agent_start", { prompt: "Continue", systemPrompt: "sys" }, ctx);
  await pi.emit("agent_start", {}, ctx);
  await pi.emit("session_before_compact", { preparation: { tokensBefore: 90000 }, reason: "threshold", willRetry: false }, ctx);
  await pi.emit(
    "session_compact",
    { compactionEntry: { summary: "Summary", firstKeptEntryId: "e20", tokensBefore: 90000 }, reason: "threshold", willRetry: false },
    ctx,
  );
  await pi.emit("agent_end", { messages: [] }, ctx);
  const compaction = captureState.spans.find((span) => span.name === "pi.compaction");
  const agentSpan = spanByLogType("agent");
  assert.ok(compaction);
  assert.equal(parentSpanId(compaction), agentSpan.spanContext().spanId);
  assert.equal(compaction.spanContext().traceId, agentSpan.spanContext().traceId);

  // Failed / aborted compactions
  captureState.spans = [];
  await pi.emit("session_before_compact", { preparation: { tokensBefore: 1 }, reason: "overflow", willRetry: true }, ctx);
  await pi.emit(
    "session_compact_failed",
    { reason: "overflow", errorMessage: "summarizer timed out", aborted: false, willRetry: true },
    ctx,
  );
  await pi.emit("session_before_compact", { preparation: { tokensBefore: 1 }, reason: "manual", willRetry: false }, ctx);
  await pi.emit("session_compact_failed", { reason: "manual", aborted: true, willRetry: false }, ctx);
  assert.equal(captureState.spans.length, 2);
  assert.equal(captureState.spans[0].attributes.status_code, 500);
  assert.equal(captureState.spans[0].attributes["error.message"], "summarizer timed out");
  assert.equal(captureState.spans[1].attributes["error.message"], "Compaction aborted");

  // Branch summary
  captureState.spans = [];
  await pi.emit(
    "session_before_tree",
    { preparation: { targetId: "e3", oldLeafId: "e12", userWantsSummary: true, label: "alt" } },
    ctx,
  );
  await pi.emit(
    "session_tree",
    {
      newLeafId: "e13",
      oldLeafId: "e12",
      summaryEntry: { type: "branch_summary", id: "b1", parentId: "e3", fromId: "e12", summary: "Abandoned branch did X." },
    },
    ctx,
  );
  assert.equal(captureState.spans.length, 1);
  const branch = captureState.spans[0];
  assert.equal(branch.name, "pi.branch_summary");
  assert.equal(branch.attributes["traceloop.entity.name"], "branch_summary");
  assert.equal(branch.attributes["respan.entity.log_type"], "task");
  const branchOutput = JSON.parse(branch.attributes["traceloop.entity.output"]);
  assert.equal(branchOutput.summary, "Abandoned branch did X.");
  assert.equal(branchOutput.id, "b1");
  assert.equal(branchOutput.newLeafId, "e13");
  assert.deepEqual(JSON.parse(branch.attributes["traceloop.entity.input"]), {
    targetId: "e3",
    oldLeafId: "e12",
    label: "alt",
  });
  // session_tree without a summary and without a pending start → no span.
  await pi.emit("session_tree", { newLeafId: "e14", oldLeafId: "e13" }, ctx);
  assert.equal(captureState.spans.length, 1);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
});

test("skill usage is detected from read of SKILL.md and the skill tool", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx();
  await pi.emit("session_start", { reason: "startup" }, ctx);
  await pi.emit("before_agent_start", { prompt: "Review", systemPrompt: "sys" }, ctx);
  await pi.emit("agent_start", {}, ctx);
  await pi.emit(
    "tool_execution_start",
    { toolCallId: "r1", toolName: "read", args: { path: "/home/dev/.pi/agent/skills/review/SKILL.md" } },
    ctx,
  );
  await pi.emit(
    "tool_execution_end",
    { toolCallId: "r1", toolName: "read", result: { content: [{ type: "text", text: "# Review skill" }] }, isError: false },
    ctx,
  );
  await pi.emit("tool_execution_start", { toolCallId: "r2", toolName: "read", args: { path: "/repo/README.md" } }, ctx);
  await pi.emit(
    "tool_execution_end",
    { toolCallId: "r2", toolName: "read", result: { content: [{ type: "text", text: "readme" }] }, isError: false },
    ctx,
  );
  await pi.emit("tool_execution_start", { toolCallId: "s1", toolName: "skill", args: { name: "deploy" } }, ctx);
  await pi.emit(
    "tool_execution_end",
    { toolCallId: "s1", toolName: "skill", result: { content: [{ type: "text", text: "ok" }] }, isError: false },
    ctx,
  );
  await pi.emit("agent_end", { messages: [] }, ctx);

  const toolSpans = spansByLogType("tool");
  assert.equal(toolSpans.length, 3);
  assert.equal(toolSpans[0].attributes["respan.metadata.skill_name"], "review");
  assert.equal(toolSpans[1].attributes["respan.metadata.skill_name"], undefined);
  assert.equal(toolSpans[2].attributes["respan.metadata.skill_name"], "deploy");
  assert.equal(toolSpans[0].name, "read.tool");
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
});

test("attach() subscribe mode traces a session and deactivate() unsubscribes", () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor({ workflowName: "mail-agent", agentName: "triage", promptCapture: "delta" });
  instrumentor.activate();
  const session = createFakeSession("sdk-session", []);
  const detach = instrumentor.attach(session, {
    threadIdentifier: "email-chain-42",
    customerIdentifier: "cust-1",
    metadata: { mailbox: "inbox" },
  });
  assert.equal(typeof detach, "function");
  assert.equal(instrumentor.activeSessionCount, 1);

  const prompt = { role: "user", content: "Hello from SDK", timestamp: 1 };
  session.emit({ type: "agent_start" });
  session.emit({ type: "message_start", message: prompt });
  session.emit({ type: "message_end", message: prompt });
  session.messages.push(prompt);
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_end", message: assistantWithTool });
  session.messages.push(assistantWithTool);
  session.emit({ type: "tool_execution_start", toolCallId: "call-1", toolName: "bash", args: { command: "ls" } });
  session.emit({
    type: "tool_execution_end",
    toolCallId: "call-1",
    toolName: "bash",
    result: { content: [{ type: "text", text: "README.md" }], details: {} },
    isError: false,
  });
  session.emit({ type: "turn_end", message: assistantWithTool, toolResults: [toolResultMessage] });
  session.messages.push(toolResultMessage);
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_start", message: { ...assistantFinal, content: [], stopReason: "pending" } });
  session.emit({ type: "message_end", message: assistantFinal });
  session.messages.push(assistantFinal);
  session.emit({ type: "agent_end", messages: session.messages, willRetry: false });

  assert.equal(captureState.spans.length, 4);
  const agentSpan = spanByLogType("agent");
  const [chat1, chat2] = spansByLogType("chat");
  const toolSpan = spanByLogType("tool");
  assert.equal(agentSpan.name, "triage.turn-1.agent");
  assert.equal(parentSpanId(agentSpan), undefined);
  assert.equal(agentSpan.attributes["traceloop.workflow.name"], "mail-agent");
  assert.equal(agentSpan.attributes["respan.metadata.agent_name"], "triage");
  // The trace group is the pi session id, not the workflow name.
  assert.equal(agentSpan.attributes["respan.trace.trace_group_identifier"], "sdk-session");
  assert.deepEqual(JSON.parse(agentSpan.attributes["traceloop.entity.input"]), [
    { role: "user", content: "Hello from SDK" },
  ]);
  assert.equal(agentSpan.attributes["traceloop.entity.output"], "The repo has a README.");
  assert.equal(agentSpan.attributes["respan.metadata.continuation"], undefined);
  assert.equal(chat1.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(chat1.attributes["gen_ai.prompt.0.content"], "Hello from SDK");
  assert.equal(chat1.attributes["gen_ai.completion.0.content"], "Listing files.");
  assert.equal(chat1.attributes["gen_ai.usage.prompt_tokens"], 9);
  assert.equal(chat2.attributes["gen_ai.prompt.0.role"], "assistant");
  assert.equal(chat2.attributes["gen_ai.prompt.1.role"], "tool");
  assert.equal(chat2.attributes["gen_ai.prompt.2.role"], undefined);
  assert.equal(chat2.attributes["respan.metadata.prompt_message_offset"], 1);
  assert.equal(toolSpan.attributes["traceloop.entity.output"], "README.md");
  for (const span of captureState.spans) {
    assert.equal(span.attributes["respan.threads.thread_identifier"], "email-chain-42");
    assert.equal(span.attributes["respan.sessions.session_identifier"], "sdk-session");
    assert.equal(span.attributes["respan.customer_params.customer_identifier"], "cust-1");
    assert.equal(span.attributes["respan.metadata.mailbox"], "inbox");
    assert.equal(span.spanContext().traceId, agentSpan.spanContext().traceId);
    assertCommonContract(span);
  }

  // deactivate() unsubscribes and stops emission.
  instrumentor.deactivate();
  assert.equal(instrumentor.isActive(), false);
  assert.equal(session.listenerCount(), 0);
  assert.equal(instrumentor.activeSessionCount, 0);
  captureState.spans = [];
  session.emit({ type: "agent_start" });
  session.emit({ type: "agent_end", messages: [], willRetry: false });
  assert.equal(captureState.spans.length, 0);

  // A re-attached, inactive instrumentor tracks state but emits nothing.
  const detach2 = instrumentor.attach(session);
  session.emit({ type: "agent_start" });
  session.emit({ type: "agent_end", messages: [], willRetry: false });
  assert.equal(captureState.spans.length, 0);
  detach2();
});

test("continuation runs without a prompt reuse the last prompt", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx();
  await replayRun(pi, ctx, { shutdown: false });
  captureState.spans = [];
  // e.g. an auto-retry in extension mode: agent_start without before_agent_start
  await pi.emit("agent_start", {}, ctx);
  await pi.emit("context", { messages: [userMessage, assistantWithTool, toolResultMessage, assistantFinal] }, ctx);
  await pi.emit("turn_start", { turnIndex: 0 }, ctx);
  await pi.emit("message_end", { message: assistantFinal }, ctx);
  await pi.emit("agent_end", { messages: [] }, ctx);
  const agentSpan = spanByLogType("agent");
  assert.equal(agentSpan.attributes["respan.metadata.continuation"], true);
  assert.deepEqual(JSON.parse(agentSpan.attributes["traceloop.entity.input"]), [
    { role: "user", content: "Inspect the repo" },
  ]);
  // First LLM call of a run captures from the last user message onward.
  const chatSpan = spanByLogType("chat");
  assert.equal(chatSpan.attributes["respan.metadata.prompt_message_offset"], 0);
  assert.equal(JSON.parse(chatSpan.attributes["traceloop.entity.input"]).length, 4);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
});

test("traceScope: session shares one trace across runs; run scope gives one trace per run", async () => {
  const uuid = "3f2504e0-4f89-11d3-9a0c-0305e82c3301";
  const expectedTraceId = "3f2504e04f8911d39a0c0305e82c3301";
  // UUID session ids map to the dash-stripped UUID (lowercased) …
  assert.equal(sessionTraceId(uuid), expectedTraceId);
  assert.equal(sessionTraceId(uuid.toUpperCase()), expectedTraceId);
  // … anything else to a SHA-256 prefix (collision-safe, unlike a repeated short hash).
  const hashed = createHash("sha256").update("sess-123").digest("hex").slice(0, 32);
  assert.equal(sessionTraceId("sess-123"), hashed);
  assert.match(hashed, /^[0-9a-f]{32}$/);

  captureState.spans = [];
  const instrumentor = new PiInstrumentor({ traceScope: "session" });
  assert.equal(instrumentor.traceScope, "session");
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx({ sessionId: uuid });
  await replayRun(pi, ctx, { shutdown: false });
  await replayRun(pi, ctx, { shutdown: false });

  // Two runs → two parentless agent (turn) roots (distinct span ids) in ONE trace.
  const agents = spansByLogType("agent");
  assert.equal(agents.length, 2);
  assert.notEqual(agents[0].spanContext().spanId, agents[1].spanContext().spanId);
  for (const agent of agents) {
    assert.equal(parentSpanId(agent), undefined);
    assert.equal(agent.spanContext().traceId, expectedTraceId);
  }
  assert.equal(captureState.spans.length, 8);
  for (const span of captureState.spans) {
    assert.equal(span.spanContext().traceId, expectedTraceId);
    // Correlation identifiers are the same in both scopes.
    assert.equal(span.attributes["respan.threads.thread_identifier"], uuid);
    assert.equal(span.attributes["respan.sessions.session_identifier"], uuid);
    assertCommonContract(span);
  }
  // Children still hang off their own run's agent span.
  const chats = spansByLogType("chat");
  assert.equal(chats.length, 4);
  assert.equal(parentSpanId(chats[0]), agents[0].spanContext().spanId);
  assert.equal(parentSpanId(chats[2]), agents[1].spanContext().spanId);

  // A compaction outside a run joins the session trace as another root.
  captureState.spans = [];
  await pi.emit(
    "session_before_compact",
    { preparation: { tokensBefore: 10 }, reason: "manual", willRetry: false },
    ctx,
  );
  await pi.emit(
    "session_compact",
    { compactionEntry: { summary: "s", tokensBefore: 10 }, reason: "manual", willRetry: false },
    ctx,
  );
  assert.equal(captureState.spans.length, 1);
  assert.equal(captureState.spans[0].spanContext().traceId, expectedTraceId);
  assert.equal(parentSpanId(captureState.spans[0]), undefined);

  // Non-UUID session id → the SHA-256 prefix.
  captureState.spans = [];
  const hashedPi = createFakePi();
  instrumentor.extension(hashedPi);
  await replayRun(hashedPi, createFakeCtx({ sessionId: "sess-123" }), { shutdown: false });
  assert.equal(captureState.spans.length, 4);
  for (const span of captureState.spans) {
    assert.equal(span.spanContext().traceId, hashed);
  }

  // Session scope is the default; run scope gives the same session a new trace per run.
  assert.equal(new PiInstrumentor().traceScope, "session");
  captureState.spans = [];
  const perRun = new PiInstrumentor({ traceScope: "run" });
  assert.equal(perRun.traceScope, "run");
  perRun.activate();
  const runPi = createFakePi();
  perRun.extension(runPi);
  const runCtx = createFakeCtx({ sessionId: uuid });
  await replayRun(runPi, runCtx, { shutdown: false });
  await replayRun(runPi, runCtx, { shutdown: false });
  const runAgents = spansByLogType("agent");
  assert.equal(runAgents.length, 2);
  assert.notEqual(runAgents[0].spanContext().traceId, runAgents[1].spanContext().traceId);
  assert.notEqual(runAgents[0].spanContext().traceId, expectedTraceId);
});

test("run scope nests under an active OTEL span; session scope always emits a root", () => {
  const manager = new SyncContextManager();
  assert.equal(context.setGlobalContextManager(manager), true);
  try {
    const parent = {
      spanContext: () => ({
        traceId: "0af7651916cd43dd8448eb211c80319c",
        spanId: "b7ad6b7169203331",
        traceFlags: 1,
      }),
    };
    const activeContext = trace.setSpan(ROOT_CONTEXT, parent);
    const prompt = { role: "user", content: "Nested", timestamp: 1 };
    const runPrompt = (session) => {
      session.emit({ type: "agent_start" });
      session.emit({ type: "message_start", message: prompt });
      session.emit({ type: "message_end", message: prompt });
      session.messages.push(prompt);
      session.emit({ type: "agent_end", messages: session.messages, willRetry: false });
    };

    captureState.spans = [];
    const runScoped = new PiInstrumentor({ traceScope: "run" });
    runScoped.activate();
    const nested = createFakeSession("3f2504e0-4f89-11d3-9a0c-0305e82c3301", []);
    const detachNested = runScoped.attach(nested);
    context.with(activeContext, () => runPrompt(nested));
    const nestedAgent = spanByLogType("agent");
    assert.equal(nestedAgent.spanContext().traceId, "0af7651916cd43dd8448eb211c80319c");
    assert.equal(parentSpanId(nestedAgent), "b7ad6b7169203331");
    detachNested();

    captureState.spans = [];
    const sessionScoped = new PiInstrumentor({ traceScope: "session" });
    sessionScoped.activate();
    const root = createFakeSession("3f2504e0-4f89-11d3-9a0c-0305e82c3301", []);
    const detachRoot = sessionScoped.attach(root);
    context.with(activeContext, () => runPrompt(root));
    const rootAgent = spanByLogType("agent");
    assert.equal(rootAgent.spanContext().traceId, "3f2504e04f8911d39a0c0305e82c3301");
    assert.equal(parentSpanId(rootAgent), undefined);
    detachRoot();
  } finally {
    context.disable();
  }
});

test("deactivate() closes open runs before it stops emitting", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const session = createFakeSession("shutdown-mid-run", []);
  instrumentor.attach(session);
  const prompt = { role: "user", content: "Long task", timestamp: 1 };
  session.emit({ type: "agent_start" });
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_start", message: prompt });
  session.emit({ type: "message_end", message: prompt });
  session.messages.push(prompt);
  session.emit({ type: "message_end", message: assistantWithTool });
  session.messages.push(assistantWithTool);
  session.emit({
    type: "tool_execution_start",
    toolCallId: "call-1",
    toolName: "bash",
    args: { command: "sleep 3600" },
  });
  assert.equal(spansByLogType("chat").length, 1);
  assert.equal(spansByLogType("agent").length, 0);

  // What Respan.shutdown() does first (e.g. an SDK worker receiving SIGTERM).
  instrumentor.deactivate();
  assert.equal(instrumentor.isActive(), false);
  assert.equal(session.listenerCount(), 0);
  assert.equal(instrumentor.activeSessionCount, 0);
  const agentSpan = spanByLogType("agent");
  const toolSpan = spanByLogType("tool");
  assert.ok(agentSpan && toolSpan, "root agent and interrupted tool spans were emitted");
  assert.equal(agentSpan.attributes.status_code, 500);
  assert.equal(
    agentSpan.attributes["error.message"],
    "Session shut down before the agent run completed",
  );
  assert.equal(toolSpan.attributes["error.message"], "Interrupted before completion");
  assert.equal(parentSpanId(agentSpan), undefined);
  assert.equal(parentSpanId(toolSpan), agentSpan.spanContext().spanId);
  assert.equal(spanByLogType("chat").spanContext().traceId, agentSpan.spanContext().traceId);
  assert.equal(agentSpan.attributes["respan.metadata.tool_call_count"], 1);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }

  // Live extension tracers are closed the same way.
  captureState.spans = [];
  const ext = new PiInstrumentor();
  ext.activate();
  const pi = createFakePi();
  ext.extension(pi);
  const ctx = createFakeCtx({ sessionId: "sess-deactivate" });
  await pi.emit("session_start", { reason: "startup" }, ctx);
  await pi.emit("before_agent_start", { prompt: "Long task", systemPrompt: "sys" }, ctx);
  await pi.emit("agent_start", {}, ctx);
  await pi.emit("context", { messages: [{ role: "user", content: "Long task" }] }, ctx);
  await pi.emit("turn_start", { turnIndex: 0 }, ctx);
  ext.deactivate();
  assert.equal(spansByLogType("agent").length, 1);
  assert.equal(spanByLogType("agent").attributes.status_code, 500);
  assert.equal(spanByLogType("chat").attributes["error.message"], "Interrupted before completion");
  // Nothing is emitted afterwards.
  captureState.spans = [];
  await pi.emit("agent_end", { messages: [] }, ctx);
  assert.equal(captureState.spans.length, 0);
});

test("a retry that never happens closes the run on auto_retry_end / agent_settled", () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor({ traceScope: "run", promptCapture: "delta" });
  instrumentor.activate();
  const session = createFakeSession("retry-abandoned", []);
  const detach = instrumentor.attach(session);
  const failed = { ...assistantFinal, content: [], stopReason: "error", errorMessage: "overloaded_error" };

  const prompt1 = { role: "user", content: "First", timestamp: 1 };
  session.emit({ type: "agent_start" });
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_start", message: prompt1 });
  session.emit({ type: "message_end", message: prompt1 });
  session.messages.push(prompt1);
  session.emit({ type: "message_end", message: failed });
  session.emit({ type: "agent_end", messages: [prompt1, failed], willRetry: true });
  assert.equal(spansByLogType("agent").length, 0);
  // The caller aborts during the backoff: pi cancels the retry and settles.
  session.emit({ type: "auto_retry_end", success: false, attempt: 1, finalError: "Retry cancelled" });
  session.emit({ type: "agent_settled" });
  assert.equal(spansByLogType("agent").length, 1);
  const first = spanByLogType("agent");
  assert.equal(first.name, "pi.turn-1.agent");
  assert.equal(first.attributes.status_code, 500);
  assert.equal(first.attributes["error.message"], "Retry cancelled");
  assert.deepEqual(JSON.parse(first.attributes["traceloop.entity.input"]), [
    { role: "user", content: "First" },
  ]);

  // The next prompt is its own run and trace, with its own input.
  captureState.spans = [];
  const prompt2 = { role: "user", content: "Second", timestamp: 2 };
  session.emit({ type: "agent_start" });
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_start", message: prompt2 });
  session.emit({ type: "message_end", message: prompt2 });
  session.messages.push(prompt2);
  session.emit({ type: "message_start", message: { ...assistantFinal, content: [], stopReason: "pending" } });
  session.emit({ type: "message_end", message: assistantFinal });
  session.messages.push(assistantFinal);
  session.emit({ type: "agent_end", messages: session.messages, willRetry: false });
  session.emit({ type: "agent_settled" });
  assert.equal(spansByLogType("agent").length, 1);
  const second = spanByLogType("agent");
  // session.messages holds one earlier user message → this is the session's turn 2.
  assert.equal(second.name, "pi.turn-2.agent");
  assert.notEqual(second.spanContext().traceId, first.spanContext().traceId);
  assert.equal(second.attributes.status_code, undefined);
  assert.deepEqual(JSON.parse(second.attributes["traceloop.entity.input"]), [
    { role: "user", content: "Second" },
  ]);
  assert.equal(spanByLogType("chat").attributes["gen_ai.prompt.0.content"], "Second");
  assert.equal(spanByLogType("agent").attributes["respan.metadata.turn_count"], 1);

  // agent_settled alone (no auto_retry_end) also closes a run kept open by willRetry,
  // and is a no-op without an open run.
  captureState.spans = [];
  session.emit({ type: "agent_start" });
  session.emit({ type: "agent_end", messages: [], willRetry: true });
  assert.equal(spansByLogType("agent").length, 0);
  session.emit({ type: "agent_settled" });
  assert.equal(spansByLogType("agent").length, 1);
  session.emit({ type: "agent_settled" });
  assert.equal(spansByLogType("agent").length, 1);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
  detach();
});

test("the tool catalog on chat spans is capped like every other captured string", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor({ maxContentChars: 200 });
  instrumentor.activate();
  const pi = createFakePi();
  pi.getAllTools = () => [
    {
      name: "mcp_search",
      description: "d".repeat(5000),
      parameters: { type: "object", properties: { q: { type: "string", description: "p".repeat(5000) } } },
    },
  ];
  instrumentor.extension(pi);
  await replayRun(pi, createFakeCtx(), { shutdown: false });
  const chat = spanByLogType("chat");
  const functions = chat.attributes["llm.request.functions"];
  assert.ok(functions.length < 300, `capped catalog, got ${functions.length} chars`);
  assert.ok(functions.includes(" …[truncated "));
  assert.ok(functions.startsWith('[{"name":"mcp_search"'));
  assert.equal(chat.attributes["respan.metadata.truncated"], true);

  // Small catalogs are untouched.
  captureState.spans = [];
  const plain = new PiInstrumentor({ maxContentChars: 200 });
  plain.activate();
  const plainPi = createFakePi();
  plainPi.getAllTools = () => [{ name: "bash", description: "Run", parameters: { type: "object" } }];
  plain.extension(plainPi);
  await replayRun(plainPi, createFakeCtx(), { shutdown: false });
  const plainChat = spanByLogType("chat");
  assert.deepEqual(JSON.parse(plainChat.attributes["llm.request.functions"]), [
    { name: "bash", description: "Run", parameters: { type: "object" } },
  ]);
  assert.equal(plainChat.attributes["respan.metadata.truncated"], undefined);
});

test("respan.propagateAttributes() overrides correlation for a run", () => {
  captureState.spans = [];
  const manager = new SyncContextManager();
  assert.equal(context.setGlobalContextManager(manager), true);
  try {
    const instrumentor = new PiInstrumentor({ metadata: { source: "sdk" } });
    instrumentor.activate();
    const session = createFakeSession("propagated-session", []);
    const detach = instrumentor.attach(session);
    const prompt = { role: "user", content: "Propagate", timestamp: 1 };
    propagateAttributes(
      {
        thread_identifier: "email-chain-7",
        customer_identifier: "cust-7",
        metadata: { mailbox: "billing", source: "override-ignored" },
      },
      () => {
        session.emit({ type: "agent_start" });
        session.emit({ type: "turn_start" });
        session.emit({ type: "message_start", message: prompt });
        session.emit({ type: "message_end", message: prompt });
        session.messages.push(prompt);
        session.emit({ type: "message_start", message: { ...assistantFinal, content: [], stopReason: "pending" } });
        session.emit({ type: "message_end", message: assistantFinal });
        session.messages.push(assistantFinal);
        session.emit({ type: "agent_end", messages: session.messages, willRetry: false });
      },
    );
    assert.equal(captureState.spans.length, 2);
    for (const span of captureState.spans) {
      assert.equal(span.attributes["respan.threads.thread_identifier"], "email-chain-7");
      assert.equal(span.attributes["respan.sessions.session_identifier"], "propagated-session");
      assert.equal(span.attributes["respan.customer_params.customer_identifier"], "cust-7");
      // Explicit tracer metadata wins over propagated metadata in the single
      // canonical metadata object.
      const metadata = JSON.parse(span.attributes["respan.metadata"]);
      assert.equal(metadata.mailbox, "billing");
      assert.equal(metadata.source, "sdk");
      assert.equal(
        Object.keys(span.attributes).some((key) => key.startsWith("respan.metadata.")),
        false,
      );
      assertCommonContract(span);
    }

    // Outside the propagation scope the session id is the thread id again.
    captureState.spans = [];
    session.emit({ type: "agent_start" });
    session.emit({ type: "agent_end", messages: [], willRetry: false });
    assert.equal(captureState.spans.length, 1);
    for (const span of captureState.spans) {
      assert.equal(span.attributes["respan.threads.thread_identifier"], "propagated-session");
      assert.equal(span.attributes["respan.customer_params.customer_identifier"], undefined);
    }

    // Explicit attach() overrides win over propagated values.
    detach();
    const pinned = instrumentor.attach(session, { threadIdentifier: "pinned-thread" });
    captureState.spans = [];
    propagateAttributes({ thread_identifier: "ignored" }, () => {
      session.emit({ type: "agent_start" });
      session.emit({ type: "agent_end", messages: [], willRetry: false });
    });
    assert.equal(captureState.spans.length, 1);
    for (const span of captureState.spans) {
      assert.equal(span.attributes["respan.threads.thread_identifier"], "pinned-thread");
    }
    pinned();
  } finally {
    context.disable();
  }
});

test("registry contract and tracer sink", () => {
  const instrumentor = new PiInstrumentor();
  assert.equal(instrumentor.name, "pi");
  assert.equal(typeof instrumentor.extension, "function");
  assert.equal(instrumentor.isActive(), false);
  assert.equal(typeof createPiExtension(), "function");
  assert.equal(typeof PiSessionTracer, "function");

  // Inactive instrumentors track state but emit nothing.
  captureState.spans = [];
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx();
  return replayRun(pi, ctx).then(() => {
    assert.equal(captureState.spans.length, 0);

    // A tracer with an explicit sink works without a tracer provider.
    const sink = [];
    const tracer = new PiSessionTracer({ emit: (span) => sink.push(span), workflowName: "custom" });
    tracer.setSession({ sessionId: "direct" });
    tracer.onBeforeAgentStart({ prompt: "hi", systemPrompt: "sys" });
    tracer.onAgentStart();
    tracer.onContext([{ role: "user", content: "hi" }]);
    tracer.onMessageEnd(assistantFinal);
    tracer.onAgentEnd({ messages: [] });
    assert.equal(sink.length, 2);
    assert.equal(sink[1].name, "pi.turn-1.agent");
    assert.equal(sink[1].attributes["traceloop.workflow.name"], "custom");
    assert.equal(sink[1].attributes["respan.threads.thread_identifier"], "direct");
    for (const span of sink) {
      assertCommonContract(span);
    }
  });
});

// ── Turn numbering ────────────────────────────────────────────────────────

/** A pi session-file entry as returned by `sessionManager.getBranch()` / `getEntries()`. */
function sessionEntry(role, content) {
  return { type: "message", message: { role, content, timestamp: 1 } };
}

function assertTurn(span, agentName, turnNumber) {
  assert.equal(span.attributes["respan.entity.log_type"], "agent");
  assert.equal(span.name, `${agentName}.turn-${turnNumber}.agent`);
  assert.equal(span.attributes["respan.metadata.turn_number"], turnNumber);
  assert.equal(span.attributes["respan.internal.span_name.kind"], "agent");
  assert.equal(span.attributes["respan.internal.span_name.detail"], `turn-${turnNumber}`);
  assert.equal(span.attributes["traceloop.entity.name"], agentName);
  assert.equal(span.attributes["traceloop.entity.path"], "");
}

test("turn numbering: extension mode counts the user messages on the session branch", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);

  // A resumed session: two prompts already on the branch (plus non-message entries).
  const branch = [
    { type: "session", id: "root", version: 3 },
    sessionEntry("user", "First question"),
    sessionEntry("assistant", [{ type: "text", text: "First answer" }]),
    sessionEntry("user", [{ type: "text", text: "Second question" }]),
    sessionEntry("assistant", [{ type: "text", text: "Second answer" }]),
    { type: "compaction", id: "c1", summary: "…" },
  ];
  // getEntries() holds abandoned branches too; getBranch() must win.
  const entries = [...branch, sessionEntry("user", "Abandoned"), sessionEntry("user", "Also abandoned")];
  const ctx = createFakeCtx({
    sessionManager: {
      getSessionId: () => "sess-resumed",
      getSessionFile: () => undefined,
      getBranch: () => branch,
      getEntries: () => entries,
    },
  });
  await replayRun(pi, ctx, { shutdown: false });

  assert.equal(captureState.spans.length, 4);
  const turn = spanByLogType("agent");
  assertTurn(turn, "pi", 3);
  assert.equal(parentSpanId(turn), undefined);
  for (const span of captureState.spans) {
    if (span !== turn) {
      assert.equal(parentSpanId(span), turn.spanContext().spanId);
    }
    assertCommonContract(span);
  }

  // Without getBranch(), getEntries() is used.
  captureState.spans = [];
  const entriesOnly = createFakeCtx({
    sessionManager: {
      getSessionId: () => "sess-entries",
      getSessionFile: () => undefined,
      getEntries: () => entries,
    },
  });
  await replayRun(pi, entriesOnly, { shutdown: false });
  assertTurn(spanByLogType("agent"), "pi", 5);
});

test("turn numbering: a prompt already appended to the session is not counted twice", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const branch = [
    sessionEntry("user", "First question"),
    sessionEntry("assistant", [{ type: "text", text: "First answer" }]),
    sessionEntry("user", "Second question"),
    sessionEntry("assistant", [{ type: "text", text: "Second answer" }]),
    // The runtime already appended the prompt of this run (replayRun's "Inspect the repo").
    sessionEntry("user", [{ type: "text", text: "Inspect the repo" }]),
  ];
  const ctx = createFakeCtx({
    sessionManager: {
      getSessionId: () => "sess-appended",
      getSessionFile: () => undefined,
      getBranch: () => branch,
    },
  });
  await replayRun(pi, ctx, { shutdown: false });
  assertTurn(spanByLogType("agent"), "pi", 3);

  // A different last prompt is a new turn.
  captureState.spans = [];
  branch.push(sessionEntry("assistant", [{ type: "text", text: "Third answer" }]));
  branch.push(sessionEntry("user", "Something else"));
  await replayRun(pi, ctx, { shutdown: false });
  assertTurn(spanByLogType("agent"), "pi", 5);

  // The same prompt sent again after an answer ("continue" twice in a row) is
  // a new turn: only a user message that is the LAST message counts as appended.
  captureState.spans = [];
  branch.push(sessionEntry("assistant", [{ type: "text", text: "Fourth answer" }]));
  branch.push(sessionEntry("user", "Inspect the repo"));
  branch.push(sessionEntry("assistant", [{ type: "text", text: "Fifth answer" }]));
  branch.push({ type: "model_change", provider: "anthropic", modelId: "claude-sonnet-4-5" });
  await replayRun(pi, ctx, { shutdown: false });
  assertTurn(spanByLogType("agent"), "pi", 6);
});

test("turn numbering: without session history, consecutive runs count up within the tracer", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor({ traceScope: "run", agentName: "helper" });
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  // createFakeCtx()'s session manager has neither getBranch() nor getEntries().
  const ctx = createFakeCtx();
  await replayRun(pi, ctx, { shutdown: false });
  await replayRun(pi, ctx, { shutdown: false });
  const turns = spansByLogType("agent");
  assert.equal(turns.length, 2);
  assertTurn(turns[0], "helper", 1);
  assertTurn(turns[1], "helper", 2);
  // Run scope: each turn is its own trace.
  assert.notEqual(turns[0].spanContext().traceId, turns[1].spanContext().traceId);

  // A bare tracer counts the same way; an explicit turnNumber wins over the count.
  const sink = [];
  const tracer = new PiSessionTracer({ emit: (span) => sink.push(span) });
  tracer.setSession({ sessionId: "direct" });
  for (const prompt of ["one", "two", "three"]) {
    tracer.onBeforeAgentStart({ prompt, systemPrompt: "sys" });
    tracer.onAgentStart();
    tracer.onAgentEnd({ messages: [] });
  }
  assert.deepEqual(
    sink.map((span) => span.name),
    ["pi.turn-1.agent", "pi.turn-2.agent", "pi.turn-3.agent"],
  );
  tracer.onBeforeAgentStart({ prompt: "resumed", turnNumber: 9 });
  tracer.onAgentEnd({ messages: [] });
  assertTurn(sink[3], "pi", 9);
  // A continuation (agent_start without before_agent_start) is a turn of its own.
  tracer.onAgentStart();
  tracer.onAgentEnd({ messages: [] });
  assertTurn(sink[4], "pi", 10);
  assert.equal(sink[4].attributes["respan.metadata.continuation"], true);
});

test("turn numbering: subscribe mode counts the user messages of session.messages", () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  instrumentor.activate();
  const history = () => [
    { role: "user", content: "Earlier question", timestamp: 1 },
    { ...assistantFinal, content: [{ type: "text", text: "Earlier answer" }] },
    { role: "user", content: [{ type: "text", text: "Another question" }], timestamp: 3 },
    { ...assistantFinal, content: [{ type: "text", text: "Another answer" }] },
  ];
  const session = createFakeSession("sdk-turns", history());
  const detach = instrumentor.attach(session);

  // pi appends the prompt to session.messages after agent_start: 2 earlier prompts → turn 3.
  const prompt = { role: "user", content: "Third question", timestamp: 5 };
  session.emit({ type: "agent_start" });
  session.emit({ type: "turn_start" });
  session.emit({ type: "message_start", message: prompt });
  session.emit({ type: "message_end", message: prompt });
  session.messages.push(prompt);
  session.emit({ type: "message_start", message: { ...assistantFinal, content: [], stopReason: "pending" } });
  session.emit({ type: "message_end", message: assistantFinal });
  session.messages.push(assistantFinal);
  session.emit({ type: "agent_end", messages: session.messages, willRetry: false });
  assert.equal(captureState.spans.length, 2);
  const third = spanByLogType("agent");
  assertTurn(third, "pi", 3);
  assert.deepEqual(JSON.parse(third.attributes["traceloop.entity.input"]), [
    { role: "user", content: "Third question" },
  ]);

  // The next prompt of the same session is turn 4.
  captureState.spans = [];
  const next = { role: "user", content: "Fourth question", timestamp: 7 };
  session.emit({ type: "agent_start" });
  session.emit({ type: "message_start", message: next });
  session.emit({ type: "message_end", message: next });
  session.messages.push(next);
  session.emit({ type: "agent_end", messages: session.messages, willRetry: false });
  assertTurn(spanByLogType("agent"), "pi", 4);
  detach();

  // A session manager on the session (its branch) wins over session.messages.
  captureState.spans = [];
  const managed = createFakeSession("sdk-managed", history());
  managed.sessionManager = {
    getSessionId: () => "sdk-managed",
    getBranch: () => [sessionEntry("user", "Only one so far")],
  };
  const detachManaged = instrumentor.attach(managed);
  managed.emit({ type: "agent_start" });
  managed.emit({ type: "agent_end", messages: [], willRetry: false });
  assertTurn(spanByLogType("agent"), "pi", 2);
  detachManaged();
});

test("turn numbering: session scope numbers the roots of the shared trace", async () => {
  captureState.spans = [];
  const uuid = "9b2d3c44-1e0f-4a6b-8c7d-0123456789ab";
  const instrumentor = new PiInstrumentor({ traceScope: "session" });
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx({ sessionId: uuid });
  await replayRun(pi, ctx, { shutdown: false });
  await replayRun(pi, ctx, { shutdown: false });

  const turns = spansByLogType("agent");
  assert.equal(turns.length, 2);
  assertTurn(turns[0], "pi", 1);
  assertTurn(turns[1], "pi", 2);
  assert.notEqual(turns[0].spanContext().spanId, turns[1].spanContext().spanId);
  for (const turn of turns) {
    assert.equal(turn.spanContext().traceId, sessionTraceId(uuid));
    assert.equal(parentSpanId(turn), undefined);
    assert.equal(turn.attributes["respan.trace.trace_group_identifier"], uuid);
  }

  // The session resumed in a new process (a fresh extension runtime) continues
  // the numbering from the session history, in the same trace.
  captureState.spans = [];
  const resumedPi = createFakePi();
  instrumentor.extension(resumedPi);
  const resumedCtx = createFakeCtx({
    sessionId: uuid,
    sessionManager: {
      getSessionId: () => uuid,
      getSessionFile: () => undefined,
      getBranch: () => [
        sessionEntry("user", "First question"),
        sessionEntry("assistant", [{ type: "text", text: "First answer" }]),
        sessionEntry("user", "Second question"),
        sessionEntry("assistant", [{ type: "text", text: "Second answer" }]),
      ],
    },
  });
  await replayRun(resumedPi, resumedCtx, { shutdown: false });
  const resumed = spanByLogType("agent");
  assertTurn(resumed, "pi", 3);
  assert.equal(resumed.spanContext().traceId, sessionTraceId(uuid));
  assert.equal(parentSpanId(resumed), undefined);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
});

test("spans produced before activate() are buffered and emitted on activation", async () => {
  captureState.spans = [];
  const instrumentor = new PiInstrumentor();
  // Not activated yet: Respan.initialize() has not resolved.
  const pi = createFakePi();
  instrumentor.extension(pi);
  await replayRun(pi, createFakeCtx(), { shutdown: false });
  assert.equal(captureState.spans.length, 0);

  instrumentor.activate();
  assert.equal(captureState.spans.length, 4);
  const names = captureState.spans.map((span) => span.name).sort();
  assert.deepEqual(names, ["bash.tool", "pi.chat", "pi.chat", "pi.turn-1.agent"]);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }

  // Once active, spans go straight through.
  await replayRun(pi, createFakeCtx(), { shutdown: false });
  assert.equal(captureState.spans.length, 8);
});

test("git metadata of the working directory lands on the turn span and onRunEnd reports the run", async () => {
  const { execFileSync } = await import("node:child_process");
  const fs = await import("node:fs");
  const os = await import("node:os");
  const path = await import("node:path");
  const { gitMetadataFor } = await import("../dist/index.js");

  const repo = fs.mkdtempSync(path.join(os.tmpdir(), "pi-git-"));
  let hasGit = true;
  try {
    const git = (...args) => execFileSync("git", args, { cwd: repo, stdio: "pipe" });
    git("init", "-q", "-b", "main");
    git("config", "user.email", "t@example.com");
    git("config", "user.name", "t");
    git("remote", "add", "origin", "https://user:secret@github.com/acme/repo.git");
    fs.writeFileSync(path.join(repo, "a.txt"), "a");
    git("add", "a.txt");
    git("commit", "-q", "-m", "init");
  } catch {
    hasGit = false;
  }
  if (!hasGit) {
    return; // git not available in this environment
  }
  const meta = gitMetadataFor(repo);
  assert.equal(meta.branch, "main");
  assert.match(meta.commit, /^[0-9a-f]{40}$/);
  assert.equal(meta.repository, "https://github.com/acme/repo.git", "credentials stripped");
  assert.equal(gitMetadataFor(path.join(os.tmpdir())), undefined);

  captureState.spans = [];
  const runs = [];
  const instrumentor = new PiInstrumentor({ onRunEnd: (info) => runs.push(info) });
  instrumentor.activate();
  const pi = createFakePi();
  instrumentor.extension(pi);
  const ctx = createFakeCtx();
  ctx.cwd = repo;
  await replayRun(pi, ctx);

  const turn = spanByLogType("agent");
  assert.equal(turn.attributes["respan.metadata.git_branch"], "main");
  assert.equal(turn.attributes["respan.metadata.git_commit"], meta.commit);
  assert.equal(turn.attributes["respan.metadata.git_repository"], "https://github.com/acme/repo.git");
  assert.equal(runs.length, 1);
  assert.equal(runs[0].turnNumber, 1);
  assert.equal(runs[0].sessionId, "sess-123");
  assert.equal(runs[0].traceId, turn.spanContext().traceId);
  for (const span of captureState.spans) {
    assertCommonContract(span);
  }
});
