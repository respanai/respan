import assert from "node:assert/strict";
import test from "node:test";

import { SpanStatusCode, trace } from "@opentelemetry/api";

import { FlueInstrumentor } from "../dist/index.js";

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

function event(partial) {
  return {
    v: 1,
    eventIndex: partial.eventIndex ?? 1,
    timestamp: partial.timestamp ?? "2026-06-21T00:00:00.000Z",
    ...partial,
  };
}

test("exports Flue workflow, operation, model turn, and tool events as canonical spans", () => {
  captureState.spans = [];
  const instrumentor = new FlueInstrumentor();

  instrumentor.handleEvent(event({
    type: "run_start",
    eventIndex: 1,
    runId: "run-flue-1",
    workflowName: "Flue Weather.workflow",
    startedAt: "2026-06-21T00:00:00.000Z",
    payload: { city: "Paris" },
  }));
  instrumentor.handleEvent(event({
    type: "operation_start",
    eventIndex: 2,
    timestamp: "2026-06-21T00:00:00.050Z",
    runId: "run-flue-1",
    operationId: "op-1",
    operationKind: "prompt",
  }));
  instrumentor.handleEvent(event({
    type: "turn_start",
    eventIndex: 3,
    timestamp: "2026-06-21T00:00:00.100Z",
    runId: "run-flue-1",
    operationId: "op-1",
    turnId: "turn-1",
    purpose: "agent",
  }));
  instrumentor.handleEvent(event({
    type: "turn_request",
    eventIndex: 4,
    timestamp: "2026-06-21T00:00:00.110Z",
    runId: "run-flue-1",
    operationId: "op-1",
    turnId: "turn-1",
    purpose: "agent",
    model: "openai/gpt-4o-mini",
    provider: "openai",
    api: "responses",
    input: {
      systemPrompt: "Answer with weather facts.",
      messages: [
        { role: "user", content: "Weather in Paris?" },
      ],
      tools: [
        {
          name: "lookup_weather",
          description: "Lookup weather.",
          parameters: { type: "object" },
        },
      ],
    },
  }));
  instrumentor.handleEvent(event({
    type: "tool_start",
    eventIndex: 5,
    timestamp: "2026-06-21T00:00:00.200Z",
    runId: "run-flue-1",
    operationId: "op-1",
    turnId: "turn-1",
    toolCallId: "tool-1",
    toolName: "lookup_weather",
    args: { city: "Paris" },
  }));
  instrumentor.handleEvent(event({
    type: "tool",
    eventIndex: 6,
    timestamp: "2026-06-21T00:00:00.260Z",
    runId: "run-flue-1",
    operationId: "op-1",
    turnId: "turn-1",
    toolCallId: "tool-1",
    toolName: "lookup_weather",
    isError: false,
    result: { forecast: "sunny" },
    durationMs: 60,
  }));
  instrumentor.handleEvent(event({
    type: "turn",
    eventIndex: 7,
    timestamp: "2026-06-21T00:00:00.900Z",
    runId: "run-flue-1",
    operationId: "op-1",
    turnId: "turn-1",
    purpose: "agent",
    durationMs: 800,
    model: "openai/gpt-4o-mini",
    provider: "openai",
    api: "responses",
    output: {
      role: "assistant",
      content: [
        { type: "text", text: "Paris is sunny." },
        { type: "toolCall", id: "tool-1", name: "lookup_weather", arguments: { city: "Paris" } },
      ],
    },
    usage: {
      input: 20,
      output: 8,
      cacheRead: 2,
      cacheWrite: 0,
      totalTokens: 28,
      cost: { input: 0.01, output: 0.02, cacheRead: 0.001, cacheWrite: 0, total: 0.031 },
    },
    stopReason: "stop",
    isError: false,
  }));
  instrumentor.handleEvent(event({
    type: "operation",
    eventIndex: 8,
    timestamp: "2026-06-21T00:00:01.000Z",
    runId: "run-flue-1",
    operationId: "op-1",
    operationKind: "prompt",
    durationMs: 950,
    isError: false,
    result: { text: "Paris is sunny." },
  }));
  instrumentor.handleEvent(event({
    type: "run_end",
    eventIndex: 9,
    timestamp: "2026-06-21T00:00:01.100Z",
    runId: "run-flue-1",
    result: { ok: true },
    isError: false,
    durationMs: 1100,
  }));

  assert.equal(captureState.spans.length, 4);

  const workflowSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "workflow",
  );
  const operationSpan = captureState.spans.find(
    (span) => span.name === "flue.operation.prompt",
  );
  const modelSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "chat",
  );
  const toolSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );

  assert.ok(workflowSpan);
  assert.ok(operationSpan);
  assert.ok(modelSpan);
  assert.ok(toolSpan);
  assert.equal(modelSpan.instrumentationScope.name, "@respan/instrumentation-flue");
  assert.equal(operationSpan.parentSpanContext?.spanId, workflowSpan.spanContext().spanId);
  assert.equal(modelSpan.parentSpanContext?.spanId, operationSpan.spanContext().spanId);
  assert.equal(toolSpan.parentSpanContext?.spanId, operationSpan.spanContext().spanId);

  assert.equal(workflowSpan.attributes["traceloop.workflow.name"], "Flue Weather.workflow");
  assert.equal(modelSpan.attributes["traceloop.workflow.name"], "Flue Weather.workflow");
  assert.equal(modelSpan.attributes["gen_ai.system"], "openai");
  assert.equal(modelSpan.attributes["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(modelSpan.attributes["llm.request.type"], "chat");
  assert.equal(modelSpan.attributes["gen_ai.prompt.0.role"], "system");
  assert.equal(modelSpan.attributes["gen_ai.prompt.0.content"], "Answer with weather facts.");
  assert.equal(modelSpan.attributes["gen_ai.prompt.1.role"], "user");
  assert.equal(modelSpan.attributes["gen_ai.prompt.1.content"], "Weather in Paris?");
  assert.deepEqual(JSON.parse(modelSpan.attributes["llm.request.functions"]), [
    { name: "lookup_weather", description: "Lookup weather.", parameters: { type: "object" } },
  ]);
  assert.equal(modelSpan.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(modelSpan.attributes["gen_ai.completion.0.content"], "Paris is sunny.");
  assert.deepEqual(JSON.parse(modelSpan.attributes["gen_ai.completion.0.tool_calls"]), [
    {
      id: "tool-1",
      type: "function",
      function: { name: "lookup_weather", arguments: JSON.stringify({ city: "Paris" }) },
    },
  ]);
  assert.equal(modelSpan.attributes["gen_ai.usage.input_tokens"], 20);
  assert.equal(modelSpan.attributes["gen_ai.usage.output_tokens"], 8);
  assert.equal(modelSpan.attributes["gen_ai.usage.prompt_tokens"], 20);
  assert.equal(modelSpan.attributes["gen_ai.usage.completion_tokens"], 8);
  assert.equal(modelSpan.attributes["llm.usage.total_tokens"], 28);
  assert.equal(modelSpan.attributes["respan.metadata.flue_usage_cache_read_tokens"], 2);

  for (const span of captureState.spans) {
    assert.equal(span.attributes["respan.span.tools"], undefined);
    assert.equal(span.attributes["respan.span.tool_calls"], undefined);
    assert.equal(span.attributes.tools, undefined);
    assert.equal(span.attributes.tool_calls, undefined);
    assert.equal(span.attributes.model, undefined);
    assert.equal(span.attributes.prompt_tokens, undefined);
    assert.equal(span.attributes.completion_tokens, undefined);
  }
});

test("exports direct agent logs and compaction with fallback workflow name", () => {
  captureState.spans = [];
  const instrumentor = new FlueInstrumentor({
    workflowName: "Flue Direct Agent.workflow",
  });

  instrumentor.handleEvent(event({
    type: "agent_start",
    eventIndex: 1,
    instanceId: "agent-1",
    harness: "default",
    session: "main",
  }));
  instrumentor.handleEvent(event({
    type: "log",
    eventIndex: 2,
    instanceId: "agent-1",
    harness: "default",
    session: "main",
    level: "info",
    message: "agent accepted input",
    attributes: { channel: "test" },
  }));
  instrumentor.handleEvent(event({
    type: "compaction_start",
    eventIndex: 3,
    timestamp: "2026-06-21T00:00:00.200Z",
    instanceId: "agent-1",
    operationId: "op-compact",
    reason: "manual",
    estimatedTokens: 1200,
  }));
  instrumentor.handleEvent(event({
    type: "compaction",
    eventIndex: 4,
    timestamp: "2026-06-21T00:00:00.500Z",
    instanceId: "agent-1",
    operationId: "op-compact",
    messagesBefore: 12,
    messagesAfter: 3,
    durationMs: 300,
    isError: false,
  }));
  instrumentor.handleEvent(event({
    type: "agent_end",
    eventIndex: 5,
    instanceId: "agent-1",
    harness: "default",
    session: "main",
    messages: [{ role: "assistant", content: "done" }],
  }));

  assert.equal(captureState.spans.length, 3);
  const logSpan = captureState.spans.find((span) => span.name === "flue.log.info");
  const compactionSpan = captureState.spans.find((span) => span.name === "flue.compaction");
  const agentSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "agent",
  );

  assert.ok(logSpan);
  assert.ok(compactionSpan);
  assert.ok(agentSpan);
  assert.equal(logSpan.attributes["traceloop.workflow.name"], "Flue Direct Agent.workflow");
  assert.equal(compactionSpan.attributes["respan.metadata.flue_compaction_reason"], "manual");
  assert.equal(agentSpan.attributes["traceloop.entity.name"], "agent-1");
});


test("marks failed Flue events with error status and backend status attributes", () => {
  captureState.spans = [];
  const instrumentor = new FlueInstrumentor({
    workflowName: "Flue Error.workflow",
  });

  instrumentor.handleEvent(event({
    type: "run_start",
    eventIndex: 1,
    runId: "run-error-1",
    workflowName: "Flue Error.workflow",
    payload: { command: "cat missing-file.txt" },
  }));
  instrumentor.handleEvent(event({
    type: "operation_start",
    eventIndex: 2,
    runId: "run-error-1",
    operationId: "op-shell",
    operationKind: "shell",
  }));
  instrumentor.handleEvent(event({
    type: "tool_start",
    eventIndex: 3,
    runId: "run-error-1",
    operationId: "op-shell",
    turnId: "turn-shell",
    toolCallId: "tool-shell",
    toolName: "shell",
    args: { command: "cat missing-file.txt" },
  }));
  instrumentor.handleEvent(event({
    type: "tool",
    eventIndex: 4,
    runId: "run-error-1",
    operationId: "op-shell",
    turnId: "turn-shell",
    toolCallId: "tool-shell",
    toolName: "shell",
    isError: true,
    result: { message: "missing-file.txt: No such file", exitCode: 1 },
  }));
  instrumentor.handleEvent(event({
    type: "operation",
    eventIndex: 5,
    runId: "run-error-1",
    operationId: "op-shell",
    operationKind: "shell",
    isError: true,
    error: { name: "ShellError", message: "Command failed" },
  }));
  instrumentor.handleEvent(event({
    type: "run_end",
    eventIndex: 6,
    runId: "run-error-1",
    isError: true,
    error: { name: "WorkflowError", message: "Unable to read required file" },
  }));

  assert.equal(captureState.spans.length, 3);
  const toolSpan = captureState.spans.find((span) => span.name === "flue.tool.shell");
  const operationSpan = captureState.spans.find((span) => span.name === "flue.operation.shell");
  const workflowSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "workflow",
  );

  for (const span of [toolSpan, operationSpan, workflowSpan]) {
    assert.ok(span);
    assert.equal(span.status.code, SpanStatusCode.ERROR);
    assert.equal(span.attributes.status_code, 500);
    assert.equal(typeof span.attributes["error.message"], "string");
  }
  assert.equal(operationSpan.parentSpanContext?.spanId, workflowSpan.spanContext().spanId);
  assert.equal(toolSpan.parentSpanContext?.spanId, operationSpan.spanContext().spanId);
});

test("activate and deactivate wire the Flue observe subscriber", async () => {
  let subscriber;
  let unsubscribeCalls = 0;
  const instrumentor = new FlueInstrumentor({
    runtimeModule: {
      observe(nextSubscriber) {
        subscriber = nextSubscriber;
        return () => {
          unsubscribeCalls += 1;
        };
      },
    },
  });

  await instrumentor.activate();
  assert.equal(instrumentor.isActive(), true);
  assert.equal(typeof subscriber, "function");

  instrumentor.deactivate();
  assert.equal(instrumentor.isActive(), false);
  assert.equal(unsubscribeCalls, 1);
});
