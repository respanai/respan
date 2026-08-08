import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { MastraInstrumentor } from "../dist/index.js";

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

test("exports Mastra agent, model, and tool spans with canonical Respan attrs", async () => {
  captureState.spans = [];
  const instrumentor = new MastraInstrumentor();

  await instrumentor.exportTracingEvent({
    type: "span_ended",
    exportedSpan: {
      id: "agent-span",
      traceId: "trace-123",
      name: "Mastra Weather Example.workflow",
      type: "agent_run",
      isRootSpan: true,
      startTime: new Date("2026-05-21T00:00:00.000Z"),
      endTime: new Date("2026-05-21T00:00:01.000Z"),
      attributes: { availableTools: ["get_weather"] },
      input: "What is the weather in Tokyo?",
      output: "Tokyo is sunny.",
      metadata: { example: "mastra" },
      entityName: "Mastra Weather Example.workflow",
    },
  });

  await instrumentor.exportTracingEvent({
    type: "span_ended",
    exportedSpan: {
      id: "model-span",
      traceId: "trace-123",
      parentSpanId: "agent-span",
      name: "model generation",
      type: "model_generation",
      startTime: new Date("2026-05-21T00:00:00.100Z"),
      endTime: new Date("2026-05-21T00:00:00.800Z"),
      attributes: {
        model: "openai/gpt-4.1-nano",
        provider: "openai",
        usage: { inputTokens: 12, outputTokens: 8, inputDetails: { cacheRead: 2 } },
        availableTools: ["get_weather"],
        finishReason: "stop",
      },
      input: [{ role: "user", content: "What is the weather in Tokyo?" }],
      output: {
        text: "Tokyo is sunny.",
        toolCalls: [{ toolCallId: "call_1", toolName: "get_weather", args: { city: "Tokyo" } }],
      },
    },
  });

  await instrumentor.exportTracingEvent({
    type: "span_ended",
    exportedSpan: {
      id: "tool-span",
      traceId: "trace-123",
      parentSpanId: "agent-span",
      name: "get_weather",
      type: "tool_call",
      startTime: new Date("2026-05-21T00:00:00.200Z"),
      endTime: new Date("2026-05-21T00:00:00.300Z"),
      attributes: { success: true },
      input: { city: "Tokyo" },
      output: { forecast: "sunny" },
    },
  });

  assert.equal(captureState.spans.length, 3);

  const agentSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "agent",
  );
  const modelSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "chat",
  );
  const toolSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );

  assert.ok(agentSpan);
  assert.ok(modelSpan);
  assert.ok(toolSpan);
  assert.equal(agentSpan.instrumentationScope.name, "@respan/instrumentation-mastra");
  assert.equal(modelSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);
  assert.equal(toolSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);
  assert.equal(modelSpan.spanContext().traceId, agentSpan.spanContext().traceId);
  assert.equal(toolSpan.spanContext().traceId, agentSpan.spanContext().traceId);

  assert.equal(agentSpan.attributes["traceloop.workflow.name"], "Mastra Weather Example.workflow");
  assert.equal(agentSpan.attributes["respan.metadata.mastra_span_type"], "agent_run");
  assert.equal(agentSpan.attributes["respan.metadata.example"], "mastra");
  assert.equal(agentSpan.attributes.tools, undefined);

  assert.equal(modelSpan.attributes["gen_ai.system"], "openai");
  assert.equal(modelSpan.attributes["gen_ai.request.model"], "gpt-4.1-nano");
  assert.equal(modelSpan.attributes["llm.request.type"], "chat");
  assert.equal(modelSpan.attributes["gen_ai.usage.input_tokens"], 12);
  assert.equal(modelSpan.attributes["gen_ai.usage.output_tokens"], 8);
  assert.equal(modelSpan.attributes["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(modelSpan.attributes["gen_ai.usage.completion_tokens"], 8);
  assert.equal(modelSpan.attributes["llm.usage.total_tokens"], 20);
  assert.equal(modelSpan.attributes["llm.usage.cache_read_input_tokens"], 2);
  assert.equal(modelSpan.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(modelSpan.attributes["gen_ai.prompt.0.content"], "What is the weather in Tokyo?");
  assert.equal(modelSpan.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(modelSpan.attributes["gen_ai.completion.0.content"], "Tokyo is sunny.");
  assert.deepEqual(JSON.parse(modelSpan.attributes["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: { name: "get_weather", arguments: JSON.stringify({ city: "Tokyo" }) },
    },
  ]);
  assert.equal(agentSpan.attributes["respan.span.tools"], undefined);
  assert.equal(modelSpan.attributes["respan.span.tools"], undefined);
  assert.equal(modelSpan.attributes["respan.span.tool_calls"], undefined);
  assert.equal(modelSpan.attributes["llm.system"], undefined);
  assert.equal(modelSpan.attributes.tools, undefined);
  assert.equal(modelSpan.attributes.tool_calls, undefined);
  assert.equal(modelSpan.attributes.model, undefined);

  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.input"]), { city: "Tokyo" });
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.output"]), { forecast: "sunny" });
});

test("parents orphan tool spans under the later agent span for the same run", async () => {
  captureState.spans = [];
  const instrumentor = new MastraInstrumentor();

  await instrumentor.exportTracingEvent({
    type: "span_ended",
    exportedSpan: {
      id: "orphan-tool-span",
      traceId: "trace-orphan",
      parentSpanId: "excluded-internal-step",
      name: "get_weather",
      type: "tool_call",
      startTime: new Date("2026-05-21T00:00:00.200Z"),
      endTime: new Date("2026-05-21T00:00:00.300Z"),
      attributes: { success: true },
      input: { city: "Tokyo" },
      output: { forecast: "sunny" },
      metadata: { runId: "run-1" },
    },
  });

  assert.equal(captureState.spans.length, 0);

  await instrumentor.exportTracingEvent({
    type: "span_ended",
    exportedSpan: {
      id: "late-agent-span",
      traceId: "trace-orphan",
      name: "Mastra Tool Example.workflow",
      type: "agent_run",
      startTime: new Date("2026-05-21T00:00:00.000Z"),
      endTime: new Date("2026-05-21T00:00:01.000Z"),
      input: "What is the weather in Tokyo?",
      output: "Tokyo is sunny.",
      metadata: { runId: "run-1" },
      entityName: "Mastra Tool Example.workflow",
    },
  });

  assert.equal(captureState.spans.length, 2);

  const agentSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "agent",
  );
  const toolSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );

  assert.ok(agentSpan);
  assert.ok(toolSpan);
  assert.equal(toolSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);
  assert.equal(toolSpan.spanContext().traceId, agentSpan.spanContext().traceId);
});

test("drops model chunk spans by default", async () => {
  captureState.spans = [];
  const instrumentor = new MastraInstrumentor();

  await instrumentor.exportTracingEvent({
    type: "span_ended",
    exportedSpan: {
      id: "chunk-span",
      traceId: "trace-456",
      name: "chunk",
      type: "model_chunk",
      startTime: new Date(),
      endTime: new Date(),
    },
  });

  assert.equal(captureState.spans.length, 0);
});
