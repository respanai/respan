import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { CursorSDKInstrumentor } from "../dist/index.js";

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

function createFakeCursorSdk() {
  class FakeRun {
    id = "run_123";
    requestId = "req_123";
    agentId = "agent_123";
    status = "running";
    model = { id: "cursor-small" };
    result = undefined;
    durationMs = undefined;
    supports() { return true; }
    unsupportedReason() { return undefined; }
    async *stream() {
      yield { type: "system", agent_id: "agent_123", run_id: "run_123", model: { id: "cursor-small" }, tools: ["search_docs"] };
      yield { type: "user", agent_id: "agent_123", run_id: "run_123", message: { role: "user", content: [{ type: "text", text: "Search Cursor docs" }] } };
      yield { type: "thinking", agent_id: "agent_123", run_id: "run_123", text: "I should use the docs tool." };
      yield { type: "assistant", agent_id: "agent_123", run_id: "run_123", message: { role: "assistant", content: [{ type: "tool_use", id: "call_docs", name: "search_docs", input: { query: "Cursor SDK tracing" } }, { type: "text", text: "Cursor SDK supports agent runs." }] } };
      yield { type: "tool_call", agent_id: "agent_123", run_id: "run_123", call_id: "call_docs", name: "search_docs", status: "running", args: { query: "Cursor SDK tracing" } };
      yield { type: "tool_call", agent_id: "agent_123", run_id: "run_123", call_id: "call_docs", name: "search_docs", status: "completed", result: { matches: 2 } };
      yield { type: "status", agent_id: "agent_123", run_id: "run_123", status: "FINISHED", message: "done" };
    }
    async wait() { return { id: "run_123", requestId: "req_123", status: "finished", result: "Cursor SDK supports agent runs.", model: { id: "cursor-small" }, durationMs: 25 }; }
    async conversation() { return []; }
    async cancel() {}
    onDidChangeStatus() { return () => {}; }
  }

  class FakeAgent {
    agentId = "agent_123";
    model = { id: "cursor-small" };
    async send(_message, options = {}) {
      await options.onStep?.({ step: { type: "demo-step" } });
      await options.onDelta?.({ update: { type: "text-delta", text: "Cursor" } });
      return new FakeRun();
    }
    close() {}
    async reload() {}
    async [Symbol.asyncDispose]() {}
    async listArtifacts() { return []; }
    async downloadArtifact() { return Buffer.from(""); }
  }

  class Agent {
    static async create(_options) { return new FakeAgent(); }
    static async resume(_agentId, _options) { return new FakeAgent(); }
    static async prompt(_message, _options) { return { id: "prompt_run", requestId: "prompt_req", status: "finished", result: "Prompt result", model: { id: "cursor-small" }, durationMs: 12 }; }
    static async getRun(_runId) { return new FakeRun(); }
  }

  return { Agent };
}

test("instrumentor emits canonical spans for Agent.create, send, stream, and tool events", async () => {
  captureState.spans = [];
  const sdk = createFakeCursorSdk();
  const originalCreate = sdk.Agent.create;
  const instrumentor = new CursorSDKInstrumentor({ sdkModule: sdk, agentName: "cursor_docs_agent" });
  instrumentor.activate();
  assert.notEqual(sdk.Agent.create, originalCreate);

  const agent = await sdk.Agent.create({ name: "cursor_docs_agent", model: { id: "cursor-small" } });
  const run = await agent.send("Search Cursor docs", {
    mcpServers: { docs: { type: "stdio", command: "node", args: ["docs-server.js"] } },
    onStep: async () => undefined,
    onDelta: async () => undefined,
  });

  const seen = [];
  for await (const message of run.stream()) seen.push(message.type);
  assert.deepEqual(seen, ["system", "user", "thinking", "assistant", "tool_call", "tool_call", "status"]);

  const agentSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "agent");
  const chatSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "chat");
  const toolSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "tool");
  const taskSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "task");
  assert.ok(agentSpan);
  assert.ok(chatSpan);
  assert.ok(toolSpan);
  assert.ok(taskSpan);

  assert.equal(agentSpan.instrumentationScope?.name, "@respan/instrumentation-cursor");
  assert.equal(agentSpan.attributes["traceloop.entity.name"], "cursor_docs_agent");
  assert.equal(agentSpan.attributes["respan.metadata.cursor_run_id"], "run_123");
  assert.equal(chatSpan.attributes["llm.request.type"], "chat");
  assert.equal(chatSpan.attributes["gen_ai.system"], "cursor");
  assert.equal(chatSpan.attributes["gen_ai.request.model"], "cursor-small");
  assert.deepEqual(JSON.parse(chatSpan.attributes["llm.request.functions"]), [
    { type: "function", function: { name: "mcp__docs", description: "Cursor MCP server docs" } },
    { type: "function", function: { name: "search_docs" } },
  ]);
  assert.deepEqual(JSON.parse(chatSpan.attributes["gen_ai.completion.0.tool_calls"]), [
    { id: "call_docs", type: "function", function: { name: "search_docs", arguments: JSON.stringify({ query: "Cursor SDK tracing" }) } },
  ]);
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.input"]), { name: "search_docs", arguments: { query: "Cursor SDK tracing" } });
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.output"]), { matches: 2 });
  assert.equal(toolSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);
  assert.equal(chatSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);

  for (const span of [agentSpan, chatSpan, toolSpan, taskSpan]) {
    assert.equal(span.attributes["traceloop.span.kind"], undefined);
    assert.equal(span.attributes["respan.span.tools"], undefined);
    assert.equal(span.attributes["respan.span.tool_calls"], undefined);
    assert.equal(span.attributes.tools, undefined);
    assert.equal(span.attributes.tool_calls, undefined);
    assert.equal(span.attributes.model, undefined);
    assert.equal(span.attributes.prompt_tokens, undefined);
    assert.equal(span.attributes.completion_tokens, undefined);
  }
  instrumentor.deactivate();
  assert.equal(sdk.Agent.create, originalCreate);
});

test("custom local tools are wrapped and emitted as tool spans", async () => {
  captureState.spans = [];
  const sdk = createFakeCursorSdk();
  class ToolAgent {
    agentId = "agent_tool";
    model = { id: "cursor-small" };
    async send(_message, options = {}) {
      await options.local.customTools.lookup_weather.execute({ city: "Tokyo" }, { toolCallId: "tool_custom" });
      return {
        id: "run_custom",
        requestId: "req_custom",
        agentId: "agent_tool",
        model: { id: "cursor-small" },
        async *stream() {},
        async wait() { return { id: "run_custom", status: "finished", result: "Tokyo is clear.", model: { id: "cursor-small" } }; },
      };
    }
  }
  sdk.Agent.create = async () => new ToolAgent();
  const instrumentor = new CursorSDKInstrumentor({ sdkModule: sdk });
  instrumentor.activate();
  const agent = await sdk.Agent.create({ name: "tool_agent" });
  const run = await agent.send("Check weather", {
    local: {
      customTools: {
        lookup_weather: {
          description: "Look up weather",
          inputSchema: { type: "object", properties: { city: { type: "string" } } },
          execute: async ({ city }) => ({ city, condition: "clear" }),
        },
      },
    },
  });
  await run.wait();
  const toolSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "tool");
  const chatSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "chat");
  assert.ok(toolSpan);
  assert.ok(chatSpan);
  assert.equal(toolSpan.attributes["traceloop.entity.name"], "lookup_weather");
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.output"]), { city: "Tokyo", condition: "clear" });
  assert.deepEqual(JSON.parse(chatSpan.attributes["llm.request.functions"]), [
    { type: "function", function: { name: "lookup_weather", description: "Look up weather", parameters: { type: "object", properties: { city: { type: "string" } } } } },
  ]);
  assert.equal(chatSpan.attributes["respan.span.tools"], undefined);
  assert.equal(chatSpan.attributes["respan.span.tool_calls"], undefined);
});

test("Agent.prompt emits summary spans and preserves the original method on deactivate", async () => {
  captureState.spans = [];
  const sdk = createFakeCursorSdk();
  const originalPrompt = sdk.Agent.prompt;
  const instrumentor = new CursorSDKInstrumentor({ sdkModule: sdk, agentName: "prompt_agent" });
  instrumentor.activate();
  const result = await sdk.Agent.prompt("Run once", { model: { id: "cursor-small" } });
  assert.equal(result.result, "Prompt result");
  const agentSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "agent");
  const chatSpan = captureState.spans.find((span) => span.attributes["respan.entity.log_type"] === "chat");
  assert.ok(agentSpan);
  assert.ok(chatSpan);
  assert.equal(agentSpan.attributes["traceloop.entity.name"], "prompt_agent");
  assert.equal(chatSpan.attributes["traceloop.entity.output"], "Prompt result");
  instrumentor.deactivate();
  assert.equal(sdk.Agent.prompt, originalPrompt);
});
