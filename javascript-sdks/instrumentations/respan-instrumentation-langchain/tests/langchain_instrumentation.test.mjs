import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  LangChainInstrumentor,
  RespanCallbackHandler,
  addRespanCallback,
  getCallbackHandler,
} from "../dist/index.js";
import {
  extractToolNamesFromSerialized,
  extractTools,
  extractUsage,
  normalizeChatMessages,
  toSerializableValue,
} from "../dist/_callback_helpers.js";

const captured = [];

function resetProvider() {
  captured.length = 0;
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  trace.setGlobalTracerProvider({
    activeSpanProcessor: {
      onStart() {},
      onEnd(span) {
        captured.push(span);
      },
      forceFlush() {
        return Promise.resolve();
      },
      shutdown() {
        return Promise.resolve();
      },
    },
    getTracer() {
      return {
        startSpan() {
          throw new Error("startSpan should not be called in these tests");
        },
      };
    },
  });
}

function runId(index) {
  return `00000000-0000-0000-0000-${String(index).padStart(12, "0")}`;
}

test("getCallbackHandler and addRespanCallback build callback configs without duplicate handlers", () => {
  const first = getCallbackHandler();
  const second = getCallbackHandler({ includeContent: false });
  const existing = { name: "existing-callback" };
  const config = { callbacks: [existing], tags: ["demo"] };

  const nextConfig = addRespanCallback(config, first);
  const dedupedConfig = addRespanCallback(nextConfig, second);

  assert.ok(first instanceof RespanCallbackHandler);
  assert.ok(second instanceof RespanCallbackHandler);
  assert.notEqual(first, second);
  assert.equal(first.groupLangflowRootRuns, true);
  assert.equal(second.includeContent, false);
  assert.deepEqual(config.callbacks, [existing]);
  assert.deepEqual(nextConfig.callbacks, [existing, first]);
  assert.equal(dedupedConfig.callbacks.length, 2);
});

test("LangChainInstrumentor exposes a reusable callback handler and lifecycle state", () => {
  const instrumentor = new LangChainInstrumentor();

  assert.equal(instrumentor.name, "langchain");
  assert.equal(instrumentor.isActive(), false);

  instrumentor.activate();
  assert.equal(instrumentor.isActive(), true);

  const config = instrumentor.addCallback({ metadata: { framework: "langchain" } });
  assert.deepEqual(config.callbacks, [instrumentor.callbackHandler]);

  instrumentor.deactivate();
  assert.equal(instrumentor.isActive(), false);
});

test("chain root and child emit workflow and task spans with parent linkage", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const rootRunId = runId(1);
  const childRunId = runId(2);

  handler.handleChainStart(
    { name: "root_chain" },
    { question: "hi" },
    rootRunId,
    undefined,
    ["langgraph"],
    { langgraph_node: "root" },
  );
  handler.handleChainStart(
    { name: "child_chain" },
    { input: "hi" },
    childRunId,
    rootRunId,
  );
  handler.handleChainEnd({ answer: "hello" }, childRunId);
  handler.handleChainEnd({ done: true }, rootRunId);

  const [childSpan, rootSpan] = captured;
  assert.equal(childSpan.attributes["respan.entity.log_type"], "task");
  assert.equal(rootSpan.attributes["respan.entity.log_type"], "workflow");
  assert.equal(childSpan.spanContext().traceId, rootSpan.spanContext().traceId);
  assert.equal(childSpan.parentSpanContext?.spanId, rootSpan.spanContext().spanId);
  assert.equal(rootSpan.attributes["langchain.framework"], "langgraph");
});

test("explicit Langflow handler groups independent root runs into one trace", () => {
  resetProvider();
  const handler = getCallbackHandler();
  const metadata = {
    framework: "langflow",
    langflow_component: "DemoComponent",
  };

  handler.handleToolStart(
    { name: "route_to_workspace" },
    { department: "security" },
    runId(3),
    undefined,
    ["langflow"],
    metadata,
  );
  handler.handleToolEnd("secops-critical", runId(3));
  handler.handleChainStart(
    { name: "component_chain" },
    { question: "hi" },
    runId(4),
    undefined,
    ["langflow"],
    metadata,
  );
  handler.handleChainEnd({ answer: "ok" }, runId(4));

  assert.equal(captured.length, 2);
  assert.equal(captured[0].spanContext().traceId, captured[1].spanContext().traceId);
  assert.equal(captured[0].parentSpanContext?.spanId, undefined);
  assert.equal(captured[1].parentSpanContext?.spanId, undefined);
});

test("chat model output maps messages, usage, model, tool calls, and strips JSON fences", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const llmRunId = runId(5);
  const fencedJson = '```json\n{"owner":"Security Operations Team"}\n```';

  handler.handleChatModelStart(
    { name: "ChatOpenAI", kwargs: { model: "gpt-4o-mini" } },
    [[{ type: "human", content: "Route this case" }]],
    llmRunId,
  );
  handler.handleLLMEnd(
    {
      generations: [[{
        message: {
          type: "ai",
          content: fencedJson,
          tool_calls: [
            { id: "call_1", name: "router", args: { department: "security" } },
          ],
          usage_metadata: { input_tokens: 12, output_tokens: 4 },
        },
      }]],
      llmOutput: { model_name: "gpt-4o-mini" },
    },
    llmRunId,
  );

  const span = captured[0];
  assert.equal(span.attributes["respan.entity.log_type"], "chat");
  assert.equal(span.attributes["llm.request.type"], "chat");
  assert.equal(span.attributes["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(span.attributes.model, "gpt-4o-mini");
  assert.equal(span.attributes["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(span.attributes["gen_ai.usage.completion_tokens"], 4);
  assert.equal(span.attributes.prompt_tokens, 12);
  assert.equal(span.attributes.completion_tokens, 4);
  assert.equal(span.attributes.total_request_tokens, 16);
  assert.equal(span.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(
    span.attributes["gen_ai.completion.0.content"],
    '{"owner":"Security Operations Team"}',
  );
  assert.equal(span.attributes.output.includes("```"), false);
  assert.ok(Array.isArray(span.attributes["respan.span.tool_calls"]));
  assert.equal(span.attributes["respan.span.tool_calls"][0].function.name, "router");
});

test("chat model start preserves full tool definitions from LangChain extra params", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const llmRunId = runId(51);
  const tools = [{
    type: "function",
    function: {
      name: "route_case",
      description: "Route the case to a department",
      parameters: {
        type: "object",
        properties: {
          department: { type: "string" },
        },
        required: ["department"],
      },
    },
  }];

  handler.handleChatModelStart(
    { name: "ChatOpenAI", kwargs: { model: "gpt-4o-mini" } },
    [[{ type: "human", content: "Route this case" }]],
    llmRunId,
    undefined,
    { invocation_params: { tools } },
  );
  handler.handleLLMEnd(
    {
      generations: [[{
        message: { type: "ai", content: "Done" },
      }]],
    },
    llmRunId,
  );

  const span = captured[0];
  assert.deepEqual(span.attributes["respan.span.tools"], tools);
  assert.deepEqual(span.attributes.tools, tools);
});

test("LLM streaming falls back to collected text when final output is empty", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const llmRunId = runId(6);

  handler.handleLLMStart(
    { name: "OpenAI", kwargs: { modelName: "text-davinci" } },
    ["Write a haiku"],
    llmRunId,
  );
  handler.handleLLMNewToken("old ", undefined, llmRunId);
  handler.handleLLMNewToken("pond", undefined, llmRunId);
  handler.handleLLMEnd(undefined, llmRunId);

  const span = captured[0];
  assert.equal(span.attributes["respan.entity.log_type"], "text");
  assert.equal(span.attributes["llm.request.type"], "completion");
  assert.equal(span.attributes.output, "old pond");
});

test("handleText streaming fallback records chain text output", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const chainRunId = runId(61);

  handler.handleChainStart({ name: "streaming_chain" }, {}, chainRunId);
  handler.handleText("hello ", chainRunId);
  handler.handleText("world", chainRunId);
  handler.handleChainEnd(undefined, chainRunId);

  assert.equal(captured[0].attributes.output, "hello world");
});

test("tool and retriever callbacks map fields and errors", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const toolRunId = runId(7);
  const retrieverRunId = runId(8);
  const errorRunId = runId(9);

  handler.handleToolStart({ name: "calculator" }, { expression: "2+2" }, toolRunId);
  handler.handleToolEnd({ answer: 4 }, toolRunId);
  handler.handleRetrieverStart({ name: "vectorstore" }, "query", retrieverRunId);
  handler.handleRetrieverEnd([{ pageContent: "doc text", metadata: { source: "unit" } }], retrieverRunId);
  handler.handleToolStart({ name: "failing_tool" }, "input", errorRunId);
  handler.handleToolError(new Error("tool failed"), errorRunId);

  const [toolSpan, retrieverSpan, errorSpan] = captured;
  assert.equal(toolSpan.attributes["respan.entity.log_type"], "tool");
  assert.equal(toolSpan.attributes["gen_ai.tool.name"], "calculator");
  assert.equal(toolSpan.attributes["gen_ai.tool.call.arguments"], '{"expression":"2+2"}');
  assert.equal(toolSpan.attributes["gen_ai.tool.call.result"], '{"answer":4}');
  assert.equal(retrieverSpan.attributes["respan.entity.log_type"], "task");
  assert.equal(retrieverSpan.attributes.output.includes("doc text"), true);
  assert.equal(errorSpan.status.code, 2);
  assert.equal(errorSpan.attributes["error.message"], "tool failed");
  assert.equal(errorSpan.attributes.status_code, 500);
});

test("chain, LLM, tool, and retriever error callbacks mark spans as failed", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const chainRunId = runId(91);
  const llmRunId = runId(92);
  const toolRunId = runId(93);
  const retrieverRunId = runId(94);

  handler.handleChainStart({ name: "chain" }, {}, chainRunId);
  handler.handleChainError(new Error("chain failed"), chainRunId);
  handler.handleLLMStart({ name: "llm" }, ["prompt"], llmRunId);
  handler.handleLLMError(new Error("llm failed"), llmRunId);
  handler.handleToolStart({ name: "tool" }, "input", toolRunId);
  handler.handleToolError(new Error("tool failed"), toolRunId);
  handler.handleRetrieverStart({ name: "retriever" }, "query", retrieverRunId);
  handler.handleRetrieverError(new Error("retriever failed"), retrieverRunId);

  assert.deepEqual(
    captured.map((span) => span.status.code),
    [2, 2, 2, 2],
  );
  assert.deepEqual(
    captured.map((span) => span.attributes["error.message"]),
    ["chain failed", "llm failed", "tool failed", "retriever failed"],
  );
  assert.deepEqual(
    captured.map((span) => span.attributes.status_code),
    [500, 500, 500, 500],
  );
});

test("agent action, agent end, and custom event emit event spans", () => {
  resetProvider();
  const handler = new RespanCallbackHandler();
  const chainRunId = runId(10);

  handler.handleChainStart({ name: "agent_chain" }, {}, chainRunId);
  handler.handleAgentAction(
    { tool: "search", toolInput: { q: "respan" }, log: "searching" },
    chainRunId,
  );
  handler.handleAgentEnd({ returnValues: { output: "done" } }, chainRunId);
  handler.handleCustomEvent("custom_step", { value: 1 }, chainRunId);
  handler.handleChainEnd({ done: true }, chainRunId);

  const [toolSpan, agentSpan, eventSpan, chainSpan] = captured;
  assert.equal(toolSpan.attributes["respan.entity.log_type"], "tool");
  assert.equal(toolSpan.attributes["gen_ai.tool.name"], "search");
  assert.equal(agentSpan.attributes["respan.entity.log_type"], "agent");
  assert.equal(eventSpan.name, "custom_step");
  assert.equal(eventSpan.attributes["respan.entity.log_type"], "task");
  assert.equal(chainSpan.attributes["respan.entity.log_type"], "workflow");
});

test("pure helpers normalize messages, usage, serializable values, and tool definitions", () => {
  const circular = { name: "root" };
  circular.self = circular;
  const serialized = toSerializableValue({
    createdAt: new Date("2026-05-08T00:00:00.000Z"),
    count: 1n,
    circular,
  });
  assert.deepEqual(serialized, {
    createdAt: "2026-05-08T00:00:00.000Z",
    count: "1",
    circular: {
      name: "root",
      self: "[Circular]",
    },
  });

  assert.deepEqual(
    normalizeChatMessages([{ type: "human", content: "hello" }]),
    [[{ role: "user", content: "hello" }]],
  );

  assert.deepEqual(
    extractUsage({
      llmOutput: {
        tokenUsage: {
          promptTokens: "10",
          completionTokens: 5.8,
        },
      },
    }),
    {
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    },
  );
  assert.deepEqual(extractUsage({ usage: { total_tokens: "7" } }), {
    totalTokens: 7,
  });
  assert.deepEqual(extractUsage({}), {});

  const langChainStyleTool = {
    name: "search",
    description: "Search docs",
    schema: {
      type: "object",
      properties: { query: { type: "string" } },
    },
  };
  const openAiStyleTool = {
    type: "function",
    function: {
      name: "route",
      parameters: { type: "object" },
    },
  };

  assert.deepEqual(
    extractTools({
      serialized: { kwargs: { tools: [langChainStyleTool] } },
      extraParams: { invocation_params: { tools: [openAiStyleTool] } },
    }),
    [
      openAiStyleTool,
      {
        type: "function",
        function: {
          name: "search",
          description: "Search docs",
          parameters: {
            type: "object",
            properties: { query: { type: "string" } },
          },
        },
      },
    ],
  );
  assert.deepEqual(
    extractToolNamesFromSerialized({ kwargs: { tools: [langChainStyleTool] } }),
    ["search"],
  );
  assert.equal(extractTools({ serialized: { tools: ["bad-tool"] } }), undefined);
});
