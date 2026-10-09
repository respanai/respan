import assert from "node:assert/strict";
import test from "node:test";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import * as bee from "beeai-framework";
import { DummyChatModel } from "beeai-framework/adapters/dummy/backend/chat";
import { ToolCallingAgent } from "beeai-framework/agents/toolCalling/agent";
import { ChatModelOutput } from "beeai-framework/backend/chat";
import { AssistantMessage, UserMessage } from "beeai-framework/backend/message";
import { CalculatorTool } from "beeai-framework/tools/calculator";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { BeeAIInstrumentor } from "../dist/index.js";

test(
  "declared minimum released SDK uses its native final text response and connected native tree",
  { skip: Number(bee.Version.split(".")[2]) >= 13 },
  async () => {
    trace.disable();
    context.disable();
    const manager = new AsyncLocalStorageContextManager().enable();
    context.setGlobalContextManager(manager);
    const exporter = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    trace.setGlobalTracerProvider(provider);
    const instrumentor = new BeeAIInstrumentor({ sdkModule: bee });
    await instrumentor.activate();
    try {
      let turn = 0;
      const llm = new DummyChatModel("dummy");
      llm._create = async () =>
        new ChatModelOutput(
          [
            ++turn === 1
              ? new AssistantMessage({
                  type: "tool-call",
                  toolCallId: "floor-call",
                  toolName: "Calculator",
                  args: { expression: "(19+23)*2" },
                })
              : new AssistantMessage("84"),
          ],
          { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
          turn === 1 ? "tool-call" : "stop",
        );
      const agent = new ToolCallingAgent({
        llm,
        memory: new UnconstrainedMemory(),
        tools: [new CalculatorTool()],
      });
      const result = await trace
        .getTracer("floor")
        .startActiveSpan(
          "floor.workflow.workflow",
          { attributes: { "traceloop.span.kind": "workflow" } },
          async (span) => {
            try {
              return await agent.run({ prompt: "Compute (19+23)*2" });
            } finally {
              span.end();
            }
          },
        );
      assert.equal(result.result.text, "84");
      assert.equal(turn, 2);
      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      const agents = spans.filter(
        (s) => s.attributes["respan.entity.log_type"] === "agent",
      );
      assert.equal(agents.length, 1);
      const chats = spans.filter(
        (s) => s.attributes["respan.entity.log_type"] === "chat",
      );
      assert.equal(chats.length, 2);
      const tools = spans.filter(
        (s) => s.attributes["respan.entity.log_type"] === "tool",
      );
      assert.equal(tools.length, 1);
      assert.equal(
        chats.at(-1).attributes["gen_ai.completion.0.content"],
        "84",
      );
      assert.equal(
        JSON.parse(tools[0].attributes["traceloop.entity.output"]),
        84,
      );
      for (const child of [...chats, ...tools])
        assert.equal(
          child.parentSpanContext.spanId,
          agents[0].spanContext().spanId,
        );
    } finally {
      instrumentor.deactivate();
      await provider.shutdown();
      manager.disable();
      trace.disable();
      context.disable();
    }
  },
);
