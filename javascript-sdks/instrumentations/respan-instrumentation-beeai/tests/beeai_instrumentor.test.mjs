import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { BeeAIInstrumentor } from "../dist/index.js";

const BEEAI_SCOPE_NAME = "@arizeai/openinference-instrumentation-beeai";

function makeSpan({
  name = "test-span",
  attributes = {},
  instrumentationScopeName = BEEAI_SCOPE_NAME,
  traceId = "trace-test",
  spanId = `${name}-span`,
  parentSpanId,
} = {}) {
  return {
    name,
    parentSpanId,
    spanContext() {
      return {
        traceId,
        spanId,
        traceFlags: 1,
      };
    },
    attributes: { ...attributes },
    instrumentationScope: {
      name: instrumentationScopeName,
      version: "1.0.0",
    },
    instrumentationScope: {
      name: instrumentationScopeName,
      version: "1.0.0",
    },
  };
}

function resetTracerProvider(provider) {
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  if (provider) {
    trace.setGlobalTracerProvider(provider);
  }
}

function createFakeTracerProvider(processor) {
  return {
    activeSpanProcessor: processor,
    getTracer() {
      return {
        startSpan() {
          throw new Error("startSpan should not be called in this test");
        },
      };
    },
  };
}

test("BeeAIInstrumentor delegates activation with the provided BeeAI module", async () => {
  class FakeBeeAIInstrumentation {}

  const calls = [];
  const sdkModule = { BeeAgent: class BeeAgent {} };
  const delegate = {
    activate() {
      calls.push(["activate"]);
    },
    deactivate() {
      calls.push(["deactivate"]);
    },
  };

  const instrumentor = new BeeAIInstrumentor({
    sdkModule,
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory(instrumentationClass, module) {
      calls.push(["factory", instrumentationClass, module]);
      return delegate;
    },
  });

  await instrumentor.activate();
  await instrumentor.activate();
  instrumentor.deactivate();

  assert.deepEqual(calls, [
    ["factory", FakeBeeAIInstrumentation, sdkModule],
    ["activate"],
    ["deactivate"],
  ]);
});


test("BeeAIInstrumentor exports only complete BeeAI event rows", async () => {
  class FakeBeeAIInstrumentation {}

  const capturedSpans = [];
  const startedSpans = [];
  const processor = {
    onStart(span) {
      startedSpans.push(span);
    },
    onEnd(span) {
      if (Array.isArray(span.attributes["respan.processors"]) && span.attributes["respan.processors"].length === 0) {
        return;
      }
      capturedSpans.push(span);
    },
  };
  resetTracerProvider(createFakeTracerProvider(processor));

  const calls = [];
  const delegate = {
    activate() {
      calls.push("activate");
    },
    deactivate() {
      calls.push("deactivate");
    },
  };
  const instrumentor = new BeeAIInstrumentor({
    sdkModule: { BeeAgent: class BeeAgent {} },
    instrumentationClass: FakeBeeAIInstrumentation,
    delegateFactory() {
      return delegate;
    },
  });

  try {
    await instrumentor.activate();

    const expression = "(19 + 23) * 2";
    const userMessage = {
      role: "user",
      content: [{ type: "text", text: `Compute ${expression}` }],
    };
    const toolCallMessage = {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          toolCallId: "call-1",
          toolName: "Calculator",
          args: { expression },
        },
      ],
    };
    const toolResultMessage = {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "call-1",
          toolName: "Calculator",
          result: "84",
          isError: false,
        },
      ],
    };
    const agentStartState = {
      memory: { messages: [userMessage] },
      iteration: 1,
    };
    const agentSuccessState = {
      memory: { messages: [userMessage, toolCallMessage, toolResultMessage] },
      iteration: 1,
    };
    const finalToolCallMessage = {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          toolCallId: "call-final",
          toolName: "final_answer",
          args: { response: "84" },
        },
      ],
    };
    const finalToolResultMessage = {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "call-final",
          toolName: "final_answer",
          result: "Message has been sent",
          isError: false,
        },
      ],
    };
    const finalState = {
      memory: {
        messages: [
          userMessage,
          toolCallMessage,
          toolResultMessage,
          finalToolCallMessage,
          finalToolResultMessage,
        ],
      },
      result: {
        role: "assistant",
        content: [{ type: "text", text: "84" }],
      },
      iteration: 2,
    };

    const normalizedToolCall = {
      id: "call-1",
      type: "function",
      function: { name: "Calculator", arguments: JSON.stringify({ expression }) },
    };
    const normalizedFinalToolCall = {
      id: "call-final",
      type: "function",
      function: { name: "final_answer", arguments: JSON.stringify({ response: "84" }) },
    };
    const normalizedUserMessages = [
      { role: "user", content: `Compute ${expression}` },
    ];
    const normalizedFollowupMessages = [
      { role: "user", content: `Compute ${expression}` },
      { role: "assistant", content: "", tool_calls: [normalizedToolCall] },
      {
        role: "tool",
        tool_call_id: "call-1",
        name: "Calculator",
        content: "84",
        is_error: false,
      },
    ];
    const normalizedAgentInput = {
      iteration: 1,
      messages: [
        { role: "user", content: `Compute ${expression}` },
        { role: "assistant", content: "", tool_calls: [normalizedToolCall] },
      ],
    };
    const basicUserMessage = {
      role: "user",
      content: [{ type: "text", text: "Explain tracing in one sentence." }],
    };
    const basicAssistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: "Tracing shows each step and value in a run." }],
    };

    const basicChatSpan = makeSpan({
      name: "backend.openai.chat.success-basic",
      traceId: "otel-trace-basic",
      spanId: "basic-chat-span",
      attributes: {
        target: "backend.openai.chat.success",
        traceId: "trace-basic",
        "input.value": JSON.stringify([basicUserMessage]),
        "output.value": JSON.stringify([basicAssistantMessage]),
        data: JSON.stringify({
          value: {
            model: "gpt-4o-mini",
            messages: [basicAssistantMessage],
            usage: { promptTokens: 7, completionTokens: 9, totalTokens: 16 },
          },
        }),
      },
    });

    const agentStartSpan = makeSpan({
      name: "agent.toolCalling.start-1",
      traceId: "otel-trace-1",
      spanId: "start-span-1",
      parentSpanId: "framework-span",
      attributes: {
        target: "agent.toolCalling.start",
        traceId: "trace-1",
        data: JSON.stringify({}),
      },
    });
    const chatSpan = makeSpan({
      name: "backend.openai.chat.success-1",
      traceId: "otel-trace-1",
      spanId: "chat-span-1",
      parentSpanId: "start-span-1",
      attributes: {
        target: "backend.openai.chat.success",
        traceId: "trace-1",
        "input.value": JSON.stringify([{}]),
        "output.value": JSON.stringify([toolCallMessage]),
        "llm.input_messages.0.message.content": `Compute ${expression}`,
        "llm.output_messages.0.message.content": "",
        data: JSON.stringify({
          value: {
            model: "gpt-4o-mini",
            messages: [toolCallMessage],
            usage: { promptTokens: 18, completionTokens: 2, totalTokens: 20 },
          },
        }),
      },
    });
    const chat2Span = makeSpan({
      name: "backend.openai.chat.success-2",
      traceId: "otel-trace-1",
      spanId: "chat-span-2",
      parentSpanId: "agent-span-1",
      attributes: {
        target: "backend.openai.chat.success",
        traceId: "trace-1",
        "input.value": JSON.stringify([userMessage, toolCallMessage, toolResultMessage]),
        "output.value": JSON.stringify([finalToolCallMessage]),
        data: JSON.stringify({
          value: {
            model: "gpt-4o-mini",
            messages: [finalToolCallMessage],
            usage: { promptTokens: 22, completionTokens: 1, totalTokens: 23 },
          },
        }),
      },
    });
    const toolSpan = makeSpan({
      name: "tool.calculator.success-1",
      traceId: "otel-trace-1",
      spanId: "tool-span-1",
      parentSpanId: "chat-span-1",
      attributes: {
        target: "tool.calculator.success",
        traceId: "trace-1",
        data: JSON.stringify({
          input: { expression },
          output: { result: 84 },
        }),
      },
    });
    const agentSpan = makeSpan({
      name: "agent.toolCalling.success-1",
      traceId: "otel-trace-1",
      spanId: "agent-span-1",
      parentSpanId: "chat-span-1",
      attributes: {
        target: "agent.toolCalling.success",
        traceId: "trace-1",
        data: JSON.stringify({ state: agentSuccessState }),
      },
    });
    const finishSpan = makeSpan({
      name: "tool.calculator.finish-1",
      traceId: "otel-trace-1",
      spanId: "finish-span-1",
      parentSpanId: "tool-span-1",
      attributes: {
        target: "tool.calculator.finish",
        traceId: "trace-1",
        metadata: JSON.stringify({
          state: agentSuccessState,
          toolCallMsg: {
            type: "tool-call",
            toolCallId: "call-1",
            toolName: "Calculator",
            args: { expression },
          },
        }),
      },
    });
    const finalAnswerSpan = makeSpan({
      name: "tool.dynamic.finalAnswer.success-1",
      traceId: "otel-trace-1",
      spanId: "final-answer-span-1",
      parentSpanId: "chat-span-2",
      attributes: {
        target: "tool.dynamic.finalAnswer.success",
        traceId: "trace-1",
        metadata: JSON.stringify({
          state: finalState,
          toolCallMsg: {
            type: "tool-call",
            toolCallId: "call-final",
            toolName: "final_answer",
            args: { response: "84" },
          },
        }),
        data: JSON.stringify({
          input: { response: "84" },
          output: { result: "Message has been sent" },
        }),
      },
    });
    const parentSpan = makeSpan({
      name: "beeai-framework-main",
      traceId: "otel-trace-1",
      spanId: "framework-span",
      parentSpanId: "workflow-span",
      attributes: {
        traceId: "trace-1",
        source: "ToolCallingAgent",
        "beeai.version": "0.1.13",
      },
    });
    const workflowSpan = makeSpan({
      name: "beeai_tool_calling_agent.workflow.workflow",
      instrumentationScopeName: "@respan/tracing",
      traceId: "otel-trace-1",
      spanId: "workflow-span",
      attributes: {
        "traceloop.span.kind": "workflow",
      },
    });

    processor.onEnd(basicChatSpan);
    processor.onStart(workflowSpan, {});
    processor.onStart(parentSpan, {});
    processor.onStart(agentStartSpan, {});
    processor.onEnd(agentStartSpan);
    processor.onEnd(chatSpan);
    processor.onEnd(toolSpan);
    processor.onEnd(agentSpan);
    processor.onEnd(chat2Span);
    processor.onEnd(finishSpan);
    processor.onEnd(finalAnswerSpan);
    processor.onEnd(parentSpan);

    assert.deepEqual(calls, ["activate"]);
    assert.equal(startedSpans.length, 3);
    assert.equal(capturedSpans.length, 6);
    assert.deepEqual(capturedSpans, [
      basicChatSpan,
      toolSpan,
      chatSpan,
      agentSpan,
      chat2Span,
      finalAnswerSpan,
    ]);
    assert.deepEqual(agentStartSpan.attributes["respan.processors"], []);
    assert.deepEqual(finishSpan.attributes["respan.processors"], []);
    assert.deepEqual(parentSpan.attributes["respan.processors"], []);

    assert.equal(basicChatSpan.attributes["respan.entity.log_type"], "chat");
    assert.equal(basicChatSpan.attributes["llm.request.type"], "chat");
    assert.equal(basicChatSpan.attributes["traceloop.entity.name"], "backend.openai.chat.success");
    assert.equal(basicChatSpan.attributes["traceloop.entity.path"], "backend.openai.chat.success-basic");
    assert.equal(basicChatSpan.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(basicChatSpan.attributes.model, "gpt-4o-mini");
    assert.equal(basicChatSpan.attributes.prompt_tokens, 7);
    assert.equal(basicChatSpan.attributes.completion_tokens, 9);
    assert.equal(basicChatSpan.attributes.total_request_tokens, 16);
    assert.equal(
      basicChatSpan.attributes["traceloop.entity.input"],
      JSON.stringify([{ role: "user", content: "Explain tracing in one sentence." }]),
    );
    assert.equal(
      basicChatSpan.attributes["traceloop.entity.output"],
      JSON.stringify({ role: "assistant", content: "Tracing shows each step and value in a run." }),
    );
    assert.equal(basicChatSpan.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(
      basicChatSpan.attributes["gen_ai.completion.0.content"],
      "Tracing shows each step and value in a run.",
    );

    assert.equal(chatSpan.parentSpanId, "workflow-span");
    assert.equal(chatSpan.attributes["respan.entity.log_type"], "chat");
    assert.equal(chatSpan.attributes["llm.request.type"], "chat");
    assert.equal(chatSpan.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(chatSpan.attributes.model, "gpt-4o-mini");
    assert.equal(chatSpan.attributes.prompt_tokens, 18);
    assert.equal(chatSpan.attributes.completion_tokens, 2);
    assert.equal(chatSpan.attributes.total_request_tokens, 20);
    assert.equal(chatSpan.attributes["traceloop.entity.input"], JSON.stringify(normalizedUserMessages));
    assert.equal(
      chatSpan.attributes["traceloop.entity.output"],
      JSON.stringify({ role: "assistant", content: "", tool_calls: [normalizedToolCall] }),
    );
    assert.equal(chatSpan.attributes["gen_ai.prompt.0.role"], "user");
    assert.equal(chatSpan.attributes["gen_ai.prompt.0.content"], `Compute ${expression}`);
    assert.equal(chatSpan.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(chatSpan.attributes["gen_ai.completion.0.content"], "");
    assert.equal(
      chatSpan.attributes["respan.span.tool_calls"],
      JSON.stringify([normalizedToolCall]),
    );
    assert.deepEqual(
      chatSpan.attributes["gen_ai.completion.0.tool_calls"],
      [normalizedToolCall],
    );

    assert.equal(toolSpan.parentSpanId, "workflow-span");
    assert.equal(toolSpan.attributes["respan.entity.log_type"], "tool");
    assert.equal(toolSpan.attributes["traceloop.entity.input"], JSON.stringify({ expression }));
    assert.equal(toolSpan.attributes["traceloop.entity.output"], JSON.stringify({ result: 84 }));

    assert.equal(agentSpan.parentSpanId, "workflow-span");
    assert.equal(agentSpan.attributes["respan.entity.log_type"], "agent");
    assert.equal(agentSpan.attributes["traceloop.entity.input"], JSON.stringify(normalizedAgentInput));
    assert.equal(
      agentSpan.attributes["traceloop.entity.output"],
      JSON.stringify({ result: "84", is_error: false }),
    );

    assert.equal(chat2Span.parentSpanId, "workflow-span");
    assert.equal(chat2Span.attributes["respan.entity.log_type"], "chat");
    assert.equal(chat2Span.attributes["llm.request.type"], "chat");
    assert.equal(chat2Span.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(chat2Span.attributes.model, "gpt-4o-mini");
    assert.equal(chat2Span.attributes.prompt_tokens, 22);
    assert.equal(chat2Span.attributes.completion_tokens, 1);
    assert.equal(chat2Span.attributes.total_request_tokens, 23);
    assert.equal(chat2Span.attributes["traceloop.entity.input"], JSON.stringify(normalizedFollowupMessages));
    assert.equal(
      chat2Span.attributes["traceloop.entity.output"],
      JSON.stringify({ role: "assistant", content: "", tool_calls: [normalizedFinalToolCall] }),
    );
    assert.equal(chat2Span.attributes["gen_ai.prompt.0.role"], "user");
    assert.equal(chat2Span.attributes["gen_ai.prompt.0.content"], `Compute ${expression}`);
    assert.equal(chat2Span.attributes["gen_ai.prompt.1.role"], "assistant");
    assert.equal(chat2Span.attributes["gen_ai.prompt.1.content"], "");
    assert.deepEqual(chat2Span.attributes["gen_ai.prompt.1.tool_calls"], [normalizedToolCall]);
    assert.equal(chat2Span.attributes["gen_ai.prompt.2.role"], "tool");
    assert.equal(chat2Span.attributes["gen_ai.prompt.2.content"], "84");
    assert.equal(chat2Span.attributes["gen_ai.prompt.2.tool_call_id"], "call-1");
    assert.equal(chat2Span.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(chat2Span.attributes["gen_ai.completion.0.content"], "");
    assert.equal(
      chat2Span.attributes["respan.span.tool_calls"],
      JSON.stringify([normalizedFinalToolCall]),
    );
    assert.deepEqual(
      chat2Span.attributes["gen_ai.completion.0.tool_calls"],
      [normalizedFinalToolCall],
    );

    assert.equal(finalAnswerSpan.parentSpanId, "workflow-span");
    assert.equal(finalAnswerSpan.attributes["respan.entity.log_type"], "tool");
    assert.equal(finalAnswerSpan.attributes["traceloop.entity.input"], JSON.stringify({ response: "84" }));
    assert.equal(finalAnswerSpan.attributes["traceloop.entity.output"], "84");

    for (const span of capturedSpans) {
      assert.equal(span.attributes.data, undefined);
      assert.equal(span.attributes.metadata, undefined);
      assert.equal(span.attributes["input.value"], undefined);
      assert.equal(span.attributes["output.value"], undefined);
      assert.equal(span.attributes["llm.input_messages.0.message.content"], undefined);
      assert.equal(span.attributes["llm.output_messages.0.message.content"], undefined);
    }

    instrumentor.deactivate();
    assert.deepEqual(calls, ["activate", "deactivate"]);

    const rawSpan = makeSpan({
      name: "backend.openai.chat.success-2",
      attributes: { target: "backend.openai.chat.success" },
    });
    processor.onEnd(rawSpan);
    assert.equal(rawSpan.attributes["respan.entity.log_type"], undefined);
  } finally {
    instrumentor.deactivate();
    resetTracerProvider();
  }
});
