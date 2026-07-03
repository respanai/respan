import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  GoogleADKInstrumentor,
  GoogleADKTranslator,
  isGoogleADKSpan,
  translateGoogleADKSpan,
} from "../dist/index.js";

function createSpan(name, attributes) {
  return {
    name,
    attributes: { ...attributes },
    instrumentationScope: { name: "gcp.vertex.agent", version: "1.2.0" },
  };
}

test("translates ADK LLM spans to canonical chat attrs and strips raw keys", () => {
  const span = createSpan("call_llm", {
    "gen_ai.system": "gcp.vertex.agent",
    "gen_ai.request.model": "deterministic-gemini",
    "gcp.vertex.agent.invocation_id": "invocation-1",
    "gcp.vertex.agent.session_id": "session-1",
    "gcp.vertex.agent.llm_request": JSON.stringify({
      model: "deterministic-gemini",
      config: {
        systemInstruction: "You are concise.",
        tools: [
          {
            functionDeclarations: [
              {
                name: "get_weather",
                description: "Get weather.",
                parameters: { type: "OBJECT" },
              },
            ],
          },
        ],
      },
      contents: [
        {
          role: "user",
          parts: [{ text: "Weather in Tokyo?" }],
        },
      ],
    }),
    "gcp.vertex.agent.llm_response": JSON.stringify({
      content: {
        role: "model",
        parts: [
          {
            functionCall: {
              id: "call_1",
              name: "get_weather",
              args: { city: "Tokyo" },
            },
          },
        ],
      },
      usageMetadata: {
        promptTokenCount: 12,
        candidatesTokenCount: 5,
        thoughtsTokenCount: 2,
        totalTokenCount: 19,
      },
      finishReason: "STOP",
    }),
  });

  assert.equal(isGoogleADKSpan(span), true);
  translateGoogleADKSpan(span);

  const attrs = span.attributes;
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["traceloop.entity.name"], "google_adk.call_llm");
  assert.equal(attrs["traceloop.entity.path"], "google_adk.call_llm");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.system"], "google");
  assert.equal(attrs["gen_ai.request.model"], "deterministic-gemini");
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.0.content"], "You are concise.");
  assert.equal(attrs["gen_ai.prompt.1.role"], "user");
  assert.equal(attrs["gen_ai.prompt.1.content"], "Weather in Tokyo?");
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      name: "get_weather",
      description: "Get weather.",
      parameters: { type: "OBJECT" },
    },
  ]);
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: JSON.stringify({ city: "Tokyo" }),
      },
    },
  ]);
  assert.equal(attrs["gen_ai.completion.0.finish_reason"], "stop");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 7);
  assert.equal(attrs["llm.usage.total_tokens"], 19);

  assert.equal(attrs["gcp.vertex.agent.llm_request"], undefined);
  assert.equal(attrs["gcp.vertex.agent.llm_response"], undefined);
  assert.equal(attrs["respan.span.tools"], undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs.tools, undefined);
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs.model, undefined);
  assert.equal(attrs.prompt_tokens, undefined);
});

test("translates ADK tool spans with normalized input and output", () => {
  const span = createSpan("execute_tool get_weather", {
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.name": "get_weather",
    "gen_ai.tool.description": "Get weather.",
    "gen_ai.tool.type": "FunctionTool",
    "gen_ai.tool.call.id": "call_1",
    "gcp.vertex.agent.tool_call_args": JSON.stringify({ city: "Tokyo" }),
    "gcp.vertex.agent.tool_response": JSON.stringify({
      forecast: "sunny",
    }),
  });

  translateGoogleADKSpan(span);

  const attrs = span.attributes;
  assert.equal(attrs["respan.entity.log_type"], "tool");
  assert.equal(attrs["traceloop.entity.name"], "get_weather");
  assert.equal(attrs["traceloop.entity.path"], "get_weather");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), {
    name: "get_weather",
    arguments: { city: "Tokyo" },
  });
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), {
    forecast: "sunny",
  });
  assert.equal(attrs["respan.metadata.google_adk_tool_call_id"], "call_1");
  assert.equal(attrs["respan.metadata.google_adk_tool_type"], "FunctionTool");
  assert.equal(attrs["gen_ai.tool.name"], undefined);
  assert.equal(attrs["gcp.vertex.agent.tool_call_args"], undefined);
  assert.equal(attrs.tool_calls, undefined);
});

test("translates ADK workflow and agent spans", () => {
  const workflowSpan = createSpan("invocation", {});
  translateGoogleADKSpan(workflowSpan);
  assert.equal(workflowSpan.attributes["respan.entity.log_type"], "workflow");
  assert.equal(workflowSpan.attributes["traceloop.entity.name"], "google_adk.invocation");
  assert.equal(workflowSpan.attributes["traceloop.entity.path"], "");

  const agentSpan = createSpan("invoke_agent weather_agent", {
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.agent.name": "weather_agent",
    "gen_ai.agent.description": "Answer weather questions.",
    "gen_ai.conversation.id": "session-1",
  });
  translateGoogleADKSpan(agentSpan);
  assert.equal(agentSpan.attributes["respan.entity.log_type"], "agent");
  assert.equal(agentSpan.attributes["traceloop.entity.name"], "weather_agent");
  assert.equal(agentSpan.attributes["respan.metadata.agent_name"], "weather_agent");
  assert.equal(
    agentSpan.attributes["respan.metadata.google_adk_agent_description"],
    "Answer weather questions.",
  );
  assert.equal(agentSpan.attributes["respan.metadata.google_adk_conversation_id"], "session-1");
  assert.equal(agentSpan.attributes["gen_ai.agent.name"], undefined);
});

test("translator marks ADK span names at start so Respan exports them", () => {
  const translator = new GoogleADKTranslator();
  const attributes = {};
  const span = {
    name: "call_llm",
    setAttribute(key, value) {
      attributes[key] = value;
    },
  };

  translator.onStart(span, {});

  assert.equal(attributes["respan.entity.log_method"], "ts_tracing");
  assert.equal(attributes["respan.entity.log_type"], "chat");
  assert.equal(attributes["traceloop.entity.name"], "google_adk.call_llm");
  assert.equal(attributes["traceloop.entity.path"], "google_adk.call_llm");
});

test("translator hook mutates ADK spans before the active processor receives them", () => {
  const capturedSpans = [];
  const originalGetTracerProvider = trace.getTracerProvider.bind(trace);
  const activeSpanProcessor = {
    onEnd(span) {
      capturedSpans.push(span);
    },
  };

  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value() {
      return { activeSpanProcessor };
    },
  });

  try {
    const originalOnEnd = activeSpanProcessor.onEnd;
    const instrumentor = new GoogleADKInstrumentor();
    instrumentor.activate();
    assert.equal(instrumentor.isActive(), true);
    assert.notEqual(activeSpanProcessor.onEnd, originalOnEnd);

    const span = createSpan("execute_tool get_weather", {
      "gen_ai.operation.name": "execute_tool",
      "gen_ai.tool.name": "get_weather",
      "gcp.vertex.agent.tool_call_args": JSON.stringify({ city: "Tokyo" }),
      "gcp.vertex.agent.tool_response": JSON.stringify({ forecast: "sunny" }),
    });
    activeSpanProcessor.onEnd(span);

    assert.equal(capturedSpans.length, 1);
    assert.equal(capturedSpans[0].attributes["respan.entity.log_type"], "tool");
    assert.equal(capturedSpans[0].attributes["gcp.vertex.agent.tool_response"], undefined);

    instrumentor.deactivate();
    assert.equal(instrumentor.isActive(), false);
    assert.equal(activeSpanProcessor.onEnd, originalOnEnd);
  } finally {
    Object.defineProperty(trace, "getTracerProvider", {
      configurable: true,
      writable: true,
      value: originalGetTracerProvider,
    });
  }
});
