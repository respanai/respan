import assert from "node:assert/strict";
import test from "node:test";

import { ROOT_CONTEXT, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";

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
    setAttribute(key, value) {
      this.attributes[key] = value;
      return this;
    },
    setAttributes(values) {
      for (const [key, value] of Object.entries(values)) {
        this.setAttribute(key, value);
      }
      return this;
    },
  };
}

async function exportCapturedLlmResponses(responses) {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [
      new GoogleADKTranslator(),
      new SimpleSpanProcessor(exporter),
    ],
  });

  try {
    const span = provider
      .getTracer("gcp.vertex.agent", "1.3.0")
      .startSpan("call_llm");
    span.setAttributes({
      "gen_ai.system": "gcp.vertex.agent",
      "gen_ai.request.model": "deterministic-gemini",
      "gcp.vertex.agent.llm_request": JSON.stringify({
        model: "deterministic-gemini",
        contents: [
          {
            role: "user",
            parts: [{ text: "Stream a response." }],
          },
        ],
      }),
    });
    for (const response of responses) {
      span.setAttribute(
        "gcp.vertex.agent.llm_response",
        JSON.stringify(response),
      );
    }
    span.end();
    await provider.forceFlush();

    const finishedSpans = exporter.getFinishedSpans();
    assert.equal(finishedSpans.length, 1);
    return finishedSpans[0];
  } finally {
    await provider.shutdown();
  }
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

test("assembles every ADK streaming response chunk with OTel 2.x", async () => {
  const span = await exportCapturedLlmResponses([
    {
      content: { role: "model", parts: [{ text: "Streaming " }] },
      partial: true,
    },
    {
      content: { role: "model", parts: [{ text: "ADK telemetry " }] },
      partial: true,
    },
    {
      content: {
        role: "model",
        parts: [{ text: "complete with propagated Respan attributes." }],
      },
      usageMetadata: {
        promptTokenCount: 14,
        candidatesTokenCount: 7,
        totalTokenCount: 21,
      },
      finishReason: "STOP",
    },
  ]);

  const attrs = span.attributes;
  assert.equal(
    attrs["gen_ai.completion.0.content"],
    "Streaming ADK telemetry complete with propagated Respan attributes.",
  );
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.finish_reason"], "stop");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 14);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 7);
  assert.equal(attrs["llm.usage.total_tokens"], 21);
  assert.equal(attrs["gcp.vertex.agent.llm_response"], undefined);
});

test("retains repeated equal text deltas instead of treating them as cumulative", async () => {
  const span = await exportCapturedLlmResponses([
    {
      content: { role: "model", parts: [{ text: "ha" }] },
      partial: true,
    },
    {
      content: { role: "model", parts: [{ text: "ha" }] },
      partial: true,
    },
    {
      usageMetadata: {
        promptTokenCount: 2,
        candidatesTokenCount: 1,
        totalTokenCount: 3,
      },
      finishReason: "STOP",
    },
  ]);

  assert.equal(span.attributes["gen_ai.completion.0.content"], "haha");
});

test("does not duplicate cumulative terminal content or streamed tool calls", async () => {
  const span = await exportCapturedLlmResponses([
    {
      content: { role: "model", parts: [{ text: "Hello " }] },
      partial: true,
    },
    {
      content: { role: "model", parts: [{ text: "world" }] },
      partial: true,
    },
    {
      content: {
        role: "model",
        parts: [
          { text: "Hello world" },
          {
            functionCall: {
              id: "call_1",
              name: "get_weather",
              args: { city: "Tokyo" },
            },
          },
        ],
      },
      finishReason: "STOP",
    },
    {
      content: {
        role: "model",
        parts: [
          { text: "Hello world" },
          {
            functionCall: {
              id: "call_1",
              name: "get_weather",
              args: { city: "Tokyo" },
            },
          },
        ],
      },
      finishReason: "STOP",
    },
  ]);

  const attrs = span.attributes;
  assert.equal(attrs["gen_ai.completion.0.content"], "Hello world");
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
});

test("merges id-less streamed function calls by name and sequence", async () => {
  const span = await exportCapturedLlmResponses([
    {
      content: {
        role: "model",
        parts: [
          {
            functionCall: {
              name: "search_places",
              partialArgs: [
                {
                  jsonPath: "$.query",
                  stringValue: "weather ",
                  willContinue: true,
                },
              ],
              willContinue: true,
            },
          },
          {
            functionCall: {
              name: "search_places",
              partialArgs: [
                {
                  jsonPath: "$.query",
                  stringValue: "muse",
                  willContinue: true,
                },
              ],
              willContinue: true,
            },
          },
        ],
      },
      partial: true,
    },
    {
      content: {
        role: "model",
        parts: [
          {
            functionCall: {
              name: "search_places",
              partialArgs: [
                {
                  jsonPath: "$.query",
                  stringValue: "Tokyo",
                  willContinue: false,
                },
                {
                  jsonPath: "$.filters[0].kind",
                  stringValue: "forecast",
                  willContinue: false,
                },
              ],
              willContinue: false,
            },
          },
          {
            functionCall: {
              name: "search_places",
              partialArgs: [
                {
                  jsonPath: "$.query",
                  stringValue: "um",
                  willContinue: false,
                },
              ],
              willContinue: false,
            },
          },
        ],
      },
      partial: true,
    },
    {
      content: {
        role: "model",
        parts: [
          {
            functionCall: {
              name: "search_places",
              args: {
                query: "weather Tokyo",
                filters: [{ kind: "forecast" }],
              },
            },
          },
          {
            functionCall: {
              name: "search_places",
              args: { query: "museum" },
            },
          },
        ],
      },
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 3,
        totalTokenCount: 13,
      },
      finishReason: "STOP",
    },
  ]);

  const attrs = span.attributes;
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      type: "function",
      function: {
        name: "search_places",
        arguments: JSON.stringify({
          query: "weather Tokyo",
          filters: [{ kind: "forecast" }],
        }),
      },
    },
    {
      type: "function",
      function: {
        name: "search_places",
        arguments: JSON.stringify({ query: "museum" }),
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 3);
  assert.equal(attrs["llm.usage.total_tokens"], 13);
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
  assert.equal(attrs["gen_ai.tool.call.id"], "call_1");
  assert.equal(
    JSON.parse(attrs["respan.metadata"]).google_adk_tool_type,
    "FunctionTool",
  );
  assert.equal(attrs["gen_ai.tool.name"], undefined);
  assert.equal(attrs["gcp.vertex.agent.tool_call_args"], undefined);
  assert.equal(attrs.tool_calls, undefined);
});

test("translates ADK workflow and agent spans", () => {
  const workflowSpan = createSpan("invocation", {});
  translateGoogleADKSpan(workflowSpan);
  assert.equal(workflowSpan.attributes["respan.entity.log_type"], "workflow");
  assert.equal(
    workflowSpan.attributes["traceloop.entity.name"],
    "google_adk.invocation",
  );
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
  assert.equal(
    agentSpan.attributes["respan.metadata.agent_name"],
    "weather_agent",
  );
  assert.equal(
    JSON.parse(agentSpan.attributes["respan.metadata"])
      .google_adk_agent_description,
    "Answer weather questions.",
  );
  assert.equal(
    JSON.parse(agentSpan.attributes["respan.metadata"])
      .google_adk_conversation_id,
    "session-1",
  );
  assert.equal(agentSpan.attributes["gen_ai.agent.name"], undefined);
});

test("translator marks ADK span names at start so Respan exports them", () => {
  const translator = new GoogleADKTranslator();
  const attributes = {};
  const span = {
    name: "call_llm",
    attributes,
    instrumentationScope: { name: "gcp.vertex.agent" },
    setAttribute(key, value) {
      attributes[key] = value;
    },
  };

  translator.onStart(span, ROOT_CONTEXT);

  assert.equal(attributes["respan.entity.log_method"], "ts_tracing");
  assert.equal(attributes["respan.entity.log_type"], "chat");
  assert.equal(attributes["traceloop.entity.name"], "google_adk.call_llm");
  assert.equal(attributes["traceloop.entity.path"], "google_adk.call_llm");
});

test("activation requires a compatible initialized Respan host", () => {
  const instrumentor = new GoogleADKInstrumentor();
  assert.throws(
    () => instrumentor.activate(),
    /No compatible Respan span-transformer host/,
  );
  assert.equal(instrumentor.isActive(), false);
  instrumentor.deactivate();
});

test("foreign GenAI operations and matching span names stay untouched", () => {
  const span = createSpan("call_llm", { "gen_ai.operation.name": "chat" });
  span.instrumentationScope = { name: "another.library" };
  const original = span.setAttribute;
  const translator = new GoogleADKTranslator();
  translator.onStart(span, ROOT_CONTEXT);
  translateGoogleADKSpan(span);
  assert.deepEqual(span.attributes, { "gen_ai.operation.name": "chat" });
  assert.equal(span.setAttribute, original);
});
