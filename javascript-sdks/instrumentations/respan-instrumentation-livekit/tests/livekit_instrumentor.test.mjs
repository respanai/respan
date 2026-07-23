import assert from "node:assert/strict";
import test from "node:test";

import { RespanSpanAttributes } from "@respan/respan-sdk";

import { LiveKitInstrumentor } from "../dist/index.js";

const capturedSpans = [];

class FakeSpan {
  constructor(name, attributes = {}) {
    this.name = name;
    this.attributes = { ...attributes };
    this.ended = false;
  }

  setAttribute(key, value) {
    this.attributes[key] = value;
    return this;
  }

  setAttributes(attributes) {
    Object.assign(this.attributes, attributes);
    return this;
  }

  end() {
    this.ended = true;
    capturedSpans.push(this);
  }
}

function createFakeTelemetry() {
  const telemetry = {
    provider: undefined,
    setTracerProvider(provider) {
      telemetry.provider = provider;
    },
    tracer: {
      startSpan(options = {}) {
        return new FakeSpan(options.name, options.attributes);
      },
      async startActiveSpan(fn, options = {}) {
        const span = new FakeSpan(options.name, options.attributes);
        try {
          return await fn(span);
        } finally {
          span.end();
        }
      },
      startActiveSpanSync(fn, options = {}) {
        const span = new FakeSpan(options.name, options.attributes);
        try {
          return fn(span);
        } finally {
          span.end();
        }
      },
    },
  };
  return telemetry;
}

test.beforeEach(() => {
  capturedSpans.length = 0;
});

test("translates LiveKit llm_node spans to canonical chat attrs", async () => {
  const telemetry = createFakeTelemetry();
  const instrumentor = new LiveKitInstrumentor({
    telemetryModule: telemetry,
    syncTracerProvider: false,
  });
  await instrumentor.activate();

  const span = telemetry.tracer.startSpan({
    name: "llm_node",
    attributes: {
      "lk.chat_ctx": JSON.stringify({
        items: [
          {
            type: "message",
            role: "user",
            content: ["What is the weather in Tokyo?"],
          },
        ],
      }),
      "lk.function_tools": JSON.stringify(["get_weather"]),
      "gen_ai.provider.name": "openai",
      "gen_ai.request.model": "gpt-4o-mini",
      tools: [{ name: "legacy" }],
      tool_calls: [{ name: "legacy_call" }],
      "respan.span.tool_calls": [{ name: "legacy_call" }],
    },
  });

  span.setAttribute("lk.response.text", "Tokyo is sunny.");
  span.end();

  assert.equal(capturedSpans.length, 1);
  assert.equal(span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE], "chat");
  assert.equal(span.attributes[RespanSpanAttributes.RESPAN_LOG_METHOD], "ts_tracing");
  assert.equal(span.attributes["traceloop.entity.name"], "openai.gpt-4o-mini");
  assert.equal(span.attributes["gen_ai.system"], "openai");
  assert.equal(span.attributes["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(span.attributes["llm.request.type"], "chat");
  assert.equal(span.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(span.attributes["gen_ai.prompt.0.content"], "What is the weather in Tokyo?");
  assert.equal(span.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(span.attributes["gen_ai.completion.0.content"], "Tokyo is sunny.");
  assert.deepEqual(JSON.parse(span.attributes["llm.request.functions"]), [
    { type: "function", function: { name: "get_weather" } },
  ]);
  assert.equal(span.attributes.tools, undefined);
  assert.equal(span.attributes.tool_calls, undefined);
  assert.equal(span.attributes["respan.span.tool_calls"], undefined);

  instrumentor.deactivate();
});

test("translates LiveKit function_tool spans without tool-call aliases", async () => {
  const telemetry = createFakeTelemetry();
  const instrumentor = new LiveKitInstrumentor({
    telemetryModule: telemetry,
    syncTracerProvider: false,
  });
  await instrumentor.activate();

  const result = await telemetry.tracer.startActiveSpan(
    async (span) => {
      span.setAttribute("lk.function_tool.name", "get_weather");
      span.setAttribute("lk.function_tool.arguments", JSON.stringify({ city: "Tokyo" }));
      span.setAttribute("lk.function_tool.output", JSON.stringify({ forecast: "sunny" }));
      span.setAttribute("lk.function_tool.is_error", false);
      span.setAttribute("tool_calls", [{ name: "get_weather" }]);
      return "ok";
    },
    { name: "function_tool" },
  );

  assert.equal(result, "ok");
  assert.equal(capturedSpans.length, 1);
  const span = capturedSpans[0];
  assert.equal(span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE], "tool");
  assert.equal(span.attributes["traceloop.entity.name"], "get_weather");
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.input"]), {
    name: "get_weather",
    arguments: { city: "Tokyo" },
  });
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.output"]), {
    output: { forecast: "sunny" },
    is_error: false,
  });
  assert.equal(span.attributes.tool_calls, undefined);
  assert.equal(span.attributes["respan.span.tool_calls"], undefined);

  instrumentor.deactivate();
});

test("translates LiveKit user_turn and tts spans", async () => {
  const telemetry = createFakeTelemetry();
  const instrumentor = new LiveKitInstrumentor({
    telemetryModule: telemetry,
    syncTracerProvider: false,
  });
  await instrumentor.activate();

  const userTurn = telemetry.tracer.startSpan({ name: "user_turn" });
  userTurn.setAttributes({
    "lk.user_transcript": "book a table",
    "lk.transcript_confidence": 0.98,
    "lk.end_of_turn_delay": 12,
  });
  userTurn.end();

  const tts = telemetry.tracer.startSpan({
    name: "tts_request",
    attributes: {
      "lk.tts.label": "fake-tts",
      "gen_ai.provider.name": "livekit",
      "gen_ai.request.model": "voice-demo",
      "lk.tts_metrics": JSON.stringify({ inputTokens: 4, outputTokens: 6 }),
    },
  });
  tts.end();

  assert.equal(capturedSpans[0].attributes[RespanSpanAttributes.RESPAN_LOG_TYPE], "task");
  assert.deepEqual(JSON.parse(capturedSpans[0].attributes["traceloop.entity.output"]), {
    transcript: "book a table",
    confidence: 0.98,
    end_of_turn_delay: 12,
  });

  assert.equal(capturedSpans[1].attributes[RespanSpanAttributes.RESPAN_LOG_TYPE], "text");
  assert.equal(capturedSpans[1].attributes["gen_ai.system"], "livekit");
  assert.equal(capturedSpans[1].attributes["gen_ai.usage.input_tokens"], 4);
  assert.equal(capturedSpans[1].attributes["gen_ai.usage.output_tokens"], 6);

  instrumentor.deactivate();
});

test("does not translate uncorrelated LiveKit activity lifecycle spans", async () => {
  const telemetry = createFakeTelemetry();
  const instrumentor = new LiveKitInstrumentor({
    telemetryModule: telemetry,
    syncTracerProvider: false,
  });
  await instrumentor.activate();

  for (const name of ["start_agent_activity", "on_enter", "on_exit", "drain_agent_activity"]) {
    const span = telemetry.tracer.startSpan({
      name,
      attributes: { "lk.agent_label": "weather_agent" },
    });
    span.end();

    assert.equal(span.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE], undefined);
    assert.equal(span.attributes[RespanSpanAttributes.RESPAN_LOG_METHOD], undefined);
    assert.equal(span.attributes["traceloop.entity.name"], undefined);
  }

  instrumentor.deactivate();
});

test("deactivate restores LiveKit tracer methods", async () => {
  const telemetry = createFakeTelemetry();
  const originalStartSpan = telemetry.tracer.startSpan;
  const instrumentor = new LiveKitInstrumentor({
    telemetryModule: telemetry,
    syncTracerProvider: false,
  });

  await instrumentor.activate();
  assert.notEqual(telemetry.tracer.startSpan, originalStartSpan);

  instrumentor.deactivate();
  assert.equal(telemetry.tracer.startSpan, originalStartSpan);
}
);
