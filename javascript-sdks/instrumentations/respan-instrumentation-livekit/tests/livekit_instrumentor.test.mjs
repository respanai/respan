import assert from "node:assert/strict";
import test from "node:test";

import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { getRegisteredSpanTransformerKeys } from "@respan/tracing";
import {
  MultiProcessorManager,
  RespanCompositeProcessor,
} from "@respan/tracing/dist/processor/index.js";

import { LiveKitInstrumentor } from "../dist/index.js";

function createOTel2Harness(spanNameStyle = "semantic") {
  const exporter = new InMemorySpanExporter();
  const manager = new MultiProcessorManager({
    disableBatch: true,
    spanNameStyle,
  });
  manager.addProcessor({
    exporter,
    name: "default",
    disableBatch: true,
  });
  const composite = new RespanCompositeProcessor(manager);
  const provider = new BasicTracerProvider({ spanProcessors: [composite] });
  return {
    exporter,
    provider,
    tracer: provider.getTracer("@livekit/agents"),
  };
}

function createInstrumentor() {
  return new LiveKitInstrumentor({
    telemetryModule: { tracer: {} },
    syncTracerProvider: false,
  });
}

function childContext(parentSpan) {
  return trace.setSpan(context.active(), parentSpan);
}

function findSpan(exporter, liveKitName) {
  return exporter
    .getFinishedSpans()
    .find((span) => span.attributes["traceloop.entity.path"] === liveKitName);
}

function assertNoRawLiveKitAttributes(span) {
  for (const key of Object.keys(span.attributes)) {
    assert.equal(
      key.startsWith("lk."),
      false,
      `raw LiveKit attr exported: ${key}`,
    );
  }
  assert.equal(span.attributes["livekit.span.name"], undefined);
  assert.equal(
    span.attributes["langfuse.observation.completion_start_time"],
    undefined,
  );
}

test("activation fails without a compatible transformer host", async () => {
  const instrumentor = createInstrumentor();
  await assert.rejects(
    instrumentor.activate(),
    /No compatible Respan span-transformer host is active/,
  );
  assert.equal(instrumentor.isActive(), false);
  assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
});

test("concurrent activate calls share one transformer registry acquisition", async () => {
  const harness = createOTel2Harness();
  const instrumentor = createInstrumentor();

  try {
    await Promise.all([
      instrumentor.activate(),
      instrumentor.activate(),
      instrumentor.activate(),
    ]);
    assert.equal(instrumentor.isActive(), true);
    assert.deepEqual(getRegisteredSpanTransformerKeys(), [
      "@respan/instrumentation-livekit",
    ]);

    instrumentor.deactivate();
    assert.equal(instrumentor.isActive(), false);
    assert.deepEqual(
      getRegisteredSpanTransformerKeys(),
      [],
      "one deactivate must release the only shared-registry lease",
    );
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("deactivate during module resolution prevents late transformer registration", async () => {
  const harness = createOTel2Harness();
  const instrumentor = createInstrumentor();
  let resolveTelemetryModule;
  const telemetryModule = new Promise((resolve) => {
    resolveTelemetryModule = resolve;
  });
  instrumentor._resolveTelemetryModule = () => telemetryModule;

  try {
    const pendingActivation = instrumentor.activate();
    instrumentor.deactivate();
    resolveTelemetryModule({ tracer: {} });
    await pendingActivation;

    assert.equal(instrumentor.isActive(), false);
    assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("a fresh activate can supersede a cancelled pending activation", async () => {
  const harness = createOTel2Harness();
  const instrumentor = createInstrumentor();
  const resolvers = [];
  instrumentor._resolveTelemetryModule = () =>
    new Promise((resolve) => {
      resolvers.push(resolve);
    });

  try {
    const cancelledActivation = instrumentor.activate();
    instrumentor.deactivate();
    const freshActivation = instrumentor.activate();
    assert.equal(resolvers.length, 2);

    resolvers[1]({ tracer: {} });
    await freshActivation;
    assert.equal(instrumentor.isActive(), true);
    assert.deepEqual(getRegisteredSpanTransformerKeys(), [
      "@respan/instrumentation-livekit",
    ]);

    // The old generation may resolve last, but it must neither register a
    // second lease nor disturb the newer active generation.
    resolvers[0]({ tracer: {} });
    await cancelledActivation;
    assert.equal(instrumentor.isActive(), true);
    assert.deepEqual(getRegisteredSpanTransformerKeys(), [
      "@respan/instrumentation-livekit",
    ]);

    instrumentor.deactivate();
    assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("real OTel 2 merges a direct llm_request into llm_node and preserves retry errors", async () => {
  const harness = createOTel2Harness();
  assert.equal(
    typeof harness.provider.addSpanProcessor,
    "undefined",
    "the regression harness must use the OTel 2 provider API",
  );
  const instrumentor = createInstrumentor();
  await instrumentor.activate();

  try {
    const node = harness.tracer.startSpan("llm_node");
    node.setAttributes({
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
      "gen_ai.provider.name": "unknown",
      "gen_ai.request.model": "unknown",
      "lk.response.text": "Tokyo is sunny.",
      "lk.response.ttft": 0.125,
    });

    const request = harness.tracer.startSpan(
      "llm_request",
      {},
      childContext(node),
    );
    request.setAttributes({
      "gen_ai.provider.name": "openai",
      "gen_ai.request.model": "gpt-4o-mini",
      "gen_ai.usage.input_tokens": 12,
      "gen_ai.usage.output_tokens": 4,
      "lk.llm_metrics": JSON.stringify({
        requestId: "req_metrics_456",
        ttftMs: 125,
        durationMs: 800,
        tokensPerSecond: 5,
        cancelled: false,
        promptTokens: 12,
        completionTokens: 4,
        promptCachedTokens: 2,
        totalTokens: 16,
        metadata: {
          modelProvider: "openai",
          modelName: "gpt-4o-mini",
        },
      }),
    });

    const attempt = harness.tracer.startSpan(
      "llm_request_run",
      {},
      childContext(request),
    );
    attempt.setAttribute("lk.retry_count", 1);
    attempt.setAttribute("lk.provider_request_ids", ["req_livekit_123"]);
    attempt.recordException(new Error("first provider attempt failed"));
    attempt.setStatus({
      code: SpanStatusCode.ERROR,
      message: "retryable provider error",
    });
    attempt.end();
    request.end();
    node.end();

    const exported = harness.exporter.getFinishedSpans();
    assert.equal(
      exported.length,
      2,
      "the correlated request wrapper is dropped",
    );

    const chat = findSpan(harness.exporter, "llm_node");
    assert.ok(chat);
    assert.equal(chat.name, "llm.gpt-4o-mini");
    assert.equal(chat.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE], "chat");
    assert.equal(chat.attributes["gen_ai.provider.name"], "openai");
    assert.equal(chat.attributes["gen_ai.system"], "openai");
    assert.equal(chat.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(chat.attributes["gen_ai.usage.prompt_tokens"], 12);
    assert.equal(chat.attributes["gen_ai.usage.completion_tokens"], 4);
    assert.equal(chat.attributes["llm.usage.total_tokens"], 16);
    assert.equal(chat.attributes["gen_ai.usage.cache_read.input_tokens"], 2);
    assert.equal(chat.attributes["gen_ai.response.time_to_first_chunk"], 0.125);
    const chatMetadata = JSON.parse(
      chat.attributes[RespanSpanAttributes.RESPAN_METADATA],
    );
    assert.deepEqual(chatMetadata.provider_request_ids, [
      "req_livekit_123",
      "req_metrics_456",
    ]);
    assert.equal(chatMetadata.time_to_first_token_seconds, 0.125);
    assert.equal(chatMetadata.llm_ttft_ms, 125);
    assert.equal(chatMetadata.llm_duration_ms, 800);
    assert.equal(chatMetadata.llm_tokens_per_second, 5);
    assert.equal(chatMetadata.llm_cancelled, false);
    assert.equal(chat.attributes["gen_ai.prompt.0.role"], "user");
    assert.equal(
      chat.attributes["gen_ai.prompt.0.content"],
      "What is the weather in Tokyo?",
    );
    assert.equal(chat.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(
      chat.attributes["gen_ai.completion.0.content"],
      "Tokyo is sunny.",
    );
    assert.deepEqual(JSON.parse(chat.attributes["traceloop.entity.output"]), {
      text: "Tokyo is sunny.",
    });
    assertNoRawLiveKitAttributes(chat);

    const retry = findSpan(harness.exporter, "llm_request_run");
    assert.ok(retry);
    assert.equal(
      retry.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE],
      "task",
    );
    assert.deepEqual(JSON.parse(retry.attributes["traceloop.entity.input"]), {
      retry_count: 1,
    });
    assert.equal(
      JSON.parse(retry.attributes[RespanSpanAttributes.RESPAN_METADATA])
        .retry_count,
      1,
    );
    assert.equal(retry.status.code, SpanStatusCode.ERROR);
    assert.equal(retry.events.length, 1);
    assert.equal(retry.parentSpanContext?.spanId, chat.spanContext().spanId);
    assertNoRawLiveKitAttributes(retry);
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("standalone llm_request remains a chat without metrics-as-output or empty completion", async () => {
  const harness = createOTel2Harness("legacy");
  const instrumentor = createInstrumentor();
  await instrumentor.activate();

  try {
    const request = harness.tracer.startSpan("llm_request", {
      attributes: {
        "gen_ai.provider.name": "livekit",
        "gen_ai.request.model": "standalone-model",
        "gen_ai.usage.input_tokens": 3,
        "gen_ai.usage.output_tokens": 0,
        "lk.llm_metrics": JSON.stringify({
          promptTokens: 3,
          completionTokens: 0,
          totalTokens: 3,
          metadata: {
            modelProvider: "livekit",
            modelName: "standalone-model",
          },
        }),
      },
    });
    request.end();

    const exported = findSpan(harness.exporter, "llm_request");
    assert.ok(exported);
    assert.equal(
      exported.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE],
      "chat",
    );
    assert.equal(
      exported.attributes["gen_ai.request.model"],
      "standalone-model",
    );
    assert.equal(exported.attributes["llm.usage.total_tokens"], 3);
    assert.equal(exported.attributes["traceloop.entity.input"], undefined);
    assert.equal(exported.attributes["traceloop.entity.output"], undefined);
    assert.equal(exported.attributes["gen_ai.completion.0.role"], undefined);
    assert.equal(exported.attributes["gen_ai.completion.0.content"], undefined);
    assert.equal(
      exported.attributes["gen_ai.response.time_to_first_chunk"],
      undefined,
    );
    assertNoRawLiveKitAttributes(exported);
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("standalone llm_request converts metrics ttftMs to time_to_first_chunk seconds", async () => {
  const harness = createOTel2Harness("legacy");
  const instrumentor = createInstrumentor();
  await instrumentor.activate();

  try {
    for (const [model, ttftMs] of [
      ["streamed-model", 250],
      ["no-chunk-model", -1],
    ]) {
      const request = harness.tracer.startSpan("llm_request", {
        attributes: {
          "gen_ai.provider.name": "livekit",
          "gen_ai.request.model": model,
          "lk.llm_metrics": JSON.stringify({
            ttftMs,
            metadata: { modelProvider: "livekit", modelName: model },
          }),
        },
      });
      request.end();
    }

    const [streamed, noChunk] = harness.exporter
      .getFinishedSpans()
      .filter((span) => span.attributes["gen_ai.request.model"] !== undefined);
    assert.equal(streamed.attributes["gen_ai.request.model"], "streamed-model");
    assert.equal(
      streamed.attributes["gen_ai.response.time_to_first_chunk"],
      0.25,
    );
    assert.equal(noChunk.attributes["gen_ai.request.model"], "no-chunk-model");
    assert.equal(
      noChunk.attributes["gen_ai.response.time_to_first_chunk"],
      undefined,
    );
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("interleaved LiveKit nodes keep correlation state isolated by trace and span id", async () => {
  const harness = createOTel2Harness("legacy");
  const first = createInstrumentor();
  const second = createInstrumentor();
  await first.activate();
  await second.activate();

  try {
    assert.deepEqual(getRegisteredSpanTransformerKeys(), [
      "@respan/instrumentation-livekit",
    ]);

    const nodeA = harness.tracer.startSpan("llm_node");
    nodeA.setAttribute("lk.response.text", "response A");
    const nodeB = harness.tracer.startSpan("llm_node");
    nodeB.setAttribute("lk.response.text", "response B");
    const requestB = harness.tracer.startSpan(
      "llm_request",
      {},
      childContext(nodeB),
    );
    requestB.setAttribute("gen_ai.request.model", "model-b");
    const requestA = harness.tracer.startSpan(
      "llm_request",
      {},
      childContext(nodeA),
    );
    requestA.setAttribute("gen_ai.request.model", "model-a");

    requestA.end();
    requestB.end();
    nodeB.end();
    nodeA.end();

    const chats = harness.exporter
      .getFinishedSpans()
      .filter(
        (span) => span.attributes["traceloop.entity.path"] === "llm_node",
      );
    assert.equal(chats.length, 2);
    const modelsByOutput = Object.fromEntries(
      chats.map((span) => [
        JSON.parse(span.attributes["traceloop.entity.output"]).text,
        span.attributes["gen_ai.request.model"],
      ]),
    );
    assert.deepEqual(modelsByOutput, {
      "response A": "model-a",
      "response B": "model-b",
    });
    harness.exporter.getFinishedSpans().forEach(assertNoRawLiveKitAttributes);

    first.deactivate();
    assert.deepEqual(getRegisteredSpanTransformerKeys(), [
      "@respan/instrumentation-livekit",
    ]);
    second.deactivate();
    assert.deepEqual(getRegisteredSpanTransformerKeys(), []);
  } finally {
    first.deactivate();
    second.deactivate();
    await harness.provider.shutdown();
  }
});

test("tool, user-turn, and TTS spans keep their span-contract mappings", async () => {
  const harness = createOTel2Harness("legacy");
  const instrumentor = createInstrumentor();
  await instrumentor.activate();

  try {
    const tool = harness.tracer.startSpan("function_tool", {
      attributes: {
        "lk.function_tool.name": "get_weather",
        "lk.function_tool.arguments": JSON.stringify({ city: "Tokyo" }),
        "lk.function_tool.output": JSON.stringify({ forecast: "sunny" }),
        "lk.function_tool.is_error": false,
        tool_calls: [{ name: "legacy_alias" }],
      },
    });
    tool.end();

    const userTurn = harness.tracer.startSpan("user_turn", {
      attributes: {
        "lk.user_transcript": "book a table",
        "lk.transcript_confidence": 0.98,
        "lk.end_of_turn_delay": 12,
      },
    });
    userTurn.end();

    const tts = harness.tracer.startSpan("tts_request", {
      attributes: {
        "lk.tts.label": "fake-tts",
        "gen_ai.provider.name": "livekit",
        "gen_ai.request.model": "voice-demo",
        "lk.tts_metrics": JSON.stringify({ inputTokens: 4, outputTokens: 6 }),
      },
    });
    tts.end();

    const exportedTool = findSpan(harness.exporter, "function_tool");
    assert.deepEqual(
      JSON.parse(exportedTool.attributes["traceloop.entity.input"]),
      {
        name: "get_weather",
        arguments: { city: "Tokyo" },
      },
    );
    assert.deepEqual(
      JSON.parse(exportedTool.attributes["traceloop.entity.output"]),
      {
        output: { forecast: "sunny" },
        is_error: false,
      },
    );
    assert.equal(exportedTool.attributes.tool_calls, undefined);
    assertNoRawLiveKitAttributes(exportedTool);

    const exportedUserTurn = findSpan(harness.exporter, "user_turn");
    assert.deepEqual(
      JSON.parse(exportedUserTurn.attributes["traceloop.entity.output"]),
      {
        transcript: "book a table",
        confidence: 0.98,
        end_of_turn_delay: 12,
      },
    );
    assertNoRawLiveKitAttributes(exportedUserTurn);

    const exportedTTS = findSpan(harness.exporter, "tts_request");
    assert.equal(exportedTTS.attributes["gen_ai.system"], "livekit");
    assert.equal(exportedTTS.attributes["gen_ai.usage.input_tokens"], 4);
    assert.equal(exportedTTS.attributes["gen_ai.usage.output_tokens"], 6);
    assert.equal(exportedTTS.attributes["traceloop.entity.output"], undefined);
    assertNoRawLiveKitAttributes(exportedTTS);
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});

test("activity lifecycle spans remain uncorrelated noise", async () => {
  const harness = createOTel2Harness("legacy");
  const instrumentor = createInstrumentor();
  await instrumentor.activate();

  try {
    for (const name of [
      "start_agent_activity",
      "on_enter",
      "on_exit",
      "drain_agent_activity",
    ]) {
      const span = harness.tracer.startSpan(name, {
        attributes: { "lk.agent_label": "weather_agent" },
      });
      span.end();
    }
    assert.equal(harness.exporter.getFinishedSpans().length, 0);
  } finally {
    instrumentor.deactivate();
    await harness.provider.shutdown();
  }
});
