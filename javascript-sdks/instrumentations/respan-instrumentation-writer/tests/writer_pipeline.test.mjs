import assert from "node:assert/strict";
import test from "node:test";
import { context, trace, ROOT_CONTEXT } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  AlwaysOffSampler,
  SimpleSpanProcessor,
  InMemorySpanExporter,
  BatchSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { WriterInstrumentor } from "../dist/index.js";
const sdk = await import(process.env.WRITER_TEST_PACKAGE || "writer-sdk");
const originalProvider = trace.getTracerProvider;
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
test.after(() => {
  trace.getTracerProvider = originalProvider;
  context.disable();
});
const request = {
  model: "pipeline-model",
  messages: [{ role: "user", content: "controlled private input" }],
};
function client() {
  return new sdk.default({
    apiKey: "fixture",
    maxRetries: 0,
    fetch: async () =>
      Response.json({
        id: "pipeline-native-id",
        model: "pipeline-response",
        choices: [
          {
            index: 0,
            finish_reason: "stop",
            message: { role: "assistant", content: "native answer" },
          },
        ],
      }),
  });
}
function processor(onEnd) {
  return {
    onStart() {},
    onEnd,
    forceFlush: async () => {},
    shutdown: async () => {},
  };
}

test("actual AlwaysOffSampler creates no Writer exports and preserves native result", async () => {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    sampler: new AlwaysOffSampler(),
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  trace.getTracerProvider = () => provider;
  const i = new WriterInstrumentor({ sdkModule: sdk });
  await i.activate();
  try {
    assert.equal(
      (await client().chat.chat(request)).choices[0].message.content,
      "native answer",
    );
    await provider.forceFlush();
    assert.equal(exporter.getFinishedSpans().length, 0);
  } finally {
    i.deactivate();
    await provider.shutdown();
  }
});

test("actual processors cannot redefine denied readable buffers before a real exporter", async () => {
  const exporter = new InMemorySpanExporter();
  let attempts = 0;
  const provider = new BasicTracerProvider({
    spanProcessors: [
      processor((span) => {
        for (const [key, value] of [
          ["attributes", { "traceloop.entity.output": "late-private-output" }],
          ["events", [{ name: "late-private-event" }]],
          ["status", { code: 2, message: "late-private-status" }],
        ]) {
          attempts++;
          assert.throws(
            () =>
              Object.defineProperty(span, key, { configurable: true, value }),
            TypeError,
          );
        }
        span.attributes = {
          ...span.attributes,
          "traceloop.entity.output": "assignment-private-output",
          "respan.metadata": "assignment-private-metadata",
        };
        span.events = [{ name: "assignment-private-event" }];
        span.status = { code: 2, message: "assignment-private-status" };
      }),
      new SimpleSpanProcessor(exporter),
    ],
  });
  trace.getTracerProvider = () => provider;
  const i = new WriterInstrumentor({ sdkModule: sdk, traceContent: false });
  await i.activate();
  try {
    assert.equal(
      (await client().chat.chat(request)).choices[0].message.content,
      "native answer",
    );
    await provider.forceFlush();
    assert.equal(attempts, 3);
    const [span] = exporter.getFinishedSpans();
    assert.ok(span);
    assert.equal(span.attributes["traceloop.entity.output"], undefined);
    assert.equal(span.attributes["respan.metadata"], undefined);
    assert.deepEqual(span.events, []);
    assert.equal(span.status.message, undefined);
  } finally {
    i.deactivate();
    await provider.shutdown();
  }
});

test("actual BatchSpanProcessor queue retains privacy after deactivation and exporter ambient suppression", async () => {
  const exporter = new InMemorySpanExporter(),
    queued = [];
  const batch = new BatchSpanProcessor(exporter, {
    scheduledDelayMillis: 60000,
  });
  const provider = new BasicTracerProvider({
    spanProcessors: [processor((s) => queued.push(s)), batch],
  });
  trace.getTracerProvider = () => provider;
  const denied = new WriterInstrumentor({
    sdkModule: sdk,
    traceContent: false,
  });
  await denied.activate();
  await client().chat.chat(request);
  denied.deactivate();
  const allowed = new WriterInstrumentor({ sdkModule: sdk });
  await allowed.activate();
  await client().chat.chat(request);
  allowed.deactivate();
  queued[0].attributes = {
    ...queued[0].attributes,
    "gen_ai.completion.0.content": "queued-private-output",
  };
  queued[0].events = [{ name: "queued-private-event" }];
  queued[0].status = { code: 1, message: "queued-private-status" };
  try {
    await context.with(suppressTracing(ROOT_CONTEXT), () =>
      provider.forceFlush(),
    );
    const spans = exporter.getFinishedSpans();
    assert.equal(spans.length, 2);
    assert.equal(spans[0].attributes["gen_ai.completion.0.content"], undefined);
    assert.equal(spans[0].status.message, undefined);
    assert.deepEqual(spans[0].events, []);
    assert.equal(
      spans[1].attributes["gen_ai.completion.0.content"],
      "native answer",
    );
  } finally {
    await provider.shutdown();
  }
});

test("default import activation is coalesced and synchronous deactivation cancels pending activation", async () => {
  const i = new WriterInstrumentor();
  const pending = i.activate();
  assert.equal(i.activate(), pending);
  i.deactivate();
  await pending;
  assert.equal(i.isActive(), false);
  await i.activate();
  assert.equal(i.isActive(), true);
  i.deactivate();
  assert.equal(i.isActive(), false);
});

test("canonical request metadata merges with actual native span propagated metadata", async () => {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [
      {
        ...processor(() => {}),
        onStart(span) {
          span.setAttributes({
            "respan.threads.thread_identifier": "native-thread",
            "respan.trace.trace_group_identifier": "native-group",
            "respan.customer_params.customer_identifier":
              "synthetic-correlation",
            "sampler.attribute": false,
          });
          span.setAttribute(
            "respan.metadata",
            JSON.stringify({
              run_id: "native-metadata-fixture",
              nested: { flag: false, zero: 0, empty: "", nil: null },
            }),
          );
        },
      },
      new SimpleSpanProcessor(exporter),
    ],
  });
  trace.getTracerProvider = () => provider;
  const i = new WriterInstrumentor({ sdkModule: sdk });
  await i.activate();
  try {
    await client().chat.chat({
      ...request,
      tool_choice: { value: "auto" },
      response_format: {
        type: "json_schema",
        json_schema: { name: "fixture", schema: { type: "object" } },
      },
    });
    await provider.forceFlush();
    const span = exporter.getFinishedSpans()[0];
    assert.deepEqual(JSON.parse(span.attributes["respan.metadata"]), {
      run_id: "native-metadata-fixture",
      nested: { flag: false, zero: 0, empty: "", nil: null },
      tool_choice: { value: "auto" },
      response_format: {
        type: "json_schema",
        json_schema: { name: "fixture", schema: { type: "object" } },
      },
    });
    assert.equal(
      span.attributes["respan.threads.thread_identifier"],
      "native-thread",
    );
    assert.equal(
      span.attributes["respan.trace.trace_group_identifier"],
      "native-group",
    );
    assert.equal(
      span.attributes["respan.customer_params.customer_identifier"],
      "synthetic-correlation",
    );
    assert.equal(span.attributes["sampler.attribute"], false);
    assert.equal(span.attributes["respan.metadata.tool_choice"], undefined);
    assert.equal(span.attributes["respan.metadata.response_format"], undefined);
  } finally {
    i.deactivate();
    await provider.shutdown();
  }
});
