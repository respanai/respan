import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { context, trace, createContextKey } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { CohereInstrumentor } from "../dist/index.js";
import { fixture, vector } from "./_native_fixture.mjs";
const require = createRequire(import.meta.url);
const sdk = require(process.env.COHERE_NATIVE_MODULE || "cohere-ai");
const current = !!sdk.CohereClientV2;
let env,
  spans = [],
  created = [],
  owners = [],
  decision = 2,
  onStart,
  onEnd,
  samples = [];
const provider = new NodeTracerProvider({
  spanLimits: { attributeCountLimit: 4096 },
  sampler: {
    shouldSample(ctx, id, name, kind, attrs) {
      samples.push({ name, attrs });
      return { decision };
    },
    toString() {
      return "controlled";
    },
  },
  spanProcessors: [
    {
      onStart(span, ctx) {
        created.push(span);
        onStart?.(span, ctx);
      },
      onEnd(span) {
        onEnd?.(span);
        spans.push(span);
      },
      forceFlush: async () => {},
      shutdown: async () => {},
    },
  ],
});
provider.register();
const tick = () => new Promise((r) => setImmediate(r));
const client = () =>
  new sdk.CohereClient({
    token: "fixture",
    environment: env.url,
    maxRetries: 0,
  });
const v2client = () =>
  new sdk.CohereClientV2({
    token: "fixture",
    environment: env.url,
    maxRetries: 0,
  });
const request = (model = "command") => ({
  model,
  message: "fixture-input",
  temperature: 0,
  p: 0,
  k: 0,
});
const request2 = (model = "command") => ({
  model,
  messages: [{ role: "user", content: "fixture-input" }],
  temperature: 0,
});
async function plugin(options = {}) {
  const p = new CohereInstrumentor({ sdkModule: sdk, ...options });
  owners.push(p);
  await p.activate();
  assert.equal(p.isActive(), true);
  return p;
}
const attrs = () => spans.at(-1).attributes;
const contentKeys = (a) =>
  Object.keys(a).filter(
    (k) =>
      k.startsWith("gen_ai.prompt.") ||
      k.startsWith("gen_ai.completion.") ||
      k === "traceloop.entity.input" ||
      k === "traceloop.entity.output" ||
      k === "llm.request.functions" ||
      k.startsWith("respan.metadata") ||
      k.startsWith("exception.") ||
      k.startsWith("error."),
  );
test.before(async () => {
  env = await fixture();
});
test.beforeEach(() => {
  spans = [];
  created = [];
  samples = [];
  decision = 2;
  onStart = onEnd = undefined;
});
test.afterEach(() => {
  for (const p of owners) p.deactivate();
  owners = [];
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
test.after(async () => {
  await env.close();
  await provider.shutdown();
});
test("native v1 chat uses real tracer spans and preserves zero scalars", async () => {
  await plugin();
  await client().chat(request());
  await tick();
  assert.equal(spans.length, 1);
  assert.equal(created[0], spans[0]);
  assert.equal(samples[0].attrs["gen_ai.request.model"], "command");
  assert.equal(samples[0].attrs["traceloop.entity.input"], undefined);
  assert.equal(attrs()["gen_ai.prompt.0.content"], "fixture-input");
  assert.equal(attrs()["gen_ai.completion.0.content"], "fixture-answer");
  assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
  assert.equal(attrs()["llm.usage.total_tokens"], undefined);
  assert.equal(attrs()["status_code"], undefined);
  assert.equal(attrs()["gen_ai.request.temperature"], 0);
});
test(
  "native v2 messages, multimodal content and 80 tools remain complete",
  { skip: !current },
  async () => {
    await plugin();
    const c = v2client();
    const messages = Array.from({ length: 80 }, (_, i) => ({
      role: "user",
      content:
        i === 79
          ? [
              { type: "text", text: "" },
              {
                type: "image_url",
                imageUrl: { url: "https://example.invalid/image.png" },
              },
            ]
          : `message-${i}`,
    }));
    const tools = Array.from({ length: 80 }, (_, i) => ({
      type: "function",
      function: {
        name: `tool_${i}`,
        description: "",
        parameters: {
          type: "object",
          properties: {
            flag: { type: "boolean", default: false },
            value: { type: "number", default: 0 },
          },
        },
      },
    }));
    await c.chat({ model: "command", messages, tools });
    await tick();
    assert.deepEqual(JSON.parse(attrs()["traceloop.entity.input"]), messages);
    assert.equal(JSON.parse(attrs()["llm.request.functions"]).length, 80);
    assert.equal(
      attrs()["gen_ai.prompt.79.content"],
      JSON.stringify(messages[79].content),
    );
    const calls = JSON.parse(attrs()["gen_ai.completion.0.tool_calls"]);
    assert.equal(calls[0].id, "call_1");
    assert.equal(calls[0].function.arguments, '{"value":0,"flag":false}');
  },
);
test(
  "native v2 streaming preserves HttpResponsePromise raw helper, stream, controller and chunks",
  { skip: !current },
  async () => {
    await plugin();
    const c = v2client(),
      pending = c.chatStream(request2());
    assert.equal(typeof pending.withRawResponse, "function");
    const raw = await pending.withRawResponse(),
      stream = await pending;
    assert.equal(raw.data, stream);
    const controller = stream.controller;
    const values = [];
    for await (const item of stream) values.push(item);
    await tick();
    assert.equal(stream.controller, controller);
    assert.equal(values[2].delta.message.content.text, "fixture-");
    assert.equal(spans.length, 1);
    assert.equal(attrs()["gen_ai.completion.0.content"], "fixture-answer");
    assert.deepEqual(JSON.parse(attrs()["gen_ai.completion.0.tool_calls"]), [
      {
        id: "call_1",
        type: "function",
        function: { name: "lookup", arguments: '{"value":0,"flag":false}' },
      },
    ]);
    assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
  },
);
test("native v1 chat stream completes once and preserves parsed final response", async () => {
  await plugin();
  const stream = await client().chatStream(request());
  const items = [];
  for await (const item of stream) items.push(item);
  await tick();
  assert.equal(items[0].text, "fixture-answer");
  assert.equal(spans.length, 1);
  assert.equal(attrs()["gen_ai.completion.0.content"], "fixture-answer");
});
test("native embeddings preserve all five returned types and 5001 vector entries", async () => {
  await plugin();
  const result = await client().embed({
    model: "embed",
    texts: ["", "value"],
    inputType: "classification",
    embeddingTypes: ["float", "int8", "uint8", "binary", "ubinary"],
  });
  await tick();
  assert.deepEqual(
    JSON.parse(attrs()["traceloop.entity.output"]),
    result.embeddings,
  );
  assert.equal(
    JSON.parse(attrs()["traceloop.entity.output"]).float[0].length,
    5001,
  );
  assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
});
test("native embedding floats remain arrays", async () => {
  await plugin();
  const result = await client().embed({ model: "embed", texts: ["value"] });
  await tick();
  assert.deepEqual(
    JSON.parse(attrs()["traceloop.entity.output"]),
    result.embeddings,
  );
  assert.equal(result.embeddings[0].length, vector.length);
});
test(
  "native v2 embeddings and rerank preserve full documents and do not invent tokens",
  { skip: !current },
  async () => {
    await plugin();
    await v2client().embed({
      model: "embed",
      texts: ["value"],
      inputType: "classification",
      embeddingTypes: ["float"],
    });
    await v2client().rerank({
      model: "rerank",
      query: "",
      documents: ["one", ""],
      topN: 0,
    });
    await tick();
    assert.equal(spans.length, 2);
    assert.equal(attrs()["llm.usage.total_tokens"], undefined);
    assert.equal(
      Object.keys(attrs()).some(
        (key) => key.startsWith("gen_ai.") || key.startsWith("llm."),
      ),
      false,
    );
    assert.equal(
      JSON.parse(attrs()["traceloop.entity.output"])[0].document.flag,
      false,
    );
    assert.deepEqual(JSON.parse(attrs()["traceloop.entity.input"]).documents, [
      "one",
      "",
    ]);
  },
);
test("native v1 rerank search units are metadata rather than token counts", async () => {
  await plugin();
  await client().rerank({
    model: "rerank",
    query: "query",
    documents: [{ text: "", extra: "kept" }],
    returnDocuments: true,
    topN: 0,
  });
  await tick();
  assert.equal(attrs()["llm.usage.total_tokens"], undefined);
  assert.equal(
    JSON.parse(attrs()["respan.metadata"]).cohere_response.meta.billedUnits
      .searchUnits,
    1,
  );
});
test("native legacy generate preserves every generation, including empty text", async () => {
  await plugin();
  await client().generate({ model: "command", prompt: "", numGenerations: 2 });
  await tick();
  assert.equal(JSON.parse(attrs()["traceloop.entity.output"]).length, 2);
  assert.equal(attrs()["gen_ai.completion.1.content"], "");
  assert.equal(attrs()["gen_ai.usage.output_tokens"], 0);
});
test("native legacy generate stream ends once", async () => {
  await plugin();
  for await (const item of await client().generateStream({
    model: "command",
    prompt: "hello",
  })) {
  }
  await tick();
  assert.equal(spans.length, 1);
  assert.equal(JSON.parse(attrs()["traceloop.entity.output"]).length, 2);
});
test("missing model and usage remain absent", async () => {
  await plugin();
  await client().chat({ message: "fixture-input" });
  await tick();
  assert.equal(attrs()["gen_ai.request.model"], undefined);
  await client().chat(request("no-usage"));
  await tick();
  assert.equal(attrs()["gen_ai.usage.input_tokens"], undefined);
  assert.equal(attrs()["llm.usage.total_tokens"], undefined);
});
test("native errors preserve identity and actual HTTP status with no completion", async () => {
  await plugin();
  let caught;
  try {
    await client().chat(request("fail"));
  } catch (e) {
    caught = e;
  }
  await tick();
  assert.ok(caught instanceof sdk.CohereError);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 2);
  assert.equal(attrs()["http.response.status_code"], 429);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
test("traceContent false suppresses input, output, events, metadata and error messages", async () => {
  await plugin({ traceContent: false });
  await client().chat(request());
  try {
    await client().chat(request("fail"));
  } catch {}
  await tick();
  for (const span of spans) {
    assert.deepEqual(contentKeys(span.attributes), []);
    assert.equal(span.events.length, 0);
    assert.equal(span.status.message, undefined);
  }
});
test("independent recordInputs and recordOutputs lanes", async () => {
  const p = await plugin({ recordInputs: false });
  await client().chat(request());
  await tick();
  assert.equal(attrs()["traceloop.entity.input"], undefined);
  assert.ok(attrs()["traceloop.entity.output"]);
  p.deactivate();
  await plugin({ recordOutputs: false });
  await client().chat(request());
  await tick();
  assert.ok(attrs()["traceloop.entity.input"]);
  assert.equal(attrs()["traceloop.entity.output"], undefined);
});
test("environment and canonical context denials cannot be restored by true", async () => {
  await plugin();
  process.env.TRACELOOP_TRACE_CONTENT = "false";
  await context.with(
    context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
    () => client().chat(request()),
  );
  await tick();
  assert.deepEqual(contentKeys(attrs()), []);
  delete process.env.TRACELOOP_TRACE_CONTENT;
  await context.with(
    context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
    () => client().chat(request()),
  );
  await tick();
  assert.deepEqual(contentKeys(attrs()), []);
});
test("DROP and RECORD_ONLY add no payload reads beyond the bare native serializer", async () => {
  let reads = 0;
  const req = request();
  Object.defineProperty(req, "telemetryOnly", {
    enumerable: true,
    get() {
      reads++;
      return "native-value";
    },
  });
  await client().chat(req);
  const bareReads = reads;
  await plugin();
  for (const sampled of [0, 1]) {
    decision = sampled;
    reads = 0;
    await client().chat(req);
    await tick();
    assert.equal(reads, bareReads);
  }
  assert.equal(samples.length, 2);
  for (const span of spans) assert.deepEqual(contentKeys(span.attributes), []);
});
test("OTel and historical language-model suppression bypass tracing", async () => {
  await plugin();
  await context.with(suppressTracing(context.active()), () =>
    client().chat(request()),
  );
  await context.with(
    context
      .active()
      .setValue(
        createContextKey("suppress_language_model_instrumentation"),
        true,
      ),
    () => client().chat(request()),
  );
  await tick();
  assert.equal(spans.length, 0);
  assert.equal(samples.length, 0);
});
test("late ancestor veto remains an immutable ceiling for following siblings", async () => {
  await plugin();
  const parent = trace.getTracer("fixture").startSpan("parent");
  await context.with(trace.setSpan(context.active(), parent), async () => {
    parent.setAttribute("allow_trace_content", false);
    await client().chat(request());
    parent.setAttribute("allow_trace_content", true);
    await client().chat(request());
  });
  await tick();
  for (const span of spans) assert.deepEqual(contentKeys(span.attributes), []);
  parent.end();
});
test("late parent veto scrubs the actual span and blocks unknown fields before traversal", async () => {
  await plugin();
  const parent = trace.getTracer("fixture").startSpan("parent");
  let reads = 0;
  onEnd = (span) => {
    if (span.name === "parent") return;
    const value = {};
    Object.defineProperty(value, "nested", {
      enumerable: true,
      get() {
        reads++;
        throw Error("must not read");
      },
    });
    span.attributes = {
      ...span.attributes,
      "gen_ai.usage.secret": value,
      "customer.secret": value,
    };
    span.events = [{ name: "secret" }];
    span.status = {
      code: { payload: "private-status-code" },
      message: "secret",
    };
  };
  await context.with(trace.setSpan(context.active(), parent), async () => {
    const pending = client().chat(request("delayed"));
    parent.setAttribute("allow_trace_content", false);
    await pending;
  });
  await tick();
  assert.equal(reads, 0);
  assert.equal(created.at(-1), spans.at(-1));
  assert.deepEqual(contentKeys(attrs()), []);
  assert.equal(attrs()["gen_ai.usage.secret"], undefined);
  assert.equal(attrs()["customer.secret"], undefined);
  assert.equal(spans.at(-1).events.length, 0);
  assert.equal(spans.at(-1).status.message, undefined);
  assert.equal(spans.at(-1).status.code, 0);
  assert.equal(
    Object.getOwnPropertyDescriptor(spans.at(-1), "attributes").configurable,
    false,
  );
  parent.end();
});
test("exporter ambient suppression does not revoke admitted payloads", async () => {
  await plugin();
  onEnd = (span) =>
    context.with(suppressTracing(context.active()), () =>
      assert.ok(span.attributes["traceloop.entity.output"]),
    );
  await client().chat(request());
  await tick();
  assert.ok(attrs()["traceloop.entity.output"]);
});
test("stream early return and controller abort drain exactly once", async () => {
  await plugin();
  const c = current ? v2client() : client();
  const stream = await c.chatStream(current ? request2() : request());
  const iterator = stream[Symbol.asyncIterator]();
  await iterator.next();
  await iterator.return();
  stream.controller?.abort();
  await tick();
  assert.equal(spans.length, 1);
});
test("deactivate drains in-flight calls and reactivation does not duplicate", async () => {
  const p = await plugin();
  const pending = client().chat(request("delayed"));
  p.deactivate();
  await pending;
  await tick();
  assert.equal(spans.length, 1);
  await Promise.all([p.activate(), p.activate()]);
  await client().chat(request());
  await tick();
  assert.equal(spans.length, 2);
});
test("multiple owners keep patches alive and apply the strictest policy", async () => {
  const a = await plugin(),
    b = await plugin({ traceContent: false });
  a.deactivate();
  await client().chat(request());
  await tick();
  assert.equal(spans.length, 1);
  assert.deepEqual(contentKeys(attrs()), []);
  b.deactivate();
  await client().chat(request());
  await tick();
  assert.equal(spans.length, 1);
});
test("foreign patches are preserved on deactivate", async () => {
  const proto = sdk.CohereClient.prototype,
    original = proto.chat;
  const p = await plugin();
  const patched = proto.chat;
  function foreign(...args) {
    return patched.apply(this, args);
  }
  proto.chat = foreign;
  p.deactivate();
  assert.equal(proto.chat, foreign);
  proto.chat = original;
});

test(
  "native v2 document parse emits one common-only tool span",
  { skip: !current },
  async () => {
    await plugin();
    const request = {
      model: "parse-v5.0",
      document: {
        type: "image_url",
        imageUrl: "https://example.invalid/document.png",
      },
      outputFormat: "markdown",
    };
    const pending = v2client().parse(request);
    assert.equal(typeof pending.withRawResponse, "function");
    const raw = await pending.withRawResponse(),
      result = await pending;
    await tick();
    assert.equal(raw.data, result);
    assert.equal(spans.length, 1);
    assert.equal(attrs()["respan.entity.log_type"], "tool");
    assert.deepEqual(JSON.parse(attrs()["traceloop.entity.input"]), {
      name: "cohere.parse",
      arguments: request,
    });
    assert.deepEqual(JSON.parse(attrs()["traceloop.entity.output"]), result);
    assert.equal(
      Object.keys(attrs()).some(
        (key) => key.startsWith("gen_ai.") || key.startsWith("llm."),
      ),
      false,
    );
  },
);

test(
  "native chat raw response status is observed without replacing the HttpResponsePromise",
  { skip: !current },
  async () => {
    await plugin();
    const pending = v2client().chat(request2());
    const raw = await pending.withRawResponse();
    const result = await pending;
    await tick();
    assert.equal(raw.rawResponse.status, 200);
    assert.equal(attrs()["http.response.status_code"], raw.rawResponse.status);
    assert.deepEqual(raw.data, result);
  },
);
test("the actual child sampler can admit a child of a non-sampled parent", async () => {
  await plugin();
  const parent = trace.wrapSpanContext({
    traceId: "11111111111111111111111111111111",
    spanId: "2222222222222222",
    traceFlags: 0,
  });
  await context.with(trace.setSpan(context.active(), parent), () =>
    client().chat(request()),
  );
  await tick();
  assert.equal(spans.length, 1);
  assert.ok(attrs()["traceloop.entity.output"]);
});

test("actual native provider default128 preserves canonical payloads before optional indexed prompts", () => {
  const output = execFileSync(
    process.execPath,
    [new URL("./_default_budget_worker.mjs", import.meta.url).pathname],
    { env: process.env, encoding: "utf8" },
  );
  assert.equal(JSON.parse(output).canonicalPayloads, true);
});
