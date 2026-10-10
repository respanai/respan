import { execFileSync } from "node:child_process";
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
const sdkVersion = createRequire(import.meta.url)(
  "@aws-sdk/client-bedrock-runtime/package.json",
).version;
import { ATTR_HTTP_RESPONSE_STATUS_CODE } from "@opentelemetry/semantic-conventions";
import { context, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import * as SDK from "@aws-sdk/client-bedrock-runtime";
import { AWSBedrockInstrumentor, buildBedrockAttrs } from "../dist/index.js";
import { handler, toolEvents, invokeEvents } from "./native-fixture.mjs";
let decision = SamplingDecision.RECORD_AND_SAMPLED;
let sampled = [];
let spans = [];
let onStart;
let onEnd;
const provider = new BasicTracerProvider({
  spanLimits: { attributeCountLimit: 10000 },
  sampler: {
    shouldSample(_ctx, _traceId, name, kind, attrs) {
      sampled.push({ name, kind, attrs });
      return { decision };
    },
    toString() {
      return "native-audit";
    },
  },
  spanProcessors: [
    {
      onStart(s) {
        onStart?.(s);
      },
      onEnd(s) {
        onEnd?.(s);
        spans.push(s);
      },
      forceFlush: async () => {},
      shutdown: async () => {},
    },
  ],
});
trace.setGlobalTracerProvider(provider);
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
const insts = [];
const clients = [];
function install(options = {}) {
  const inst = new AWSBedrockInstrumentor(options);
  insts.push(inst);
  return inst;
}
function client(options = {}) {
  const transport = handler(options);
  const client = new SDK.BedrockRuntimeClient({
    region: "us-east-1",
    credentials: { accessKeyId: "fixture", secretAccessKey: "fixture" },
    maxAttempts: 1,
    requestHandler: transport,
  });
  clients.push(client);
  return { client, transport };
}
const input = () => ({
  modelId: "native-model",
  messages: [{ role: "user", content: [{ text: "native prompt" }] }],
});
const converse = () => new SDK.ConverseCommand(input());
function attrs() {
  return spans.at(-1).attributes;
}
function noPayload(a) {
  for (const key of Object.keys(a))
    assert.ok(
      !key.startsWith("gen_ai.prompt.") &&
        !key.startsWith("gen_ai.completion.") &&
        ![
          "traceloop.entity.input",
          "traceloop.entity.output",
          "llm.request.functions",
          "respan.metadata",
          "error.message",
        ].includes(key),
      key,
    );
}
test.beforeEach(() => {
  decision = SamplingDecision.RECORD_AND_SAMPLED;
  sampled = [];
  spans = [];
  onStart = onEnd = undefined;
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
test.afterEach(() => {
  for (const i of insts.splice(0)) i.deactivate();
  for (const c of clients.splice(0)) c.destroy();
});
test.after(async () => {
  await provider.shutdown();
  context.disable();
  trace.disable();
});
test("native Converse captures full messages, tool definitions, response, cache usage and zero values", async () => {
  const i = install();
  await i.activate();
  const { client: c, transport } = client();
  const req = input();
  req.messages = Array.from({ length: 76 }, (_, n) => ({
    role: "user",
    content: [{ text: `message-${n}` }],
  }));
  req.toolConfig = {
    tools: Array.from({ length: 76 }, (_, n) => ({
      toolSpec: {
        name: `tool_${n}`,
        description: "",
        inputSchema: {
          json: {
            type: "object",
            properties: {
              enabled: { const: false },
              count: { const: 0 },
              empty: { const: "" },
              nil: { const: null },
            },
          },
        },
      },
    })),
  };
  const r = await c.send(new SDK.ConverseCommand(req));
  assert.equal(r.output.message.content[0].text, "native response");
  assert.equal(transport.calls.length, 1);
  assert.equal(spans.length, 1);
  assert.equal(sampled.length, 1);
  assert.equal(sampled[0].attrs["gen_ai.request.model"], "native-model");
  assert.equal(sampled[0].attrs["traceloop.entity.input"], undefined);
  const a = attrs();
  assert.equal(JSON.parse(a["traceloop.entity.input"]).length, 76);
  assert.equal(JSON.parse(a["llm.request.functions"]).length, 76);
  assert.equal(a["gen_ai.prompt.75.content"], "message-75");
  assert.equal(a["gen_ai.usage.input_tokens"], 0);
  assert.equal(a["gen_ai.usage.output_tokens"], 0);
  assert.equal(a["llm.usage.total_tokens"], 0);
  assert.equal(a["gen_ai.usage.cache_read.input_tokens"], 0);
  assert.equal(
    JSON.parse(a["traceloop.entity.output"]).output.message.content[1].toolUse
      .input.enabled,
    false,
  );
  assert.equal(a["respan.span.tools"], undefined);
});
test("native Invoke body and response preserve bytes, errors and provider-only token totals", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client({ mode: "invoke" });
  const r = await c.send(
    new SDK.InvokeModelCommand({
      modelId: "native-model",
      contentType: "application/json",
      body: JSON.stringify({
        messages: [{ role: "user", content: "invoke" }],
        tools: [{ name: "lookup", input_schema: { type: "object" } }],
      }),
    }),
  );
  assert.ok(r.body instanceof Uint8Array);
  assert.equal(attrs()["gen_ai.prompt.0.content"], "invoke");
  assert.equal(attrs()["gen_ai.completion.0.content"], "native invoke");
  assert.equal(attrs()["llm.usage.total_tokens"], undefined);
});
test("native ConverseStream preserves response, stream and chunk identity and assembles fragmented tool arguments", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client();
  let native;
  c.middlewareStack.add(
    (next) => async (args) => {
      const r = await next(args);
      native = r.output;
      return r;
    },
    { step: "initialize", name: "nativeCapture" },
  );
  const r = await c.send(new SDK.ConverseStreamCommand(input()));
  assert.equal(r, native);
  const nativeStream = native.stream;
  assert.equal(r.stream, nativeStream);
  assert.equal(spans.length, 0);
  const seen = [];
  for await (const e of r.stream) seen.push(e);
  assert.equal(seen.length, toolEvents.length);
  assert.equal(spans.length, 1);
  assert.equal(attrs()["gen_ai.completion.0.content"], "native stream");
  assert.equal(
    JSON.parse(attrs()["gen_ai.completion.0.tool_calls"])[0].function.arguments,
    '{"city":"Paris","enabled":false,"count":0,"empty":"","nil":null}',
  );
});
test("native Invoke eventstream captures Anthropic fragmented tool arguments", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client({ mode: "invoke", events: invokeEvents });
  const r = await c.send(
    new SDK.InvokeModelWithResponseStreamCommand({
      modelId: "native-model",
      body: '{"messages":[]}',
    }),
  );
  for await (const e of r.body) assert.ok(e.chunk.bytes instanceof Uint8Array);
  assert.equal(attrs()["gen_ai.completion.0.content"], "invoke stream");
  assert.equal(
    JSON.parse(attrs()["gen_ai.completion.0.tool_calls"])[0].function.arguments,
    '{"city":"Paris"}',
  );
  assert.equal(attrs()["llm.usage.total_tokens"], undefined);
});
test("native callback overload preserves void return, callback this and complete response", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client();
  await new Promise((resolve, reject) => {
    const ret = c.send(converse(), function (error, r) {
      try {
        assert.equal(error, null);
        assert.equal(r.output.message.content[0].text, "native response");
        assert.equal(spans.length, 1);
        resolve();
      } catch (e) {
        reject(e);
      }
    });
    assert.equal(ret, undefined);
  });
  assert.equal(attrs()["gen_ai.completion.0.content"], "native response");
});
test("native callback with request options emits controlled service errors once", async () => {
  const i = install();
  await i.activate();
  const { client: c, transport } = client({
    status: 429,
    response: {
      __type: "ThrottlingException",
      message: "private failure body",
    },
  });
  await new Promise((resolve, reject) => {
    assert.equal(
      c.send(converse(), {}, (error, r) => {
        try {
          assert.equal(error.name, "ThrottlingException");
          assert.equal(r, undefined);
          resolve();
        } catch (e) {
          reject(e);
        }
      }),
      undefined,
    );
  });
  assert.equal(transport.calls.length, 1);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 2);
  assert.equal(attrs()[ATTR_HTTP_RESPONSE_STATUS_CODE], 429);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
test("native Promise result identity and rejection identity remain unchanged", async () => {
  const original = SDK.BedrockRuntimeClient.prototype.send;
  let nativePromise;
  SDK.BedrockRuntimeClient.prototype.send = function (...args) {
    return (nativePromise = original.apply(this, args));
  };
  const foreign = SDK.BedrockRuntimeClient.prototype.send;
  try {
    const i = install();
    await i.activate();
    const { client: c } = client();
    const result = c.send(converse());
    assert.equal(result, nativePromise);
    await result;
    i.deactivate();
    assert.equal(SDK.BedrockRuntimeClient.prototype.send, foreign);
  } finally {
    SDK.BedrockRuntimeClient.prototype.send = original;
  }
});
for (const [label, d] of [
  ["DROP", SamplingDecision.NOT_RECORD],
  ["RECORD_ONLY", SamplingDecision.RECORD],
])
  test(`${label} reaches native sampler before any telemetry payload traversal`, async () => {
    decision = d;
    onEnd = (span) => {
      span.setAttribute("gen_ai.usage.untrusted", "private");
      span.setAttribute("gen_ai.prompt.0.content", "private");
      span.events = [{ name: "private" }];
      span.status = { code: 2, message: "private" };
    };
    const i = install();
    await i.activate();
    const { client: c } = client();
    let reads = 0;
    const req = input();
    Object.defineProperty(req, "unknown", {
      enumerable: true,
      get() {
        reads++;
        return "native getter value";
      },
    });
    i.deactivate();
    await c.send(new SDK.ConverseCommand(req));
    const nativeReads = reads;
    reads = 0;
    await i.activate();
    await c.send(new SDK.ConverseCommand(req));
    assert.equal(reads, nativeReads);
    assert.equal(sampled.length, 1);
    assert.equal(sampled[0].attrs["traceloop.entity.input"], undefined);
    if (d === SamplingDecision.RECORD) noPayload(attrs());
    else assert.equal(spans.length, 0);
  });
for (const option of [
  { traceContent: false },
  { recordInputs: false, recordOutputs: false },
])
  test(`native denied options ${JSON.stringify(option)} keep actual span private through late processors`, async () => {
    const i = install(option);
    await i.activate();
    let reads = 0;
    onStart = (s) => {
      s.setAttribute("unknown", { secret: "bad" });
      s.setAttribute("error.message", "secret");
    };
    onEnd = (s) => {
      const replacement = {
        allow_trace_content: true,
        "gen_ai.usage.untrusted": "secret",
        "gen_ai.usage.input_tokens": 0,
      };
      Object.defineProperty(replacement, "unknown", {
        enumerable: true,
        get() {
          reads++;
          throw Error("getter");
        },
      });
      s.attributes = replacement;
      s.events = [{ name: "secret-event" }];
      s.status = { code: 2, message: "secret-status" };
    };
    const { client: c } = client();
    await c.send(converse());
    assert.equal(reads, 0);
    noPayload(attrs());
    assert.equal(attrs()["gen_ai.usage.untrusted"], undefined);
    assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
    assert.equal(spans[0].events.length, 0);
    assert.deepEqual(spans[0].status, { code: 2 });
    assert.throws(() =>
      Object.defineProperty(spans[0], "attributes", {
        value: { secret: "bad" },
      }),
    );
  });
for (const lane of ["recordInputs", "recordOutputs"])
  test(`native independent ${lane} lane`, async () => {
    const i = install({ [lane]: false });
    await i.activate();
    const { client: c } = client();
    await c.send(converse());
    assert.equal(
      attrs()["gen_ai.prompt.0.content"],
      lane === "recordInputs" ? undefined : "native prompt",
    );
    assert.equal(
      attrs()["gen_ai.completion.0.content"],
      lane === "recordOutputs" ? undefined : "native response",
    );
  });
for (const gate of [
  "RESPAN_TRACE_CONTENT",
  "TRACELOOP_TRACE_CONTENT",
  "context",
  "parent",
])
  test(`native ${gate} false is an immutable content ceiling`, async () => {
    const i = install();
    await i.activate();
    const { client: c } = client();
    let ctx = context.active();
    let parent;
    if (gate === "context")
      ctx = ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
    else if (gate === "parent") {
      parent = trace.getTracer("native-test").startSpan("parent");
      parent.setAttribute("allow_trace_content", false);
      ctx = trace.setSpan(ctx, parent);
    } else process.env[gate] = "false";
    await context.with(ctx, () => c.send(converse()));
    noPayload(attrs());
    parent?.end();
  });
test("native guarded parent late veto removes admitted child content and cannot be restored", async () => {
  const i = install();
  await i.activate();
  let parent;
  onStart = (span) => {
    parent ??= span;
  };
  const { client: c } = client();
  const outer = await c.send(new SDK.ConverseStreamCommand(input()));
  const { client: delayed } = client({ delay: 15 });
  const promise = context.with(trace.setSpan(context.active(), parent), () =>
    delayed.send(converse()),
  );
  parent.setAttribute("allow_trace_content", false);
  parent.setAttribute("allow_trace_content", true);
  await promise;
  noPayload(attrs());
  assert.equal(spans[0].parentSpanContext.spanId, parent.spanContext().spanId);
  for await (const event of outer.stream) {
  }
  noPayload(attrs());
});
test("native canonical context suppression invokes AWS without telemetry", async () => {
  const i = install();
  await i.activate();
  const { client: c, transport } = client();
  await context.with(suppressTracing(context.active()), () =>
    c.send(converse()),
  );
  assert.equal(transport.calls.length, 1);
  assert.equal(spans.length, 0);
  assert.equal(sampled.length, 0);
});
test("native early iterator return before next and partial break end the span once", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client();
  const r = await c.send(new SDK.ConverseStreamCommand(input()));
  await r.stream[Symbol.asyncIterator]().return();
  assert.equal(spans.length, 1);
  const second = await c.send(new SDK.ConverseStreamCommand(input()));
  for await (const e of second.stream) break;
  assert.equal(spans.length, 2);
});
test("native stream transport error is rethrown and has no fabricated completion", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client({ failAfter: 2 });
  const r = await c.send(new SDK.ConverseStreamCommand(input()));
  await assert.rejects(async () => {
    for await (const e of r.stream) {
    }
  }, /native stream transport failure/);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 2);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
test("native owners, activation coalescing, pending cancellation and foreign patches are preserved", async () => {
  const original = SDK.BedrockRuntimeClient.prototype.send;
  const a = install(),
    b = install();
  const pending = a.activate();
  assert.equal(a.activate(), pending);
  a.deactivate();
  await pending;
  assert.equal(a.isActive(), false);
  await Promise.all([a.activate(), b.activate()]);
  const wrapped = SDK.BedrockRuntimeClient.prototype.send;
  assert.notEqual(wrapped, original);
  a.deactivate();
  assert.equal(SDK.BedrockRuntimeClient.prototype.send, wrapped);
  const { client: c } = client();
  await c.send(converse());
  assert.equal(spans.length, 1);
  const foreign = function (...args) {
    return wrapped.apply(this, args);
  };
  SDK.BedrockRuntimeClient.prototype.send = foreign;
  b.deactivate();
  assert.equal(SDK.BedrockRuntimeClient.prototype.send, foreign);
  SDK.BedrockRuntimeClient.prototype.send = original;
});
test("native deactivation drains admitted in-flight calls", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client({ delay: 15 });
  const pending = c.send(converse());
  i.deactivate();
  await pending;
  assert.equal(spans.length, 1);
});
test("native Invoke Titan embeddings preserve all 5001 dimensions and no invented usage", async () => {
  const i = install();
  await i.activate();
  const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
  const { client: c } = client({
    response: { embedding: vector, inputTextTokenCount: 0 },
  });
  await c.send(
    new SDK.InvokeModelCommand({
      modelId: "amazon.titan-embed-text-v2:0",
      body: '{"inputText":"native vector"}',
    }),
  );
  assert.equal(attrs()["respan.entity.log_type"], "embedding");
  assert.equal(JSON.parse(attrs()["traceloop.entity.output"]).length, 5001);
  assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
  assert.equal(attrs()["llm.usage.total_tokens"], undefined);
});
test("canonical helper preserves multimodal structured output without executing toJSON", () => {
  let calls = 0;
  const a = buildBedrockAttrs({
    operationName: "Converse",
    apiParams: {
      modelId: "model",
      messages: [
        {
          role: "user",
          content: [
            { text: "" },
            {
              image: {
                format: "png",
                source: { bytes: new Uint8Array([0, 1]) },
              },
            },
            { json: { enabled: false, count: 0, empty: "", nil: null } },
          ],
        },
      ],
      toJSON() {
        calls++;
        throw Error("no");
      },
    },
    responsePayload: {
      output: {
        message: {
          role: "assistant",
          content: [
            {
              reasoningContent: {
                reasoningText: { text: "reason", signature: "fixture" },
              },
            },
            { text: "" },
          ],
        },
      },
      usage: { inputTokens: 0, outputTokens: 0 },
    },
  });
  assert.equal(calls, 0);
  assert.ok(a["traceloop.entity.input"].includes("image"));
  assert.ok(a["traceloop.entity.output"].includes("reasoningContent"));
  assert.equal(a["llm.usage.total_tokens"], undefined);
});

test(
  "native newer Converse configuration and multimodal output remain complete canonical JSON",
  {
    skip:
      Number(sdkVersion.split(".")[1]) < 1000
        ? "AWS3.704.0 has no audio/cache/system-tool/outputConfig schema"
        : false,
  },
  async () => {
    const i = install();
    await i.activate();
    const req = input();
    req.messages[0].content.push(
      {
        image: {
          format: "png",
          source: { bytes: new Uint8Array([0, 1, 2, 255]) },
        },
      },
      {
        audio: {
          format: "wav",
          source: { bytes: new Uint8Array([0, 2, 0, 3]) },
        },
      },
      { cachePoint: { type: "default", ttl: "1h" } },
    );
    req.toolConfig = {
      tools: [
        { systemTool: { name: "web_search" } },
        {
          toolSpec: {
            name: "strict_tool",
            strict: true,
            inputSchema: { json: { type: "object" } },
          },
        },
      ],
    };
    req.outputConfig = {
      textFormat: {
        type: "json_schema",
        structure: {
          jsonSchema: { name: "fixture", schema: '{"type":"object"}' },
        },
      },
      effort: "high",
    };
    req.requestMetadata = { fixture: "native-metadata" };
    const response = {
      output: {
        message: {
          role: "assistant",
          content: [
            { text: '{"enabled":false,"count":0,"empty":"","nil":null}' },
            {
              reasoningContent: {
                reasoningText: {
                  text: "reasoning fixture",
                  signature: "fixture-signature",
                },
              },
            },
            {
              image: {
                format: "png",
                source: {
                  bytes: Buffer.from([0, 1, 2, 255]).toString("base64"),
                },
              },
            },
          ],
        },
      },
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      stopReason: "malformed_tool_use",
    };
    const { client: c, transport } = client({ response });
    await c.send(new SDK.ConverseCommand(req));
    const metadata = JSON.parse(attrs()["respan.metadata"]);
    assert.deepEqual(metadata.request.outputConfig, req.outputConfig);
    assert.equal(metadata.request.requestMetadata.fixture, "native-metadata");
    assert.ok(attrs()["traceloop.entity.input"].includes("[0,1,2,255]"));
    assert.equal(
      JSON.parse(attrs()["llm.request.functions"])[0].system.name,
      "web_search",
    );
    assert.equal(
      JSON.parse(attrs()["llm.request.functions"])[1].function.strict,
      true,
    );
    assert.ok(attrs()["traceloop.entity.output"].includes("reasoning fixture"));
    assert.deepEqual(attrs()["gen_ai.response.finish_reasons"], [
      "malformed_tool_use",
    ]);
    // Features absent in older SDK serializers remain a declared native boundary.
    const wire = JSON.parse(transport.calls[0].body);
    if (wire.outputConfig)
      assert.deepEqual(wire.outputConfig, req.outputConfig);
  },
);
test("native inherited canonical metadata merges with request metadata", async () => {
  const i = install();
  await i.activate();
  const parent = trace.getTracer("parent").startSpan("parent");
  parent.setAttribute(
    "respan.metadata",
    '{"run_id":"native-parent-marker","nested":{"retained":true}}',
  );
  const { client: c } = client();
  await context.with(trace.setSpan(context.active(), parent), () =>
    c.send(converse()),
  );
  assert.equal(
    JSON.parse(attrs()["respan.metadata"]).run_id,
    "native-parent-marker",
  );
  assert.equal(JSON.parse(attrs()["respan.metadata"]).nested.retained, true);
  parent.end();
});
test("observed foreign ancestor false persists after later true without evaluating getters", async () => {
  const i = install();
  await i.activate();
  const parent = trace.getTracer("parent").startSpan("parent");
  parent.setAttribute("allow_trace_content", false);
  const { client: c } = client();
  const ctx = trace.setSpan(context.active(), parent);
  await context.with(ctx, () => c.send(converse()));
  parent.setAttribute("allow_trace_content", true);
  await context.with(ctx, () => c.send(converse()));
  noPayload(attrs());
  parent.end();
});
test("exporter ambient suppression leaves previously admitted content intact", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client();
  await c.send(converse());
  const span = spans[0];
  context.with(suppressTracing(context.active()), () =>
    assert.equal(span.attributes["gen_ai.prompt.0.content"], "native prompt"),
  );
});

test("native historical tool result preserves call ID without inventing tool executions", async () => {
  const i = install();
  await i.activate();
  const { client: c } = client();
  const req = input();
  req.messages = [
    {
      role: "assistant",
      content: [
        {
          toolUse: {
            toolUseId: "historical-tool",
            name: "lookup",
            input: { city: "Tokyo" },
          },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          toolResult: {
            toolUseId: "historical-tool",
            content: [
              { json: { enabled: false, count: 0, empty: "", nil: null } },
            ],
            status: "success",
          },
        },
      ],
    },
  ];
  await c.send(new SDK.ConverseCommand(req));
  const a = attrs();
  assert.equal(a["gen_ai.prompt.1.role"], "tool");
  assert.equal(a["gen_ai.prompt.1.tool_call_id"], "historical-tool");
  assert.equal(spans.length, 1);
  assert.equal(
    JSON.parse(a["gen_ai.completion.0.tool_calls"])[0].id,
    "response-tool",
  );
});

test("native stream keeps its actual iterator and observed HTTP response status", async () => {
  await install().activate();
  const { client: c } = client();
  let nativeIterator;
  c.middlewareStack.add(
    (next) => async (args) => {
      const result = await next(args);
      const stream = result.output.stream;
      const factory = stream[Symbol.asyncIterator];
      stream[Symbol.asyncIterator] = function () {
        nativeIterator = Reflect.apply(factory, this, []);
        return nativeIterator;
      };
      return result;
    },
    { step: "initialize", name: "captureNativeIterator" },
  );
  const result = await c.send(new SDK.ConverseStreamCommand(input()));
  const iterator = result.stream[Symbol.asyncIterator]();
  assert.equal(iterator, nativeIterator);
  while (!(await iterator.next()).done) {}
  assert.equal(attrs()[ATTR_HTTP_RESPONSE_STATUS_CODE], 200);
  assert.equal(attrs().status_code, undefined);
});

test("native child sampler decides independently for an unsampled remote parent", async () => {
  await install().activate();
  const { client: c } = client();
  const parent = trace.wrapSpanContext({
    traceId: "12345678901234567890123456789012",
    spanId: "1234567890123456",
    traceFlags: 0,
  });
  await context.with(trace.setSpan(context.active(), parent), () =>
    c.send(converse()),
  );
  assert.equal(sampled.length, 1);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].parentSpanContext.spanId, "1234567890123456");
});

test("native private late status accepts only an OTel numeric status code", async () => {
  await install({ traceContent: false }).activate();
  const { client: c } = client();
  onEnd = (span) => {
    span.status = { code: { secret: "private-status" }, message: "private" };
  };
  await c.send(converse());
  assert.deepEqual(spans[0].status, { code: 0 });
});

test("default native 128-attribute budget retains complete canonical Bedrock payloads", () => {
  execFileSync(
    process.execPath,
    [new URL("./default_budget.mjs", import.meta.url).pathname],
    { stdio: "pipe" },
  );
});
