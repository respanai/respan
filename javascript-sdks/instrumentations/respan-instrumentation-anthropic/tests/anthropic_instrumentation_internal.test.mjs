import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { context, trace, TraceFlags } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { AnthropicInstrumentor } from "../dist/index.js";
import { transport, MODEL, message, events } from "./native-fixture.mjs";
const version = JSON.parse(
  readFileSync(
    join(
      dirname(createRequire(import.meta.url).resolve("@anthropic-ai/sdk")),
      "package.json",
    ),
    "utf8",
  ),
).version;
const current = version === "0.133.0";
let decision = SamplingDecision.RECORD_AND_SAMPLED;
let starts = [];
let spans = [];
let onStart;
let onEnd;
const provider = new BasicTracerProvider({
  spanLimits: { attributeCountLimit: 10000 },
  sampler: {
    shouldSample(_ctx, _id, name, kind, attrs) {
      starts.push({ name, kind, attrs });
      return { decision };
    },
    toString() {
      return "native fixture";
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
const owners = [];
const input = () => ({
  model: MODEL,
  max_tokens: 100,
  messages: [{ role: "user", content: "fixture prompt" }],
});
async function setup(options = {}, native = {}) {
  const owner = new AnthropicInstrumentor({
    clientClass: Anthropic,
    ...options,
  });
  owners.push(owner);
  await owner.activate();
  const wire = transport(native);
  return {
    owner,
    wire,
    client: new Anthropic({
      apiKey: "fixture",
      maxRetries: 0,
      baseURL: "https://fixture.invalid",
      fetch: wire.fetch,
    }),
  };
}
const attrs = () => spans.at(-1).attributes;
function privateAttrs(a) {
  for (const k of Object.keys(a))
    assert.ok(
      !k.startsWith("gen_ai.prompt.") &&
        !k.startsWith("gen_ai.completion.") &&
        ![
          "traceloop.entity.input",
          "traceloop.entity.output",
          "llm.request.functions",
          "respan.metadata",
          "error.message",
        ].includes(k),
      k,
    );
}
test.beforeEach(() => {
  starts = [];
  spans = [];
  decision = SamplingDecision.RECORD_AND_SAMPLED;
  onStart = onEnd = undefined;
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
test.afterEach(() => {
  for (const o of owners.splice(0)) o.deactivate();
});
test.after(async () => {
  await provider.shutdown();
  context.disable();
  trace.disable();
});
test("native APIPromise preserves lazy parser, raw helpers, response and message identities", async () => {
  const { client, wire } = await setup();
  const proto = Object.getPrototypeOf(client.messages);
  const native = proto.create;
  let returned;
  proto.create = function (...args) {
    return (returned = native.apply(this, args));
  };
  try {
    const p = client.messages.create(input());
    assert.equal(p, returned);
    for (const name of [
      "then",
      "catch",
      "finally",
      "asResponse",
      "withResponse",
    ])
      assert.equal(Object.hasOwn(p, name), false, name);
    const raw = await p.asResponse();
    assert.equal(raw, wire.responses[0]);
    assert.equal(raw.bodyUsed, false);
    assert.equal(spans.length, 1);
    assert.equal(attrs()["traceloop.entity.output"], undefined);
    const r = await p.withResponse();
    assert.equal(r.response, raw);
    assert.equal(r.data, await p);
    assert.equal(spans.length, 1);
    assert.equal(attrs()["http.response.status_code"], 201);
    assert.equal(attrs()["llm.usage.total_tokens"], undefined);
  } finally {
    proto.create = native;
  }
});
test("native messages preserve76messages/tools, multimodal/thinking/cache/strict schemas and zero values", async () => {
  const { client } = await setup();
  const b = input();
  b.messages = Array.from({ length: 76 }, (_, i) => ({
    role: "user",
    content: [
      { type: "text", text: `fixture-${i}` },
      {
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "AAEC/w==" },
      },
    ],
  }));
  b.tools = Array.from({ length: 76 }, (_, i) => ({
    name: `tool_${i}`,
    strict: true,
    input_schema: {
      type: "object",
      properties: {
        flag: { const: false },
        zero: { const: 0 },
        empty: { const: "" },
        nil: { const: null },
      },
    },
  }));
  b.thinking = { type: "enabled", budget_tokens: 1024 };
  b.output_config = {
    format: { type: "json_schema", schema: { type: "object" } },
  };
  await client.messages.create(b);
  const a = attrs();
  assert.equal(JSON.parse(a["traceloop.entity.input"]).length, 76);
  assert.equal(JSON.parse(a["llm.request.functions"]).length, 76);
  assert.ok(a["gen_ai.prompt.75.content"].includes("AAEC/w=="));
  assert.ok(a["traceloop.entity.output"].includes("fixture thought"));
  assert.ok(a["gen_ai.completion.0.content"].includes("fixture signature"));
  assert.equal(a["gen_ai.usage.input_tokens"], 0);
  assert.equal(a["gen_ai.usage.output_tokens"], 0);
  assert.equal(a["gen_ai.usage.cache_read.input_tokens"], 0);
  assert.equal(a["llm.usage.total_tokens"], undefined);
  assert.deepEqual(
    JSON.parse(a["respan.metadata"]).request.output_config,
    b.output_config,
  );
});
test("native SSE preserves native stream/controller/chunks/iterator and full fragmented content", async () => {
  const { client } = await setup();
  const stream = await client.messages.create({ ...input(), stream: true });
  let iterator;
  const original = stream.iterator;
  stream.iterator = function (...a) {
    iterator = original.apply(this, a);
    return iterator;
  };
  const iter = stream[Symbol.asyncIterator]();
  assert.equal(iter, iterator);
  const controller = stream.controller;
  const values = [];
  for await (const item of { [Symbol.asyncIterator]: () => iter })
    values.push(item);
  assert.equal(stream.controller, controller);
  assert.equal(values.length, events.length);
  assert.equal(spans.length, 1);
  const a = attrs();
  assert.equal(
    JSON.parse(a["gen_ai.completion.0.tool_calls"])[0].function.arguments,
    '{"enabled":false,"count":0,"empty":"","nil":null}',
  );
  assert.ok(a["traceloop.entity.output"].includes("fixture thought"));
  assert.equal(a["http.response.status_code"], 201);
});
test("native stream tee and toReadableStream consume once", async () => {
  const { client } = await setup();
  const stream = await client.messages.create({ ...input(), stream: true });
  const [left, right] = stream.tee();
  const a = [],
    b = [];
  await Promise.all([
    (async () => {
      for await (const e of left) a.push(e);
    })(),
    (async () => {
      for await (const e of right) b.push(e);
    })(),
  ]);
  assert.equal(a.length, b.length);
  assert.equal(a[0], b[0]);
  assert.equal(spans.length, 1);
});
test("native SSE early return before first next ends once", async () => {
  const { client } = await setup();
  const stream = await client.messages.create({ ...input(), stream: true });
  await stream[Symbol.asyncIterator]().return();
  assert.equal(spans.length, 1);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
test("native stream failure preserves original SDK error without fabricated completion", async () => {
  const { client } = await setup(
    {},
    {
      streamEvents: [
        events[0],
        {
          type: "error",
          error: {
            type: "overloaded_error",
            message: "fixture stream failure",
          },
        },
      ],
    },
  );
  const stream = await client.messages.create({ ...input(), stream: true });
  await assert.rejects(async () => {
    for await (const e of stream) {
    }
  }, /fixture stream failure/);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 2);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
test("native controlled API error keeps native error and actual429", async () => {
  const { client } = await setup({}, { error: true });
  let error;
  await assert.rejects(
    client.messages.create(input()).catch((e) => {
      error = e;
      throw e;
    }),
    (e) => e === error,
  );
  assert.equal(spans.length, 1);
  assert.equal(attrs()["http.response.status_code"], 429);
  assert.equal(spans[0].status.code, 2);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
for (const [name, value] of [
  ["DROP", SamplingDecision.NOT_RECORD],
  ["RECORD_ONLY", SamplingDecision.RECORD],
])
  test(`native ${name} samples before telemetry payload snapshot and guards late mutation`, async () => {
    decision = value;
    onEnd = (s) => {
      s.attributes = {
        allow_trace_content: true,
        "gen_ai.usage.unknown": "private",
        "gen_ai.prompt.0.content": "private",
      };
      s.status = { code: { private: "bad" }, message: "private" };
      s.events = [{ name: "private" }];
    };
    const { client } = await setup();
    const b = input();
    let reads = 0;
    Object.defineProperty(b, "unknown", {
      get() {
        reads++;
        return "native";
      },
      enumerable: false,
    });
    await client.messages.create(b);
    assert.equal(reads, 0);
    assert.equal(starts.length, 1);
    assert.equal(starts[0].attrs["gen_ai.request.model"], MODEL);
    assert.equal(starts[0].attrs["traceloop.entity.input"], undefined);
    if (value === SamplingDecision.RECORD) {
      privateAttrs(attrs());
      assert.equal(attrs()["gen_ai.usage.unknown"], undefined);
      assert.deepEqual(spans[0].status, { code: 0 });
    } else assert.equal(spans.length, 0);
  });
for (const gate of [
  "constructor",
  "RESPAN_TRACE_CONTENT",
  "TRACELOOP_TRACE_CONTENT",
  "context",
  "parent",
])
  test(`native ${gate} privacy false is immutable`, async () => {
    const { client } = await setup(
      gate === "constructor" ? { traceContent: false } : {},
    );
    let ctx = context.active();
    let parent;
    if (gate === "context")
      ctx = ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
    else if (gate === "parent") {
      parent = trace.getTracer("parent").startSpan("parent");
      parent.setAttribute("allow_trace_content", false);
      ctx = trace.setSpan(ctx, parent);
    } else if (gate !== "constructor") process.env[gate] = "false";
    await context.with(ctx, () => client.messages.create(input()));
    const span = spans.at(-1);
    privateAttrs(span.attributes);
    span.attributes = {
      allow_trace_content: true,
      "gen_ai.usage.unknown": "private",
      "gen_ai.usage.input_tokens": 0,
    };
    span.events = [{ name: "private" }];
    span.status = { code: { secret: true }, message: "private" };
    assert.equal(span.attributes["gen_ai.usage.unknown"], undefined);
    assert.equal(span.attributes["gen_ai.usage.input_tokens"], 0);
    assert.deepEqual(span.status, { code: 0 });
    assert.equal(span.events.length, 0);
    assert.throws(() =>
      Object.defineProperty(span, "attributes", { value: { secret: "bad" } }),
    );
    parent?.end();
  });
for (const key of ["recordInputs", "recordOutputs"])
  test(`native independent ${key}`, async () => {
    const { client } = await setup({ [key]: false });
    await client.messages.create(input());
    assert.equal(
      attrs()["traceloop.entity.input"] !== undefined,
      key !== "recordInputs",
    );
    assert.equal(
      attrs()["traceloop.entity.output"] !== undefined,
      key !== "recordOutputs",
    );
  });
test("native unsampled ancestor reaches actualAlwaysOn sampler", async () => {
  const { client } = await setup();
  const parent = trace.wrapSpanContext({
    traceId: "1".repeat(32),
    spanId: "2".repeat(16),
    traceFlags: TraceFlags.NONE,
  });
  await context.with(trace.setSpan(context.active(), parent), () =>
    client.messages.create(input()),
  );
  assert.equal(starts.length, 1);
  assert.equal(spans.length, 1);
});
test("native suppression preserves SDK call and exporter suppression preserves admitted content", async () => {
  const { client, wire } = await setup();
  await context.with(suppressTracing(context.active()), () =>
    client.messages.create(input()),
  );
  assert.equal(wire.requests.length, 1);
  assert.equal(starts.length, 0);
  await client.messages.create(input());
  const s = spans[0];
  context.with(suppressTracing(context.active()), () =>
    assert.ok(s.attributes["traceloop.entity.input"]),
  );
});
test("native history never fabricates execution spans and tool result ID stays in input", async () => {
  const { client } = await setup();
  const b = input();
  b.messages = [
    {
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "history-tool",
          name: "lookup",
          input: { flag: false },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "history-tool",
          content: "historic result",
        },
      ],
    },
  ];
  await client.messages.create(b);
  assert.equal(spans.length, 1);
  assert.equal(attrs()["gen_ai.prompt.1.tool_call_id"], "history-tool");
  assert.equal(attrs()["respan.entity.log_type"], "chat");
});
test(
  "native stable and beta structured parse helpers delegate create once",
  { skip: !current },
  async () => {
    const { client } = await setup();
    for (const resource of [client.messages, client.beta.messages]) {
      const r = await resource.parse({
        ...input(),
        output_config: {
          format: { type: "json_schema", schema: { type: "object" } },
        },
      });
      assert.equal(r.parsed_output.enabled, false);
    }
    assert.equal(spans.length, 2);
  },
);
test("native available MessageStream helpers retain events/finalMessage/response", async () => {
  const basic = [
    events[0],
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "minimum stream" },
    },
    { type: "content_block_stop", index: 0 },
    events.at(-2),
    events.at(-1),
  ];
  const { client } = await setup({}, current ? {} : { streamEvents: basic });
  const resources = [client.messages, client.beta.messages].filter(
    (r) => typeof r?.stream === "function",
  );
  for (const resource of resources) {
    const stream = resource.stream(input());
    const controller = stream.controller;
    let emitted = 0;
    stream.on("streamEvent", () => emitted++);
    const r = await stream.finalMessage();
    if (current) assert.equal(r.content[0].thinking, "fixture thought");
    else assert.equal(r.content[0].text, "minimum stream");
    assert.equal(stream.controller, controller);
    assert.ok(emitted > 0);
    if (typeof stream.withResponse === "function") {
      const raw = await stream.withResponse();
      assert.equal(raw.data, stream);
    }
  }
  assert.equal(spans.length, resources.length);
});
test(
  "native stable and beta countTokens are common-only tasks",
  { skip: !current },
  async () => {
    const { client } = await setup();
    for (const resource of [client.messages, client.beta.messages])
      assert.equal(
        (
          await resource.countTokens({
            model: MODEL,
            messages: input().messages,
          })
        ).input_tokens,
        0,
      );
    assert.equal(spans.length, 2);
    for (const span of spans) {
      assert.equal(span.attributes["respan.entity.log_type"], "task");
      assert.equal(span.attributes["gen_ai.system"], undefined);
    }
  },
);
test("native available batches submission/results remain actual common-only task rows", async () => {
  const { client } = await setup();
  const resources = [client.messages, client.beta.messages].filter(
    (r) => r?.batches,
  );
  for (const resource of resources) {
    await resource.batches.create({
      requests: [{ custom_id: "fixture-row", params: input() }],
    });
    const rows = await resource.batches.results("msgbatch_fixture");
    const out = [];
    for await (const row of rows) out.push(row);
    assert.equal(out.length, 2);
  }
  assert.equal(spans.length, resources.length * 2);
  for (const span of spans)
    assert.equal(span.attributes["respan.entity.log_type"], "task");
  assert.equal(
    JSON.parse(spans[1].attributes["traceloop.entity.output"]).length,
    2,
  );
});
test("native legacy completions returns original completion and atext span", async () => {
  const { client } = await setup();
  const r = await client.completions.create({
    model: MODEL,
    max_tokens_to_sample: 20,
    prompt: "\n\nHuman: fixture\n\nAssistant:",
  });
  assert.equal(r.completion, "legacy fixture");
  assert.equal(attrs()["respan.entity.log_type"], "text");
});
test(
  "native toolRunner actual callbacks receive IDs and create common-only agent/tools",
  { skip: !current },
  async () => {
    const { client } = await setup({}, { runner: true });
    let calls = 0;
    const tool = {
      name: "lookup",
      input_schema: { type: "object" },
      parse: (v) => v,
      run(args, ctx) {
        calls++;
        assert.equal(ctx.toolUse.id, "runner-tool");
        assert.equal(args.enabled, false);
        return "tool fixture output";
      },
    };
    const original = tool.run;
    const runner = client.beta.messages.toolRunner({
      ...input(),
      tools: [tool],
    });
    const result = await runner;
    assert.equal(result.stop_reason, "end_turn");
    assert.equal(calls, 1);
    assert.equal(tool.run, original);
    const agent = spans.find(
      (s) => s.attributes["respan.entity.log_type"] === "agent",
    );
    const execution = spans.find(
      (s) => s.attributes["respan.entity.log_type"] === "tool",
    );
    assert.equal(execution.attributes["gen_ai.tool.call.id"], "runner-tool");
    assert.deepEqual(
      JSON.parse(execution.attributes["traceloop.entity.input"]),
      {
        name: "lookup",
        arguments: { enabled: false, count: 0, empty: "", nil: null },
      },
    );
    assert.equal(
      execution.parentSpanContext.spanId,
      agent.spanContext().spanId,
    );
    assert.equal(agent.attributes["llm.request.functions"], undefined);
    assert.equal(agent.attributes["gen_ai.system"], undefined);
    assert.equal(
      spans.filter((s) => s.attributes["respan.entity.log_type"] === "chat")
        .length,
      2,
    );
  },
);
test(
  "native toolRunner changes callbacks before currentstate clones and leaves customer functions unchanged",
  { skip: !current },
  async () => {
    const { client } = await setup({}, { runner: true });
    let calls = 0;
    const one = {
      name: "lookup",
      input_schema: { type: "object" },
      parse: (v) => v,
      run() {
        throw Error("replaced callback ran");
      },
    };
    const two = {
      ...one,
      run() {
        calls++;
        return "replacement result";
      },
    };
    const native = two.run;
    const runner = client.beta.messages.toolRunner({
      ...input(),
      tools: [one],
    });
    runner.setMessagesParams((params) => ({ ...params, tools: [two] }));
    await runner;
    assert.equal(calls, 1);
    assert.equal(two.run, native);
    assert.equal(
      spans.filter((s) => s.attributes["respan.entity.log_type"] === "tool")
        .length,
      1,
    );
  },
);
test("native owner sharing/coalescing/pending cancellation/foreign patch preservation", async () => {
  const a = new AnthropicInstrumentor(),
    b = new AnthropicInstrumentor({ clientClass: Anthropic });
  owners.push(a, b);
  const pending = a.activate();
  assert.equal(a.activate(), pending);
  a.deactivate();
  await pending;
  assert.equal(a.isActive(), false);
  await Promise.all([a.activate(), b.activate()]);
  const wire = transport();
  const client = new Anthropic({ apiKey: "fixture", fetch: wire.fetch });
  const proto = Object.getPrototypeOf(client.messages);
  const wrapped = proto.create;
  a.deactivate();
  assert.equal(proto.create, wrapped);
  await client.messages.create(input());
  assert.equal(spans.length, 1);
  const foreign = function (...args) {
    return wrapped.apply(this, args);
  };
  proto.create = foreign;
  b.deactivate();
  assert.equal(proto.create, foreign);
  proto.create = wrapped;
});
test("native deactivation drains admittedAPIPromise", async () => {
  const { client, owner } = await setup({}, { delay: 15 });
  const pending = client.messages.create(input());
  owner.deactivate();
  await pending;
  assert.equal(spans.length, 1);
});

test(
  "native toolRunner repeated currentstate edits do not duplicate callback spans",
  { skip: !current },
  async () => {
    const { client } = await setup({}, { runner: true });
    let calls = 0;
    const tool = {
      name: "lookup",
      input_schema: { type: "object" },
      parse: (v) => v,
      run() {
        calls++;
        return "tool result";
      },
    };
    const original = tool.run;
    const runner = client.beta.messages.toolRunner({
      ...input(),
      tools: [tool],
    });
    runner.pushMessages({ role: "user", content: "one" });
    runner.pushMessages({ role: "user", content: "two" });
    await runner;
    assert.equal(calls, 1);
    assert.equal(
      spans.filter((s) => s.attributes["respan.entity.log_type"] === "tool")
        .length,
      1,
    );
    assert.equal(tool.run, original);
    assert.ok(
      spans.find((s) => s.attributes["respan.entity.log_type"] === "agent")
        .attributes["traceloop.entity.output"],
    );
  },
);

test(
  "native admitted toolRunner drains both model legs after deactivate while outside calls bypass",
  { skip: !current },
  async () => {
    const bare = new Anthropic({ apiKey: "fixture" });
    const proto = Object.getPrototypeOf(bare.beta.messages);
    const native = proto.create;
    const { client, owner } = await setup({}, { runner: true });
    const tool = {
      name: "lookup",
      input_schema: { type: "object" },
      parse: (v) => v,
      run() {
        return "native drain tool";
      },
    };
    const runner = client.beta.messages.toolRunner({
      ...input(),
      tools: [tool],
    });
    owner.deactivate();
    assert.equal(owner.isActive(), false);
    const outsideWire = transport();
    const outside = new Anthropic({
      apiKey: "fixture",
      fetch: outsideWire.fetch,
    });
    await outside.messages.create(input());
    assert.equal(spans.length, 0);
    await runner;
    assert.equal(
      spans.filter((s) => s.attributes["respan.entity.log_type"] === "chat")
        .length,
      2,
    );
    assert.equal(
      spans.filter((s) => s.attributes["respan.entity.log_type"] === "tool")
        .length,
      1,
    );
    assert.equal(proto.create, native);
  },
);
test("native parser Promise identity stays unchanged", async () => {
  const bare = new Anthropic({ apiKey: "fixture" });
  const proto = Object.getPrototypeOf(bare.messages);
  const original = proto.create;
  let nativeParsed;
  let observedParsed;
  proto.create = function (...args) {
    const p = original.apply(this, args);
    const parse = p.parseResponse;
    p.parseResponse = function (...a) {
      return (nativeParsed = parse.apply(this, a));
    };
    return p;
  };
  try {
    const { client } = await setup();
    const p = client.messages.create(input());
    const observer = p.parseResponse;
    p.parseResponse = function (...a) {
      return (observedParsed = observer.apply(this, a));
    };
    await p;
    assert.equal(observedParsed, nativeParsed);
    assert.ok(attrs()["traceloop.entity.output"]);
  } finally {
    for (const o of owners) o.deactivate();
    proto.create = original;
  }
});
test("native raw-only Response ends at actual headers without body traversal", async () => {
  const { client } = await setup();
  const p = client.messages.create(input());
  const response = await p.asResponse();
  assert.equal(response.bodyUsed, false);
  assert.equal(spans.length, 1);
  assert.equal(attrs()["http.response.status_code"], 201);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
test("native controller abort before first iterator next ends without fabricated status/output", async () => {
  const { client } = await setup();
  const stream = await client.messages.create({ ...input(), stream: true });
  stream.controller.abort();
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 0);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
for (const mode of ["private", "record-only"])
  test(
    `native ${mode} tool arrays have no telemetry Proxy traversal`,
    { skip: !current },
    async () => {
      if (mode === "record-only") decision = SamplingDecision.RECORD;
      const { client, owner } = await setup(
        mode === "private" ? { traceContent: false } : {},
      );
      owner.deactivate();
      let reads = 0;
      const tools = new Proxy(
        [
          {
            name: "lookup",
            input_schema: { type: "object" },
            parse: (v) => v,
            run() {
              return "result";
            },
          },
        ],
        {
          get(t, k, r) {
            reads++;
            return Reflect.get(t, k, r);
          },
          ownKeys(t) {
            reads++;
            return Reflect.ownKeys(t);
          },
          getOwnPropertyDescriptor(t, k) {
            reads++;
            return Reflect.getOwnPropertyDescriptor(t, k);
          },
        },
      );
      const params = { ...input(), tools };
      client.beta.messages.toolRunner(params);
      const bareReads = reads;
      reads = 0;
      await owner.activate();
      const runner = client.beta.messages.toolRunner(params);
      assert.equal(reads, bareReads);
      await runner[Symbol.asyncIterator]().return();
    },
  );
test(
  "native current compaction fragments and unknown deltas remain complete",
  { skip: !current },
  async () => {
    const special = [
      events[0],
      {
        type: "content_block_start",
        index: 0,
        content_block: { type: "compaction", content: "" },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "compaction_delta", content: "first " },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "compaction_delta", content: "second" },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: {
          type: "future_delta",
          payload: { flag: false, count: 0, empty: "", nil: null },
        },
      },
      { type: "content_block_stop", index: 0 },
      events.at(-2),
      events.at(-1),
    ];
    const { client } = await setup({}, { streamEvents: special });
    const stream = await client.messages.create({ ...input(), stream: true });
    for await (const event of stream) {
    }
    const output = JSON.parse(attrs()["traceloop.entity.output"]);
    assert.equal(output.content[0].type, "compaction");
    assert.equal(output.content[0].content, "second");
    assert.deepEqual(
      JSON.parse(attrs()["respan.metadata"]).stream_events,
      special,
    );
  },
);
test(
  "native agent canonical metadata merges inherited attribution",
  { skip: !current },
  async () => {
    const { client } = await setup({}, { runner: true });
    const parent = trace.getTracer("parent").startSpan("parent");
    parent.setAttribute(
      "respan.metadata",
      '{"run_id":"inherited-marker","custom":{"kept":true}}',
    );
    const tool = {
      name: "lookup",
      input_schema: { type: "object" },
      parse: (v) => v,
      run() {
        return "metadata tool";
      },
    };
    await context.with(trace.setSpan(context.active(), parent), () =>
      client.beta.messages.toolRunner({ ...input(), tools: [tool] }),
    );
    const agent = spans.find(
      (s) => s.attributes["respan.entity.log_type"] === "agent",
    );
    const metadata = JSON.parse(agent.attributes["respan.metadata"]);
    assert.equal(metadata.run_id, "inherited-marker");
    assert.equal(metadata.custom.kept, true);
    assert.equal(metadata.tools.length, 1);
    parent.end();
  },
);
test("native legacy completion SSE preserves actual chunks/model/stop reason", async () => {
  const chunks = [
    {
      type: "completion",
      id: "completion_native",
      model: MODEL,
      completion: "first ",
      stop_reason: null,
    },
    {
      type: "completion",
      id: "completion_native",
      model: MODEL,
      completion: "second",
      stop_reason: "stop_sequence",
    },
  ];
  const { client } = await setup({}, { streamEvents: chunks });
  const stream = await client.completions.create({
    model: MODEL,
    max_tokens_to_sample: 20,
    prompt: "\n\nHuman: test\n\nAssistant:",
    stream: true,
  });
  for await (const chunk of stream) {
  }
  assert.equal(attrs()["gen_ai.completion.0.content"], "first second");
  assert.equal(attrs()["gen_ai.response.model"], MODEL);
  assert.equal(attrs()["llm.request.type"], "completion");
  assert.deepEqual(attrs()["gen_ai.response.finish_reasons"], [
    "stop_sequence",
  ]);
});
test("native default128 budget preserves canonical input/tools/output/usage before indexed history", async () => {
  const limited = new BasicTracerProvider({
    spanLimits: { attributeCountLimit: 128 },
    spanProcessors: [
      {
        onStart() {},
        onEnd(s) {
          spans.push(s);
        },
        forceFlush: async () => {},
        shutdown: async () => {},
      },
    ],
  });
  trace.disable();
  trace.setGlobalTracerProvider(limited);
  try {
    const { client } = await setup();
    const b = input();
    b.messages = Array.from({ length: 76 }, (_, n) => ({
      role: "user",
      content: `native-${n}`,
    }));
    b.tools = Array.from({ length: 76 }, (_, n) => ({
      name: `tool_${n}`,
      input_schema: { type: "object" },
    }));
    await client.messages.create(b);
    assert.equal(JSON.parse(attrs()["traceloop.entity.input"]).length, 76);
    assert.equal(JSON.parse(attrs()["llm.request.functions"]).length, 76);
    assert.ok(attrs()["traceloop.entity.output"]);
    assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
    assert.equal(attrs()["gen_ai.usage.output_tokens"], 0);
    assert.equal(attrs()["http.response.status_code"], 201);
    assert.ok(spans[0].droppedAttributesCount > 0);
  } finally {
    trace.disable();
    trace.setGlobalTracerProvider(provider);
    await limited.shutdown();
  }
});

for (const final of [
  { content: "complete compacted context", encrypted_content: "encrypted-b" },
  { content: null, encrypted_content: null },
  { content: "", encrypted_content: "" },
])
  test(
    `native beta compaction replacement matches finalMessage ${JSON.stringify(final)}`,
    { skip: !current },
    async () => {
      const values = [
        events[0],
        {
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "compaction",
            content: null,
            encrypted_content: null,
          },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "compaction_delta",
            content: "first compaction state",
            encrypted_content: "encrypted-a",
          },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "compaction_delta", ...final },
        },
        { type: "content_block_stop", index: 0 },
        {
          type: "message_delta",
          delta: { stop_reason: "end_turn", stop_sequence: null },
          usage: { output_tokens: 0 },
        },
        events.at(-1),
      ];
      const { client } = await setup({}, { streamEvents: values });
      const native = await client.beta.messages.stream(input()).finalMessage();
      const output = JSON.parse(attrs()["traceloop.entity.output"]);
      assert.deepEqual(output.content[0], native.content[0]);
      assert.equal(output.content[0].content, final.content);
      assert.equal(
        output.content[0].encrypted_content,
        final.encrypted_content,
      );
    },
  );
