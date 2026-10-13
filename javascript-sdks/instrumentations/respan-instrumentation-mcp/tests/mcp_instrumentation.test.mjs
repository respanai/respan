import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import { createRequire } from "node:module";
import { context, trace } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import {
  SimpleSpanProcessor,
  InMemorySpanExporter,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import * as clientV1 from "@modelcontextprotocol/sdk/client/index.js";
import * as serverV1 from "@modelcontextprotocol/sdk/server/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import * as schemas from "@modelcontextprotocol/sdk/types.js";
import {
  MCPInstrumentor,
  safeJson,
  toSerializableValue,
  clientMethodInput,
  resolveClientEntityName,
  resolveClientSpanName,
} from "../dist/index.js";

const require = createRequire(import.meta.url);
const sdkVersion = JSON.parse(
  fs.readFileSync(
    require
      .resolve("@modelcontextprotocol/sdk/client/index.js")
      .replace(/dist\/((esm|cjs)\/)?client\/index.js$/, "package.json"),
  ),
).version;
const minimum = sdkVersion === "1.0.0";
let decision = SamplingDecision.RECORD_AND_SAMPLED;
let samplerSawInput;
let chooseSampling;
const exporter = new InMemorySpanExporter();
const raw = [];
const nativeProcessor = {
  onStart(span) {
    raw.push(span);
  },
  onEnd() {},
  async forceFlush() {},
  async shutdown() {},
};
const provider = new NodeTracerProvider({
  spanProcessors: [nativeProcessor, new SimpleSpanProcessor(exporter)],
  sampler: {
    shouldSample(_context, _id, _name, _kind, attributes) {
      samplerSawInput = attributes["traceloop.entity.input"];
      return { decision: chooseSampling?.(_name, _kind) ?? decision };
    },
    toString() {
      return "native-MCP-test";
    },
  },
});
provider.register();
const instrumentors = new Set();
async function activate(options = {}) {
  const p = new MCPInstrumentor({
    clientModule: clientV1,
    serverModule: serverV1,
    ...options,
  });
  instrumentors.add(p);
  await p.activate();
  return p;
}
async function environment({
  clientModule = clientV1,
  serverModule = serverV1,
  Transport = InMemoryTransport,
  plugin = true,
  clientCapabilities = {},
  serverCapabilities = {
    tools: {},
    resources: {},
    prompts: {},
    logging: {},
    completions: {},
  },
} = {}) {
  const p = plugin ? await activate({ clientModule, serverModule }) : undefined;
  const server = new serverModule.Server(
    { name: "native-controlled-server", version: "1" },
    { capabilities: serverCapabilities },
  );
  const client = new clientModule.Client(
    { name: "native-controlled-client", version: "1" },
    { capabilities: clientCapabilities },
  );
  const set = (method, handler) =>
    server.setRequestHandler(
      typeof method === "string" ? method : method,
      handler,
    );
  const [a, b] = Transport.createLinkedPair();
  const connect = async () => {
    await server.connect(b);
    await client.connect(a);
  };
  const close = async () => {
    await client.close();
    await server.close();
    p?.deactivate();
  };
  return { server, client, p, set, connect, close, transports: [a, b] };
}
function toolSpans() {
  return exporter
    .getFinishedSpans()
    .filter((s) => s.attributes["mcp.method"] === "tools/call");
}
function output(span) {
  return JSON.parse(span.attributes["traceloop.entity.output"]);
}
function input(span) {
  return JSON.parse(span.attributes["traceloop.entity.input"]);
}
test.beforeEach(() => {
  exporter.reset();
  raw.length = 0;
  decision = SamplingDecision.RECORD_AND_SAMPLED;
  chooseSampling = undefined;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
test.afterEach(() => {
  for (const p of instrumentors) p.deactivate();
  instrumentors.clear();
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
test.after(async () => {
  await provider.shutdown();
  trace.disable();
  context.disable();
});

test("native v1 tool request retains full data and actual protocol IDs", async () => {
  const env = await environment();
  const text = "native-full-content-".repeat(2500);
  const definitions = Array.from({ length: 80 }, (_, i) => ({
    name: `tool_${i}`,
    description: `desc_${i}`,
    inputSchema: { type: "object", properties: { n: { type: "number" } } },
  }));
  env.set(schemas.ListToolsRequestSchema, () => ({
    tools: definitions,
    nextCursor: "native-page-2",
  }));
  env.set(schemas.CallToolRequestSchema, (req) => ({
    content: [
      { type: "text", text },
      { type: "image", data: "YmluYXJ5", mimeType: "image/png" },
    ],
    ...(minimum
      ? {}
      : {
          structuredContent: JSON.parse(
            '{"flag":false,"zero":0,"empty":"","null":null,"nested":{"a":{"b":{"c":{"d":{"e":{"f":{"g":"native-deep"}}}}}}}}',
          ),
          _meta: { native: "result-meta" },
        }),
  }));
  await env.connect();
  const listed = await env.client.listTools({ cursor: "native-page-1" });
  assert.equal(listed.tools.length, 80);
  const result = await env.client.callTool({
    name: "large_tool",
    arguments: {
      messages: Array.from({ length: 80 }, (_, i) => ({
        role: "user",
        content: `message_${i}`,
      })),
      vector: Array.from({ length: 5001 }, (_, i) => i),
      flag: false,
      zero: 0,
      empty: "",
      null: null,
    },
  });
  assert.equal(result.content[0].text, text);
  const spans = toolSpans();
  assert.equal(spans.length, 2);
  assert.equal(output(spans[0]).content[0].text, text);
  assert.equal(input(spans[1]).arguments.vector.length, 5001);
  assert.equal(input(spans[1]).arguments.messages.length, 80);
  assert.equal(samplerSawInput, undefined);
  assert.equal(
    spans[0].parentSpanContext?.spanId ?? spans[0].parentSpanId,
    spans[1].spanContext().spanId,
  );
  assert.equal(
    spans[0].attributes["mcp.request.id"],
    spans[1].attributes["mcp.request.id"],
  );
  assert.equal(spans[0].attributes["gen_ai.tool.call.id"], undefined);
  for (const span of spans) {
    assert.equal(span.attributes["respan.entity.log_type"], "tool");
    for (const key of [
      "traceloop.span.kind",
      "gen_ai.system",
      "gen_ai.request.model",
      "llm.request.type",
      "tool_calls",
      "respan.span.tool_calls",
    ])
      assert.equal(span.attributes[key], undefined);
    assert.ok(Object.keys(span.attributes).length < 128);
  }
  const list = exporter
    .getFinishedSpans()
    .find((s) => s.attributes["mcp.operation"] === "client.tools/list");
  assert.deepEqual(output(list).tools, definitions);
  assert.equal(input(list).params.cursor, "native-page-1");
  await env.close();
});

test("native resource and prompt pagination, content, completion and logging", async () => {
  const env = await environment();
  env.set(schemas.ListResourcesRequestSchema, (r) => ({
    resources: [{ uri: "demo://native", name: "native" }],
    nextCursor: r.params?.cursor ? undefined : "next",
  }));
  env.set(schemas.ListResourceTemplatesRequestSchema, () => ({
    resourceTemplates: [{ name: "native", uriTemplate: "demo://{id}" }],
  }));
  env.set(schemas.ReadResourceRequestSchema, (r) => ({
    contents: [
      {
        uri: r.params.uri,
        blob: "YmxvYg==",
        mimeType: "application/octet-stream",
      },
    ],
  }));
  env.set(schemas.ListPromptsRequestSchema, () => ({
    prompts: [
      { name: "native_prompt", arguments: [{ name: "text", required: true }] },
    ],
    nextCursor: "p2",
  }));
  env.set(schemas.GetPromptRequestSchema, () => ({
    messages: [
      { role: "user", content: { type: "text", text: "native prompt" } },
    ],
  }));
  env.set(schemas.SetLevelRequestSchema, () => ({}));
  env.set(schemas.CompleteRequestSchema, () => ({
    completion: { values: ["native"], total: 1, hasMore: false },
  }));
  await env.connect();
  await env.client.listResources({ cursor: "r1" });
  await env.client.listResourceTemplates();
  await env.client.readResource({ uri: "demo://native" });
  await env.client.listPrompts({ cursor: "p1" });
  await env.client.getPrompt({
    name: "native_prompt",
    arguments: { text: "input" },
  });
  await env.client.complete({
    ref: { type: "ref/prompt", name: "native_prompt" },
    argument: { name: "text", value: "n" },
  });
  await env.client.setLoggingLevel("info");
  const spans = exporter.getFinishedSpans();
  assert.ok(
    spans.some((s) => s.attributes["mcp.method"] === "completion/complete"),
  );
  assert.ok(
    spans.some((s) => s.attributes["mcp.method"] === "logging/setLevel"),
  );
  for (const span of spans.filter(
    (s) => s.attributes["respan.entity.log_type"] === "task",
  )) {
    assert.equal(span.attributes["gen_ai.system"], undefined);
    assert.equal(span.attributes["gen_ai.tool.call.id"], undefined);
  }
  await env.close();
});

test("native server sampling and client roots callbacks stay protocol tasks", async () => {
  const env = await environment({
    clientCapabilities: { sampling: {}, roots: { listChanged: true } },
  });
  env.client.setRequestHandler(schemas.CreateMessageRequestSchema, () => ({
    role: "assistant",
    model: "native-client-model",
    content: { type: "text", text: "native sampled result" },
  }));
  env.client.setRequestHandler(schemas.ListRootsRequestSchema, () => ({
    roots: [{ uri: "file:///controlled", name: "controlled" }],
  }));
  await env.connect();
  const value = await env.server.createMessage({
    messages: [
      {
        role: "user",
        content: { type: "text", text: "native sampling input" },
      },
    ],
    maxTokens: 4,
  });
  assert.equal(value.model, "native-client-model");
  await env.server.listRoots();
  for (const span of exporter
    .getFinishedSpans()
    .filter((s) => s.attributes["mcp.method"] === "sampling/createMessage")) {
    assert.equal(span.attributes["respan.entity.log_type"], "task");
    assert.equal(span.attributes["gen_ai.request.model"], undefined);
    assert.equal(output(span).model, "native-client-model");
  }
  await env.close();
});

test(
  "native elicitation exposed only after the v1 minimum",
  {
    skip: minimum
      ? "SDK1.0.0 has no elicitation schemas or Server.elicitInput"
      : false,
  },
  async () => {
    const env = await environment({
      clientCapabilities: { elicitation: { form: {} } },
    });
    env.client.setRequestHandler(schemas.ElicitRequestSchema, () => ({
      action: "accept",
      content: { approved: false, count: 0 },
    }));
    await env.connect();
    const result = await env.server.elicitInput({
      message: "controlled elicitation",
      requestedSchema: {
        type: "object",
        properties: {
          approved: { type: "boolean" },
          count: { type: "number" },
        },
      },
    });
    assert.deepEqual(result.content, { approved: false, count: 0 });
    assert.equal(
      exporter
        .getFinishedSpans()
        .filter((s) => s.attributes["mcp.method"] === "elicitation/create")
        .length,
      2,
    );
    await env.close();
  },
);

test("native notifications preserve callback results and fields", async () => {
  const env = await environment({
    clientCapabilities: { roots: { listChanged: true } },
  });
  let received;
  env.client.setNotificationHandler(
    schemas.LoggingMessageNotificationSchema,
    (n) => {
      received = n;
    },
  );
  await env.connect();
  await env.server.sendLoggingMessage({
    level: "info",
    data: { native: false, zero: 0 },
  });
  await new Promise((r) => setImmediate(r));
  assert.equal(received.params.data.zero, 0);
  assert.equal(
    exporter
      .getFinishedSpans()
      .filter((s) => s.attributes["mcp.method"] === "notifications/message")
      .length,
    2,
  );
  await env.close();
});

test("native tool error result marks ERROR and keeps exact successful protocol result", async () => {
  const env = await environment();
  env.set(schemas.CallToolRequestSchema, () => ({
    isError: true,
    content: [{ type: "text", text: "native controlled error" }],
  }));
  await env.connect();
  const result = await env.client.callTool({
    name: "native_error",
    arguments: {},
  });
  assert.equal(result.isError, true);
  for (const span of toolSpans()) {
    assert.equal(span.status.code, 2);
    assert.equal(output(span).isError, true);
  }
  await env.close();
});

test("native rejection and AbortSignal cancellation preserve errors without fabricated outputs", async () => {
  const env = await environment();
  const failure = new Error("native rejection secret");
  env.set(schemas.CallToolRequestSchema, () => {
    throw failure;
  });
  await env.connect();
  await assert.rejects(
    env.client.callTool({ name: "reject", arguments: {} }),
    /native rejection secret/,
  );
  assert.ok(toolSpans().every((s) => s.status.code === 2));
  assert.ok(
    toolSpans().every(
      (s) => s.attributes["traceloop.entity.output"] === undefined,
    ),
  );
  let releaseCancelled;
  env.set(
    schemas.CallToolRequestSchema,
    (_r, extra) =>
      new Promise((resolve, reject) => {
        releaseCancelled = resolve;
        extra.signal.addEventListener(
          "abort",
          () => reject(new Error("native cancelled")),
          { once: true },
        );
      }),
  );
  const controller = new AbortController();
  const promise = env.client.callTool(
    { name: "cancel", arguments: {} },
    undefined,
    { signal: controller.signal },
  );
  controller.abort(new Error("cancel native request"));
  await assert.rejects(promise, /cancel native request|aborted|cancel/i);
  await new Promise((r) => setImmediate(r));
  const cancelled = raw.filter(
    (span) =>
      span.attributes["mcp.method"] === "tools/call" &&
      span.attributes["traceloop.entity.name"] === "cancel",
  );
  assert.equal(cancelled.length, 2);
  if (minimum) {
    // SDK1.0.0 sends "cancelled" but listens for "notifications/cancelled".
    // The client rejects, while the native server signal remains live.
    assert.equal(
      cancelled.filter((span) => !span.isRecording() && span.status.code === 2)
        .length,
      1,
    );
    releaseCancelled({
      content: [{ type: "text", text: "native minimum server completed" }],
    });
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(cancelled.every((span) => !span.isRecording()));
  } else
    assert.ok(
      cancelled.every((span) => !span.isRecording() && span.status.code === 2),
    );
  await env.close();
});

for (const [label, options, scope] of [
  ["constructor false", { traceContent: false }, (fn) => fn()],
  [
    "environment false",
    {},
    (fn) => {
      process.env.TRACELOOP_TRACE_CONTENT = "false";
      return fn();
    },
  ],
  [
    "canonical context false",
    {},
    (fn) =>
      context.with(
        context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
        fn,
      ),
  ],
  [
    "record-only sampler",
    {},
    (fn) => {
      decision = SamplingDecision.RECORD;
      return fn();
    },
  ],
  [
    "drop sampler",
    {},
    (fn) => {
      decision = SamplingDecision.NOT_RECORD;
      return fn();
    },
  ],
])
  test(`native privacy ${label} denies inputs, outputs and late span mutations`, async () => {
    const p = await activate(options);
    const env = await environment({ plugin: false });
    let activeSpan;
    env.set(schemas.CallToolRequestSchema, () => {
      activeSpan = trace.getSpan(context.active());
      activeSpan.setAttribute("private.native", "late secret");
      activeSpan.addEvent("private event", { secret: "late" });
      activeSpan.setStatus({ code: 2, message: "private status" });
      return { content: [{ type: "text", text: "private native output" }] };
    });
    await env.connect();
    exporter.reset();
    raw.length = 0;
    await scope(() =>
      env.client.callTool({
        name: "private_tool",
        arguments: { secret: "private native input" },
      }),
    );
    for (const span of raw) {
      assert.equal(span.attributes["traceloop.entity.input"], undefined);
      assert.equal(span.attributes["traceloop.entity.output"], undefined);
      assert.equal(span.attributes["private.native"], undefined);
      assert.equal(span.status.message, undefined);
      assert.deepEqual(span.events, []);
      assert.throws(
        () =>
          Object.defineProperty(span, "attributes", {
            value: { secret: "replacement" },
          }),
        TypeError,
      );
      span.attributes = {
        "unknown.secret": "replacement",
        "mcp.method": "tools/call",
      };
      assert.equal(span.attributes["unknown.secret"], undefined);
      span.status = { code: "private-code", message: "private" };
      assert.equal(span.status.code, 0);
    }
    await env.close();
    p.deactivate();
  });

test("native independent input and output lanes", async () => {
  const p = await activate({ recordInputs: false });
  const env = await environment({ plugin: false });
  env.set(schemas.CallToolRequestSchema, () => ({
    content: [{ type: "text", text: "native output" }],
  }));
  await env.connect();
  await env.client.callTool({ name: "lanes", arguments: { private: "input" } });
  for (const s of toolSpans()) {
    assert.equal(s.attributes["traceloop.entity.input"], undefined);
    assert.equal(output(s).content[0].text, "native output");
  }
  await env.close();
  p.deactivate();
  exporter.reset();
  const q = await activate({ recordOutputs: false });
  const next = await environment({ plugin: false });
  next.set(schemas.CallToolRequestSchema, () => ({
    content: [{ type: "text", text: "private output" }],
  }));
  await next.connect();
  await next.client.callTool({ name: "lanes", arguments: { value: 0 } });
  for (const s of toolSpans()) {
    assert.equal(input(s).arguments.value, 0);
    assert.equal(s.attributes["traceloop.entity.output"], undefined);
  }
  await next.close();
  q.deactivate();
});

test("native suppression calls through and late veto remains a ceiling for descendants", async () => {
  const env = await environment({ clientCapabilities: { sampling: {} } });
  env.set(schemas.CallToolRequestSchema, async (req) => {
    if (req.params.name === "outer")
      return context.with(
        context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
        async () => {
          trace.getSpan(context.active()).setAttribute("late.private", "veto");
          return context
            .with(
              context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
              () =>
                env.server.createMessage({
                  messages: [
                    {
                      role: "user",
                      content: { type: "text", text: "child private" },
                    },
                  ],
                  maxTokens: 2,
                }),
            )
            .then(() => ({
              content: [{ type: "text", text: "outer private" }],
            }));
        },
      );
    return { content: [] };
  });
  env.client.setRequestHandler(schemas.CreateMessageRequestSchema, () => ({
    role: "assistant",
    model: "native",
    content: { type: "text", text: "private child output" },
  }));
  await env.connect();
  exporter.reset();
  await context.with(suppressTracing(context.active()), () =>
    env.client.callTool({ name: "suppressed", arguments: {} }),
  );
  assert.equal(toolSpans().length, 0);
  await env.client.callTool({ name: "outer", arguments: {} });
  for (const s of exporter
    .getFinishedSpans()
    .filter((s) => s.attributes["mcp.method"] === "sampling/createMessage")) {
    assert.equal(s.attributes["traceloop.entity.input"], undefined);
    assert.equal(s.attributes["traceloop.entity.output"], undefined);
  }
  await env.close();
});

test("native payload snapshots skip getters/toJSON/proxies and preserve deep values", () => {
  let reads = 0;
  const value = {
    zero: 0,
    flag: false,
    empty: "",
    null: null,
    _meta: { native: true },
    nested: { a: { b: { c: { d: { e: { f: { g: "deep" } } } } } } },
    toJSON() {
      reads++;
      return "bad";
    },
  };
  Object.defineProperty(value, "secret", {
    enumerable: true,
    get() {
      reads++;
      return "private";
    },
  });
  value.proxy = new Proxy(
    { secret: "private" },
    {
      ownKeys() {
        reads++;
        throw new Error("proxy");
      },
    },
  );
  assert.equal(JSON.parse(safeJson(value)).nested.a.b.c.d.e.f.g, "deep");
  assert.equal(JSON.parse(safeJson(value))._meta.native, true);
  assert.equal(reads, 0);
  assert.deepEqual(toSerializableValue([false, 0, "", null]), [
    false,
    0,
    "",
    null,
  ]);
  assert.equal(typeof clientMethodInput, "function");
  assert.equal(typeof resolveClientEntityName, "function");
  assert.equal(typeof resolveClientSpanName, "function");
});

test("activation coalesces, cancellation prevents patches, and native foreign wrappers survive", async () => {
  const original = clientV1.Client.prototype.request;
  const p = new MCPInstrumentor({
    clientModule: clientV1,
    serverModule: serverV1,
  });
  instrumentors.add(p);
  const first = p.activate();
  assert.equal(first, p.activate());
  p.deactivate();
  await first;
  assert.equal(p.isActive(), false);
  assert.equal(clientV1.Client.prototype.request, original);
  await p.activate();
  const instrumented = clientV1.Client.prototype.request;
  const foreign = function (...args) {
    return instrumented.apply(this, args);
  };
  clientV1.Client.prototype.request = foreign;
  p.deactivate();
  assert.equal(clientV1.Client.prototype.request, foreign);
  clientV1.Client.prototype.request = original;
});

test("native registered callbacks stop after deactivation, reactivate, and in-flight spans drain", async () => {
  const env = await environment();
  let resolve;
  env.set(schemas.CallToolRequestSchema, (req) =>
    req.params.name === "wait"
      ? new Promise((r) => {
          resolve = r;
        })
      : { content: [{ type: "text", text: "native" }] },
  );
  await env.connect();
  exporter.reset();
  const pending = env.client.callTool({ name: "wait", arguments: {} });
  await new Promise((r) => setImmediate(r));
  env.p.deactivate();
  resolve({ content: [{ type: "text", text: "native drain" }] });
  await pending;
  assert.equal(toolSpans().length, 2);
  exporter.reset();
  await env.client.callTool({ name: "off", arguments: {} });
  assert.equal(toolSpans().length, 0);
  await env.p.activate();
  await env.client.callTool({ name: "on", arguments: {} });
  assert.equal(toolSpans().length, 2);
  await env.close();
});

test(
  "native high-level McpServer registerTool and legacy tool APIs",
  { skip: minimum ? "SDK1.0.0 has no high-level server/mcp.js" : false },
  async () => {
    const high = await import("@modelcontextprotocol/sdk/server/mcp.js");
    const p = await activate({ serverModule: { ...serverV1, ...high } });
    const server = new high.McpServer({
      name: "native-high-server",
      version: "1",
    });
    server.registerTool("registered", { description: "native" }, async () => ({
      content: [{ type: "text", text: "registered" }],
      structuredContent: { zero: 0 },
    }));
    server.tool("legacy", async () => ({
      content: [{ type: "text", text: "legacy" }],
    }));
    const client = new clientV1.Client({
      name: "native-high-client",
      version: "1",
    });
    const [a, b] = InMemoryTransport.createLinkedPair();
    await server.connect(b);
    await client.connect(a);
    await client.callTool({ name: "registered", arguments: {} });
    await client.callTool({ name: "legacy", arguments: {} });
    assert.equal(toolSpans().length, 4);
    await client.close();
    await server.close();
    p.deactivate();
  },
);

test("native split v2 Client/Server share owners with v1 without duplicate or missing spans", async (t) => {
  let cm, sm;
  try {
    cm = await import("@modelcontextprotocol/client");
    sm = await import("@modelcontextprotocol/server");
  } catch {
    t.skip("split v2 packages are optional for v1-only consumers");
    return;
  }
  const p = await activate({
    clientModules: [clientV1, cm],
    serverModules: [serverV1, sm],
  });
  const q = await activate({ clientModule: cm, serverModule: sm });
  const env = await environment({
    clientModule: cm,
    serverModule: sm,
    Transport: cm.InMemoryTransport,
    plugin: false,
    clientCapabilities: { sampling: {}, elicitation: { form: {} } },
  });
  env.set("tools/call", () => ({
    content: [{ type: "text", text: "native v2" }],
    structuredContent: { flag: false, zero: 0, empty: "" },
  }));
  env.set("tools/list", () => ({
    tools: [{ name: "native_v2", inputSchema: { type: "object" } }],
  }));
  env.client.setRequestHandler("sampling/createMessage", () => ({
    role: "assistant",
    model: "native-v2-client",
    content: { type: "text", text: "sampling v2" },
  }));
  env.client.setRequestHandler("elicitation/create", () => ({
    action: "accept",
    content: { flag: false },
  }));
  await env.connect();
  await env.client.listTools();
  await env.client.callTool({ name: "native_v2", arguments: { zero: 0 } });
  assert.equal(toolSpans().length, 2);
  assert.equal(output(toolSpans()[0]).structuredContent.zero, 0);
  q.deactivate();
  await env.client.callTool({ name: "native_v2", arguments: { zero: 0 } });
  assert.equal(toolSpans().length, 4);
  await env.server.createMessage({
    messages: [{ role: "user", content: { type: "text", text: "input v2" } }],
    maxTokens: 1,
  });
  await env.server.elicitInput({
    message: "native v2 elicitation",
    requestedSchema: {
      type: "object",
      properties: { flag: { type: "boolean" } },
    },
  });
  assert.equal(env.client.experimental, undefined);
  assert.equal(env.server.experimental, undefined);
  await env.close();
  p.deactivate();
});

test(
  "native experimental task stream, status, result, list and cancel preserve iterator identity",
  { skip: minimum ? "SDK1.0.0 has no experimental task API" : false },
  async () => {
    const { InMemoryTaskStore, InMemoryTaskMessageQueue } =
      await import("@modelcontextprotocol/sdk/experimental/tasks/stores/in-memory.js");
    const p = await activate();
    const store = new InMemoryTaskStore();
    const server = new serverV1.Server(
      { name: "native-task-server", version: "1" },
      {
        capabilities: {
          tools: {},
          tasks: { list: {}, cancel: {}, requests: { tools: { call: {} } } },
        },
        taskStore: store,
        taskMessageQueue: new InMemoryTaskMessageQueue(),
      },
    );
    server.setRequestHandler(schemas.ListToolsRequestSchema, () => ({
      tools: [
        {
          name: "native_task",
          inputSchema: { type: "object" },
          execution: { taskSupport: "required" },
        },
      ],
    }));
    server.setRequestHandler(
      schemas.CallToolRequestSchema,
      async (req, extra) => {
        const task = await extra.taskStore.createTask({
          ttl: 60000,
          pollInterval: 1,
        });
        await extra.taskStore.storeTaskResult(task.taskId, "completed", {
          content: [{ type: "text", text: "native task result" }],
          structuredContent: { flag: false, zero: 0 },
        });
        return { task };
      },
    );
    const client = new clientV1.Client({
      name: "native-task-client",
      version: "1",
    });
    const [a, b] = InMemoryTransport.createLinkedPair();
    await server.connect(b);
    await client.connect(a);
    await client.listTools();
    const iterator = client.experimental.tasks.callToolStream({
      name: "native_task",
      arguments: { zero: 0 },
    });
    const nativeNext = iterator.next;
    assert.equal(iterator[Symbol.asyncIterator](), iterator);
    const messages = [];
    for await (const message of iterator) messages.push(message);
    assert.equal(iterator.next, nativeNext);
    assert.deepEqual(
      messages.map((m) => m.type),
      ["taskCreated", "taskStatus", "result"],
    );
    const taskId = messages[0].task.taskId;
    assert.equal(messages.at(-1).result.structuredContent.zero, 0);
    assert.equal(
      (await client.experimental.tasks.getTask(taskId)).status,
      "completed",
    );
    assert.equal(
      (
        await client.experimental.tasks.getTaskResult(
          taskId,
          schemas.CallToolResultSchema,
        )
      ).content[0].text,
      "native task result",
    );
    assert.equal((await client.experimental.tasks.listTasks()).tasks.length, 1);
    const pending = await store.createTask({ ttl: 60000 }, 9000, {
      method: "tools/call",
      params: { name: "native_task" },
    });
    assert.equal(
      (await client.experimental.tasks.cancelTask(pending.taskId)).status,
      "cancelled",
    );
    for (const method of [
      "tasks/get",
      "tasks/result",
      "tasks/list",
      "tasks/cancel",
    ]) {
      const spans = exporter
        .getFinishedSpans()
        .filter((s) => s.attributes["mcp.method"] === method);
      assert.ok(spans.length >= 2);
      for (const span of spans)
        assert.equal(span.attributes["respan.entity.log_type"], "task");
    }
    await client.close();
    await server.close();
    store.cleanup();
    p.deactivate();
  },
);

test("native dropped client ancestor does not revoke sampled server payload admission", async () => {
  const env = await environment();
  env.set(schemas.CallToolRequestSchema, () => ({
    content: [{ type: "text", text: "sampled native child" }],
  }));
  await env.connect();
  exporter.reset();
  chooseSampling = (name, kind) =>
    name === "mcp.tool.sampled_child"
      ? kind === 2
        ? SamplingDecision.NOT_RECORD
        : SamplingDecision.RECORD_AND_SAMPLED
      : SamplingDecision.RECORD_AND_SAMPLED;
  await env.client.callTool({ name: "sampled_child", arguments: { value: 0 } });
  const spans = toolSpans();
  assert.equal(spans.length, 1);
  assert.equal(input(spans[0]).arguments.value, 0);
  assert.equal(output(spans[0]).content[0].text, "sampled native child");
  await env.close();
});

test("native actual-span and external ancestor attribute vetoes remain immutable", async () => {
  const env = await environment({ clientCapabilities: { sampling: {} } });
  env.client.setRequestHandler(schemas.CreateMessageRequestSchema, () => ({
    role: "assistant",
    model: "native",
    content: { type: "text", text: "private descendant" },
  }));
  env.set(schemas.CallToolRequestSchema, async (request) => {
    if (request.params.name === "own_veto") {
      const span = trace.getSpan(context.active());
      span.setAttribute("allow_trace_content", false);
      span.setAttribute("allow_trace_content", true);
      span.setAttribute("unknown.private", "private");
      await env.server.createMessage({
        messages: [
          {
            role: "user",
            content: { type: "text", text: "private descendant input" },
          },
        ],
        maxTokens: 1,
      });
    }
    return { content: [{ type: "text", text: "native private result" }] };
  });
  await env.connect();
  exporter.reset();
  await env.client.callTool({
    name: "own_veto",
    arguments: { secret: "input" },
  });
  const ownSpan = toolSpans().find(
    (span) => span.attributes["mcp.operation"] === "server.tools/call",
  );
  assert.equal(ownSpan.attributes["traceloop.entity.input"], undefined);
  assert.equal(ownSpan.attributes["traceloop.entity.output"], undefined);
  assert.equal(ownSpan.attributes["unknown.private"], undefined);
  for (const span of exporter
    .getFinishedSpans()
    .filter(
      (span) => span.attributes["mcp.method"] === "sampling/createMessage",
    )) {
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
    assert.equal(span.attributes["traceloop.entity.output"], undefined);
  }
  exporter.reset();
  const parent = trace
    .getTracer("native-external-parent")
    .startSpan("external-parent");
  parent.setAttribute("allow_trace_content", false);
  await context.with(trace.setSpan(context.active(), parent), () =>
    env.client.callTool({
      name: "parent_veto",
      arguments: { secret: "input" },
    }),
  );
  parent.setAttribute("allow_trace_content", true);
  await context.with(trace.setSpan(context.active(), parent), () =>
    env.client.callTool({
      name: "parent_veto",
      arguments: { secret: "still input" },
    }),
  );
  for (const span of toolSpans()) {
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
    assert.equal(span.attributes["traceloop.entity.output"], undefined);
  }
  parent.end();
  await env.close();
});

test("native RESPAN_TRACE_CONTENT false remains a constructor and operation ceiling", async () => {
  process.env.RESPAN_TRACE_CONTENT = "false";
  const p = await activate();
  process.env.RESPAN_TRACE_CONTENT = "true";
  const env = await environment({ plugin: false });
  env.set(schemas.CallToolRequestSchema, () => ({
    content: [{ type: "text", text: "private environment" }],
  }));
  await env.connect();
  await env.client.callTool({
    name: "private_environment",
    arguments: { secret: "private" },
  });
  for (const span of toolSpans()) {
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
    assert.equal(span.attributes["traceloop.entity.output"], undefined);
  }
  await env.close();
  p.deactivate();
  delete process.env.RESPAN_TRACE_CONTENT;
});

test("native independent lanes reject ambiguous late attributes events and status text", async () => {
  const p = await activate({ recordInputs: false });
  const env = await environment({ plugin: false });
  env.set(schemas.CallToolRequestSchema, () => {
    const span = trace.getSpan(context.active());
    span.setAttribute("unknown.private.input", "private");
    span.setAttributes({
      "unknown.private.replacement": {
        get secret() {
          throw new Error("must not inspect");
        },
      },
    });
    span.addEvent("private input event", { private: "payload" });
    span.setStatus({ code: 2, message: "private input status" });
    return { content: [{ type: "text", text: "visible native output" }] };
  });
  await env.connect();
  await env.client.callTool({
    name: "lanes_private",
    arguments: { secret: "private" },
  });
  const span = toolSpans().find(
    (span) => span.attributes["mcp.operation"] === "server.tools/call",
  );
  assert.equal(span.attributes["unknown.private.input"], undefined);
  assert.equal(span.attributes["unknown.private.replacement"], undefined);
  assert.deepEqual(span.events, []);
  assert.equal(span.status.message, undefined);
  assert.equal(output(span).content[0].text, "visible native output");
  await env.close();
  p.deactivate();
});
