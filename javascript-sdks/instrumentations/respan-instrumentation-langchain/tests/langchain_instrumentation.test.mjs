import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import test from "node:test";
const requireNative = createRequire(
  process.env.LANGCHAIN_NATIVE_PROFILE
    ? `${process.env.LANGCHAIN_NATIVE_PROFILE}/package.json`
    : import.meta.url,
);
const load = (specifier) =>
  import(pathToFileURL(requireNative.resolve(specifier)).href);
const { context, trace, SpanStatusCode, TraceFlags } =
  await load("@opentelemetry/api");
const { suppressTracing } = await load("@opentelemetry/core");
const { NodeTracerProvider } = await load("@opentelemetry/sdk-trace-node");
const { CONTEXT_KEY_ALLOW_TRACE_CONTENT } = await load(
  "@traceloop/ai-semantic-conventions",
);
const { RunnableLambda, RunnableSequence } = await load(
  "@langchain/core/runnables",
);
const { DynamicTool } = await load("@langchain/core/tools");
const { HumanMessage, AIMessage } = await load("@langchain/core/messages");
const { ChatOpenAI } = await load("@langchain/openai");
const {
  RespanCallbackHandler,
  LangChainInstrumentor,
  addRespanCallback,
  getCallbackHandler,
} = await import(
  process.env.LANGCHAIN_INSTRUMENTATION_MODULE ?? "../dist/index.js"
);
let samplerOverride;
let decision = 2,
  calls = 0,
  onStart;
const captured = [];
const provider = new NodeTracerProvider({
  sampler: {
    shouldSample(_ctx, _id, _name, _kind, attrs) {
      calls++;
      assert.equal(attrs["traceloop.entity.input"], undefined);
      return { decision: samplerOverride ? samplerOverride(_name) : decision };
    },
    toString() {
      return "ControlledSampler";
    },
  },
  spanProcessors: [
    {
      onStart(span) {
        onStart?.(span);
      },
      onEnd(span) {
        captured.push(span);
      },
      forceFlush: async () => {},
      shutdown: async () => {},
    },
  ],
});
provider.register();
function reset() {
  captured.length = 0;
  decision = 2;
  calls = 0;
  onStart = undefined;
  samplerOverride = undefined;
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
}
function handler(options = {}) {
  return new RespanCallbackHandler(options);
}
function fixtureModel({
  tools = false,
  error = false,
  usage = true,
  fetchHook,
} = {}) {
  const fetch = async (_url, init) => {
    const body = JSON.parse(init.body);
    fetchHook?.(body);
    if (error)
      return new Response(
        JSON.stringify({
          error: {
            message: "controlled native failure",
            type: "invalid_request_error",
          },
        }),
        { status: 429, headers: { "content-type": "application/json" } },
      );
    const response = {
      id: "chatcmpl-fixture",
      object: "chat.completion",
      created: 0,
      model: "fixture-model",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: tools ? "" : "fixture result",
            ...(tools
              ? {
                  tool_calls: [
                    {
                      id: "call_native_fixture",
                      type: "function",
                      function: { name: "lookup", arguments: '{"value":0}' },
                    },
                  ],
                }
              : {}),
          },
          finish_reason: tools ? "tool_calls" : "stop",
        },
      ],
      ...(usage
        ? { usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } }
        : {}),
    };
    if (body.stream) {
      const chunks = [
        {
          id: response.id,
          object: "chat.completion.chunk",
          created: 0,
          model: response.model,
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "fixture " },
              finish_reason: null,
            },
          ],
        },
        {
          id: response.id,
          object: "chat.completion.chunk",
          created: 0,
          model: response.model,
          choices: [
            { index: 0, delta: { content: "result" }, finish_reason: "stop" },
          ],
        },
        {
          id: response.id,
          object: "chat.completion.chunk",
          created: 0,
          model: response.model,
          choices: [],
          usage: response.usage,
        },
      ];
      return new Response(
        chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") +
          "data: [DONE]\n\n",
        { headers: { "content-type": "text/event-stream" } },
      );
    }
    return new Response(JSON.stringify(response), {
      headers: { "content-type": "application/json" },
    });
  };
  return new ChatOpenAI({
    model: "fixture-model",
    modelName: "fixture-model",
    apiKey: "fixture-local-only",
    openAIApiKey: "fixture-local-only",
    maxRetries: 0,
    configuration: { baseURL: "http://fixture.invalid/v1", fetch },
  });
}
function attrs(type) {
  return captured.find((s) => s.attributes["respan.entity.log_type"] === type)
    ?.attributes;
}
const noAliases = (a) => {
  for (const key of [
    "status_code",
    "traceloop.span.kind",
    "gen_ai.tool.name",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "respan.span.tools",
    "respan.span.tool_calls",
    "tools",
    "tool_calls",
  ])
    assert.equal(a[key], undefined, key);
};
test("public exports, callback deduplication, and instrumentor lifecycle", async () => {
  reset();
  const inst = new LangChainInstrumentor();
  assert.equal(inst.isActive(), false);
  inst.activate();
  assert.equal(inst.isActive(), true);
  const cfg = inst.addCallback({ tags: ["native"] });
  assert.equal(
    addRespanCallback(cfg, getCallbackHandler()).callbacks.length,
    1,
  );
  await RunnableLambda.from((x) => x).invoke(false, cfg);
  assert.equal(captured.length, 1);
  assert.equal(
    JSON.parse(attrs("workflow")["traceloop.entity.output"]).output,
    false,
  );
  inst.deactivate();
  await RunnableLambda.from((x) => x).invoke(false, cfg);
  assert.equal(captured.length, 1);
  inst.activate();
  await RunnableLambda.from((x) => x).invoke(null, cfg);
  assert.equal(captured.length, 2);
});
test("released RunnableSequence preserves result identity and actual parent contexts", async () => {
  reset();
  const value = { answer: 0 };
  const first = RunnableLambda.from(() => value);
  const second = RunnableLambda.from((x) => x);
  const sequence = RunnableSequence.from([first, second]);
  const result = await sequence.invoke("native input", {
    callbacks: [handler()],
    runName: "native_workflow",
  });
  assert.equal(result, value);
  const root = captured.find((s) => s.name === "native_workflow");
  assert.ok(root);
  assert.equal(root.attributes["traceloop.entity.path"], "");
  const children = captured.filter((s) => s !== root);
  assert.equal(children.length, 2);
  for (const child of children) {
    assert.equal(child.parentSpanContext.spanId, root.spanContext().spanId);
    assert.equal(child.spanContext().traceId, root.spanContext().traceId);
    noAliases(child.attributes);
  }
});
for (const sampling of [0, 1])
  test(`native RunnableLambda sampler decision ${sampling} prevents telemetry traversal`, async () => {
    reset();
    decision = sampling;
    let reads = 0;
    const payload = {
      toJSON() {
        reads++;
        return { secret: "no" };
      },
      get secret() {
        reads++;
        return "no";
      },
    };
    await RunnableLambda.from((x) => x).invoke(payload, {
      callbacks: [handler()],
    });
    assert.equal(calls, 1);
    assert.equal(reads, 0);
    assert.equal(captured.length, sampling === 1 ? 1 : 0);
    if (captured[0])
      assert.equal(captured[0].attributes["traceloop.entity.input"], undefined);
  });
test("disabled payload capture never calls getters, toJSON, or proxy traps", async () => {
  reset();
  let reads = 0;
  const value = {
    toJSON() {
      reads++;
      return "secret";
    },
    get secret() {
      reads++;
      return "secret";
    },
  };
  const cb = handler({ includeContent: false, includeMetadata: false });
  await RunnableLambda.from((x) => x).invoke(value, { callbacks: [cb] });
  assert.equal(reads, 0);
  cb.includeContent = true;
  await RunnableLambda.from((x) => x).invoke(value, { callbacks: [cb] });
  assert.equal(reads, 0);
  for (const span of captured) {
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
    assert.equal(span.attributes["traceloop.entity.output"], undefined);
  }
});
test("independent input and output lanes preserve false, zero, empty and null", async () => {
  reset();
  await RunnableLambda.from(() => false).invoke(0, {
    callbacks: [handler({ recordInputs: false })],
  });
  let a = attrs("workflow");
  assert.equal(a["traceloop.entity.input"], undefined);
  assert.equal(JSON.parse(a["traceloop.entity.output"]).output, false);
  reset();
  await RunnableLambda.from(() => null).invoke("", {
    callbacks: [handler({ recordOutputs: false })],
  });
  a = attrs("workflow");
  assert.equal(JSON.parse(a["traceloop.entity.input"]).input, "");
  assert.equal(a["traceloop.entity.output"], undefined);
  reset();
  await RunnableLambda.from(() => null).invoke(0, { callbacks: [handler()] });
  assert.equal(
    JSON.parse(attrs("workflow")["traceloop.entity.output"]).output,
    null,
  );
});
test("constructor, env, canonical context and suppression remain ceilings", async () => {
  reset();
  const cb = handler();
  process.env.RESPAN_TRACE_CONTENT = "false";
  await RunnableLambda.from((x) => x).invoke(
    { secret: 1 },
    { callbacks: [cb] },
  );
  assert.equal(attrs("workflow")["traceloop.entity.input"], undefined);
  reset();
  await context.with(
    context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
    () =>
      RunnableLambda.from((x) => x).invoke({ secret: 1 }, { callbacks: [cb] }),
  );
  assert.equal(attrs("workflow")["traceloop.entity.input"], undefined);
  reset();
  await context.with(suppressTracing(context.active()), () =>
    RunnableLambda.from((x) => x).invoke({ secret: 1 }, { callbacks: [cb] }),
  );
  assert.equal(calls, 0);
  assert.equal(captured.length, 0);
});
test("late actual RecordingSpan replacements stay private and status only accepts numeric codes", async () => {
  reset();
  const cb = handler({ includeContent: false });
  let raw;
  onStart = (span) => {
    raw = span;
  };
  await RunnableLambda.from((x) => x).invoke("secret", { callbacks: [cb] });
  let reads = 0;
  raw.setAttributes({
    "gen_ai.usage.secret": {
      get payload() {
        reads++;
        return "secret";
      },
    },
    "gen_ai.usage.input_tokens": 0,
  });
  raw.attributes = {
    "traceloop.entity.input": "secret",
    "gen_ai.usage.secret": "secret",
    "gen_ai.usage.input_tokens": 0,
  };
  raw.events = [{ name: "secret" }];
  raw.status = { code: "2", message: "secret" };
  assert.equal(raw.attributes["traceloop.entity.input"], undefined);
  assert.equal(raw.attributes["gen_ai.usage.secret"], undefined);
  assert.equal(raw.attributes["gen_ai.usage.input_tokens"], 0);
  assert.deepEqual(raw.events, []);
  assert.deepEqual(raw.status, { code: 0 });
  assert.equal(reads, 0);
  assert.throws(() =>
    Object.defineProperty(raw, "attributes", { value: { secret: 1 } }),
  );
});
test("late observed ancestor veto cannot be restored and exporter suppression cannot revoke spans", async () => {
  reset();
  const parent = trace.getTracer("native-parent").startSpan("parent");
  parent.setAttribute("allow_trace_content", false);
  await context.with(trace.setSpan(context.active(), parent), () =>
    RunnableLambda.from((x) => x).invoke("private", { callbacks: [handler()] }),
  );
  parent.setAttribute("allow_trace_content", true);
  await context.with(trace.setSpan(context.active(), parent), () =>
    RunnableLambda.from((x) => x).invoke("still private", {
      callbacks: [handler()],
    }),
  );
  for (const span of captured)
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
  reset();
  await RunnableLambda.from((x) => x).invoke("visible", {
    callbacks: [handler()],
  });
  const span = captured[0];
  context.with(suppressTracing(context.active()), () =>
    assert.equal(
      JSON.parse(span.attributes["traceloop.entity.input"]).input,
      "visible",
    ),
  );
  parent.end();
});
test("released ChatOpenAI native HTTP transport maps input, output, provider usage and full tools", async () => {
  reset();
  let request;
  const model = fixtureModel({ tools: true, fetchHook: (x) => (request = x) });
  const tools = Array.from({ length: 80 }, (_, i) => ({
    type: "function",
    function: {
      name: i === 0 ? "lookup" : `lookup_${i}`,
      description: "fixture",
      parameters: { type: "object", properties: { value: { type: "number" } } },
    },
  }));
  const messages = Array.from(
    { length: 80 },
    (_, i) =>
      new HumanMessage(
        i === 79
          ? [
              { type: "text", text: "last message" },
              {
                type: "image_url",
                image_url: { url: "data:image/png;base64,AAAA" },
              },
            ]
          : `message_${i}`,
      ),
  );
  const response = await model
    .bindTools(tools)
    .invoke(messages, { callbacks: [handler()] });
  assert.equal(response.tool_calls[0].id, "call_native_fixture");
  assert.equal(request.tools.length, 80);
  const a = attrs("chat");
  assert.equal(JSON.parse(a["llm.request.functions"]).length, 80);
  assert.equal(JSON.parse(a["traceloop.entity.input"])[0].length, 80);
  assert.equal(
    JSON.parse(a["traceloop.entity.output"])[0][0].tool_calls[0].id,
    "call_native_fixture",
  );
  assert.equal(a["gen_ai.completion.0.role"], "assistant");
  const min = requireNative("@langchain/core/package.json").version === "0.2.0";
  assert.equal(a["gen_ai.usage.input_tokens"], min ? undefined : 0);
  assert.equal(a["llm.usage.total_tokens"], min ? undefined : 0);
  assert.equal(a["gen_ai.request.model"], "fixture-model");
  noAliases(a);
});
test("released ChatOpenAI native SSE preserves chunks, stream methods and final text", async () => {
  reset();
  const stream = await fixtureModel().stream("native stream", {
    callbacks: [handler()],
  });
  assert.equal(typeof stream.next, "function");
  assert.equal(typeof stream.return, "function");
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  assert.equal(chunks.map((c) => c.content).join(""), "fixture result");
  const a = attrs("chat");
  assert.ok(a);
  assert.equal(
    JSON.parse(a["traceloop.entity.output"])[0][0].content,
    "fixture result",
  );
});
test("released model errors keep native identity and never fabricate success/status/output", async () => {
  reset();
  const cb = handler();
  let callbackError;
  const observer = {
    name: "observer",
    handleLLMError(error) {
      callbackError = error;
    },
  };
  await assert.rejects(
    fixtureModel({ error: true }).invoke("native error", {
      callbacks: [cb, observer],
    }),
    (error) => error === callbackError,
  );
  const span = captured[0];
  assert.equal(span.status.code, SpanStatusCode.ERROR);
  assert.equal(span.attributes["http.response.status_code"], 429);
  assert.equal(span.attributes["traceloop.entity.output"], undefined);
  assert.equal(span.attributes["gen_ai.completion.0.content"], undefined);
  noAliases(span.attributes);
});
test("released tools preserve actual execution input/output and callback invocation ID when supported", async () => {
  reset();
  const tool = new DynamicTool({
    name: "lookup",
    description: "fixture",
    func: async () => "",
  });
  const version = requireNative("@langchain/core/package.json").version;
  if (version.startsWith("0.2."))
    await tool.invoke("native argument", { callbacks: [handler()] });
  else
    await tool.invoke(
      {
        type: "tool_call",
        id: "call_native_execution",
        name: "lookup",
        args: {},
      },
      { callbacks: [handler()] },
    );
  const a = attrs("tool");
  assert.equal(JSON.parse(a["traceloop.entity.input"]).name, "lookup");
  assert.equal(a["traceloop.entity.output"], '""');
  if (!version.startsWith("0.2."))
    assert.equal(a["gen_ai.tool.call.id"], "call_native_execution");
  noAliases(a);
});
test("native RunnableLambda failure preserves exact error and drain after deactivation", async () => {
  reset();
  const error = new Error("native runnable failure");
  await assert.rejects(
    RunnableLambda.from(() => {
      throw error;
    }).invoke("x", { callbacks: [handler()] }),
    (value) => value === error,
  );
  assert.equal(captured[0].status.code, 2);
  reset();
  let release;
  const wait = new Promise((resolve) => (release = resolve));
  const inst = new LangChainInstrumentor();
  inst.activate();
  const result = RunnableLambda.from(async () => {
    await wait;
    return 0;
  }).invoke("x", inst.addCallback());
  await new Promise((resolve) => setImmediate(resolve));
  inst.deactivate();
  release();
  assert.equal(await result, 0);
  assert.equal(captured.length, 1);
  assert.equal(
    JSON.parse(captured[0].attributes["traceloop.entity.output"]).output,
    0,
  );
});
test("shared callback owners stay active until the final deactivation", async () => {
  reset();
  const cb = handler();
  const first = new LangChainInstrumentor({ callbackHandler: cb });
  first.activate();
  const second = new LangChainInstrumentor({ callbackHandler: cb });
  await RunnableLambda.from((x) => x).invoke(1, first.addCallback());
  assert.equal(captured.length, 1);
  second.activate();
  first.deactivate();
  await RunnableLambda.from((x) => x).invoke(2, second.addCallback());
  assert.equal(captured.length, 2);
  second.deactivate();
  await RunnableLambda.from((x) => x).invoke(3, second.addCallback());
  assert.equal(captured.length, 2);
});
test("a non-recording parent does not preempt the actual child sampler", async () => {
  reset();
  decision = 0;
  const parent = trace.getTracer("parent").startSpan("dropped_parent");
  decision = 2;
  await context.with(trace.setSpan(context.active(), parent), () =>
    RunnableLambda.from((x) => x).invoke("admitted child", {
      callbacks: [handler()],
    }),
  );
  assert.equal(calls, 2);
  assert.ok(attrs("workflow")["traceloop.entity.input"]);
});
test("late guarded parent veto removes captured descendants without restoration", async () => {
  reset();
  const cb = handler();
  let root;
  onStart = (span) => {
    if (span.name === "late_parent") root = span;
  };
  const child = RunnableLambda.from((x) => {
    root.setAttribute("allow_trace_content", false);
    return x;
  });
  await RunnableSequence.from([RunnableLambda.from((x) => x), child]).invoke(
    "private after start",
    { callbacks: [cb], runName: "late_parent" },
  );
  root.setAttribute("allow_trace_content", true);
  for (const span of captured) {
    assert.equal(span.attributes["traceloop.entity.input"], undefined);
    assert.equal(span.attributes["traceloop.entity.output"], undefined);
  }
});
test("native missing provider usage stays absent and partial streams never fabricate errors", async () => {
  reset();
  await fixtureModel({ usage: false }).invoke("no usage", {
    callbacks: [handler()],
  });
  const a = attrs("chat");
  assert.equal(a["gen_ai.usage.input_tokens"], undefined);
  assert.equal(a["llm.usage.total_tokens"], undefined);
  reset();
  const inst = new LangChainInstrumentor();
  inst.activate();
  let pending;
  onStart = (span) => {
    pending = span;
  };
  const stream = await fixtureModel().stream(
    "cancel stream",
    inst.addCallback(),
  );
  const first = await stream.next();
  assert.equal(first.done, false);
  await stream.return();
  for (const span of captured)
    assert.equal(span.attributes.status_code, undefined);
});
test("released tool output preserves a 5001-value vector without translator truncation", async () => {
  reset();
  const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
  const tool = new DynamicTool({
    name: "vector",
    description: "fixture",
    func: async () => JSON.stringify(vector),
  });
  await tool.invoke("x", { callbacks: [handler()] });
  const text = JSON.parse(attrs("tool")["traceloop.entity.output"]);
  assert.equal(JSON.parse(text).length, 5001);
});
test("a dropped native callback parent does not revoke a child sampled by the real sampler", async () => {
  reset();
  samplerOverride = (name) => (name === "dropped_native_parent" ? 0 : 2);
  await RunnableSequence.from([
    RunnableLambda.from((x) => x),
    RunnableLambda.from((x) => x),
  ]).invoke("sampled child payload", {
    callbacks: [handler()],
    runName: "dropped_native_parent",
  });
  assert.equal(calls, 3);
  assert.equal(captured.length, 2);
  for (const span of captured)
    assert.ok(span.attributes["traceloop.entity.input"]);
});
test("late canonical context veto on the actual span is immutable", async () => {
  reset();
  let actual;
  onStart = (span) => {
    actual = span;
  };
  await RunnableLambda.from((value) => {
    context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
      () => actual.setAttribute("late.payload", "controlled late payload"),
    );
    context.with(
      context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
      () => actual.setAttribute("traceloop.entity.input", "cannot restore"),
    );
    return value;
  }).invoke("initial capture", { callbacks: [handler()] });
  assert.equal(captured.length, 1);
  assert.equal(actual.attributes["traceloop.entity.input"], undefined);
  assert.equal(actual.attributes["traceloop.entity.output"], undefined);
  assert.equal(actual.attributes["late.payload"], undefined);
});
test.after(() => provider.shutdown());
