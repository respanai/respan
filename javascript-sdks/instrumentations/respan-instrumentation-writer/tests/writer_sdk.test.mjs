import assert from "node:assert/strict";
import test from "node:test";
import { z } from "zod";
import {
  context,
  trace,
  ROOT_CONTEXT,
  createContextKey,
} from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import {
  BasicTracerProvider,
  AlwaysOffSampler,
  SimpleSpanProcessor,
  InMemorySpanExporter,
  BatchSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { WriterInstrumentor } from "../dist/index.js";
const sdk = await import(process.env.WRITER_TEST_PACKAGE || "writer-sdk");
const spans = [];
const originalProvider = trace.getTracerProvider;
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
const captureProvider = new BasicTracerProvider({
  spanProcessors: [
    {
      onStart() {},
      onEnd(span) {
        spans.push(span);
      },
      forceFlush: async () => {},
      shutdown: async () => {},
    },
  ],
});
test.before(() => {
  trace.getTracerProvider = () => captureProvider;
});
test.after(() => {
  trace.getTracerProvider = originalProvider;
  context.disable();
});
const request = {
  model: "request-model",
  messages: [{ role: "user", content: "fixture" }],
};
const response = {
  id: "native-id",
  object: "chat.completion",
  created: 1,
  model: "response-model",
  choices: [
    {
      index: 0,
      finish_reason: "stop",
      message: { role: "assistant", content: "native text", refusal: null },
    },
  ],
  usage: { prompt_tokens: 0, completion_tokens: 2, total_tokens: 2 },
};
function transport(input, init) {
  const body = JSON.parse(init?.body || "{}");
  if (body.model === "error")
    return Promise.resolve(
      Response.json({ error: { message: "native failure" } }, { status: 429 }),
    );
  if (body.model === "network")
    return Promise.reject(new Error("native network"));
  if (body.stream) {
    const chunks = [
      {
        id: "stream-id",
        model: "response-model",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "native " },
            finish_reason: null,
          },
        ],
      },
      {
        id: "stream-id",
        model: "response-model",
        choices: [
          { index: 0, delta: { content: "stream" }, finish_reason: "stop" },
        ],
        usage: response.usage,
      },
    ];
    return Promise.resolve(
      new Response(
        chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") +
          "data: [DONE]\n\n",
        { headers: { "content-type": "text/event-stream" } },
      ),
    );
  }
  const r = body.response_format
    ? {
        ...response,
        choices: [
          {
            ...response.choices[0],
            message: {
              role: "assistant",
              content:
                body.model === "bad-json"
                  ? "not JSON"
                  : '{"zero":0,"flag":false,"empty":"","nil":null}',
            },
          },
        ],
      }
    : response;
  if (String(input).includes("completions"))
    return Promise.resolve(
      Response.json(
        {
          model: "response-completion",
          choices: [{ text: "native completion" }],
          usage: response.usage,
        },
        { status: 201 },
      ),
    );
  return Promise.resolve(Response.json(r, { status: 201 }));
}
function client(fetch = transport) {
  return new sdk.default({ apiKey: "fixture-only", maxRetries: 0, fetch });
}
async function instrument(options = {}) {
  const i = new WriterInstrumentor({ sdkModule: sdk, ...options });
  await i.activate();
  return i;
}
function last() {
  return spans.at(-1);
}
function noContent(span) {
  const a = span.attributes;
  assert.equal(a["traceloop.entity.input"], undefined);
  assert.equal(a["traceloop.entity.output"], undefined);
  assert.equal(a["llm.request.functions"], undefined);
  assert.ok(
    !Object.keys(a).some((k) =>
      /^(gen_ai\.(prompt|completion)\.|respan.metadata|error\.|exception\.)/.test(
        k,
      ),
    ),
  );
  assert.deepEqual(span.events, []);
  assert.equal(span.status.message, undefined);
}

test("native APIPromise identity, lazy parsing, raw Response and withResponse data identities", async () => {
  const c = client(),
    post = c.post;
  let raw;
  c.post = function (...args) {
    return (raw = post.apply(this, args));
  };
  const i = await instrument();
  try {
    const p = c.chat.chat(request);
    assert.equal(p, raw);
    assert.ok(p instanceof Promise);
    assert.equal(p.parsedPromise, undefined);
    const before = spans.length;
    const rawResponse = await p.asResponse();
    assert.equal(rawResponse.status, 201);
    assert.equal(rawResponse.bodyUsed, false);
    assert.equal(spans.length, before + 1);
    assert.equal(last().attributes["traceloop.entity.output"], undefined);
    assert.equal(last().attributes["http.response.status_code"], 201);
    const next = c.chat.chat(request);
    const result = await next.withResponse();
    assert.equal(result.data, await next);
    assert.equal(result.response, await next.asResponse());
    assert.equal(result.data.choices[0].message.content, "native text");
    assert.equal(last().attributes["gen_ai.usage.input_tokens"], 0);
    assert.equal(last().attributes["gen_ai.response.model"], "response-model");
  } finally {
    i.deactivate();
  }
});
test("initiating parent and concurrent workflow contexts survive delayed consumption", async () => {
  const c = client(),
    i = await instrument();
  try {
    const parents = ["a", "b"].map((x) =>
      trace.wrapSpanContext({
        traceId: x.repeat(32),
        spanId: x.repeat(16),
        traceFlags: 1,
        isRemote: true,
      }),
    );
    const before = spans.length;
    const pending = parents.map((parent) =>
      context.with(trace.setSpan(ROOT_CONTEXT, parent), () =>
        c.chat.chat(request),
      ),
    );
    await Promise.all(pending);
    const done = spans.slice(before);
    assert.deepEqual(
      new Set(done.map((s) => s.parentSpanContext.spanId)),
      new Set(parents.map((p) => p.spanContext().spanId)),
    );
    assert.equal(done.length, 2);
  } finally {
    i.deactivate();
  }
});
test("historical tools stay input; only this response supplies current tool calls", async () => {
  const c = client(async () =>
      Response.json({
        ...response,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "current",
                  type: "function",
                  function: { name: "new", arguments: "" },
                },
              ],
            },
          },
        ],
      }),
    ),
    i = await instrument();
  try {
    const before = spans.length;
    await c.chat.chat({
      ...request,
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "history",
              type: "function",
              function: { name: "old", arguments: '{"flag":false}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "history", content: "history result" },
      ],
    });
    assert.equal(spans.length, before + 1);
    assert.equal(last().attributes["respan.entity.log_type"], "chat");
    assert.equal(
      JSON.parse(last().attributes["gen_ai.prompt.0.tool_calls"])[0].id,
      "history",
    );
    assert.equal(
      JSON.parse(last().attributes["gen_ai.completion.0.tool_calls"])[0].id,
      "current",
    );
  } finally {
    i.deactivate();
  }
});
test("native SSE stream, iterator results, early return and tee preserve native resources", async () => {
  const c = client(),
    i = await instrument();
  try {
    let before = spans.length;
    const p = c.chat.chat({ ...request, stream: true });
    assert.equal(typeof p.withResponse, "function");
    const stream = await p;
    const controller = stream.controller;
    assert.equal(spans.length, before);
    const iterator = stream[Symbol.asyncIterator]();
    assert.equal(iterator[Symbol.asyncIterator](), iterator);
    assert.equal(
      Object.prototype.toString.call(iterator),
      "[object AsyncGenerator]",
    );
    const item = await iterator.next();
    assert.equal(item.value.choices[0].delta.content, "native ");
    await iterator.return("native return");
    assert.equal(stream.controller, controller);
    assert.equal(controller.signal.aborted, true);
    assert.equal(spans.length, before + 1);
    assert.equal(last().attributes["gen_ai.completion.0.content"], "native ");
    before = spans.length;
    const split = await c.chat.chat({ ...request, stream: true });
    const [left, right] = split.tee();
    const collect = async (x) => {
      const chunks = [];
      for await (const chunk of x) chunks.push(chunk);
      return chunks;
    };
    const [l, r] = await Promise.all([collect(left), collect(right)]);
    assert.equal(l.length, 2);
    assert.equal(l[0], r[0]);
    assert.equal(spans.length, before + 1);
    assert.equal(
      last().attributes["gen_ai.completion.0.content"],
      "native stream",
    );
  } finally {
    i.deactivate();
  }
});
test("native stream cancellation and transport errors retain exact outcomes", async () => {
  const c = client(),
    i = await instrument();
  try {
    let before = spans.length;
    const stream = await c.chat.chat({ ...request, stream: true });
    stream.controller.abort();
    for await (const chunk of stream) {
      void chunk;
    }
    assert.equal(spans.length, before + 1);
    const promise = c.chat.chat({ ...request, model: "error" });
    let actual;
    try {
      await promise;
    } catch (e) {
      actual = e;
    }
    assert.ok(actual instanceof sdk.APIError);
    await assert.rejects(promise, (e) => e === actual);
    assert.equal(last().status.code, 2);
    assert.equal(last().attributes["http.response.status_code"], 429);
    assert.match(last().attributes["error.message"], /native failure/);
    await assert.rejects(c.chat.chat({ ...request, model: "network" }));
    assert.equal(last().attributes["http.response.status_code"], undefined);
  } finally {
    i.deactivate();
  }
});
test(
  "native structured parse preserves parsed fields and parse failures",
  { skip: typeof client().chat.parse !== "function" },
  async () => {
    const { zodResponseFormat } = await import(
      `${process.env.WRITER_TEST_PACKAGE || "writer-sdk"}/helpers/zod`
    );
    const format = zodResponseFormat(
      z.object({
        zero: z.number(),
        flag: z.boolean(),
        empty: z.string(),
        nil: z.null(),
      }),
      "fixture",
    );
    const c = client(),
      i = await instrument();
    try {
      let before = spans.length;
      const r = await c.chat.parse({ ...request, response_format: format });
      assert.deepEqual(r.choices[0].message.parsed, {
        zero: 0,
        flag: false,
        empty: "",
        nil: null,
      });
      assert.equal(spans.length, before + 1);
      assert.deepEqual(
        JSON.parse(last().attributes["traceloop.entity.output"])[0].parsed,
        r.choices[0].message.parsed,
      );
      before = spans.length;
      await assert.rejects(
        c.chat.parse({
          ...request,
          model: "bad-json",
          response_format: format,
        }),
      );
      assert.equal(spans.length, before + 1);
      assert.equal(last().status.code, 2);
    } finally {
      i.deactivate();
    }
  },
);
test(
  "native event helper callbacks and final completion stay native",
  { skip: typeof client().chat.stream !== "function" },
  async () => {
    const c = client(),
      i = await instrument();
    try {
      const before = spans.length;
      const runner = c.chat.stream(request);
      const events = [];
      assert.equal(
        runner.on("content", (...args) => events.push(args)),
        runner,
      );
      const final = await runner.finalChatCompletion();
      assert.equal(final.choices[0].message.content, "native stream");
      assert.equal(events.length, 2);
      assert.equal(spans.length, before + 1);
      assert.equal(
        last().attributes["gen_ai.completion.0.content"],
        "native stream",
      );
    } finally {
      i.deactivate();
    }
  },
);
test("text completions preserve native result, usage and actual response status", async () => {
  const c = client(),
    i = await instrument();
  try {
    const before = spans.length;
    const p = c.completions.create({
      model: "request-completion",
      prompt: "fixture",
      temperature: 0,
    });
    const r = await p;
    assert.equal(r.choices[0].text, "native completion");
    assert.equal(r, await p);
    assert.equal(spans.length, before + 1);
    assert.equal(last().attributes["respan.entity.log_type"], "text");
    assert.equal(
      last().attributes["gen_ai.completion.0.content"],
      "native completion",
    );
    assert.equal(last().attributes["http.response.status_code"], 201);
  } finally {
    i.deactivate();
  }
});
test("overlapping owners, foreign overrides, reactivation and in-flight deactivation", async () => {
  const c = client(),
    proto = Object.getPrototypeOf(c.chat),
    original = proto.chat;
  const a = await instrument(),
    b = await instrument();
  try {
    a.deactivate();
    const before = spans.length;
    await c.chat.chat(request);
    assert.equal(spans.length, before + 1);
    b.deactivate();
    assert.equal(proto.chat, original);
    const foreign = function (...args) {
      return original.apply(this, args);
    };
    proto.chat = foreign;
    await a.activate();
    const pending = c.chat.chat(request);
    a.deactivate();
    assert.equal(proto.chat, foreign);
    await pending;
    assert.equal(spans.length, before + 2);
    await a.activate();
    const foreign2 = function (...args) {
      return foreign.apply(this, args);
    };
    proto.chat = foreign2;
    a.deactivate();
    assert.equal(proto.chat, foreign2);
  } finally {
    a.deactivate();
    b.deactivate();
    proto.chat = original;
  }
});
test("sampling, general suppression and historical LM compatibility gates suppress spans", async () => {
  const c = client(),
    i = await instrument();
  try {
    const unsampled = trace.wrapSpanContext({
      traceId: "a".repeat(32),
      spanId: "b".repeat(16),
      traceFlags: 0,
      isRemote: true,
    });
    for (const ctx of [
      trace.setSpan(ROOT_CONTEXT, unsampled),
      suppressTracing(ROOT_CONTEXT),
      ROOT_CONTEXT.setValue(
        createContextKey("suppress_language_model_instrumentation"),
        true,
      ),
    ]) {
      const before = spans.length;
      await context.with(ctx, () => c.chat.chat(request));
      assert.equal(spans.length, before);
    }
  } finally {
    i.deactivate();
  }
});
test("constructor, environment and canonical context are sticky capture ceilings", async () => {
  const c = client();
  for (const options of [
    { traceContent: false },
    { recordInputs: false },
    { recordOutputs: false },
    {},
  ]) {
    const i = await instrument(options);
    try {
      let p;
      if (Object.keys(options).length === 0)
        p = context.with(
          ROOT_CONTEXT.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
          () => c.chat.chat(request),
        );
      else p = c.chat.chat(request);
      await p;
      const s = last();
      if (options.recordInputs === false) {
        assert.equal(s.attributes["traceloop.entity.input"], undefined);
        assert.ok(s.attributes["traceloop.entity.output"]);
      } else if (options.recordOutputs === false) {
        assert.ok(s.attributes["traceloop.entity.input"]);
        assert.equal(s.attributes["traceloop.entity.output"], undefined);
      } else noContent(s);
    } finally {
      i.deactivate();
    }
  }
  const i = await instrument();
  try {
    process.env.RESPAN_TRACE_CONTENT = "false";
    const p = c.chat.chat(request);
    delete process.env.RESPAN_TRACE_CONTENT;
    await p;
    noContent(last());
    process.env.TRACELOOP_TRACE_CONTENT = "off";
    await c.chat.chat(request);
    delete process.env.TRACELOOP_TRACE_CONTENT;
    noContent(last());
  } finally {
    delete process.env.RESPAN_TRACE_CONTENT;
    delete process.env.TRACELOOP_TRACE_CONTENT;
    i.deactivate();
  }
});
test("actual readable late attributes, events, status and queue export obey frozen veto after deactivation", async () => {
  const c = client(),
    parent = {
      attributes: { allow_trace_content: false },
      spanContext: () => ({
        traceId: "a".repeat(32),
        spanId: "b".repeat(16),
        traceFlags: 1,
      }),
    },
    i = await instrument();
  try {
    const p = context.with(trace.setSpan(ROOT_CONTEXT, parent), () =>
      c.chat.chat(request),
    );
    parent.attributes.allow_trace_content = true;
    await p;
    const s = last();
    i.deactivate();
    s.attributes = {
      ...s.attributes,
      "traceloop.entity.input": "late input",
      "traceloop.entity.output": "late output",
      "gen_ai.prompt.0.content": "late prompt",
      "respan.metadata": "late metadata",
      "exception.message": "late error",
    };
    s.events = [{ name: "late event", attributes: { secret: "late" } }];
    s.status = { code: 2, message: "late message" };
    context.with(suppressTracing(ROOT_CONTEXT), () => noContent(s));
    assert.equal(s.status.code, 2);
  } finally {
    i.deactivate();
  }
});
test("telemetry processing faults cannot reject successful native calls", async () => {
  const provider = trace.getTracerProvider;
  const faulty = new BasicTracerProvider({
    spanProcessors: [
      {
        onStart() {},
        onEnd() {
          throw Error("fixture telemetry fault");
        },
        forceFlush: async () => {},
        shutdown: async () => {},
      },
    ],
  });
  trace.getTracerProvider = () => faulty;
  const c = client(),
    i = await instrument();
  try {
    assert.equal(
      (await c.chat.chat(request)).choices[0].message.content,
      "native text",
    );
  } finally {
    i.deactivate();
    trace.getTracerProvider = provider;
  }
});

test("native SDK spans retain observed ancestor denial through a later true child context", async () => {
  const { BasicTracerProvider } = await import("@opentelemetry/sdk-trace-base");
  const {
    acquireSpanTransformerHost,
    releaseSpanTransformerHost,
    runSpanTransformersOnStart,
    runSpanTransformersOnEnd,
  } = await import("@respan/tracing/dist/processor/transformers.js");
  acquireSpanTransformerHost();
  const processor = {
    onStart: runSpanTransformersOnStart,
    onEnd: runSpanTransformersOnEnd,
    forceFlush: async () => {},
    shutdown: async () => {},
  };
  const provider = new BasicTracerProvider({ spanProcessors: [processor] });
  const tracer = provider.getTracer("native-ancestor-test");
  const c = client(),
    i = await instrument();
  try {
    const denied = ROOT_CONTEXT.setValue(
      CONTEXT_KEY_ALLOW_TRACE_CONTENT,
      false,
    );
    const parent = tracer.startSpan("native ancestor", {}, denied);
    const allowed = trace.setSpan(
      ROOT_CONTEXT.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
      parent,
    );
    await context.with(allowed, () => c.chat.chat(request));
    noContent(last());
    parent.end();
  } finally {
    i.deactivate();
    await provider.shutdown();
    releaseSpanTransformerHost();
  }
});

test("late actual readable denial strips existing and replaced buffers; permissive queued spans ignore exporter suppression", async () => {
  const c = client(),
    i = await instrument();
  try {
    await c.chat.chat(request);
    const denied = last();
    denied.attributes.allow_trace_content = false;
    denied.events.push({ name: "late", attributes: { secret: "fixture" } });
    denied.status.message = "late status";
    assert.equal(denied.status.message, undefined);
    noContent(denied);
    denied.attributes = {
      ...denied.attributes,
      allow_trace_content: true,
      "gen_ai.completion.0.content": "replacement",
    };
    denied.events = [{ name: "replacement" }];
    denied.status = { code: 1, message: "replacement" };
    noContent(denied);
    await c.chat.chat(request);
    const allowed = last();
    i.deactivate();
    context.with(suppressTracing(ROOT_CONTEXT), () =>
      assert.equal(
        allowed.attributes["gen_ai.completion.0.content"],
        "native text",
      ),
    );
  } finally {
    i.deactivate();
  }
});

test("native text SSE completion reconstruction omits unavailable usage", async () => {
  const c = client(
      async () =>
        new Response(
          'data: {"value":"text "}\n\ndata: {"value":"stream"}\n\ndata: [DONE]\n\n',
          { headers: { "content-type": "text/event-stream" } },
        ),
    ),
    i = await instrument();
  try {
    const before = spans.length;
    const stream = await c.completions.create({
      model: "request-completion",
      prompt: "fixture",
      stream: true,
    });
    const chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    assert.deepEqual(chunks, [{ value: "text " }, { value: "stream" }]);
    assert.equal(spans.length, before + 1);
    assert.equal(
      last().attributes["gen_ai.completion.0.content"],
      "text stream",
    );
    assert.equal(last().attributes["gen_ai.usage.input_tokens"], undefined);
  } finally {
    i.deactivate();
  }
});
