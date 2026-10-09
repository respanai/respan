import test, { before, beforeEach, afterEach, after } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { context, trace, createContextKey } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import {
  BasicTracerProvider,
  AlwaysOffSampler,
} from "@opentelemetry/sdk-trace-base";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { OpenRouter } from "@openrouter/sdk";
import * as z from "zod/v4";
import { HTTPClient } from "@openrouter/sdk/lib/http.js";
import { ClientSDK } from "@openrouter/sdk/lib/sdks.js";
import { chatSend } from "@openrouter/sdk/funcs/chatSend.js";
import { APIPromise } from "@openrouter/sdk/types/async.js";
import { EventStream } from "@openrouter/sdk/lib/event-streams.js";
import { OpenRouterInstrumentor } from "../dist/index.js";
const require = createRequire(import.meta.url);
const sdkVersion = require("@openrouter/sdk/package.json").version;
let instrumentor;
const spans = [];
const provider = new BasicTracerProvider({
  spanLimits: {
    attributeCountLimit: Infinity,
    attributeValueLengthLimit: Infinity,
  },
  spanProcessors: [
    {
      onStart() {},
      onEnd(span) {
        spans.push(span);
      },
      async forceFlush() {},
      async shutdown() {},
    },
  ],
});
const cm = new AsyncLocalStorageContextManager().enable();
before(() => {
  context.setGlobalContextManager(cm);
  trace.setGlobalTracerProvider(provider);
});
beforeEach(async () => {
  spans.length = 0;
  instrumentor = new OpenRouterInstrumentor();
  await instrumentor.activate();
});
afterEach(async () => {
  await instrumentor.deactivate();
  delete process.env.RESPAN_TRACE_CONTENT;
  delete process.env.TRACELOOP_TRACE_CONTENT;
});
after(() => {
  trace.disable();
  context.disable();
  cm.disable();
});
function chat(content = "", extra = {}) {
  return {
    id: "native-chat-id",
    object: "chat.completion",
    created: 1,
    system_fingerprint: "synthetic",
    model: "resolved/model",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    ...extra,
  };
}
function response(
  output = [
    {
      type: "message",
      id: "msg-native",
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: "response text",
          annotations: [],
          logprobs: [],
        },
      ],
    },
  ],
  extra = {},
) {
  return {
    id: "native-response-id",
    object: "response",
    created_at: 1,
    completed_at: 2,
    error: null,
    frequency_penalty: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    model: "resolved/responses",
    output,
    parallel_tool_calls: false,
    presence_penalty: null,
    status: "completed",
    temperature: null,
    tool_choice: "auto",
    tools: [],
    top_p: null,
    usage: {
      input_tokens: 2,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 1,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 3,
    },
    ...extra,
  };
}
function client(payload = chat(), fetcher) {
  return new OpenRouter({
    apiKey: "controlled-synthetic-key",
    retryConfig: { strategy: "none" },
    httpClient: new HTTPClient({
      fetcher: fetcher ?? (async () => Response.json(payload)),
    }),
  });
}
const request = {
  chatRequest: {
    model: "requested/model",
    messages: [{ role: "user", content: "controlled prompt" }],
  },
};
function sse(events, onCancel) {
  const encoder = new TextEncoder();
  let i = 0;
  return new Response(
    new ReadableStream({
      pull(c) {
        if (i < events.length)
          c.enqueue(encoder.encode(`data: ${JSON.stringify(events[i++])}\n\n`));
        else c.close();
      },
      cancel: onCancel,
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
}
function chunk(delta, index = 0, extra = {}) {
  return {
    id: "native-stream-id",
    object: "chat.completion.chunk",
    created: 1,
    model: "resolved/stream",
    choices: [{ index, delta, finish_reason: null }],
    ...extra,
  };
}
function attrs() {
  assert.equal(spans.length, 1);
  return spans[0].attributes;
}
function parent(flags = 1) {
  return {
    attributes: {},
    spanContext: () => ({
      traceId: "a".repeat(32),
      spanId: "b".repeat(16),
      traceFlags: flags,
    }),
  };
}
const runContext = (ctx, fn) => context.with(ctx, fn);

test("native chat preserves empty content, zero usage, requested/resolved model and response identity", async () => {
  const result = await client().chat.send(request);
  assert.equal(result.choices[0].message.content, "");
  const a = attrs();
  assert.equal(a["gen_ai.completion.0.content"], "");
  assert.equal(a["gen_ai.usage.input_tokens"], 0);
  assert.equal(a["llm.usage.total_tokens"], 0);
  assert.equal(a["gen_ai.request.model"], "requested/model");
  assert.equal(a["gen_ai.response.model"], "resolved/model");
  assert.equal(a["http.response.status_code"], 200);
  assert.equal(spans[0].status.code, 0);
});
test("native history 75, full schema and current tool calls remain distinct", async () => {
  const history = Array.from({ length: 75 }, (_, i) => ({
    role: "user",
    content: `history-${i}`,
  }));
  history[0] = {
    role: "assistant",
    content: null,
    toolCalls: [
      {
        id: "old",
        type: "function",
        function: { name: "old_fn", arguments: "{}" },
      },
    ],
  };
  const schema = {
    type: "object",
    properties: Object.fromEntries(
      Array.from({ length: 90 }, (_, i) => [
        `field${i}`,
        { type: "string", description: "x".repeat(70) },
      ]),
    ),
    additionalProperties: false,
  };
  await client(
    chat(null, {
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
                function: { name: "now", arguments: "{}" },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    }),
  ).chat.send({
    chatRequest: {
      model: "test",
      messages: history,
      tools: [
        {
          type: "function",
          function: { name: "now", parameters: schema, strict: false },
        },
      ],
    },
  });
  const a = attrs();
  assert.equal(JSON.parse(a["traceloop.entity.input"]).length, 75);
  assert.equal(a["gen_ai.prompt.74.content"], "history-74");
  assert.equal(
    JSON.parse(a["llm.request.functions"])[0].function.parameters.properties
      .field89.description.length,
    70,
  );
  assert.equal(
    JSON.parse(a["llm.request.functions"])[0].function.strict,
    false,
  );
  assert.equal(
    JSON.parse(a["gen_ai.completion.0.tool_calls"])[0].id,
    "current",
  );
  assert.equal(JSON.parse(a["gen_ai.prompt.0.tool_calls"])[0].id, "old");
  assert.equal(a["gen_ai.completion.0.content"], "null");
});
test("all native chat choices retain their indexes", async () => {
  await client(
    chat("one", {
      choices: [
        {
          index: 2,
          message: { role: "assistant", content: "two" },
          finish_reason: "stop",
        },
        {
          index: 4,
          message: { role: "assistant", content: "four" },
          finish_reason: "length",
        },
      ],
    }),
  ).chat.send(request);
  assert.equal(attrs()["gen_ai.completion.4.content"], "four");
  assert.equal(attrs()["gen_ai.completion.2.content"], "two");
});
test("native embedding captures all 5001 vector elements", async () => {
  const vector = Array.from({ length: 5001 }, (_, i) => i / 5001);
  const result = await client({
    object: "list",
    model: "embedding-resolved",
    data: [{ object: "embedding", embedding: vector, index: 0 }],
    usage: { prompt_tokens: 3, total_tokens: 3 },
  }).embeddings.generate({
    requestBody: { model: "embedding-requested", input: ["a", "b"] },
  });
  assert.equal(result.data[0].embedding.length, 5001);
  assert.equal(JSON.parse(attrs()["traceloop.entity.output"])[0].length, 5001);
  assert.equal(attrs()["llm.request.type"], "embedding");
});
test("native standalone APIPromise and inspect HTTP Response remain intact", async () => {
  const raw = Response.json(chat("standalone"));
  const promise = chatSend(
    client(undefined, async () => raw),
    request,
  );
  assert.ok(promise instanceof APIPromise);
  assert.equal(Object.prototype.toString.call(promise), "[object APIPromise]");
  const inspected = promise.$inspect();
  assert.equal(promise.$inspect(), inspected);
  const [result, meta] = await inspected;
  assert.equal(result.ok, true);
  assert.equal(meta.response, raw);
  assert.equal(raw.bodyUsed, true);
  assert.equal(attrs()["gen_ai.completion.0.content"], "standalone");
});
test("native ERROR status and error object are preserved for response validation", async () => {
  let error;
  try {
    await client({ broken: true }).chat.send(request);
  } catch (e) {
    error = e;
  }
  assert.equal(error.name, "ResponseValidationError");
  assert.equal(spans[0].status.code, 2);
  assert.equal(attrs()["error.type"], "ResponseValidationError");
  assert.equal(error.rawResponse.status, 200);
});
test("native provider HTTP error retains rawResponse and actual status", async () => {
  let error;
  const raw = Response.json(
    { error: { code: 401, message: "controlled unauthorized" } },
    { status: 401 },
  );
  try {
    await client(undefined, async () => raw).chat.send(request);
  } catch (e) {
    error = e;
  }
  assert.ok(error);
  assert.equal(error.rawResponse, raw);
  assert.equal(attrs()["traceloop.entity.output"], undefined);
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
  assert.equal(attrs()["http.response.status_code"], 401);
  assert.equal(spans[0].status.code, 2);
});
test("native input validation rejects without contacting transport", async () => {
  let calls = 0;
  await assert.rejects(
    client(undefined, async () => {
      calls++;
      return Response.json(chat());
    }).chat.send({ chatRequest: { messages: 42 } }),
  );
  assert.equal(calls, 0);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 2);
});
test("native streaming EventStream and lazy consumer semantics are retained", async () => {
  const events = [
    chunk({ role: "assistant", content: "A" }),
    chunk({ content: "B" }),
    chunk({}, 0, {
      usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
    }),
  ];
  const stream = await client(undefined, async () => sse(events)).chat.send({
    chatRequest: { ...request.chatRequest, stream: true },
  });
  assert.ok(stream instanceof EventStream);
  assert.equal(spans.length, 0);
  const values = [];
  for await (const value of stream) values.push(value);
  assert.equal(values.length, 3);
  assert.equal(attrs()["gen_ai.completion.0.content"], "AB");
  assert.equal(attrs()["gen_ai.usage.input_tokens"], 2);
});
test("native split stream tool deltas assemble per choice/index", async () => {
  const events = [
    chunk({
      role: "assistant",
      tool_calls: [
        {
          index: 0,
          id: "call-native",
          type: "function",
          function: { name: "look", arguments: '{"q":' },
        },
      ],
    }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: '"x"}' } }] }),
    chunk({}, 0, {
      choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    }),
  ];
  const stream = await client(undefined, async () => sse(events)).chat.send({
    chatRequest: { ...request.chatRequest, stream: true },
  });
  for await (const _ of stream) {
  }
  const calls = JSON.parse(attrs()["gen_ai.completion.0.tool_calls"]);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].function.arguments, '{"q":"x"}');
  assert.equal(calls[0].id, "call-native");
});
test("native reader and tee consumers complete once without replacing streams", async () => {
  const stream = await client(undefined, async () =>
    sse([chunk({ content: "tee" })]),
  ).chat.send({ chatRequest: { ...request.chatRequest, stream: true } });
  const branches = stream.tee();
  assert.equal(branches.length, 2);
  await Promise.all(
    branches.map(async (branch) => {
      const reader = branch.getReader();
      while (!(await reader.read()).done) {}
      reader.releaseLock();
    }),
  );
  assert.equal(attrs()["gen_ai.completion.0.content"], "tee");
});
test("native cancellation records outcome without inventing provider error", async () => {
  const stream = await client(undefined, async () =>
    sse([chunk({ content: "partial" }), chunk({ content: "later" })]),
  ).chat.send({ chatRequest: { ...request.chatRequest, stream: true } });
  const iterator = stream[Symbol.asyncIterator]();
  await iterator.next();
  await iterator.return();
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 0);
  assert.equal(
    JSON.parse(attrs()["respan.metadata"]).openrouter_cancelled,
    true,
  );
});
test("actual unsampled and general suppressed parents emit no spans", async () => {
  await runContext(trace.setSpan(context.active(), parent(0)), () =>
    client().chat.send(request),
  );
  await runContext(suppressTracing(context.active()), () =>
    client().chat.send(request),
  );
  assert.equal(spans.length, 0);
});
test("local language-model suppression compatibility emits no spans", async () => {
  await runContext(
    context
      .active()
      .setValue(
        createContextKey("suppress_language_model_instrumentation"),
        true,
      ),
    () => client().chat.send(request),
  );
  assert.equal(spans.length, 0);
});
for (const privacy of [
  "constructor",
  "canonical",
  "respan-env",
  "traceloop-env",
])
  test(`content veto before conversion: ${privacy}`, async () => {
    if (privacy === "constructor") {
      await instrumentor.deactivate();
      instrumentor = new OpenRouterInstrumentor({ traceContent: false });
      await instrumentor.activate();
    }
    if (privacy === "respan-env") process.env.RESPAN_TRACE_CONTENT = "false";
    if (privacy === "traceloop-env")
      process.env.TRACELOOP_TRACE_CONTENT = "false";
    const ctx =
      privacy === "canonical"
        ? context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false)
        : context.active();
    await runContext(ctx, () =>
      client(chat("secret-response")).chat.send(request),
    );
    const a = attrs();
    assert.equal(a["traceloop.entity.input"], undefined);
    assert.equal(a["gen_ai.completion.0.content"], undefined);
    assert.doesNotMatch(
      JSON.stringify(spans[0]),
      /secret-response|controlled prompt/,
    );
  });
test("observed ancestor false remains false after a later true", async () => {
  const p = parent();
  const ctx = trace.setSpan(context.active(), p);
  await runContext(ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false), () =>
    client().chat.send(request),
  );
  await runContext(ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true), () =>
    client().chat.send(request),
  );
  assert.equal(spans.length, 2);
  assert.ok(spans.every((s) => !s.attributes["traceloop.entity.input"]));
});
test("actual ReadableSpan denies late attributes/events/status and replacements after deactivation", async () => {
  process.env.RESPAN_TRACE_CONTENT = "false";
  await client().chat.send(request);
  const s = spans[0];
  await instrumentor.deactivate();
  delete process.env.RESPAN_TRACE_CONTENT;
  s.attributes["traceloop.entity.output"] = "late-secret";
  s.attributes["respan.metadata"] = '{"secret":"late-secret"}';
  s.attributes = { secret: "late-secret" };
  s.events = [{ name: "late-secret" }];
  s.status = { code: 2, message: "late-secret" };
  assert.doesNotMatch(JSON.stringify(s), /late-secret/);
  assert.equal(s.status.code, 0);
});
test("native parent IDs and frozen in-flight gate survive deactivation", async () => {
  let resolveFetch;
  const raw = new Promise((r) => {
    resolveFetch = r;
  });
  const pending = runContext(trace.setSpan(context.active(), parent()), () =>
    client(undefined, () => raw).chat.send(request),
  );
  await new Promise(setImmediate);
  await instrumentor.deactivate();
  resolveFetch(Response.json(chat("inflight")));
  await pending;
  assert.equal(spans.length, 1);
  assert.equal(spans[0].spanContext().traceId, "a".repeat(32));
  assert.equal(
    spans[0].parentSpanContext?.spanId ?? spans[0].parentSpanId,
    "b".repeat(16),
  );
});
test("overlapping owners retain patches until last disable", async () => {
  const second = new OpenRouterInstrumentor();
  const patched = ClientSDK.prototype._do;
  await second.activate();
  await instrumentor.deactivate();
  assert.equal(ClientSDK.prototype._do, patched);
  await client().chat.send(request);
  assert.equal(spans.length, 1);
  await second.deactivate();
  assert.notEqual(ClientSDK.prototype._do, patched);
});
test("disable cancels an activation still awaiting native imports", async () => {
  await instrumentor.deactivate();
  const i = new OpenRouterInstrumentor();
  const activation = i.activate();
  i.disable();
  await activation;
  assert.equal(i.isActive(), false);
});
test("foreign wrapper survives disable and reactivation does not duplicate spans", async () => {
  const own = ClientSDK.prototype._do;
  const foreign = function (...args) {
    return own.apply(this, args);
  };
  ClientSDK.prototype._do = foreign;
  await instrumentor.deactivate();
  assert.equal(ClientSDK.prototype._do, foreign);
  await instrumentor.activate();
  await client().chat.send(request);
  assert.equal(spans.length, 1);
  await instrumentor.deactivate();
  ClientSDK.prototype._do = own;
});
test("native Responses result retains full output and usage", async () => {
  const sdk = client(response());
  const endpoint = sdk.responses ?? sdk.beta.responses;
  const result = await endpoint.send({
    responsesRequest: {
      model: "requested/responses",
      input: "response prompt",
      stream: false,
    },
  });
  assert.equal(result.id, "native-response-id");
  assert.equal(attrs()["gen_ai.response.model"], "resolved/responses");
  assert.equal(
    JSON.parse(attrs()["traceloop.entity.output"])[0].id,
    "msg-native",
  );
});
test("native lazy callModel keeps its ModelResult and shared promises", async () => {
  let calls = 0;
  const sdk = client(response(), async () => {
    calls++;
    return Response.json(response());
  });
  const result = sdk.callModel({
    model: "requested/callmodel",
    input: "lazy prompt",
  });
  assert.equal(calls, 0);
  const text = result.getText();
  assert.equal(result.getText(), text);
  const [value, full] = await Promise.all([text, result.getResponse()]);
  assert.equal(value, "response text");
  assert.equal(full.id, "native-response-id");
  assert.equal(calls, 1);
  assert.equal(spans.length, 1);
  assert.equal(attrs()["gen_ai.request.model"], "requested/callmodel");
});
test("native callModel creation gate and parent remain frozen for lazy consumption", async () => {
  const ctx = trace
    .setSpan(context.active(), parent())
    .setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false);
  const result = runContext(ctx, () =>
    client(response()).callModel({ model: "test", input: "private-lazy" }),
  );
  await result.getText();
  assert.equal(spans.length, 1);
  assert.equal(
    spans[0].parentSpanContext?.spanId ?? spans[0].parentSpanId,
    "b".repeat(16),
  );
  assert.doesNotMatch(JSON.stringify(spans[0]), /private-lazy|response text/);
});

test("native callModel tool loop preserves callbacks and complete continuation history", async () => {
  let requests = 0;
  let executed = 0;
  const wire = [];
  const turns = [];
  const sdk = client(undefined, async (request) => {
    wire.push(await request.json());
    return Response.json(
      requests++ === 0
        ? response([
            {
              type: "function_call",
              id: "native-item",
              call_id: "native-call",
              name: "lookup",
              arguments: '{"q":"x"}',
              status: "completed",
            },
          ])
        : response(undefined, { id: "native-followup" }),
    );
  });
  const tools = [
    {
      type: "function",
      function: {
        name: "lookup",
        inputSchema: z.object({ q: z.string() }),
        outputSchema: z.object({ found: z.boolean() }),
        execute: async (args) => {
          executed++;
          assert.equal(args.q, "x");
          return { found: false };
        },
      },
    },
  ];
  const result = sdk.callModel({
    model: "tool-loop",
    input: "find",
    tools,
    onTurnStart: (ctx) => {
      turns.push(ctx);
    },
    onTurnEnd: (ctx, output) => {
      assert.ok(turns.includes(ctx));
      assert.ok(output.id);
    },
  });
  assert.equal(await result.getText(), "response text");
  assert.equal(executed, 1);
  assert.equal(requests, 2);
  assert.equal(spans.length, 2);
  const followup = wire[1].input;
  assert.ok(
    followup.some(
      (item) =>
        item.call_id === "native-call" && item.type === "function_call_output",
    ),
  );
  const captured = JSON.parse(spans[1].attributes["traceloop.entity.input"]);
  assert.deepEqual(captured, followup);
  assert.equal(
    JSON.parse(spans[0].attributes["gen_ai.completion.0.tool_calls"])[0]
      .call_id,
    "native-call",
  );
  assert.equal(
    spans[1].attributes["gen_ai.completion.0.tool_calls"],
    undefined,
  );
});
test("telemetry does not invoke customer request getters beyond bare SDK behavior", async () => {
  async function run(enabled) {
    if (!enabled) await instrumentor.deactivate();
    else await instrumentor.activate();
    let visits = 0;
    const message = {
      role: "user",
      get content() {
        visits++;
        return "getter text";
      },
    };
    await client().chat.send({
      chatRequest: { model: "test", messages: [message] },
    });
    return visits;
  }
  const bare = await run(false);
  const traced = await run(true);
  assert.equal(traced, bare);
});
test("native stream parse failure matches bare SDK and records real ERROR", async () => {
  async function run(enabled) {
    if (!enabled) await instrumentor.deactivate();
    else await instrumentor.activate();
    let caught;
    const stream = await client(undefined, async () =>
      sse([{ malformed: true }]),
    ).chat.send({ chatRequest: { ...request.chatRequest, stream: true } });
    try {
      for await (const _ of stream) {
      }
    } catch (error) {
      caught = error;
    }
    return caught;
  }
  const bare = await run(false);
  const traced = await run(true);
  assert.equal(traced.name, bare.name);
  assert.equal(traced.message, bare.message);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].status.code, 2);
});
test("native EventStream next promises and reader results retain identities", async () => {
  let nativePromise;
  const nativeResults = [];
  let nativeReader;
  const raw = sse([chunk({ content: "identity" })]);
  const original = raw.body.getReader;
  raw.body.getReader = function (...args) {
    const reader = original.apply(this, args);
    nativeReader = reader;
    const read = reader.read;
    reader.read = function (...readArgs) {
      const promise = read.apply(this, readArgs);
      nativePromise = promise;
      promise.then((value) => {
        nativeResults.push(value);
      });
      return promise;
    };
    return reader;
  };
  const stream = await client(undefined, async () => raw).chat.send({
    chatRequest: { ...request.chatRequest, stream: true },
  });
  assert.ok(stream instanceof EventStream);
  const reader = stream.getReader();
  const first = await reader.read();
  assert.equal(first.value.choices[0].delta.content, "identity");
  assert.ok(nativeReader);
  assert.ok(nativePromise instanceof Promise);
  assert.ok(nativeResults.some((value) => value.value instanceof Uint8Array));
  while (!(await reader.read()).done) {}
  assert.equal(spans.length, 1);
});
test("export transport suppression does not change a frozen readable decision", async () => {
  await client().chat.send(request);
  const span = spans[0];
  const a = runContext(
    suppressTracing(context.active()),
    () => span.attributes,
  );
  assert.equal(a["gen_ai.prompt.0.content"], "controlled prompt");
});

test("queued allowed readable observes actual parent denial and never re-allows", async () => {
  const p = { ...parent(), attributes: {} };
  await runContext(trace.setSpan(context.active(), p), () =>
    client(chat("queued-secret")).chat.send(request),
  );
  const span = spans[0];
  assert.equal(span.attributes["gen_ai.prompt.0.content"], "controlled prompt");
  p.attributes.allow_trace_content = false;
  assert.equal(span.attributes["gen_ai.prompt.0.content"], undefined);
  p.attributes.allow_trace_content = true;
  span.attributes["arbitrary.payload"] = "late-secret";
  span.events = [{ name: "late-secret" }];
  span.status = { code: 2, message: "late-secret" };
  assert.doesNotMatch(
    JSON.stringify(span),
    /queued-secret|late-secret|controlled prompt/,
  );
  assert.equal(span.attributes["gen_ai.usage.input_tokens"], 0);
});
test("actual readable own denial strips arbitrary later payloads", async () => {
  await client(chat("own-secret")).chat.send(request);
  const span = spans[0];
  span.attributes["arbitrary.payload"] = "late-secret";
  span.attributes.allow_trace_content = false;
  span.attributes.allow_trace_content = true;
  assert.doesNotMatch(
    JSON.stringify(span),
    /own-secret|late-secret|controlled prompt/,
  );
});
test("denied spans retain native scalar model usage and ID without payloads", async () => {
  process.env.RESPAN_TRACE_CONTENT = "false";
  await client().chat.send(request);
  assert.equal(attrs()["gen_ai.request.model"], "requested/model");
  assert.equal(attrs()["gen_ai.response.model"], "resolved/model");
  assert.equal(attrs()["gen_ai.response.id"], "native-chat-id");
  assert.equal(attrs()["gen_ai.usage.output_tokens"], 0);
  assert.equal(attrs()["traceloop.entity.input"], undefined);
});

test("cancelling one concurrent activation does not cancel another owner", async () => {
  await instrumentor.deactivate();
  const first = new OpenRouterInstrumentor();
  const second = new OpenRouterInstrumentor();
  const a = first.activate();
  const b = second.activate();
  first.disable();
  await Promise.all([a, b]);
  assert.equal(first.isActive(), false);
  assert.equal(second.isActive(), true);
  await client().chat.send(request);
  assert.equal(spans.length, 1);
  await second.deactivate();
});
test("callModel general suppression remains frozen across lazy consumption", async () => {
  const result = runContext(suppressTracing(context.active()), () =>
    client(response()).callModel({ model: "test", input: "suppressed-lazy" }),
  );
  assert.equal(await result.getText(), "response text");
  assert.equal(spans.length, 0);
});

test("actual no-parent AlwaysOffSampler vetoes capture and emission", async () => {
  trace.disable();
  const off = new BasicTracerProvider({
    sampler: new AlwaysOffSampler(),
    spanProcessors: [
      {
        onStart() {},
        onEnd(span) {
          spans.push(span);
        },
        async forceFlush() {},
        async shutdown() {},
      },
    ],
  });
  trace.setGlobalTracerProvider(off);
  try {
    await client(chat("root-secret")).chat.send(request);
    assert.equal(spans.length, 0);
  } finally {
    trace.disable();
    trace.setGlobalTracerProvider(provider);
    await off.shutdown();
  }
});

test("standalone in-flight APIPromise settles after deactivation before first await", async () => {
  let release;
  const fetchPromise = new Promise((resolve) => {
    release = resolve;
  });
  const promise = chatSend(
    client(undefined, () => fetchPromise),
    request,
  );
  await new Promise(setImmediate);
  await instrumentor.deactivate();
  release(Response.json(chat("standalone-inflight")));
  const result = await promise;
  assert.equal(result.ok, true);
  assert.equal(spans.length, 1);
  assert.equal(attrs()["gen_ai.completion.0.content"], "standalone-inflight");
});

for (const mode of ["general", "language-model", "unsampled"])
  test(`allowed callModel creation obeys later application admission veto: ${mode}`, async () => {
    const result = client(response()).callModel({
      model: "test",
      input: "lazy-admission",
    });
    const ctx =
      mode === "general"
        ? suppressTracing(context.active())
        : mode === "language-model"
          ? context
              .active()
              .setValue(
                createContextKey("suppress_language_model_instrumentation"),
                true,
              )
          : trace.setSpan(context.active(), parent(0));
    assert.equal(
      await runContext(ctx, () => result.getText()),
      "response text",
    );
    assert.equal(spans.length, 0);
  });

test("successful native explicit empty choices remains an explicit empty output", async () => {
  const result = await client(chat("", { choices: [] })).chat.send(request);
  assert.deepEqual(result.choices, []);
  assert.equal(attrs()["traceloop.entity.output"], "[]");
  assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
});
