import { execFileSync } from "node:child_process";
import assert from "node:assert/strict";
import { test } from "node:test";
import { createRequire } from "node:module";
import { Readable } from "node:stream";
import * as OpenAI from "openai";
import * as Minimum from "openai-min";
import * as LegacyMin from "azure-openai-legacy-min";
import * as Legacy from "azure-openai-legacy";
import {
  context,
  trace,
  ROOT_CONTEXT,
  SpanStatusCode,
} from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  SamplingDecision,
} from "@opentelemetry/sdk-trace-base";
import { suppressTracing } from "@opentelemetry/core";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { AzureOpenAIInstrumentor } from "../dist/index.js";

const spans = [],
  started = [];
let decision = SamplingDecision.RECORD_AND_SAMPLED;
const sampler = {
  shouldSample(_ctx, _trace, _name, _kind, attrs) {
    assert(!("traceloop.entity.input" in attrs));
    return { decision };
  },
  toString() {
    return "ControlledSampler";
  },
};
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
trace.setGlobalTracerProvider(
  new BasicTracerProvider({
    sampler,
    spanLimits: { attributeCountLimit: 10000 },
    spanProcessors: [
      {
        onStart(span) {
          started.push(span);
        },
        onEnd(span) {
          spans.push(span);
        },
        forceFlush: async () => {},
        shutdown: async () => {},
      },
    ],
  }),
);
const chat = (content = "hello", model = "native-model") => ({
  id: "chat-1",
  object: "chat.completion",
  created: 1,
  model,
  choices: [
    {
      index: 0,
      message: { role: "assistant", content },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
});
const response = (text = "response") => ({
  id: "resp-1",
  object: "response",
  created_at: 1,
  status: "completed",
  model: "native-model",
  output: [
    {
      id: "msg-1",
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text, annotations: [] }],
    },
  ],
  usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5 },
});
const json = (value, status = 201) =>
  new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "x-request-id": "native-request",
    },
  });
const sse = (values) =>
  new Response(
    values.map((v) => `data: ${JSON.stringify(v)}\n\n`).join("") +
      "data: [DONE]\n\n",
    { status: 202, headers: { "content-type": "text/event-stream" } },
  );
function client(module = OpenAI, fetch = async () => json(chat())) {
  return new module.AzureOpenAI({
    apiKey: "fixture",
    endpoint: "https://fixture.openai.azure.com",
    apiVersion: "2024-10-21",
    deployment: "deployment",
    maxRetries: 0,
    fetch,
  });
}
const params = {
  model: "deployment",
  messages: [{ role: "user", content: "input" }],
};
const last = () => spans.at(-1);
const attrs = () => last().attributes;
async function using(options, fn) {
  const i = new AzureOpenAIInstrumentor(options);
  await i.activate();
  try {
    return await fn(i);
  } finally {
    i.deactivate();
  }
}

for (const [label, module] of [
  ["current", OpenAI],
  ["minimum", Minimum],
]) {
  test(`${label}: native APIPromise identity, helpers, complete input/output and status`, async () => {
    const messages = Array.from({ length: 76 }, (_, i) => ({
      role: "user",
      content: `message-${i}`,
    }));
    messages[0].content = [
      { type: "text", text: "x".repeat(20001) },
      { type: "image_url", image_url: { url: "data:image/png;base64,Zg==" } },
    ];
    const tools = Array.from({ length: 76 }, (_, i) => ({
      type: "function",
      function: {
        name: `tool${i}`,
        parameters: {
          type: "object",
          properties: {
            a: {
              type: "object",
              properties: {
                b: {
                  type: "object",
                  properties: {
                    c: {
                      type: "object",
                      properties: { d: { type: "string" } },
                    },
                  },
                },
              },
            },
          },
        },
      },
    }));
    const c = client(module, async (_url, init) => {
      const body = JSON.parse(init.body);
      assert.equal(body.extraAttributes, undefined);
      return json(chat("o".repeat(22000)));
    });
    const native = c.chat.completions.create(params);
    const prototype = Object.getPrototypeOf(native);
    await native;
    await using({ openAIModule: module }, async () => {
      const p = c.chat.completions.create({
        model: "deployment",
        messages,
        tools,
        temperature: 0,
        max_tokens: 0,
        extraAttributes: { "respan.metadata": '{"flag":false}' },
      });
      assert.equal(Object.getPrototypeOf(p), prototype);
      assert.equal(typeof p.withResponse, "function");
      assert.equal(typeof p.asResponse, "function");
      const { data, response: raw } = await p.withResponse();
      assert.equal(raw.status, 201);
      assert.equal(data.choices[0].message.content.length, 22000);
      assert.equal(attrs()["http.response.status_code"], 201);
      assert.equal(attrs()["gen_ai.prompt.75.content"], "message-75");
      assert.equal(JSON.parse(attrs()["llm.request.functions"]).length, 76);
      assert.deepEqual(
        JSON.parse(attrs()["llm.request.functions"])[75],
        tools[75],
      );
      assert.equal(
        JSON.parse(attrs()["traceloop.entity.input"])[0].content[0].text.length,
        20001,
      );
      assert.equal(attrs()["gen_ai.completion.0.content"].length, 22000);
      assert.equal(attrs()["gen_ai.request.temperature"], 0);
      assert.equal(attrs()["llm.usage.total_tokens"], 5);
      assert.equal(attrs()["respan.entity.log_type"], "chat");
    });
  });
  test(`${label}: raw-only response leaves body readable`, async () =>
    using({ openAIModule: module }, async () => {
      const p = client(module).chat.completions.create(params);
      const raw = await p.asResponse();
      assert.equal(raw.bodyUsed, false);
      assert.equal((await raw.json()).id, "chat-1");
      assert.equal(attrs()["traceloop.entity.output"], undefined);
      assert.equal(attrs()["http.response.status_code"], 201);
    }));
  test(`${label}: text and 5001-dimension embedding output, no estimated total`, async () =>
    using({ openAIModule: module }, async () => {
      const c = client(module, async (url) =>
        String(url).includes("embeddings")
          ? json({
              object: "list",
              data: [
                {
                  object: "embedding",
                  index: 0,
                  embedding: Array.from({ length: 5001 }, (_, i) => i / 10000),
                },
              ],
              model: "embedding-native",
              usage: { prompt_tokens: 0 },
            })
          : json({
              id: "text-1",
              object: "text_completion",
              created: 1,
              model: "text-native",
              choices: [{ index: 0, text: "", finish_reason: "stop" }],
              usage: { prompt_tokens: 0, completion_tokens: 0 },
            }),
      );
      await c.completions.create({
        model: "deployment",
        prompt: "",
        temperature: 0,
      });
      assert.equal(attrs()["gen_ai.completion.0.content"], "");
      assert.equal(attrs()["llm.usage.total_tokens"], undefined);
      await c.embeddings.create({
        model: "deployment",
        input: ["", "text"],
        encoding_format: "float",
      });
      assert.equal(
        JSON.parse(attrs()["traceloop.entity.output"])[0].length,
        5001,
      );
      assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
    }));
  test(`${label}: streaming tools retain native Stream/controller/tee and partial return`, async () =>
    using({ openAIModule: module }, async () => {
      const chunks = [
        {
          id: "chat-1",
          object: "chat.completion.chunk",
          created: 1,
          model: "native-model",
          choices: [
            {
              index: 0,
              delta: {
                role: "assistant",
                content: "hi",
                tool_calls: [
                  {
                    index: 0,
                    id: "call-1",
                    type: "function",
                    function: { name: "tool", arguments: '{"a":' },
                  },
                ],
              },
            },
          ],
        },
        {
          id: "chat-1",
          object: "chat.completion.chunk",
          created: 1,
          model: "native-model",
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [{ index: 0, function: { arguments: "0}" } }],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        },
      ];
      const stream = await client(module, async () =>
        sse(chunks),
      ).chat.completions.create({ ...params, stream: true });
      assert.equal(typeof stream.tee, "function");
      assert(stream.controller instanceof AbortController);
      const observed = [];
      for await (const chunk of stream) observed.push(chunk);
      assert.equal(observed[0].choices[0].delta.tool_calls[0].id, "call-1");
      assert.equal(
        JSON.parse(attrs()["gen_ai.completion.0.tool_calls"])[0].function
          .arguments,
        '{"a":0}',
      );
      assert.equal(attrs()["http.response.status_code"], 202);
      const early = await client(module, async () =>
        sse(chunks),
      ).chat.completions.create({ ...params, stream: true });
      for await (const _ of early) break;
      assert.equal(attrs()["gen_ai.completion.0.content"], "hi");
      assert(early.controller.signal.aborted);
    }));
  test(`${label}: transport error uses real HTTP status and SDK error identity`, async () =>
    using({ openAIModule: module }, async () => {
      const c = client(module, async () =>
        json(
          {
            error: {
              message: "provider denied",
              type: "invalid_request_error",
              code: "fixture",
            },
          },
          429,
        ),
      );
      await assert.rejects(c.chat.completions.create(params), (e) => {
        assert.equal(e.status, 429);
        assert.equal(e.message.includes("provider denied"), true);
        return true;
      });
      assert.equal(attrs()["http.response.status_code"], 429);
      assert.equal(last().status.code, SpanStatusCode.ERROR);
      assert.equal(attrs()["traceloop.entity.output"], undefined);
    }));
}

test(
  "current: Responses output/function calls, stream and parse helpers",
  { skip: !OpenAI.AzureOpenAI.Responses },
  async () =>
    using({ openAIModule: OpenAI }, async () => {
      const toolResponse = response("structured");
      toolResponse.output.push({
        id: "fc-1",
        type: "function_call",
        call_id: "call-native",
        name: "native_tool",
        arguments: '{"enabled":false}',
      });
      const c = client(OpenAI, async (_url, init) =>
        JSON.parse(init.body).stream
          ? sse([
              {
                type: "response.created",
                response: { ...response(""), output: [] },
              },
              {
                type: "response.output_item.added",
                output_index: 0,
                item: {
                  id: "msg-1",
                  type: "message",
                  role: "assistant",
                  content: [],
                },
              },
              {
                type: "response.content_part.added",
                output_index: 0,
                content_index: 0,
                item_id: "msg-1",
                part: { type: "output_text", text: "" },
              },
              {
                type: "response.output_text.delta",
                output_index: 0,
                content_index: 0,
                item_id: "msg-1",
                delta: "structured",
              },
              { type: "response.completed", response: toolResponse },
            ])
          : json(toolResponse),
      );
      const before = spans.length;
      const result = await c.responses.create({
        model: "deployment",
        input: "input",
        instructions: "",
        tools: [
          {
            type: "function",
            name: "native_tool",
            parameters: { type: "object", properties: {} },
            strict: true,
          },
        ],
      });
      assert.equal(result.output_text, "structured");
      assert.equal(
        JSON.parse(attrs()["gen_ai.completion.0.tool_calls"])[0].id,
        "call-native",
      );
      assert.equal(spans.length, before + 1);
      const stream = await c.responses.create({
        model: "deployment",
        input: "input",
        stream: true,
      });
      for await (const _ of stream) {
      }
      assert.equal(attrs()["gen_ai.completion.0.content"], "structured");
      const n = spans.length;
      const parsed = await c.responses.parse({
        model: "deployment",
        input: "input",
      });
      assert.equal(parsed.output_text, "structured");
      assert.equal(spans.length, n + 1);
      const helper = c.responses.stream({
        model: "deployment",
        input: "input",
      });
      assert.equal(typeof helper.finalResponse, "function");
      assert.equal((await helper.finalResponse()).output_text, "structured");
    }),
);

test(
  "current: chat parse and stream helpers trace each native request once",
  { skip: !OpenAI.AzureOpenAI.Chat.Completions.prototype.parse },
  async () =>
    using({ openAIModule: OpenAI }, async () => {
      const c = client(OpenAI, async (_u, init) =>
        JSON.parse(init.body).stream
          ? sse([
              {
                id: "chat-1",
                object: "chat.completion.chunk",
                created: 1,
                model: "native-model",
                choices: [
                  {
                    index: 0,
                    delta: { role: "assistant", content: "hello" },
                    finish_reason: "stop",
                  },
                ],
              },
            ])
          : json(chat()),
      );
      const n = spans.length;
      const parsed = await c.chat.completions.parse(params);
      assert.equal(parsed.choices[0].message.content, "hello");
      assert.equal(spans.length, n + 1);
      const runner = c.chat.completions.stream(params);
      assert.equal(typeof runner.finalChatCompletion, "function");
      assert.equal(
        (await runner.finalChatCompletion()).choices[0].message.content,
        "hello",
      );
      assert.equal(spans.length, n + 2);
    }),
);

for (const [label, module] of [
  ["legacy-min", LegacyMin],
  ["legacy-last", Legacy],
]) {
  test(`${label}: actual native chat, text, embeddings, streaming and real status`, async () => {
    const require = createRequire(import.meta.url);
    const { createHttpHeaders } = require(
      require.resolve("@azure/core-rest-pipeline", {
        paths: [
          require.resolve(
            label === "legacy-min"
              ? "azure-openai-legacy-min"
              : "azure-openai-legacy",
          ),
        ],
      }),
    );
    const transport = {
      async sendRequest(request) {
        const body = JSON.parse(request.body);
        let data = String(request.url).includes("embeddings")
          ? {
              data: [{ index: 0, embedding: [0, 0.5] }],
              usage: { prompt_tokens: 0, total_tokens: 0 },
            }
          : String(request.url).includes("/chat/")
            ? chat()
            : {
                id: "text-1",
                created: 1,
                choices: [
                  {
                    index: 0,
                    text: "text",
                    finish_reason: "stop",
                    logprobs: {
                      tokens: [],
                      token_logprobs: [],
                      top_logprobs: [],
                      text_offset: [],
                    },
                  },
                ],
                usage: {
                  prompt_tokens: 1,
                  completion_tokens: 2,
                  total_tokens: 3,
                },
              };
        const raw = JSON.stringify(data);
        return {
          request,
          status: 200,
          headers: createHttpHeaders({
            "content-type": body.stream
              ? "text/event-stream"
              : "application/json",
          }),
          ...(body.stream
            ? {
                readableStreamBody: Readable.from([
                  `data: ${raw}\n\ndata: [DONE]\n\n`,
                ]),
              }
            : { bodyAsText: raw }),
        };
      },
    };
    const c = new module.OpenAIClient(
      "https://fixture.openai.azure.com",
      new module.AzureKeyCredential("fixture"),
      { httpClient: transport, retryOptions: { maxRetries: 0 } },
    );
    await using(
      { openAIModule: OpenAI, azureOpenAIModule: module },
      async () => {
        const result = await c.getChatCompletions(
          "deployment",
          params.messages,
        );
        assert.equal(result.choices[0].message.content, "hello");
        assert.equal(attrs()["gen_ai.completion.0.content"], "hello");
        assert.equal(attrs()["http.response.status_code"], 200);
        await c.getCompletions("deployment", ["prompt"]);
        assert.equal(attrs()["gen_ai.completion.0.content"], "text");
        await c.getEmbeddings("deployment", ["text"]);
        assert.deepEqual(JSON.parse(attrs()["traceloop.entity.output"]), [
          [0, 0.5],
        ]);
        for await (const _ of await (
          c.listChatCompletions ?? c.streamChatCompletions
        ).call(c, "deployment", params.messages)) {
        }
        assert.equal(attrs()["respan.entity.log_type"], "chat");
      },
    );
  });
}

for (const [label, options, scope] of [
  [
    "constructor",
    { traceContent: false },
    (ctx) => ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
  ],
  [
    "context",
    {},
    (ctx) => ctx.setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
  ],
  ["inputs", { recordInputs: false }, (ctx) => ctx],
  ["outputs", { recordOutputs: false }, (ctx) => ctx],
])
  test(`privacy: ${label} ceiling and native serialization only`, async () => {
    let reads = 0;
    const msg = {
      role: "user",
      get content() {
        reads++;
        return "private";
      },
    };
    await using({ openAIModule: OpenAI, ...options }, async () =>
      context.with(scope(context.active()), async () => {
        await client().chat.completions.create({ ...params, messages: [msg] });
        assert.equal(reads, 1); // Native SDK JSON serialization; telemetry never evaluates the getter.
        assert.equal(
          attrs()["traceloop.entity.input"] !== undefined,
          label === "outputs",
        );
        assert.equal(
          attrs()["traceloop.entity.output"] !== undefined,
          label === "inputs",
        );
      }),
    );
  });
test("privacy: DROP and RECORD_ONLY sample before payload copying", async () => {
  for (const sampled of [
    SamplingDecision.NOT_RECORD,
    SamplingDecision.RECORD,
  ]) {
    decision = sampled;
    const n = spans.length;
    let reads = 0;
    await using({ openAIModule: OpenAI }, async () => {
      await client().chat.completions.create({
        ...params,
        messages: [
          {
            role: "user",
            get content() {
              reads++;
              return "secret";
            },
          },
        ],
      });
    });
    assert.equal(reads, 1);
    if (sampled === SamplingDecision.RECORD)
      assert.equal(attrs()["traceloop.entity.output"], undefined);
    else assert.equal(spans.length, n);
  }
  decision = SamplingDecision.RECORD_AND_SAMPLED;
});
test("privacy: late actual span veto, replacements/events/status, immutable observed ancestor veto", async () =>
  using({ openAIModule: OpenAI }, async () => {
    const c = client(OpenAI, async () => {
      const span = trace.getSpan(context.active());
      span.setAttribute("allow_trace_content", false);
      span.setAttribute("allow_trace_content", true);
      let touched = 0;
      span.attributes = {
        "gen_ai.usage.secret": {
          get x() {
            touched++;
            return "secret";
          },
        },
        "gen_ai.request.model": "deployment",
        "http.response.status_code": 201,
      };
      span.events = [{ name: "secret", attributes: { payload: "private" } }];
      span.status = { code: 2, message: "private" };
      assert.equal(touched, 0);
      assert.equal(
        Object.getOwnPropertyDescriptor(span, "attributes").configurable,
        false,
      );
      return json(chat("private-output"));
    });
    await c.chat.completions.create(params);
    assert.equal(attrs()["traceloop.entity.output"], undefined);
    assert.equal(attrs()["gen_ai.usage.secret"], undefined);
    assert.deepEqual(last().events, []);
    assert.equal(last().status.message, undefined);
    const parent = trace.getTracer("test").startSpan("parent");
    await context.with(trace.setSpan(context.active(), parent), async () => {
      parent.setAttribute("allow_trace_content", false);
      await client().chat.completions.create(params);
      parent.setAttribute("allow_trace_content", true);
      await client().chat.completions.create(params);
      assert.equal(attrs()["traceloop.entity.input"], undefined);
    });
    parent.end();
  }));
test("privacy: suppression performs native call without an instrumentation span", async () =>
  using({ openAIModule: OpenAI }, async () => {
    const n = spans.length;
    await context.with(suppressTracing(context.active()), () =>
      client().chat.completions.create(params),
    );
    assert.equal(spans.length, n);
  }));
test("lifecycle: concurrent activation, owners/modules, foreign restoration and in-flight drain", async () => {
  const original = OpenAI.AzureOpenAI.Chat.Completions.prototype.create;
  const a = new AzureOpenAIInstrumentor({ openAIModule: OpenAI }),
    b = new AzureOpenAIInstrumentor({ openAIModule: OpenAI }),
    m = new AzureOpenAIInstrumentor({ openAIModule: Minimum });
  await Promise.all([a.activate(), a.activate(), b.activate(), m.activate()]);
  assert(a.isActive());
  a.deactivate();
  await client().chat.completions.create(params);
  await client(Minimum).chat.completions.create(params);
  m.deactivate();
  let resolve;
  const gate = new Promise((r) => (resolve = r));
  const p = client(OpenAI, async () => {
    await gate;
    return json(chat());
  }).chat.completions.create(params);
  b.deactivate();
  assert.equal(OpenAI.AzureOpenAI.Chat.Completions.prototype.create, original);
  resolve();
  await p;
  assert.equal(attrs()["gen_ai.completion.0.content"], "hello");
  await a.activate();
  const foreign = function (...args) {
    return original.apply(this, args);
  };
  OpenAI.AzureOpenAI.Chat.Completions.prototype.create = foreign;
  a.deactivate();
  assert.equal(OpenAI.AzureOpenAI.Chat.Completions.prototype.create, foreign);
  OpenAI.AzureOpenAI.Chat.Completions.prototype.create = original;
  const cancel = new AzureOpenAIInstrumentor({ openAIModule: OpenAI });
  const activating = cancel.activate();
  cancel.deactivate();
  await activating;
  assert.equal(cancel.isActive(), false);
  await cancel.activate();
  cancel.deactivate();
});
test("native OpenAI clients are excluded and stable Azure 2 companion adds no client", async () =>
  using({ openAIModule: OpenAI }, async () => {
    const c = new OpenAI.OpenAI({
      apiKey: "fixture",
      fetch: async () => json(chat()),
    });
    const n = spans.length;
    await c.chat.completions.create(params);
    assert.equal(spans.length, n);
    const stable = await import("@azure/openai");
    assert.equal(stable.OpenAIClient, undefined);
  }));

test("streamed native refusal/audio fields are retained without changing delivered chunks", async () =>
  using({ openAIModule: OpenAI }, async () => {
    const chunks = [
      {
        id: "chat-1",
        object: "chat.completion.chunk",
        created: 1,
        model: "native-model",
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              refusal: "cannot ",
              audio: { id: "audio-1", data: "YW", transcript: "hello " },
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: "chat-1",
        object: "chat.completion.chunk",
        created: 1,
        model: "native-model",
        choices: [
          {
            index: 0,
            delta: {
              refusal: "answer",
              audio: { data: "Jj", transcript: "world" },
            },
            finish_reason: "stop",
          },
        ],
      },
    ];
    const stream = await client(OpenAI, async () =>
      sse(chunks),
    ).chat.completions.create({ ...params, stream: true });
    const delivered = [];
    for await (const chunk of stream) delivered.push(chunk);
    assert.equal(delivered[0].choices[0].delta.audio.data, "YW");
    const output = JSON.parse(attrs()["traceloop.entity.output"]);
    assert.equal(output[0].refusal, "cannot answer");
    assert.equal(output[0].audio.data, "YWJj");
  }));
test("privacy: environment ceiling, output lanes, and private streamed usage", async () => {
  const previous = process.env.RESPAN_TRACE_CONTENT;
  process.env.RESPAN_TRACE_CONTENT = "false";
  try {
    await using({ openAIModule: OpenAI, traceContent: true }, async () =>
      context.with(
        context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, true),
        async () => {
          const stream = await client(OpenAI, async () =>
            sse([
              {
                id: "chat-1",
                object: "chat.completion.chunk",
                model: "native-model",
                choices: [{ index: 0, delta: { content: "secret" } }],
                usage: {
                  prompt_tokens: 0,
                  completion_tokens: 0,
                  total_tokens: 0,
                },
              },
            ]),
          ).chat.completions.create({ ...params, stream: true });
          for await (const _ of stream) {
          }
          assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
          assert.equal(attrs()["gen_ai.usage.input_tokens"], 0);
          assert.equal(attrs()["http.response.status_code"], 202);
        },
      ),
    );
  } finally {
    if (previous === undefined) delete process.env.RESPAN_TRACE_CONTENT;
    else process.env.RESPAN_TRACE_CONTENT = previous;
  }
});
test("native parse failures and stream controller cancellation finish spans", async () =>
  using({ openAIModule: OpenAI }, async () => {
    await assert.rejects(
      client(
        OpenAI,
        async () =>
          new Response("{broken", {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
      ).chat.completions.create(params),
    );
    assert.equal(last().status.code, SpanStatusCode.ERROR);
    assert.equal(attrs()["http.response.status_code"], 200);
    assert.equal(attrs()["traceloop.entity.output"], undefined);
    const n = spans.length;
    const stream = await client(OpenAI, async () =>
      sse([]),
    ).chat.completions.create({ ...params, stream: true });
    stream.controller.abort();
    assert.equal(spans.length, n + 1);
    assert.equal(attrs()["traceloop.entity.output"], undefined);
  }));

test("minimum: native beta chat streaming helper delegates once", async () =>
  using({ openAIModule: Minimum }, async () => {
    const c = client(Minimum, async () =>
      sse([
        {
          id: "chat-1",
          object: "chat.completion.chunk",
          created: 1,
          model: "native-model",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "minimum helper" },
              finish_reason: "stop",
            },
          ],
        },
      ]),
    );
    const n = spans.length;
    const runner = c.beta.chat.completions.stream(params);
    assert.equal(
      (await runner.finalChatCompletion()).choices[0].message.content,
      "minimum helper",
    );
    assert.equal(spans.length, n + 1);
  }));

test("peer regression: every active owner restricts capture and an in-flight restriction stays immutable", async () => {
  const a = new AzureOpenAIInstrumentor({ openAIModule: OpenAI });
  const opts = { openAIModule: OpenAI, traceContent: false };
  const b = new AzureOpenAIInstrumentor(opts);
  opts.traceContent = true;
  await a.activate();
  await b.activate();
  try {
    await client().chat.completions.create(params);
    assert.equal(attrs()["traceloop.entity.input"], undefined);
    assert.equal(attrs()["traceloop.entity.output"], undefined);
    let resolve;
    const gate = new Promise((r) => (resolve = r));
    const pending = client(OpenAI, async () => {
      await gate;
      return json(chat());
    }).chat.completions.create(params);
    b.deactivate();
    resolve();
    await pending;
    assert.equal(attrs()["traceloop.entity.output"], undefined);
    await client().chat.completions.create(params);
    assert.ok(attrs()["traceloop.entity.output"]);
  } finally {
    a.deactivate();
    b.deactivate();
  }
  const input = new AzureOpenAIInstrumentor({
      openAIModule: OpenAI,
      recordInputs: false,
    }),
    output = new AzureOpenAIInstrumentor({
      openAIModule: OpenAI,
      recordOutputs: false,
    });
  await input.activate();
  await output.activate();
  try {
    await client().chat.completions.create(params);
    assert.equal(attrs()["traceloop.entity.input"], undefined);
    assert.equal(attrs()["traceloop.entity.output"], undefined);
  } finally {
    input.deactivate();
    output.deactivate();
  }
});
test("peer regression: provider stream failure preserves error/status and omits completions", async () =>
  using({ openAIModule: OpenAI }, async () => {
    const stream = await client(OpenAI, async () =>
      sse([
        {
          id: "chat-1",
          object: "chat.completion.chunk",
          created: 1,
          model: "native-model",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "observed before failure" },
              finish_reason: null,
            },
          ],
        },
        {
          error: {
            message: "Native stream failure",
            type: "server_error",
            code: "fixture",
          },
        },
      ]),
    ).chat.completions.create({ ...params, stream: true });
    let observed = 0;
    await assert.rejects(async () => {
      for await (const chunk of stream) {
        assert.equal(chunk.choices[0].delta.content, "observed before failure");
        observed++;
      }
    });
    assert.equal(observed, 1);
    assert.equal(last().status.code, SpanStatusCode.ERROR);
    assert.equal(attrs()["http.response.status_code"], 202);
    assert.equal(attrs()["traceloop.entity.output"], undefined);
    assert.equal(attrs()["gen_ai.completion.0.content"], undefined);
  }));

test("default native 128-attribute budget retains complete canonical Azure payloads", () => {
  for (const module of ["openai", "openai-min"])
    execFileSync(
      process.execPath,
      [new URL("./default_budget.mjs", import.meta.url).pathname],
      { env: { ...process.env, AZURE_BUDGET_MODULE: module }, stdio: "pipe" },
    );
});
