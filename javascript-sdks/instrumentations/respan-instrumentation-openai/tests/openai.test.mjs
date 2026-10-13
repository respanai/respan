import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { context, trace, SpanStatusCode } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import { OpenAIInstrumentor } from "../dist/index.js";

const exporter = new InMemorySpanExporter();
const provider = new BasicTracerProvider({
  spanProcessors: [new SimpleSpanProcessor(exporter)],
});
trace.setGlobalTracerProvider(provider);
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
const chatParams = {
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Say hello" }],
};
const toolCall = {
  id: "call_now",
  type: "function",
  function: { name: "weather", arguments: '{"city":"Paris"}' },
};
const tools = [
  {
    type: "function",
    function: { name: "weather", parameters: { type: "object" } },
  },
];
const chatResult = (message = { role: "assistant", content: "hello" }) => ({
  id: "chat_fixture",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o-mini",
  choices: [{ index: 0, message, finish_reason: "stop", logprobs: null }],
  usage: {
    prompt_tokens: 10,
    completion_tokens: 2,
    total_tokens: 12,
    prompt_tokens_details: { cached_tokens: 4 },
  },
});
const chatChunk = (delta, usage) => ({
  id: "chat_fixture",
  object: "chat.completion.chunk",
  created: 1,
  model: "gpt-4o-mini",
  choices: delta ? [{ index: 0, delta, finish_reason: null }] : [],
  ...(usage ? { usage } : {}),
});
const responseResult = (
  output = [
    {
      id: "msg_fixture",
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "hello", annotations: [] }],
    },
  ],
) => ({
  id: "resp_fixture",
  object: "response",
  created_at: 1,
  model: "gpt-4o-mini",
  status: "completed",
  output,
  usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
  error: null,
  incomplete_details: null,
});
const responseEvents = () => [
  {
    type: "response.created",
    response: { ...responseResult([]), status: "in_progress" },
  },
  {
    type: "response.output_item.added",
    output_index: 0,
    item: {
      id: "msg_fixture",
      type: "message",
      role: "assistant",
      status: "in_progress",
      content: [],
    },
  },
  {
    type: "response.content_part.added",
    item_id: "msg_fixture",
    output_index: 0,
    content_index: 0,
    part: { type: "output_text", text: "", annotations: [] },
  },
  {
    type: "response.output_text.delta",
    item_id: "msg_fixture",
    output_index: 0,
    content_index: 0,
    delta: "hello",
  },
  { type: "response.completed", response: responseResult() },
];

function clientFor(OpenAI, payload, { stream = false, status = 200 } = {}) {
  return new OpenAI({
    apiKey: "fixture-key",
    maxRetries: 0,
    fetch: async () =>
      new Response(
        stream
          ? payload
              .map(
                (item) =>
                  `data: ${typeof item === "string" ? item : JSON.stringify(item)}\n\n`,
              )
              .join("") + "data: [DONE]\n\n"
          : JSON.stringify(payload),
        {
          status,
          headers: {
            "content-type": stream ? "text/event-stream" : "application/json",
            "x-request-id": "fixture-request",
          },
        },
      ),
  });
}

function onlySpan(type = "chat") {
  const spans = exporter.getFinishedSpans();
  assert.equal(
    spans.length,
    1,
    `expected one span; got ${spans.map((span) => span.name)}`,
  );
  const attrs = spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], type);
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  for (const key of [
    "traceloop.span.kind",
    "respan.span.tools",
    "respan.span.tool_calls",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "span_tools",
    "has_tool_calls",
  ]) {
    assert.equal(attrs[key], undefined, `off-contract attribute ${key}`);
  }
  return spans[0];
}

test("real OpenAI SDK compatibility matrix", async (t) => {
  for (const moduleName of [
    "openai-v4",
    "openai-v4-last",
    "openai-v5",
    "openai-v6",
    "openai-v7",
    "openai",
  ]) {
    await t.test(moduleName, async (t) => {
      const sdk = await import(moduleName);
      const OpenAI = sdk.default;
      const instrumentor = new OpenAIInstrumentor({ openAI: sdk });
      await instrumentor.activate();
      t.after(() => instrumentor.deactivate());
      t.beforeEach(() => exporter.reset());

      await t.test(
        "chat APIPromise withResponse, canonical messages and usage",
        async () => {
          const client = clientFor(OpenAI, chatResult());
          const request = client.chat.completions.create(chatParams);
          assert.equal(typeof request._thenUnwrap, "function");
          assert.equal(typeof request.asResponse, "function");
          const { data, response } = await request.withResponse();
          assert.equal(data.choices[0].message.content, "hello");
          assert.equal(response.headers.get("x-request-id"), "fixture-request");
          assert.equal(await request, data);
          const attrs = onlySpan().attributes;
          assert.equal(attrs["gen_ai.prompt.0.content"], "Say hello");
          assert.equal(attrs["gen_ai.completion.0.content"], "hello");
          assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
          assert.equal(attrs["gen_ai.usage.prompt_tokens"], 10);
          assert.equal(attrs["gen_ai.usage.output_tokens"], 2);
          assert.equal(attrs["gen_ai.usage.completion_tokens"], 2);
          assert.equal(attrs["llm.usage.total_tokens"], 12);
        },
      );

      await t.test(
        "raw access stays unconsumed and ends a request-only span",
        async () => {
          const response = await clientFor(OpenAI, chatResult())
            .chat.completions.create(chatParams)
            .asResponse();
          assert.equal(response.bodyUsed, false);
          assert.deepEqual(await response.json(), chatResult());
          assert.equal(
            onlySpan().attributes["gen_ai.completion.0.content"],
            undefined,
          );
        },
      );

      await t.test(
        "request history and new tool calls remain separate",
        async () => {
          const client = clientFor(
            OpenAI,
            chatResult({
              role: "assistant",
              content: null,
              tool_calls: [toolCall],
            }),
          );
          const history = { ...toolCall, id: "call_history" };
          await client.chat.completions.create({
            ...chatParams,
            tools,
            messages: [
              { role: "assistant", content: null, tool_calls: [history] },
              { role: "tool", content: "sunny", tool_call_id: history.id },
              ...chatParams.messages,
            ],
          });
          const attrs = onlySpan().attributes;
          assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.0.tool_calls"]), [
            history,
          ]);
          assert.deepEqual(
            JSON.parse(attrs["gen_ai.completion.0.tool_calls"]),
            [toolCall],
          );
          assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), tools);
        },
      );

      await t.test(
        "chat streams keep SDK objects, usage and fragmented tools",
        async () => {
          const chunks = [
            chatChunk({
              role: "assistant",
              content: "hel",
              tool_calls: [
                {
                  index: 0,
                  id: "call_now",
                  type: "function",
                  function: { name: "weather", arguments: '{"city":' },
                },
              ],
            }),
            chatChunk({
              content: "lo",
              tool_calls: [{ index: 0, function: { arguments: '"Paris"}' } }],
            }),
            chatChunk(null, {
              prompt_tokens: 10,
              completion_tokens: 2,
              total_tokens: 12,
            }),
          ];
          const request = clientFor(OpenAI, chunks, {
            stream: true,
          }).chat.completions.create({ ...chatParams, stream: true });
          const { data: stream } = await request.withResponse();
          assert.equal(typeof stream.controller.abort, "function");
          assert.equal(exporter.getFinishedSpans().length, 0);
          const actual = [];
          for await (const chunk of stream) actual.push(chunk);
          assert.deepEqual(actual, chunks);
          const attrs = onlySpan().attributes;
          assert.equal(attrs["gen_ai.completion.0.content"], "hello");
          assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
          assert.deepEqual(
            JSON.parse(attrs["gen_ai.completion.0.tool_calls"]),
            [toolCall],
          );
        },
      );

      await t.test(
        "stream early return closes one span with partial output",
        async () => {
          const stream = await clientFor(
            OpenAI,
            [
              chatChunk({ content: "partial" }),
              chatChunk({ content: "ignored" }),
            ],
            { stream: true },
          ).chat.completions.create({ ...chatParams, stream: true });
          for await (const chunk of stream) {
            assert.equal(chunk.choices[0].delta.content, "partial");
            break;
          }
          assert.equal(
            onlySpan().attributes["gen_ai.completion.0.content"],
            "partial",
          );
          assert.equal(
            onlySpan().attributes["gen_ai.usage.input_tokens"],
            undefined,
          );
        },
      );

      await t.test(
        "explicit abort before iteration closes the span",
        async () => {
          const stream = await clientFor(
            OpenAI,
            [chatChunk({ content: "unused" })],
            { stream: true },
          ).chat.completions.create({ ...chatParams, stream: true });
          stream.controller.abort();
          await delay(5);
          onlySpan();
        },
      );

      if (moduleName !== "openai-v4") {
        await t.test(
          "tee preserves both consumers and emits one completed span",
          async () => {
            const chunks = [
              chatChunk({ content: "hello" }),
              chatChunk(null, {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
              }),
            ];
            const stream = await clientFor(OpenAI, chunks, {
              stream: true,
            }).chat.completions.create({ ...chatParams, stream: true });
            const [left, right] = stream.tee();
            const collect = async (source) => {
              const result = [];
              for await (const item of source) result.push(item);
              return result;
            };
            assert.deepEqual(
              await Promise.all([collect(left), collect(right)]),
              [chunks, chunks],
            );
            assert.equal(
              onlySpan().attributes["gen_ai.completion.0.content"],
              "hello",
            );
          },
        );

        await t.test(
          "provider SSE error closes with error status",
          async () => {
            const stream = await clientFor(
              OpenAI,
              [
                chatChunk({ content: "partial" }),
                { error: { message: "provider fixture failure" } },
              ],
              { stream: true },
            ).chat.completions.create({ ...chatParams, stream: true });
            await assert.rejects(async () => {
              for await (const _ of stream) {
              }
            }, /provider fixture failure/);
            assert.equal(onlySpan().status.code, SpanStatusCode.ERROR);
          },
        );
      }

      await t.test(
        "HTTP failures retain SDK errors and end error spans",
        async () => {
          const client = clientFor(
            OpenAI,
            {
              error: {
                message: "fixture rejection",
                type: "invalid_request_error",
              },
            },
            { status: 401 },
          );
          await assert.rejects(
            client.chat.completions.create(chatParams),
            (error) => error.status === 401,
          );
          assert.equal(onlySpan().status.code, SpanStatusCode.ERROR);
          assert.equal(onlySpan().attributes["http.response.status_code"], 401);
          assert.equal(onlySpan().attributes["http.status_code"], 401);
        },
      );

      await t.test(
        "stream errors retain SDK errors and end error spans",
        async () => {
          const stream = await clientFor(
            OpenAI,
            [chatChunk({ content: "partial" }), "{invalid json"],
            { stream: true },
          ).chat.completions.create({ ...chatParams, stream: true });
          await assert.rejects(async () => {
            for await (const _ of stream) {
            }
          }, SyntaxError);
          const span = onlySpan();
          assert.equal(span.status.code, SpanStatusCode.ERROR);
          assert.equal(
            span.attributes["gen_ai.completion.0.content"],
            "partial",
          );
        },
      );

      await t.test("text completion and embedding payloads", async () => {
        await clientFor(OpenAI, {
          model: "gpt-4o-mini",
          choices: [{ index: 0, text: "hello" }],
          usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
        }).completions.create({ model: "gpt-4o-mini", prompt: "Say hello" });
        assert.equal(
          onlySpan("text").attributes["gen_ai.completion.0.content"],
          "hello",
        );
        exporter.reset();
        await clientFor(OpenAI, {
          model: "gpt-4o-mini",
          data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
          usage: { prompt_tokens: 3, total_tokens: 3 },
        }).embeddings.create({
          model: "gpt-4o-mini",
          input: ["embed me"],
          encoding_format: "float",
        });
        const attrs = onlySpan("embedding").attributes;
        assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [
          [0.1, 0.2],
        ]);
        assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
          "embed me",
        ]);
        assert.equal(attrs["gen_ai.usage.input_tokens"], 3);
      });

      if (OpenAI.Responses) {
        await t.test(
          "Responses managed prompts use the provider-resolved model",
          async () => {
            await clientFor(OpenAI, responseResult()).responses.create({
              prompt: { id: "pmpt_fixture" },
            });
            assert.equal(
              onlySpan().attributes["gen_ai.request.model"],
              "gpt-4o-mini",
            );
          },
        );

        await t.test(
          "Responses multi-turn input retains order and assistant boundaries",
          async () => {
            const input = [
              { role: "assistant", content: "first" },
              { role: "user", content: "second" },
              {
                type: "function_call",
                call_id: "history",
                name: "weather",
                arguments: "{}",
              },
              {
                type: "function_call_output",
                call_id: "history",
                output: "sunny",
              },
              { role: "assistant", content: "third" },
              { role: "user", content: "fourth" },
            ];
            await clientFor(OpenAI, responseResult()).responses.create({
              model: "gpt-4o-mini",
              input,
            });
            const attrs = onlySpan().attributes;
            assert.deepEqual(
              [0, 1, 2, 3, 4, 5].map(
                (index) => attrs[`gen_ai.prompt.${index}.role`],
              ),
              ["assistant", "user", "assistant", "tool", "assistant", "user"],
            );
            assert.equal(attrs["gen_ai.prompt.0.content"], "first");
            assert.equal(attrs["gen_ai.prompt.4.content"], "third");
          },
        );

        await t.test(
          "Responses messages, instructions and function output",
          async () => {
            const output = [
              {
                type: "function_call",
                id: "item_now",
                call_id: "call_now",
                name: "weather",
                arguments: '{"city":"Paris"}',
              },
            ];
            const input = [
              {
                type: "function_call",
                call_id: "history",
                name: "weather",
                arguments: "{}",
              },
              {
                type: "function_call_output",
                call_id: "history",
                output: "sunny",
              },
              { role: "user", content: [{ type: "input_text", text: "next" }] },
            ];
            await clientFor(OpenAI, responseResult(output))
              .responses.create({
                model: "gpt-4o-mini",
                instructions: "Be concise",
                input,
                tools: [
                  {
                    type: "function",
                    name: "weather",
                    parameters: { type: "object" },
                  },
                ],
              })
              .withResponse();
            const attrs = onlySpan().attributes;
            assert.equal(attrs["gen_ai.prompt.0.content"], "Be concise");
            assert.equal(attrs["gen_ai.prompt.2.content"], "sunny");
            assert.deepEqual(
              JSON.parse(attrs["gen_ai.completion.0.tool_calls"]),
              [toolCall],
            );
            assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
          },
        );

        await t.test(
          "Responses stream completion and early return",
          async () => {
            const stream = await clientFor(OpenAI, responseEvents(), {
              stream: true,
            }).responses.create({
              model: "gpt-4o-mini",
              input: "hello",
              stream: true,
            });
            let terminal;
            let output;
            for await (const event of stream) {
              if (event.type === "response.completed") {
                terminal = event;
                output = event.response.output;
              }
            }
            assert.equal(
              terminal.response.output,
              output,
              "instrumentation must not replace SDK event arrays",
            );
            assert.deepEqual(terminal, responseEvents().at(-1));
            assert.equal(
              onlySpan().attributes["gen_ai.completion.0.content"],
              "hello",
            );
            assert.equal(
              onlySpan().attributes["gen_ai.usage.output_tokens"],
              2,
            );
            exporter.reset();
            const partial = await clientFor(OpenAI, responseEvents(), {
              stream: true,
            }).responses.create({
              model: "gpt-4o-mini",
              input: "hello",
              stream: true,
            });
            for await (const event of partial) {
              if (event.type === "response.output_text.delta") break;
            }
            assert.equal(
              onlySpan().attributes["gen_ai.completion.0.content"],
              "hello",
            );
            assert.equal(
              onlySpan().attributes["gen_ai.usage.output_tokens"],
              undefined,
            );
          },
        );

        await t.test(
          "Responses stream helper and parse helper delegate once",
          async () => {
            const client = clientFor(OpenAI, responseEvents(), {
              stream: true,
            });
            const stream = client.responses.stream({
              model: "gpt-4o-mini",
              input: "hello",
            });
            assert.equal(
              (await stream.finalResponse()).output[0].content[0].text,
              "hello",
            );
            assert.equal(
              onlySpan().attributes["gen_ai.completion.0.content"],
              "hello",
            );
            exporter.reset();
            const parsed = await clientFor(
              OpenAI,
              responseResult(),
            ).responses.parse({ model: "gpt-4o-mini", input: "hello" });
            assert.equal(parsed.output[0].content[0].text, "hello");
            onlySpan();
          },
        );
      }

      if (OpenAI.Chat.Completions.prototype.parse) {
        await t.test(
          "chat parse and stream helpers retain native contracts",
          async () => {
            const parsed = await clientFor(
              OpenAI,
              chatResult(),
            ).chat.completions.parse(chatParams);
            assert.equal(parsed.choices[0].message.content, "hello");
            onlySpan();
            exporter.reset();
            const completed = chatChunk({
              role: "assistant",
              content: "hello",
            });
            completed.choices[0].finish_reason = "stop";
            const helper = clientFor(OpenAI, [completed], {
              stream: true,
            }).chat.completions.stream(chatParams);
            assert.equal(
              (await helper.finalChatCompletion()).choices[0].message.content,
              "hello",
            );
            onlySpan();
          },
        );
      }

      await t.test(
        "content context opt-out and tracing suppression",
        async () => {
          await context.with(
            context.active().setValue(CONTEXT_KEY_ALLOW_TRACE_CONTENT, false),
            () =>
              clientFor(OpenAI, chatResult()).chat.completions.create(
                chatParams,
              ),
          );
          const attrs = onlySpan().attributes;
          assert.equal(attrs["gen_ai.prompt.0.content"], undefined);
          assert.equal(attrs["gen_ai.completion.0.content"], undefined);
          assert.equal(attrs["traceloop.entity.input"], undefined);
          assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
          exporter.reset();
          await context.with(suppressTracing(context.active()), () =>
            clientFor(OpenAI, chatResult()).chat.completions.create(chatParams),
          );
          assert.equal(exporter.getFinishedSpans().length, 0);
        },
      );

      await t.test(
        "environment content opt-out leaves usage and request untouched",
        async () => {
          const before = process.env.TRACELOOP_TRACE_CONTENT;
          process.env.TRACELOOP_TRACE_CONTENT = "false";
          try {
            await clientFor(OpenAI, chatResult()).chat.completions.create(
              Object.freeze({ ...chatParams }),
            );
            const attrs = onlySpan().attributes;
            assert.equal(attrs["traceloop.entity.input"], undefined);
            assert.equal(attrs["traceloop.entity.output"], undefined);
            assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
          } finally {
            if (before === undefined)
              delete process.env.TRACELOOP_TRACE_CONTENT;
            else process.env.TRACELOOP_TRACE_CONTENT = before;
          }
        },
      );

      await t.test(
        "async parent context and shared-instance lifecycle",
        async () => {
          const second = new OpenAIInstrumentor({ openAI: sdk });
          await Promise.all([
            second.activate(),
            second.activate(),
            instrumentor.activate(),
          ]);
          instrumentor.deactivate();
          const parent = trace.getTracer("test").startSpan("parent");
          await context.with(
            trace.setSpan(context.active(), parent),
            async () => {
              await delay(0);
              await clientFor(OpenAI, chatResult()).chat.completions.create(
                chatParams,
              );
            },
          );
          assert.equal(
            onlySpan().parentSpanContext.spanId,
            parent.spanContext().spanId,
          );
          second.deactivate();
          exporter.reset();
          await clientFor(OpenAI, chatResult()).chat.completions.create(
            chatParams,
          );
          assert.equal(exporter.getFinishedSpans().length, 0);
          await instrumentor.activate();
        },
      );
    });
  }
});

const GATEWAY_LOG_ID = "4f1c1b0e5c8a4d3e9b7a6f5e4d3c2b1a";

function gatewayClientFor(OpenAI, payload, { stream = false, status = 200, logId = GATEWAY_LOG_ID } = {}) {
  return new OpenAI({
    apiKey: "fixture-key",
    maxRetries: 0,
    fetch: async () =>
      new Response(
        stream
          ? payload.map((item) => `data: ${JSON.stringify(item)}\n\n`).join("") +
              "data: [DONE]\n\n"
          : JSON.stringify(payload),
        {
          status,
          headers: {
            "content-type": stream ? "text/event-stream" : "application/json",
            ...(logId ? { "x-respan-log-id": logId } : {}),
          },
        },
      ),
  });
}

function assertLinkOnly(attrs) {
  assert.equal(attrs["respan.gateway.log_id"], GATEWAY_LOG_ID);
  const leaked = Object.keys(attrs).filter(
    (key) =>
      key.startsWith("gen_ai.usage.") ||
      key.startsWith("llm.usage.") ||
      key.startsWith("gen_ai.prompt.") ||
      key.startsWith("gen_ai.completion.") ||
      key === "traceloop.entity.input" ||
      key === "traceloop.entity.output",
  );
  assert.deepEqual(leaked, [], "a linked span carries no usage or content");
}

test("a Respan gateway response links the span instead of re-reporting it", async (t) => {
  for (const moduleName of ["openai-v4", "openai-v5", "openai-v6", "openai-v7", "openai"]) {
    await t.test(moduleName, async (t) => {
      const sdk = await import(moduleName);
      const OpenAI = sdk.default;
      const instrumentor = new OpenAIInstrumentor({ openAI: sdk });
      await instrumentor.activate();
      t.after(() => instrumentor.deactivate());
      t.beforeEach(() => exporter.reset());

      await t.test("chat create", async () => {
        const result = await gatewayClientFor(OpenAI, chatResult()).chat.completions.create(chatParams);
        assert.equal(result.choices[0].message.content, "hello");
        assertLinkOnly(onlySpan().attributes);
      });

      await t.test("chat stream", async () => {
        const stream = await gatewayClientFor(
          OpenAI,
          [chatChunk({ role: "assistant", content: "hel" }), chatChunk({ content: "lo" }), chatChunk(null, { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 })],
          { stream: true },
        ).chat.completions.create({ ...chatParams, stream: true, stream_options: { include_usage: true } });
        let text = "";
        for await (const chunk of stream) text += chunk.choices[0]?.delta?.content ?? "";
        assert.equal(text, "hello");
        assertLinkOnly(onlySpan().attributes);
      });

      await t.test("error response", async () => {
        await assert.rejects(
          gatewayClientFor(OpenAI, { error: { message: "bad key", type: "invalid_request_error" } }, { status: 401 })
            .chat.completions.create(chatParams),
        );
        const span = onlySpan();
        assert.equal(span.status.code, SpanStatusCode.ERROR);
        assertLinkOnly(span.attributes);
      });

      await t.test("a direct provider response keeps usage and content", async () => {
        await gatewayClientFor(OpenAI, chatResult(), { logId: null }).chat.completions.create(chatParams);
        const attrs = onlySpan().attributes;
        assert.equal(attrs["respan.gateway.log_id"], undefined);
        assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
        assert.equal(attrs["gen_ai.prompt.0.content"], "Say hello");
      });
    });
  }
});

test("default resolution covers host ESM and CommonJS copies and Azure exclusion", async () => {
  exporter.reset();
  const sdk = await import("openai");
  const commonjs = createRequire(import.meta.url)("openai");
  const instrumentor = new OpenAIInstrumentor();
  await instrumentor.activate();
  try {
    for (const OpenAI of [sdk.default, commonjs.OpenAI]) {
      exporter.reset();
      await clientFor(OpenAI, chatResult()).chat.completions.create(chatParams);
      onlySpan();
    }
    exporter.reset();
    const azure = new sdk.AzureOpenAI({
      apiKey: "fixture-key",
      apiVersion: "2024-10-21",
      endpoint: "https://fixture.openai.azure.com",
      fetch: async () =>
        new Response(JSON.stringify(chatResult()), {
          headers: { "content-type": "application/json" },
        }),
    });
    await azure.chat.completions.create(chatParams);
    await azure.responses.create({ model: "gpt-4o-mini", input: "hello" });
    await azure.embeddings.create({
      model: "gpt-4o-mini",
      input: "hello",
      encoding_format: "float",
    });
    assert.equal(exporter.getFinishedSpans().length, 0);
  } finally {
    instrumentor.deactivate();
  }
});

test("reactivation after cancelling an in-flight activation installs patches", async () => {
  const sdk = await import("openai");
  for (const options of [{ openAI: sdk }, {}]) {
    exporter.reset();
    const instrumentor = new OpenAIInstrumentor(options);
    const first = instrumentor.activate();
    instrumentor.deactivate();
    await instrumentor.activate();
    await first;
    try {
      await clientFor(sdk.default, chatResult()).chat.completions.create(
        chatParams,
      );
      onlySpan();
    } finally {
      instrumentor.deactivate();
    }
  }
});
