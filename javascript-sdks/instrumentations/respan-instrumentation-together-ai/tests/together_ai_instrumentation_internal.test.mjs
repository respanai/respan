import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { patchResourceMethod } from "../dist/_patching.js";
const CHAT_SPEC = {
  kind: "chat",
  method: "create",
  spanName: "together.chat.completions",
  logType: "chat",
  requestType: "chat",
};

const EMBEDDING_SPEC = {
  kind: "embedding",
  method: "create",
  spanName: "together.embeddings",
  logType: "embedding",
  requestType: "embedding",
};

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);

test.before(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value() {
      return {
        activeSpanProcessor: {
          onEnd(span) {
            captureState.spans.push(span);
          },
        },
      };
    },
  });
});

test.after(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
});

test("patchResourceMethod emits canonical chat attributes without off-contract aliases", async () => {
  captureState.spans = [];

  const resourcePrototype = {
    create(_body) {
      return Promise.resolve({
        model: "meta-llama/test",
        choices: [
          {
            message: {
              role: "assistant",
              content: "I will call the weather tool.",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: {
                    name: "get_weather",
                    arguments: "{\"city\":\"Tokyo\"}",
                  },
                },
              ],
            },
          },
        ],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 7,
          total_tokens: 19,
        },
      });
    },
  };

  const patchedTarget = patchResourceMethod(resourcePrototype, CHAT_SPEC);
  assert.ok(patchedTarget);

  const result = await resourcePrototype.create({
    model: "meta-llama/test",
    messages: [
      { role: "user", content: "Use the weather tool for Tokyo." },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_existing",
            type: "function",
            function: {
              name: "get_weather",
              arguments: "{\"city\":\"Osaka\"}",
            },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_existing",
        content: "{\"forecast\":\"clear\"}",
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather by city.",
          parameters: {
            type: "object",
            properties: {
              city: { type: "string" },
            },
          },
        },
      },
    ],
  }).then((value) => value);

  assert.equal(result.model, "meta-llama/test");
  assert.equal(captureState.spans.length, 2);

  const toolSpan = captureState.spans[0];
  assert.equal(toolSpan.attributes["respan.entity.log_type"], "tool");
  assert.equal(toolSpan.attributes["traceloop.entity.name"], "get_weather");
  assert.equal(toolSpan.attributes.tool_calls, undefined);
  assert.equal(toolSpan.attributes["respan.span.tool_calls"], undefined);

  const chatSpan = captureState.spans[1];
  const attrs = chatSpan.attributes;
  assert.equal(chatSpan.instrumentationScope?.name, "@respan/instrumentation-together-ai");
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["gen_ai.system"], "together");
  assert.equal(attrs["gen_ai.request.model"], "meta-llama/test");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Use the weather tool for Tokyo." },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_existing",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "{\"city\":\"Osaka\"}",
          },
        },
      ],
    },
    {
      role: "tool",
      content: "{\"forecast\":\"clear\"}",
      tool_call_id: "call_existing",
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather by city.",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
        },
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 7);
  assert.equal(attrs["llm.usage.total_tokens"], 19);

  for (const bannedKey of [
    "respan.span.tools",
    "respan.span.tool_calls",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
  ]) {
    assert.equal(attrs[bannedKey], undefined, `${bannedKey} should not be emitted`);
  }

  resourcePrototype.create = patchedTarget.original;
});

test("streaming chat calls emit after async iteration completes", async () => {
  captureState.spans = [];

  const stream = {
    async *[Symbol.asyncIterator]() {
      yield {
        model: "meta-llama/stream",
        choices: [{ delta: { role: "assistant", content: "Hello " }, finish_reason: null }],
      };
      yield {
        model: "meta-llama/stream",
        choices: [{ delta: { content: "stream" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      };
    },
  };

  const resourcePrototype = {
    create(_body) {
      return Promise.resolve(stream);
    },
  };

  const patchedTarget = patchResourceMethod(resourcePrototype, CHAT_SPEC);
  const result = await resourcePrototype.create({
    model: "meta-llama/stream",
    stream: true,
    messages: [{ role: "user", content: "Say hello." }],
  }).then((value) => value);

  for await (const _chunk of result) {
    // Consume the stream to trigger final span emission.
  }

  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["gen_ai.completion.0.content"], "Hello stream");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 3);

  resourcePrototype.create = patchedTarget.original;
});

test("embedding spans preserve vectors in traceloop entity output", async () => {
  captureState.spans = [];

  const resourcePrototype = {
    create(_body) {
      return Promise.resolve({
        model: "togethercomputer/m2-bert-80M-8k-retrieval",
        data: [
          {
            index: 0,
            embedding: [0.1, 0.2, 0.3],
          },
        ],
      });
    },
  };

  const patchedTarget = patchResourceMethod(resourcePrototype, EMBEDDING_SPEC);
  await resourcePrototype.create({
    model: "togethercomputer/m2-bert-80M-8k-retrieval",
    input: "semantic search text",
  }).then((value) => value);

  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], "embedding");
  assert.equal(attrs["llm.request.type"], "embedding");
  assert.equal(attrs["gen_ai.request.model"], "togethercomputer/m2-bert-80M-8k-retrieval");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), [
    { index: 0, embedding: [0.1, 0.2, 0.3] },
  ]);

  resourcePrototype.create = patchedTarget.original;
});
