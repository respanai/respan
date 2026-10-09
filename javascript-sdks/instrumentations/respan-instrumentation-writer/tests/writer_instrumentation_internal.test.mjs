import assert from "node:assert/strict";
import test from "node:test";
import { context, ROOT_CONTEXT } from "@opentelemetry/api";
import { CONTEXT_KEY_ALLOW_TRACE_CONTENT } from "@traceloop/ai-semantic-conventions";
import {
  buildSuccessAttrs,
  buildErrorAttrs,
  createChatStreamState,
  updateChatStreamState,
  buildChatCompletionFromStreamState,
  createTextStreamState,
  updateTextStreamState,
  buildCompletionFromStreamState,
} from "../dist/index.js";

test("canonical mapping retains full messages, null, schemas, tools and zero usage", () => {
  const schema = {
    type: "object",
    properties: Object.fromEntries(
      Array.from({ length: 75 }, (_, i) => [
        `field${i}`,
        { type: "string", description: `schema${i}` },
      ]),
    ),
  };
  const messages = Array.from({ length: 75 }, (_, i) => ({
    role: i % 2 ? "assistant" : "user",
    content:
      i === 0
        ? null
        : [
            { type: "text", text: `message${i}` },
            { type: "image", image_url: { url: "fixture-image" } },
          ],
    tool_calls:
      i === 74
        ? [
            {
              id: "history",
              type: "function",
              function: { name: "old", arguments: '{"zero":0,"flag":false}' },
            },
          ]
        : undefined,
  }));
  const tools = [
    {
      type: "function",
      function: { name: "tool", parameters: schema, strict: false },
    },
  ];
  const response = {
    id: "actual-id",
    model: "actual-model",
    choices: [
      {
        message: {
          role: "assistant",
          content: null,
          refusal: null,
          tool_calls: [
            {
              id: "current",
              type: "function",
              function: { name: "new", arguments: "" },
            },
          ],
          parsed: { zero: 0, flag: false, empty: "", nil: null },
        },
      },
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      prompt_token_details: { cached_tokens: 0 },
    },
  };
  const attrs = buildSuccessAttrs(
    "chat",
    {
      model: "requested-model",
      messages,
      tools,
      response_format: { type: "json_schema", json_schema: schema },
      temperature: 0,
    },
    response,
  );
  assert.equal(attrs["gen_ai.request.model"], "requested-model");
  assert.equal(attrs["gen_ai.response.model"], "actual-model");
  assert.equal(attrs["gen_ai.response.id"], "actual-id");
  assert.deepEqual(
    JSON.parse(attrs["traceloop.entity.input"]),
    JSON.parse(JSON.stringify(messages)),
  );
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), tools);
  assert.deepEqual(
    JSON.parse(attrs["traceloop.entity.output"])[0],
    response.choices[0].message,
  );
  assert.equal(
    JSON.parse(attrs["gen_ai.completion.0.tool_calls"])[0].id,
    "current",
  );
  assert.equal(attrs["gen_ai.usage.input_tokens"], 0);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 0);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 0);
  assert.equal(attrs["llm.usage.total_tokens"], undefined);
  assert.equal(attrs["gen_ai.request.temperature"], 0);
  for (const key of [
    "respan.span.tools",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "status_code",
    "traceloop.span.kind",
  ])
    assert.equal(attrs[key], undefined);
});
test("stream reconstruction preserves choice indices, native IDs and tool fragments without invented fields", () => {
  const s = createChatStreamState({ model: "request" });
  updateChatStreamState(s, {
    id: "native-id",
    model: "response",
    choices: [
      { index: 1, delta: { role: "assistant", content: "" } },
      {
        index: 0,
        delta: {
          role: "assistant",
          content: "Hi ",
          tool_calls: [
            {
              index: 0,
              id: "current",
              type: "function",
              function: { name: "loo", arguments: '{"zero":' },
            },
          ],
        },
      },
    ],
  });
  updateChatStreamState(s, {
    choices: [
      {
        index: 0,
        finish_reason: "tool_calls",
        delta: {
          content: "there",
          tool_calls: [
            { index: 0, function: { name: "kup", arguments: "0}" } },
          ],
        },
      },
    ],
    usage: { prompt_tokens: 0, completion_tokens: 2, total_tokens: 2 },
  });
  const r = buildChatCompletionFromStreamState(s, { model: "request" });
  assert.equal(r.id, "native-id");
  assert.equal(r.model, "response");
  assert.equal(r.created, undefined);
  assert.equal(r.choices[0].message.content, "Hi there");
  assert.equal(r.choices[0].message.tool_calls[0].function.name, "lookup");
  assert.equal(
    r.choices[0].message.tool_calls[0].function.arguments,
    '{"zero":0}',
  );
  assert.equal(r.choices[1].message.content, "");
  const text = createTextStreamState();
  updateTextStreamState(text, { value: "" });
  updateTextStreamState(text, { value: "text" });
  assert.deepEqual(buildCompletionFromStreamState(text), {
    choices: [{ text: "text" }],
  });
});
test("error status comes only from an actual status and safe error messages", () => {
  const attrs = buildErrorAttrs(
    "chat",
    {},
    Object.assign(new Error("fixture rejection"), { status: 429 }),
  );
  assert.equal(attrs["http.response.status_code"], 429);
  assert.equal(attrs["error.message"], "fixture rejection");
  assert.equal(
    buildErrorAttrs("chat", {}, new Error("network"))[
      "http.response.status_code"
    ],
    undefined,
  );
  let reads = 0;
  const e = {
    get message() {
      reads++;
      throw Error("telemetry getter");
    },
    toString() {
      reads++;
      throw Error("telemetry string");
    },
  };
  buildErrorAttrs("chat", {}, e);
  assert.equal(reads, 0);
});
test("mapping never invokes getters, toJSON, toString, custom iterators or proxy traps", () => {
  let reads = 0;
  const poison = {
    get secret() {
      reads++;
      throw Error("getter");
    },
    toJSON() {
      reads++;
      throw Error("toJSON");
    },
    toString() {
      reads++;
      throw Error("toString");
    },
    [Symbol.iterator]() {
      reads++;
      throw Error("iterator");
    },
  };
  const proxy = new Proxy(
    {},
    {
      ownKeys() {
        reads++;
        throw Error("proxy");
      },
      getOwnPropertyDescriptor() {
        reads++;
        throw Error("proxy");
      },
      get() {
        reads++;
        throw Error("proxy");
      },
    },
  );
  const attrs = buildSuccessAttrs(
    "chat",
    { messages: [{ role: "user", content: poison }], tools: [proxy] },
    { choices: [{ message: { role: "assistant", content: proxy } }] },
  );
  assert.equal(reads, 0);
  assert.equal(
    JSON.parse(attrs["traceloop.entity.input"])[0].content.secret,
    undefined,
  );
});
