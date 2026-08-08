import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  buildChatCompletionFromStreamState,
  buildCompletionFromStreamState,
  buildErrorAttrs,
  buildSuccessAttrs,
  createChatStreamState,
  createTextStreamState,
  updateChatStreamState,
  updateTextStreamState,
} from "../dist/index.js";
import { patchWriterMethod } from "../dist/_streaming.js";

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

test("buildSuccessAttrs maps Writer chat tools and usage to canonical fields only", () => {
  const attrs = buildSuccessAttrs(
    "chat",
    {
      model: "palmyra-x5",
      messages: [
        { role: "system", content: "Be brief." },
        { role: "user", content: "Find the weather in Tokyo." },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Lookup weather.",
            parameters: {
              type: "object",
              properties: { city: { type: "string" } },
            },
          },
        },
      ],
      tool_choice: "auto",
      response_format: { type: "json_schema", json_schema: { type: "object" } },
    },
    {
      model: "palmyra-x5",
      choices: [
        {
          message: {
            role: "assistant",
            content: "",
            tool_calls: [
              {
                id: "call_123",
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
        prompt_tokens: 11,
        completion_tokens: 5,
        total_tokens: 16,
        prompt_token_details: { cached_tokens: 2 },
      },
    },
  );

  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.system"], "writer");
  assert.equal(attrs["gen_ai.request.model"], "palmyra-x5");
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.1.content"], "Find the weather in Tokyo.");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_123",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Lookup weather.",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
        },
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 11);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 11);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 5);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 5);
  assert.equal(attrs["llm.usage.total_tokens"], 16);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 2);
  assert.equal(attrs["respan.span.tools"], undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs.tools, undefined);
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs.model, undefined);
  assert.equal(attrs.prompt_tokens, undefined);
  assert.equal(attrs.completion_tokens, undefined);
});

test("buildSuccessAttrs maps Writer text completions", () => {
  const attrs = buildSuccessAttrs(
    "completion",
    {
      model: "palmyra-x-003-instruct",
      prompt: "Write one sentence about observability.",
      max_tokens: 32,
      temperature: 0.2,
      top_p: 0.9,
    },
    {
      model: "palmyra-x-003-instruct",
      choices: [{ text: "Observability turns runtime behavior into evidence." }],
      usage: { prompt_tokens: 7, completion_tokens: 6, total_tokens: 13 },
    },
  );

  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Write one sentence about observability.");
  assert.equal(attrs["gen_ai.completion.0.content"], "Observability turns runtime behavior into evidence.");
  assert.equal(attrs["gen_ai.request.max_tokens"], 32);
  assert.equal(attrs["gen_ai.request.temperature"], 0.2);
  assert.equal(attrs["gen_ai.request.top_p"], 0.9);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 6);
  assert.equal(attrs["llm.usage.total_tokens"], 13);
  assert.equal(attrs.tool_calls, undefined);
});

test("stream states reconstruct Writer chat and completion responses", () => {
  const chatBody = { model: "palmyra-x5" };
  const chatState = createChatStreamState(chatBody);
  updateChatStreamState(chatState, {
    id: "chatcmpl_1",
    model: "palmyra-x5",
    choices: [{ delta: { content: "Hello " }, finish_reason: null, index: 0 }],
  });
  updateChatStreamState(chatState, {
    choices: [
      {
        delta: {
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "lookup", arguments: "{\"q\":\"res" },
            },
          ],
        },
      },
    ],
  });
  updateChatStreamState(chatState, {
    choices: [
      {
        delta: {
          content: "world",
          tool_calls: [
            {
              index: 0,
              function: { arguments: "pan\"}" },
            },
          ],
        },
      },
    ],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  });

  const chatCompletion = buildChatCompletionFromStreamState(chatState, chatBody);
  assert.equal(chatCompletion.choices[0].message.content, "Hello world");
  assert.deepEqual(chatCompletion.choices[0].message.tool_calls, [
    {
      id: "call_1",
      type: "function",
      function: {
        name: "lookup",
        arguments: "{\"q\":\"respan\"}",
      },
    },
  ]);

  const textState = createTextStreamState({ model: "palmyra-x-003-instruct" });
  updateTextStreamState(textState, { value: "first " });
  updateTextStreamState(textState, { value: "second" });
  assert.deepEqual(buildCompletionFromStreamState(textState, { model: "fallback" }), {
    model: "palmyra-x-003-instruct",
    choices: [{ text: "first second" }],
  });
});

test("patchWriterMethod emits success, error, and tool execution spans", async () => {
  captureState.spans = [];

  const target = {
    chat(body) {
      return Promise.resolve({
        model: body.model,
        choices: [{ message: { role: "assistant", content: "Done." } }],
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      });
    },
  };

  const patchedTarget = patchWriterMethod(target, "chat", "chat");
  assert.ok(patchedTarget);

  const result = await target.chat({
    model: "palmyra-x5",
    messages: [
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "lookup", arguments: "{\"q\":\"respan\"}" },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_1",
        content: "{\"answer\":\"ok\"}",
      },
      { role: "user", content: "Summarize it." },
    ],
  });

  assert.equal(result.choices[0].message.content, "Done.");
  assert.equal(captureState.spans.length, 2);
  const [toolSpan, chatSpan] = captureState.spans;
  assert.equal(toolSpan.attributes["respan.entity.log_type"], "tool");
  assert.equal(toolSpan.attributes["traceloop.entity.name"], "lookup");
  assert.equal(chatSpan.instrumentationScope?.name, "@respan/instrumentation-writer");
  assert.equal(chatSpan.attributes["respan.entity.log_method"], "ts_tracing");
  assert.equal(chatSpan.attributes["respan.entity.log_type"], "chat");
  assert.equal(chatSpan.attributes["gen_ai.usage.completion_tokens"], 1);

  target.chat = function () {
    return Promise.reject(Object.assign(new Error("Writer failed"), { status: 429 }));
  };
  patchWriterMethod(target, "chat", "chat");
  await assert.rejects(
    target.chat({ model: "palmyra-x5", messages: [{ role: "user", content: "Hi" }] }),
    /Writer failed/,
  );
  const errorSpan = captureState.spans.at(-1);
  assert.equal(errorSpan.attributes["error.message"], "Writer failed");
  assert.equal(errorSpan.attributes.status_code, 429);
});

test("buildErrorAttrs emits backend-visible status and error message", () => {
  const attrs = buildErrorAttrs(
    "chat",
    { model: "palmyra-x5", messages: [{ role: "user", content: "Hi" }] },
    Object.assign(new Error("Rate limited"), { status: 429 }),
  );

  assert.equal(attrs["error.message"], "Rate limited");
  assert.equal(attrs.status_code, 429);
  assert.equal(attrs.tool_calls, undefined);
});
