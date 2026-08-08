import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  formatInputMessages,
} from "../dist/_helpers.js";
import {
  buildMessageFromStreamState,
  createStreamState,
  patchMessagesPrototype,
  updateStreamState,
} from "../dist/_streaming.js";

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

test("formatInputMessages normalizes system prompts, tool calls, and tool results", () => {
  const messages = formatInputMessages(
    [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Checking weather" },
          {
            type: "tool_use",
            id: "toolu_123",
            name: "get_weather",
            input: { city: "Tokyo" },
          },
          {
            type: "tool_result",
            tool_use_id: "toolu_123",
            content: { forecast: "sunny" },
          },
        ],
      },
    ],
    ["Be concise.", { text: "Use tools when helpful." }],
  );

  assert.deepEqual(messages, [
    {
      role: "system",
      content: "Be concise.\nUse tools when helpful.",
    },
    {
      role: "assistant",
      content: "Checking weather",
      tool_calls: [
        {
          id: "toolu_123",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "{\"city\":\"Tokyo\"}",
          },
        },
      ],
    },
    {
      role: "tool",
      content: "{\"forecast\":\"sunny\"}",
      tool_call_id: "toolu_123",
    },
  ]);
});

test("stream reconstruction reassembles tool input deltas and usage", () => {
  const state = createStreamState();

  updateStreamState(state, {
    type: "message_start",
    message: {
      model: "claude-3-7-sonnet",
      usage: { input_tokens: 11 },
      content: [],
    },
  });
  updateStreamState(state, {
    type: "content_block_start",
    index: 0,
    content_block: {
      type: "tool_use",
      id: "toolu_123",
      name: "get_weather",
      input: "",
    },
  });
  updateStreamState(state, {
    type: "content_block_delta",
    index: 0,
    delta: {
      type: "input_json_delta",
      partial_json: "{\"city\":\"Tok",
    },
  });
  updateStreamState(state, {
    type: "content_block_delta",
    index: 0,
    delta: {
      type: "input_json_delta",
      partial_json: "yo\"}",
    },
  });
  updateStreamState(state, {
    type: "content_block_start",
    index: 1,
    content_block: { type: "text", text: "" },
  });
  updateStreamState(state, {
    type: "content_block_delta",
    index: 1,
    delta: { type: "text_delta", text: "Tokyo is sunny." },
  });
  updateStreamState(state, {
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: { output_tokens: 7 },
  });

  const message = buildMessageFromStreamState(state, { model: "fallback-model" });

  assert.deepEqual(message, {
    model: "claude-3-7-sonnet",
    usage: {
      input_tokens: 11,
      output_tokens: 7,
    },
    content: [
      {
        type: "tool_use",
        id: "toolu_123",
        name: "get_weather",
        input: { city: "Tokyo" },
      },
      {
        type: "text",
        text: "Tokyo is sunny.",
      },
    ],
    stop_reason: "end_turn",
    stop_sequence: null,
  });
});

test("patchMessagesPrototype emits a chat span for successful create calls", async () => {
  captureState.spans = [];

  const messagesPrototype = {
    create(_body) {
      return Promise.resolve({
        model: "claude-3-5-haiku",
        content: [
          { type: "text", text: "Hello from Anthropic." },
          {
            type: "tool_use",
            id: "toolu_123",
            name: "get_weather",
            input: { city: "Tokyo" },
          },
        ],
        usage: {
          input_tokens: 4,
          output_tokens: 2,
        },
      });
    },
  };

  const patchedTarget = patchMessagesPrototype(messagesPrototype);
  assert.ok(patchedTarget);

  const result = await messagesPrototype.create({
    model: "claude-3-5-haiku",
    messages: [{ role: "user", content: "Hi" }],
    tools: [
      {
        name: "get_weather",
        description: "Lookup the current weather.",
        input_schema: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
        },
      },
    ],
  }).then((value) => value);

  assert.equal(result.model, "claude-3-5-haiku");
  assert.equal(captureState.spans.length, 1);

  const [span] = captureState.spans;
  assert.equal(span.instrumentationScope?.name, "@respan/instrumentation-anthropic");
  assert.equal(span.attributes["respan.entity.log_method"], "ts_tracing");
  assert.equal(span.attributes["respan.entity.log_type"], "chat");
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.input"]), [
    { role: "user", content: "Hi" },
  ]);
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.output"]), [
    {
      role: "assistant",
      content: "Hello from Anthropic.",
      tool_calls: [
        {
          id: "toolu_123",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "{\"city\":\"Tokyo\"}",
          },
        },
      ],
    },
  ]);
  assert.deepEqual(span.attributes["respan.span.tools"], [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Lookup the current weather.",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
        },
      },
    },
  ]);
  assert.deepEqual(span.attributes["respan.span.tool_calls"], [
    {
      id: "toolu_123",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.deepEqual(span.attributes.tools, [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Lookup the current weather.",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
        },
      },
    },
  ]);
  assert.deepEqual(span.attributes.tool_calls, [
    {
      id: "toolu_123",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);

  messagesPrototype.create = patchedTarget.originalCreate;
});
