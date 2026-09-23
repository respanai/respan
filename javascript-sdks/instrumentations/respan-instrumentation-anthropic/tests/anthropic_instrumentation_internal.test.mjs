import assert from "node:assert/strict";
import { AsyncLocalStorage } from "node:async_hooks";
import test from "node:test";

import { context, ROOT_CONTEXT, SpanStatusCode, trace } from "@opentelemetry/api";
import { hrTime } from "@opentelemetry/core";
import { propagateAttributes } from "@respan/tracing";

import {
  formatInputMessages,
} from "../dist/_helpers.js";
import {
  buildMessageFromStreamState,
  createStreamState,
  patchMessagesPrototype,
  updateStreamState,
  wrapStreamingCreateResult,
} from "../dist/_streaming.js";

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);
const contextStorage = new AsyncLocalStorage();
const contextManager = {
  active() {
    return contextStorage.getStore() ?? ROOT_CONTEXT;
  },
  with(ctx, fn, thisArg, ...args) {
    return contextStorage.run(ctx, () => fn.apply(thisArg, args));
  },
  bind(_ctx, target) {
    return target;
  },
  enable() {
    return this;
  },
  disable() {
    contextStorage.disable();
    return this;
  },
};

test.before(() => {
  context.setGlobalContextManager(contextManager);
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
  context.disable();
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

test("stream consumption emits one streaming chat span", async () => {
  captureState.spans = [];
  const events = [
    {
      type: "message_start",
      message: {
        model: "claude-3-7-sonnet",
        usage: { input_tokens: 11 },
        content: [],
      },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "Streamed response." },
    },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 3 },
    },
  ];
  const stream = {
    async *[Symbol.asyncIterator]() {
      yield* events;
    },
  };

  const wrapped = wrapStreamingCreateResult(
    stream,
    {
      model: "claude-3-7-sonnet",
      stream: true,
      messages: [{ role: "user", content: "Stream this" }],
    },
    hrTime(),
  );
  for await (const _event of wrapped) {
    // Consume the complete stream so the instrumentation emits its final span.
  }

  assert.equal(captureState.spans.length, 1);
  const [span] = captureState.spans;
  assert.equal(span.attributes["llm.is_streaming"], true);
  assert.equal(span.attributes["gen_ai.usage.input_tokens"], 11);
  assert.equal(span.attributes["gen_ai.usage.output_tokens"], 3);
  assert.equal(span.attributes["llm.usage.total_tokens"], 14);
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
  assert.equal(span.instrumentationScope?.version, "1.1.2");
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
  assert.deepEqual(JSON.parse(span.attributes["llm.request.functions"]), [
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
  assert.deepEqual(JSON.parse(span.attributes["gen_ai.completion.0.tool_calls"]), [
    {
      id: "toolu_123",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(span.attributes["gen_ai.usage.input_tokens"], 4);
  assert.equal(span.attributes["gen_ai.usage.output_tokens"], 2);
  assert.equal(span.attributes["gen_ai.usage.prompt_tokens"], 4);
  assert.equal(span.attributes["gen_ai.usage.completion_tokens"], 2);
  assert.equal(span.attributes["llm.usage.total_tokens"], 6);
  assert.equal(span.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(span.attributes["gen_ai.prompt.0.content"], "Hi");
  assert.equal(span.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(span.attributes["gen_ai.completion.0.content"], "Hello from Anthropic.");
  assert.equal(span.attributes["llm.is_streaming"], false);

  for (const alias of ["respan.span.tools", "respan.span.tool_calls", "tools", "tool_calls"]) {
    assert.equal(span.attributes[alias], undefined, `${alias} must not be emitted`);
  }

  messagesPrototype.create = patchedTarget.originalCreate;
});

test("rejecting create calls preserve status and propagated audit attributes", async () => {
  captureState.spans = [];
  const rejection = Object.assign(new Error("Anthropic model was not found"), {
    status: 404,
  });
  const messagesPrototype = {
    create() {
      return Promise.reject(rejection);
    },
  };

  const patchedTarget = patchMessagesPrototype(messagesPrototype);
  assert.ok(patchedTarget);

  await propagateAttributes(
    {
      custom_identifier: "anthropic-error-case",
      trace_group_identifier: "anthropic-error-group",
      metadata: {
        run_id: "anthropic-error-marker",
        case_id: "failure",
      },
    },
    async () => {
      await messagesPrototype.create({
        model: "missing-model",
        messages: [
          {
            role: "assistant",
            content: [
              {
                type: "tool_use",
                id: "toolu_history",
                name: "lookup_weather",
                input: { city: "Tokyo" },
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_history",
                content: "sunny",
              },
            ],
          },
        ],
      }).catch((error) => {
        assert.equal(error, rejection);
      });
    },
  );

  assert.equal(captureState.spans.length, 2);
  const span = captureState.spans.find(
    (candidate) => candidate.attributes["respan.entity.log_type"] === "chat",
  );
  const toolSpan = captureState.spans.find(
    (candidate) => candidate.attributes["respan.entity.log_type"] === "tool",
  );
  assert.ok(span);
  assert.ok(toolSpan);
  assert.equal(span.status.code, SpanStatusCode.ERROR);
  assert.equal(span.status.message, "Error: Anthropic model was not found");
  assert.equal(span.attributes.status_code, 404);
  assert.equal(span.attributes["error.message"], "Error: Anthropic model was not found");
  assert.equal(span.attributes["respan.span_params.custom_identifier"], "anthropic-error-case");
  assert.equal(span.attributes["respan.trace.trace_group_identifier"], "anthropic-error-group");
  assert.deepEqual(JSON.parse(span.attributes["respan.metadata"]), {
    run_id: "anthropic-error-marker",
    case_id: "failure",
  });
  assert.equal(
    Object.keys(span.attributes).some((key) => key.startsWith("respan.metadata.")),
    false,
  );
  assert.deepEqual(JSON.parse(span.attributes["gen_ai.prompt.0.tool_calls"]), [
    {
      id: "toolu_history",
      type: "function",
      function: {
        name: "lookup_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(span.attributes["gen_ai.prompt.1.role"], "tool");
  assert.equal(span.attributes["gen_ai.prompt.1.content"], "sunny");
  assert.equal(span.attributes["gen_ai.prompt.1.tool_call_id"], "toolu_history");
  assert.equal(span.attributes["gen_ai.completion.0.tool_calls"], undefined);
  assert.equal(span.instrumentationScope?.version, "1.1.2");
  assert.deepEqual(JSON.parse(toolSpan.attributes["respan.metadata"]), {
    run_id: "anthropic-error-marker",
    case_id: "failure",
  });
  assert.equal(
    Object.keys(toolSpan.attributes).some((key) => key.startsWith("respan.metadata.")),
    false,
  );
  assert.equal(toolSpan.instrumentationScope?.version, "1.1.2");

  messagesPrototype.create = patchedTarget.originalCreate;
});
