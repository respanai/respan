import assert from "node:assert/strict";
import test from "node:test";

import * as HeliconeHelpers from "@helicone/helpers";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  ENTITY_NAME_KEY,
  propagateAttributes,
} from "@respan/tracing";

import {
  HeliconeInstrumentor,
  instrumentHelicone,
} from "../dist/index.js";

const captureState = { spans: [] };
const fetchState = { calls: [] };
const originalFetch = globalThis.fetch;
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);
const originalConsoleError = console.error;
const contextManager = new AsyncLocalStorageContextManager();

test.before(() => {
  context.setGlobalContextManager(contextManager.enable());
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
  console.error = () => {};
});

test.after(() => {
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
  globalThis.fetch = originalFetch;
  console.error = originalConsoleError;
  contextManager.disable();
});

test.beforeEach(() => {
  captureState.spans = [];
  fetchState.calls = [];
  globalThis.fetch = async (input, init) => {
    fetchState.calls.push({ input: String(input), init });
    return new Response(null, { status: 204 });
  };
});

function createLogger(headers = {
  Authorization: "Bearer provider-secret",
  "X-Internal-Token": "another-secret",
  "Helicone-User-Id": "constructor-user",
  "Helicone-Session-Id": "constructor-session",
  "Helicone-Property-service": "constructor-service",
}) {
  return new HeliconeHelpers.HeliconeManualLogger({
    apiKey: "helicone-secret-that-must-not-be-traced",
    headers,
    loggingEndpoint: "http://127.0.0.1:43199",
  });
}

async function withInstrumentor(fn, options = {}) {
  const instrumentor = new HeliconeInstrumentor({
    sdkModule: HeliconeHelpers,
    ...options,
  });
  await instrumentor.activate();
  try {
    return await fn(createLogger(), instrumentor);
  } finally {
    instrumentor.deactivate();
  }
}

function attributesAt(index = 0) {
  assert.ok(captureState.spans[index], `missing captured span at index ${index}`);
  return captureState.spans[index].attributes;
}

function assertNoOffContractAliases(attrs) {
  for (const key of [
    "respan.span.tools",
    "respan.span.tool_calls",
    "respan.span.handoffs",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
    "span_tools",
    "has_tool_calls",
    "parallel_tool_calls",
  ]) {
    assert.equal(attrs[key], undefined, `${key} must not be emitted`);
  }
}

test("logRequest emits one canonical chat span with tools, usage, and safe correlations", async () => {
  await withInstrumentor(async (logger) => {
    const request = {
      model: "gpt-4o-mini",
      api_key: "request-secret",
      messages: [
        { role: "system", content: "Use the weather tool." },
        { role: "user", content: "Weather in Tokyo?" },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Look up weather.",
            parameters: {
              type: "object",
              properties: { city: { type: "string" }, apiKey: { type: "string" } },
            },
          },
        },
      ],
    };
    const response = {
      id: "chatcmpl-helicone-1",
      model: "gpt-4o-mini-2026-08-01",
      choices: [
        {
          message: {
            role: "assistant",
            content: "Calling the weather tool.",
            tool_calls: [
              {
                id: "call_weather",
                type: "function",
                function: {
                  name: "get_weather",
                  arguments: JSON.stringify({ city: "Tokyo" }),
                },
              },
            ],
          },
        },
      ],
      usage: { prompt_tokens: 14, completion_tokens: 6, total_tokens: 20 },
    };

    const result = await logger.logRequest(
      request,
      async (recorder) => {
        recorder.appendResults(response);
        return response;
      },
      {
        "Helicone-User-Id": "user-42",
        "Helicone-Session-Id": "session-42",
        "Helicone-Property-environment": "test",
        Authorization: "Bearer must-not-leak",
      },
      "openai",
    );
    assert.equal(result, response);
  });

  assert.equal(captureState.spans.length, 1, "success path must not duplicate");
  assert.equal(fetchState.calls.length, 1, "Helicone logging behavior must remain intact");
  const span = captureState.spans[0];
  const attrs = span.attributes;
  assert.equal(span.instrumentationScope.name, "@respan/instrumentation-helicone");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["gen_ai.response.model"], "gpt-4o-mini-2026-08-01");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.prompt.1.role"], "user");
  assert.equal(attrs["gen_ai.prompt.1.content"], "Weather in Tokyo?");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Calling the weather tool.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    responseToolCall("call_weather", "get_weather", { city: "Tokyo" }),
  ]);
  const definitions = JSON.parse(attrs["llm.request.functions"]);
  assert.equal(definitions[0].function.name, "get_weather");
  assert.equal(
    definitions[0].function.parameters.properties.apiKey,
    "[REDACTED]",
  );
  assert.equal(attrs["gen_ai.usage.input_tokens"], 14);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 6);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 14);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 6);
  assert.equal(attrs["llm.usage.total_tokens"], 20);
  assert.equal(attrs["respan.customer_params.customer_identifier"], "user-42");
  assert.equal(attrs["respan.threads.thread_identifier"], "session-42");
  assert.deepEqual(JSON.parse(attrs["traceloop.association.properties"]), {
    service: "constructor-service",
    environment: "test",
  });
  assert.equal(attrs.authorization, undefined);
  assert.equal(attrs["x-internal-token"], undefined);
  assert.ok(!JSON.stringify(attrs).includes("provider-secret"));
  assert.ok(!JSON.stringify(attrs).includes("must-not-leak"));
  assertNoOffContractAliases(attrs);
});

test("Anthropic content blocks, tools, tool use, and cache usage map canonically", async () => {
  const multimodalPrompt = [
    { type: "text", text: "Inspect this order label." },
    {
      type: "image",
      source: { type: "base64", media_type: "image/png", data: "sample-image-data" },
    },
  ];
  const historicalToolUse = {
    type: "tool_use",
    id: "toolu_history",
    name: "lookup_order",
    input: { order_id: "H-1024" },
  };
  const twoTextBlocks = [
    { type: "text", text: "Keep this boundary. " },
    { type: "text", text: "And this one." },
  ];
  const responseContent = [
    { type: "text", text: "I will check that order." },
    {
      type: "tool_use",
      id: "toolu_current",
      name: "lookup_order",
      input: { order_id: "H-2048" },
    },
  ];

  await withInstrumentor(async (logger) => {
    await logger.sendLog(
      {
        model: "claude-sonnet-4-20250514",
        system: "Be concise.",
        messages: [
          { role: "user", content: multimodalPrompt },
          {
            role: "assistant",
            content: [
              { type: "text", text: "I need the lookup tool." },
              historicalToolUse,
            ],
          },
          {
            role: "user",
            content: [{
              type: "tool_result",
              tool_use_id: "toolu_history",
              content: "Order found.",
            }],
          },
          { role: "user", content: twoTextBlocks },
        ],
        tools: [{
          name: "lookup_order",
          description: "Look up an order.",
          input_schema: {
            type: "object",
            properties: { order_id: { type: "string" } },
            required: ["order_id"],
          },
        }],
      },
      {
        id: "msg_anthropic_1",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-20250514-rev1",
        content: responseContent,
        usage: {
          input_tokens: 21,
          output_tokens: 8,
          cache_read_input_tokens: 13,
        },
      },
      { startTime: Date.now() - 10, endTime: Date.now(), status: 200 },
    );
  });

  const attrs = attributesAt();
  assert.equal(attrs["gen_ai.system"], "anthropic");
  assert.equal(attrs["gen_ai.request.model"], "claude-sonnet-4-20250514");
  assert.equal(attrs["gen_ai.response.model"], "claude-sonnet-4-20250514-rev1");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Be concise.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.1.content"]), multimodalPrompt);
  assert.deepEqual(
    JSON.parse(attrs["gen_ai.prompt.2.tool_calls"]),
    [responseToolCall("toolu_history", "lookup_order", { order_id: "H-1024" })],
  );
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.4.content"]), twoTextBlocks);
  const entityInput = JSON.parse(attrs["traceloop.entity.input"]);
  assert.deepEqual(entityInput[0], { role: "system", content: "Be concise." });
  const tools = JSON.parse(attrs["llm.request.functions"]);
  assert.equal(tools[0].function.name, "lookup_order");
  assert.equal(tools[0].function.parameters.type, "object");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.content"]), responseContent);
  assert.deepEqual(
    JSON.parse(attrs["gen_ai.completion.0.tool_calls"]),
    [responseToolCall("toolu_current", "lookup_order", { order_id: "H-2048" })],
  );
  assert.equal(attrs["gen_ai.usage.input_tokens"], 21);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 8);
  assert.equal(attrs["gen_ai.usage.cache_read.input_tokens"], 13);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 13);
  assert.equal(attrs["llm.usage.total_tokens"], 29);
  assertNoOffContractAliases(attrs);
});

test("prompt indexing is capped, missing roles default, and JSON strings redact", async () => {
  const serializedPrompt = JSON.stringify({
    privateKey: "serialized-private-secret",
    nested: { authToken: "serialized-auth-secret" },
    promptTokens: 12,
    tokenizer: "cl100k_base",
  });
  const serializedCompletion = JSON.stringify({
    clientSecret: "serialized-client-secret",
    completionTokens: 4,
    tokenCount: 16,
  });
  const messages = Array.from({ length: 130 }, (_, index) => ({
    content: index === 0 ? serializedPrompt : `message-${index}`,
  }));

  await withInstrumentor(async (logger) => {
    const now = Date.now();
    await logger.sendLog(
      { model: "bounded-history-model", messages },
      {
        choices: [{ message: { content: serializedCompletion } }],
        usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
      },
      { startTime: now - 2, endTime: now, status: 200 },
    );
  });

  const attrs = attributesAt();
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.0.content"]), {
    privateKey: "[REDACTED]",
    nested: { authToken: "[REDACTED]" },
    promptTokens: 12,
    tokenizer: "cl100k_base",
  });
  assert.equal(attrs["gen_ai.prompt.127.role"], "user");
  assert.equal(attrs["gen_ai.prompt.127.content"], "message-127");
  assert.equal(attrs["gen_ai.prompt.128.role"], undefined);
  assert.equal(attrs["gen_ai.prompt.128.content"], undefined);

  const entityInput = JSON.parse(attrs["traceloop.entity.input"]);
  assert.equal(entityInput.length, 130);
  assert.equal(entityInput[0].role, "user");
  assert.deepEqual(JSON.parse(entityInput[0].content), {
    privateKey: "[REDACTED]",
    nested: { authToken: "[REDACTED]" },
    promptTokens: 12,
    tokenizer: "cl100k_base",
  });
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.content"]), {
    clientSecret: "[REDACTED]",
    completionTokens: 4,
    tokenCount: 16,
  });
  const entityOutput = JSON.parse(attrs["traceloop.entity.output"]);
  assert.deepEqual(JSON.parse(entityOutput.content), {
    clientSecret: "[REDACTED]",
    completionTokens: 4,
    tokenCount: 16,
  });
  assert.ok(!JSON.stringify(attrs).includes("serialized-private-secret"));
  assert.ok(!JSON.stringify(attrs).includes("serialized-auth-secret"));
  assert.ok(!JSON.stringify(attrs).includes("serialized-client-secret"));
});

test("direct sendLog maps text completions and logical timing", async () => {
  const startTime = Date.now() - 250;
  const endTime = startTime + 100;
  await withInstrumentor(async (logger) => {
    await logger.sendLog(
      { model: "text-model-1", prompt: "Write one line." },
      {
        model: "text-model-1",
        choices: [{ text: "One traced line." }],
        usage: { input_tokens: 5, output_tokens: 4 },
      },
      {
        startTime,
        endTime,
        status: 200,
        timeToFirstToken: 18,
        provider: "custom-provider",
      },
    );
  });

  assert.equal(captureState.spans.length, 1);
  const span = captureState.spans[0];
  const attrs = span.attributes;
  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.system"], "custom-provider");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Write one line.");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Write one line." },
  ]);
  assert.equal(attrs["gen_ai.completion.0.content"], "One traced line.");
  assert.deepEqual(JSON.parse(attrs["respan.metadata"]).helicone, {
    status: 200,
    time_to_first_token_ms: 18,
    streaming: true,
  });
  assert.equal(attrs["respan.metadata.helicone_status"], undefined);
  assert.equal(attrs["respan.metadata.helicone_time_to_first_token_ms"], undefined);
  assert.equal(attrs["respan.metadata.helicone_streaming"], undefined);
  assert.equal(attrs["gen_ai.response.time_to_first_chunk"], 0.018);
  assert.equal(span.startTime[0], Math.trunc(startTime / 1000));
  assert.equal(span.endTime[0], Math.trunc(endTime / 1000));
});

test("single request, builder, and direct send each emit exactly once", async () => {
  await withInstrumentor(async (logger) => {
    await logger.logSingleRequest(
      { model: "single-model", messages: [{ role: "user", content: "single" }] },
      JSON.stringify({
        choices: [{ message: { role: "assistant", content: "single response" } }],
        usage: { prompt_tokens: 1, completion_tokens: 2 },
      }),
      { latencyMs: 25 },
    );

    const builder = logger.logBuilder({
      model: "builder-model",
      messages: [{ role: "user", content: "builder" }],
    });
    builder.setResponse(JSON.stringify({
      choices: [{ message: { role: "assistant", content: "builder response" } }],
      usage: { prompt_tokens: 3, completion_tokens: 4 },
    }));
    await builder.sendLog();

    const attachedBuilder = logger.logBuilder({
      model: "builder-attach-model",
      messages: [{ role: "user", content: "builder attach" }],
    });
    await attachedBuilder.attachStream(heliconeValueStream([
      { choices: [{ delta: { role: "assistant", content: "attached" } }] },
    ]));
    await attachedBuilder.sendLog();

    const readableBuilder = logger.logBuilder({
      model: "builder-readable-model",
      messages: [{ role: "user", content: "builder readable" }],
    });
    const readable = readableBuilder.toReadableStream(heliconeValueStream([
      { choices: [{ delta: { role: "assistant", content: "readable" } }] },
    ]));
    const reader = readable.getReader();
    while (!(await reader.read()).done) {
      // Drain the public readable stream so the builder captures its chunks.
    }
    await readableBuilder.sendLog();

    await logger.sendLog(
      { model: "direct-model", messages: [{ role: "user", content: "direct" }] },
      { choices: [{ message: { role: "assistant", content: "direct response" } }] },
      { startTime: Date.now() - 1, endTime: Date.now(), status: 200 },
    );
  });

  assert.equal(captureState.spans.length, 5);
  assert.equal(fetchState.calls.length, 5);
  assert.deepEqual(
    captureState.spans.map((span) => span.attributes["gen_ai.request.model"]),
    [
      "single-model",
      "builder-model",
      "builder-attach-model",
      "builder-readable-model",
      "direct-model",
    ],
  );
  assert.equal(
    captureState.spans[1].attributes["gen_ai.system"],
    undefined,
    "unknown builder providers must be omitted",
  );
});

test("delayed builders retain creation parent and every propagated correlation", async () => {
  await withInstrumentor(async () => {
    const logger = createLogger({});
    const parentA = trace.wrapSpanContext({
      traceId: "a".repeat(32),
      spanId: "b".repeat(16),
      traceFlags: 1,
    });
    const parentB = trace.wrapSpanContext({
      traceId: "c".repeat(32),
      spanId: "d".repeat(16),
      traceFlags: 1,
    });
    const contextA = trace
      .setSpan(context.active(), parentA)
      .setValue(ENTITY_NAME_KEY, "workflow.creation_a");
    const contextB = trace
      .setSpan(context.active(), parentB)
      .setValue(ENTITY_NAME_KEY, "workflow.send_b");
    let builder;

    context.with(contextA, () => propagateAttributes(
      {
        custom_identifier: "custom-a",
        trace_group_identifier: "trace-group-a",
        customer_identifier: "customer-a",
        thread_identifier: "thread-a",
        metadata: { run_id: "run-a", example_run_id: "run-a" },
      },
      () => {
        builder = logger.logBuilder({
          model: "delayed-builder-model",
          messages: [{ role: "user", content: "created in A" }],
        });
      },
    ));

    assert.ok(builder);
    builder.setResponse(JSON.stringify({
      choices: [{ message: { role: "assistant", content: "sent in B" } }],
    }));
    await context.with(contextB, () => propagateAttributes(
      {
        custom_identifier: "custom-b",
        trace_group_identifier: "trace-group-b",
        customer_identifier: "customer-b",
        thread_identifier: "thread-b",
        metadata: { run_id: "run-b", example_run_id: "run-b" },
      },
      () => builder.sendLog(),
    ));
  });

  assert.equal(captureState.spans.length, 1);
  const span = captureState.spans[0];
  assert.equal(span.spanContext().traceId, "a".repeat(32));
  assert.equal(span.parentSpanContext?.spanId, "b".repeat(16));
  assert.equal(span.attributes["traceloop.entity.path"], "workflow.creation_a");
  assert.equal(span.attributes["respan.span_params.custom_identifier"], "custom-a");
  assert.equal(span.attributes["respan.trace.trace_group_identifier"], "trace-group-a");
  assert.equal(span.attributes["respan.customer_params.customer_identifier"], "customer-a");
  assert.equal(span.attributes["respan.threads.thread_identifier"], "thread-a");
  assert.deepEqual(JSON.parse(span.attributes["respan.metadata"]), {
    run_id: "run-a",
    example_run_id: "run-a",
    helicone: { status: 200, operation: "logBuilder" },
  });
  assert.equal(span.attributes["respan.metadata.run_id"], undefined);
  assert.equal(span.attributes["respan.metadata.example_run_id"], undefined);
  const serializedAttributes = JSON.stringify(span.attributes);
  for (const value of ["custom-b", "trace-group-b", "customer-b", "thread-b", "run-b"]) {
    assert.ok(!serializedAttributes.includes(value), `${value} leaked from send context`);
  }
});

test("logStream and logSingleStream aggregate streamed content and usage", async () => {
  const chunks = [
    { choices: [{ delta: { role: "assistant", content: "streamed " } }] },
    { choices: [{ delta: { content: "answer" } }] },
    { choices: [{ delta: {} }], usage: { prompt_tokens: 8, completion_tokens: 3 } },
  ];

  await withInstrumentor(async (logger) => {
    const result = await logger.logStream(
      { model: "stream-model", messages: [{ role: "user", content: "stream" }] },
      async (recorder) => {
        recorder.attachStream(jsonLineStream(chunks));
        return "stream-result";
      },
      { "Helicone-Property-mode": "logStream" },
    );
    assert.equal(result, "stream-result");

    await logger.logSingleStream(
      { model: "single-stream-model", messages: [{ role: "user", content: "stream2" }] },
      jsonLineStream(chunks),
      { "Helicone-Property-mode": "logSingleStream" },
    );
  });

  assert.equal(captureState.spans.length, 2);
  for (const [index, span] of captureState.spans.entries()) {
    const attrs = span.attributes;
    assert.equal(attrs["gen_ai.completion.0.content"], "streamed answer");
    assert.equal(attrs["gen_ai.request.stream"], true);
    assert.equal(attrs["gen_ai.usage.input_tokens"], 8);
    assert.equal(attrs["gen_ai.usage.output_tokens"], 3);
    assert.equal(JSON.parse(attrs["respan.metadata"]).helicone.streaming, true);
    assert.equal(attrs["respan.metadata.helicone_streaming"], undefined);
    assert.equal(JSON.parse(attrs["respan.metadata"]).helicone.operation,
      index === 0 ? "logStream" : "logSingleStream");
  }
});

test("Anthropic, Google, and fragmented OpenAI stream payloads aggregate canonically", async () => {
  const anthropicChunks = [
    {
      type: "message_start",
      message: {
        role: "assistant",
        model: "claude-sonnet-4-stream-rev",
        content: [],
        usage: { input_tokens: 12, cache_read_input_tokens: 4 },
      },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Anthropic " } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "streamed." } },
    {
      type: "content_block_start",
      index: 1,
      content_block: {
        type: "tool_use",
        id: "toolu_stream",
        name: "lookup_order",
        input: {},
      },
    },
    {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: '{"order_id":' },
    },
    {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: '"H-2048"}' },
    },
    { type: "message_delta", usage: { output_tokens: 7 } },
  ];
  const googleChunks = [
    {
      modelVersion: "gemini-2.5-flash-rev",
      candidates: [{ content: { role: "model", parts: [{ text: "Google " }] } }],
    },
    {
      candidates: [{ content: { role: "model", parts: [{ text: "streamed." }] } }],
    },
    {
      candidates: [{
        content: {
          role: "model",
          parts: [{ functionCall: { name: "lookup_order", args: { order_id: "G-42" } } }],
        },
      }],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 5,
        totalTokenCount: 15,
        cachedContentTokenCount: 3,
      },
    },
  ];
  const openAiChunks = [
    {
      model: "gpt-4o-mini-stream-rev",
      choices: [{ delta: {
        role: "assistant",
        tool_calls: [{
          index: 0,
          id: "call_stream",
          type: "function",
          function: { name: "lookup_order", arguments: "" },
        }],
      } }],
    },
    {
      choices: [{ delta: {
        tool_calls: [{ index: 0, function: { arguments: '{"order_id":' } }],
      } }],
    },
    {
      choices: [{ delta: {
        tool_calls: [{ index: 0, function: { arguments: '"O-42"}' } }],
      } }],
      usage: { prompt_tokens: 9, completion_tokens: 4, total_tokens: 13 },
    },
  ];

  await withInstrumentor(async (logger) => {
    const now = Date.now();
    await logger.sendLog(
      {
        model: "claude-sonnet-4-stream",
        messages: [{ role: "user", content: "Stream with Anthropic." }],
        tools: [{ name: "lookup_order", input_schema: { type: "object" } }],
      },
      anthropicChunks.map((chunk) => `data: ${JSON.stringify(chunk)}`).join("\n"),
      { startTime: now - 20, endTime: now, status: 200, timeToFirstToken: 2 },
    );
    await logger.sendLog(
      {
        model: "gemini-2.5-flash",
        contents: [{ role: "user", parts: [{ text: "Stream with Google." }] }],
        tools: [{ functionDeclarations: [{
          name: "lookup_order",
          description: "Look up an order.",
          parameters: { type: "object" },
        }] }],
      },
      googleChunks.map((chunk) => JSON.stringify(chunk)).join("\n"),
      { startTime: now - 15, endTime: now, status: 200, timeToFirstToken: 3 },
    );
    await logger.sendLog(
      {
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Stream one tool call." }],
      },
      openAiChunks.map((chunk) => `data: ${JSON.stringify(chunk)}`).join("\n"),
      { startTime: now - 10, endTime: now, status: 200, timeToFirstToken: 1 },
    );
  });

  assert.equal(captureState.spans.length, 3);
  const [anthropic, google, openai] = captureState.spans.map((span) => span.attributes);
  assert.equal(anthropic["gen_ai.system"], "anthropic");
  assert.equal(anthropic["gen_ai.response.model"], "claude-sonnet-4-stream-rev");
  assert.deepEqual(JSON.parse(anthropic["gen_ai.completion.0.content"]), [
    { type: "text", text: "Anthropic streamed." },
    {
      type: "tool_use",
      id: "toolu_stream",
      name: "lookup_order",
      input: { order_id: "H-2048" },
    },
  ]);
  assert.deepEqual(JSON.parse(anthropic["gen_ai.completion.0.tool_calls"]), [
    responseToolCall("toolu_stream", "lookup_order", { order_id: "H-2048" }),
  ]);
  assert.equal(anthropic["gen_ai.usage.input_tokens"], 12);
  assert.equal(anthropic["gen_ai.usage.output_tokens"], 7);
  assert.equal(anthropic["llm.usage.cache_read_input_tokens"], 4);

  assert.equal(google["gen_ai.system"], "google");
  assert.equal(google["gen_ai.response.model"], "gemini-2.5-flash-rev");
  assert.deepEqual(JSON.parse(google["gen_ai.prompt.0.content"]), [
    { text: "Stream with Google." },
  ]);
  assert.deepEqual(JSON.parse(google["gen_ai.completion.0.content"]), [
    { text: "Google streamed." },
    { functionCall: { name: "lookup_order", args: { order_id: "G-42" } } },
  ]);
  assert.deepEqual(JSON.parse(google["gen_ai.completion.0.tool_calls"]), [{
    type: "function",
    function: { name: "lookup_order", arguments: JSON.stringify({ order_id: "G-42" }) },
  }]);
  assert.equal(JSON.parse(google["llm.request.functions"])[0].function.name, "lookup_order");
  assert.equal(google["gen_ai.usage.input_tokens"], 10);
  assert.equal(google["gen_ai.usage.output_tokens"], 5);
  assert.equal(google["llm.usage.cache_read_input_tokens"], 3);

  assert.deepEqual(JSON.parse(openai["gen_ai.completion.0.tool_calls"]), [
    responseToolCall("call_stream", "lookup_order", { order_id: "O-42" }),
  ]);
  assert.equal(openai["gen_ai.usage.input_tokens"], 9);
  assert.equal(openai["gen_ai.usage.output_tokens"], 4);
});

test("Google non-stream response objects map candidates, parts, tools, and usage", async () => {
  await withInstrumentor(async (logger) => {
    const now = Date.now();
    await logger.sendLog(
      {
        model: "gemini-2.5-pro",
        contents: [{ role: "user", parts: [{ text: "Call the lookup function." }] }],
        tools: [{ functionDeclarations: [{
          name: "lookup_order",
          parameters: { type: "object", properties: { id: { type: "string" } } },
        }] }],
      },
      {
        modelVersion: "gemini-2.5-pro-rev",
        candidates: [{
          content: {
            role: "model",
            parts: [
              { text: "Calling lookup." },
              { functionCall: { name: "lookup_order", args: { id: "G-99" } } },
            ],
          },
        }],
        usageMetadata: {
          promptTokenCount: 8,
          candidatesTokenCount: 4,
          totalTokenCount: 12,
          cachedContentTokenCount: 2,
        },
      },
      { startTime: now - 4, endTime: now, status: 200 },
    );
  });

  const attrs = attributesAt();
  assert.equal(attrs["gen_ai.system"], "google");
  assert.equal(attrs["gen_ai.response.model"], "gemini-2.5-pro-rev");
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.0.content"]), [
    { text: "Call the lookup function." },
  ]);
  assert.equal(JSON.parse(attrs["llm.request.functions"])[0].function.name, "lookup_order");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [{
    type: "function",
    function: { name: "lookup_order", arguments: JSON.stringify({ id: "G-99" }) },
  }]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 8);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 4);
  assert.equal(attrs["llm.usage.total_tokens"], 12);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 2);
});

test("synthetic bare usage.tokens is not reported as provider usage", async () => {
  await withInstrumentor(async (logger) => {
    const now = Date.now();
    await logger.sendLog(
      { model: "synthetic-usage-model", messages: [{ role: "user", content: "usage" }] },
      {
        choices: [{ message: { role: "assistant", content: "ignored synthetic usage" } }],
        usage: { tokens: 999 },
      },
      { startTime: now - 1, endTime: now, status: 200 },
    );
  });

  const attrs = attributesAt();
  assert.equal(attrs["gen_ai.usage.input_tokens"], undefined);
  assert.equal(attrs["gen_ai.usage.output_tokens"], undefined);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], undefined);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], undefined);
  assert.equal(attrs["llm.usage.total_tokens"], undefined);
});

test("outer operation failures emit one error span and preserve the original error", async () => {
  const leakedCredential = "sk-proj-this-must-not-leak";
  await withInstrumentor(async (logger) => {
    await assert.rejects(
      logger.logRequest(
        { model: "failed-model", messages: [{ role: "user", content: "fail" }] },
        async () => {
          throw new TypeError(
            `provider operation failed Authorization: Bearer ${leakedCredential}`,
          );
        },
        { "Helicone-Session-Id": "failed-session" },
        "openai",
      ),
      /provider operation failed/,
    );

    await assert.rejects(
      logger.logStream(
        { model: "failed-stream-model", messages: [{ role: "user", content: "fail stream" }] },
        async () => {
          throw new Error("stream setup failed");
        },
      ),
      /stream setup failed/,
    );
  });

  assert.equal(captureState.spans.length, 2);
  assert.equal(fetchState.calls.length, 0, "failed operations never reached Helicone sendLog");
  assert.equal(captureState.spans[0].status.code, 2);
  assert.equal(
    captureState.spans[0].status.message,
    "provider operation failed Authorization: [REDACTED]",
  );
  assert.equal(attributesAt(0)["error.type"], "TypeError");
  assert.equal(attributesAt(0)["status_code"], undefined);
  assert.equal(attributesAt(0)["http.response.status_code"], 500);
  assert.equal(
    attributesAt(0)["error.message"],
    "provider operation failed Authorization: [REDACTED]",
  );
  assert.deepEqual(JSON.parse(attributesAt(0)["traceloop.entity.output"]), {
    error: "provider operation failed Authorization: [REDACTED]",
  });
  assert.ok(!JSON.stringify(captureState.spans[0]).includes(leakedCredential));
  assert.equal(attributesAt(0)["gen_ai.completion.0.content"], undefined);
  assert.equal(attributesAt(0)["respan.threads.thread_identifier"], "failed-session");
  assert.equal(captureState.spans[1].status.message, "stream setup failed");
});

test("nested and direct callback sends do not suppress the outer failure", async () => {
  await withInstrumentor(async (logger) => {
    const parentA = trace.wrapSpanContext({
      traceId: "1".repeat(32),
      spanId: "2".repeat(16),
      traceFlags: 1,
    });
    const parentB = trace.wrapSpanContext({
      traceId: "3".repeat(32),
      spanId: "4".repeat(16),
      traceFlags: 1,
    });
    const contextA = trace
      .setSpan(context.active(), parentA)
      .setValue(ENTITY_NAME_KEY, "workflow.a");
    const contextB = trace
      .setSpan(context.active(), parentB)
      .setValue(ENTITY_NAME_KEY, "workflow.b");
    const inA = (fn) => context.with(contextA, () => propagateAttributes(
      { metadata: { run_id: "run-a" } },
      fn,
    ));
    const inB = (fn) => context.with(contextB, () => propagateAttributes(
      { metadata: { run_id: "run-b" } },
      fn,
    ));

    await assert.rejects(inA(() => logger.logRequest(
      { model: "outer-nested-model", messages: [{ role: "user", content: "outer" }] },
      async () => {
        await inB(() => logger.logRequest(
          { model: "inner-model", messages: [{ role: "user", content: "inner" }] },
          async (recorder) => {
            const response = {
              choices: [{ message: { role: "assistant", content: "inner success" } }],
            };
            recorder.appendResults(response);
            return response;
          },
        ));
        throw new Error("outer failed after inner helper");
      },
    )), /outer failed after inner helper/);

    await assert.rejects(inA(() => logger.logRequest(
      { model: "outer-direct-model", messages: [{ role: "user", content: "outer direct" }] },
      async () => {
        const now = Date.now();
        await inB(() => logger.sendLog(
          { model: "callback-direct-model", messages: [{ role: "user", content: "direct" }] },
          { choices: [{ message: { role: "assistant", content: "direct success" } }] },
          { startTime: now - 1, endTime: now, status: 200 },
        ));
        throw new Error("outer failed after direct send");
      },
    )), /outer failed after direct send/);
  });

  assert.equal(captureState.spans.length, 4);
  assert.equal(fetchState.calls.length, 2);
  assert.deepEqual(
    captureState.spans.map((span) => ({
      model: span.attributes["gen_ai.request.model"],
      status: span.status.code,
    })),
    [
      { model: "inner-model", status: 1 },
      { model: "outer-nested-model", status: 2 },
      { model: "callback-direct-model", status: 1 },
      { model: "outer-direct-model", status: 2 },
    ],
  );
  assert.deepEqual(
    captureState.spans.map((span) => ({
      traceId: span.spanContext().traceId,
      parentId: span.parentSpanContext?.spanId,
      entityPath: span.attributes["traceloop.entity.path"],
      runId: JSON.parse(span.attributes["respan.metadata"]).run_id,
    })),
    [
      {
        traceId: "3".repeat(32),
        parentId: "4".repeat(16),
        entityPath: "workflow.b",
        runId: "run-b",
      },
      {
        traceId: "1".repeat(32),
        parentId: "2".repeat(16),
        entityPath: "workflow.a",
        runId: "run-a",
      },
      {
        traceId: "3".repeat(32),
        parentId: "4".repeat(16),
        entityPath: "workflow.b",
        runId: "run-b",
      },
      {
        traceId: "1".repeat(32),
        parentId: "2".repeat(16),
        entityPath: "workflow.a",
        runId: "run-a",
      },
    ],
  );
});

test("builder error status produces one failed span with its response context", async () => {
  await withInstrumentor(async (logger) => {
    const builder = logger.logBuilder({
      model: "builder-error-model",
      messages: [{ role: "user", content: "builder error" }],
    });
    builder.setError(new Error("builder failed"));
    await builder.sendLog();
  });

  assert.equal(captureState.spans.length, 1);
  const span = captureState.spans[0];
  assert.equal(span.status.code, 2);
  assert.match(span.status.message, /builder failed/);
  assert.equal(span.attributes.status_code, undefined);
  assert.equal(span.attributes["http.response.status_code"], 500);
  assert.match(span.attributes["error.message"], /builder failed/);
  assert.equal(span.attributes["gen_ai.completion.0.content"], undefined);
  assert.equal(JSON.parse(span.attributes["respan.metadata"]).helicone.status, 500);
});

test("JSON-shaped error messages redact nested secrets everywhere", async () => {
  const unsafeMessage = JSON.stringify({
    password: "hunter2",
    token: "plain-secret",
  });
  const safeMessage = JSON.stringify({
    password: "[REDACTED]",
    token: "[REDACTED]",
  });

  await withInstrumentor(async (logger) => {
    const now = Date.now();
    await logger.sendLog(
      { model: "failed-json-model", messages: [{ role: "user", content: "fail" }] },
      { error: { message: unsafeMessage } },
      { startTime: now - 1, endTime: now, status: 500 },
    );
  });

  const span = captureState.spans[0];
  assert.equal(span.status.message, safeMessage);
  assert.equal(span.attributes["error.message"], safeMessage);
  assert.deepEqual(JSON.parse(span.attributes["traceloop.entity.output"]), {
    error: safeMessage,
  });
  assert.ok(!JSON.stringify({
    status: span.status,
    attributes: span.attributes,
  }).includes("hunter2"));
  assert.ok(!JSON.stringify(span.attributes).includes("plain-secret"));
});

test("custom tool, vector_db, and data events map to canonical span types", async () => {
  await withInstrumentor(async (logger) => {
    const now = Date.now();
    await logger.sendLog(
      { _type: "tool", toolName: "lookup_order", input: { orderId: "O-42" } },
      { eta: "tomorrow" },
      { startTime: now - 4, endTime: now, status: 200 },
    );
    await logger.sendLog(
      {
        _type: "vector_db",
        operation: "search",
        text: "respan tracing",
        vector: [0.1, 0.2, 0.3],
        topK: 3,
        databaseName: "docs",
      },
      { matches: [{ id: "doc-1", score: 0.99 }] },
      { startTime: now - 3, endTime: now, status: 200 },
    );
    await logger.sendLog(
      { _type: "data", name: "quality_score", meta: { score: 0.98 } },
      { accepted: true },
      { startTime: now - 2, endTime: now, status: 200 },
    );
  });

  assert.equal(captureState.spans.length, 3);
  const [tool, vector, data] = captureState.spans;
  assert.equal(tool.attributes["respan.entity.log_type"], "tool");
  assert.equal(tool.attributes["traceloop.entity.name"], "lookup_order");
  assert.deepEqual(JSON.parse(tool.attributes["traceloop.entity.input"]), {
    name: "lookup_order",
    arguments: { orderId: "O-42" },
  });
  assertNoOffContractAliases(tool.attributes);

  assert.equal(vector.attributes["respan.entity.log_type"], "task");
  assert.equal(vector.attributes["db.system"], undefined);
  assert.equal(vector.attributes["db.vector.query.top_k"], 3);
  assert.equal(vector.attributes["db.vector.table_name"], "docs");
  assert.deepEqual(JSON.parse(vector.attributes["traceloop.entity.input"]).vector, [0.1, 0.2, 0.3]);

  assert.equal(data.attributes["respan.entity.log_type"], "task");
  assert.deepEqual(JSON.parse(data.attributes["traceloop.entity.input"]), { score: 0.98 });
});

test("traceContent=false redacts bodies while retaining model, usage, status, and correlations", async () => {
  await withInstrumentor(async (logger) => {
    await logger.sendLog(
      {
        model: "private-model",
        messages: [{ role: "user", content: "private prompt" }],
        tools: [{
          name: "private_tool",
          description: "private tool description",
          parameters: { type: "object", properties: { secret: { default: "private" } } },
        }],
      },
      {
        choices: [{ message: { role: "assistant", content: "private response" } }],
        usage: { prompt_tokens: 2, completion_tokens: 3 },
      },
      {
        startTime: Date.now() - 1,
        endTime: Date.now(),
        status: 200,
        additionalHeaders: { "Helicone-User-Id": "private-user" },
      },
    );
  }, { traceContent: false });

  const attrs = attributesAt();
  assert.equal(attrs["traceloop.entity.input"], undefined);
  assert.equal(attrs["traceloop.entity.output"], undefined);
  assert.equal(attrs["gen_ai.prompt.0.content"], undefined);
  assert.equal(attrs["gen_ai.completion.0.content"], undefined);
  assert.equal(attrs["llm.request.functions"], undefined);
  assert.equal(attrs["gen_ai.request.model"], "private-model");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 2);
  assert.equal(attrs["respan.customer_params.customer_identifier"], "private-user");
});

test("serialization redacts secret keys, preserves ordinary vectors, and bounds oversized payloads", async () => {
  await withInstrumentor(async (logger) => {
    const now = Date.now();
    await logger.sendLog(
      {
        _type: "data",
        name: "redaction",
        meta: {
          apiKey: "top-secret",
          nested: {
            password: "hidden",
            authToken: "auth-token-sentinel",
            bearerToken: "bearer-token-sentinel",
            idToken: "id-token-sentinel",
            sessionToken: "session-token-sentinel",
            privateKey: "private-key-sentinel",
            clientSecret: "client-secret-sentinel",
            credential: "credential-sentinel",
            credentials: "credentials-sentinel",
            heliconeAuth: "helicone-auth-sentinel",
            promptTokens: 12,
            completionTokens: 4,
            tokenCount: 16,
            tokenizer: "cl100k_base",
            vector: [0.01, 0.02, 0.03],
          },
        },
      },
      { ok: true },
      { startTime: now - 1, endTime: now, status: 200 },
    );
    await logger.sendLog(
      { _type: "data", name: "oversized", meta: { text: "x".repeat(1_100_000) } },
      { ok: true },
      { startTime: now - 1, endTime: now, status: 200 },
    );
  });

  const redacted = JSON.parse(attributesAt(0)["traceloop.entity.input"]);
  assert.equal(redacted.apiKey, "[REDACTED]");
  assert.equal(redacted.nested.password, "[REDACTED]");
  for (const key of [
    "authToken",
    "bearerToken",
    "idToken",
    "sessionToken",
    "privateKey",
    "clientSecret",
    "credential",
    "credentials",
    "heliconeAuth",
  ]) {
    assert.equal(redacted.nested[key], "[REDACTED]", `${key} must be redacted`);
  }
  assert.equal(redacted.nested.promptTokens, 12);
  assert.equal(redacted.nested.completionTokens, 4);
  assert.equal(redacted.nested.tokenCount, 16);
  assert.equal(redacted.nested.tokenizer, "cl100k_base");
  assert.deepEqual(redacted.nested.vector, [0.01, 0.02, 0.03]);
  assert.ok(!JSON.stringify(attributesAt(0)).includes("top-secret"));
  const oversized = attributesAt(1)["traceloop.entity.input"];
  assert.ok(Buffer.byteLength(oversized, "utf8") < 600_000);
  assert.equal(JSON.parse(oversized).truncated, true);
});

test("patch ownership is reference-counted and methods restore after the last owner", async () => {
  const prototype = HeliconeHelpers.HeliconeManualLogger.prototype;
  const originalSendLog = prototype.sendLog;
  const first = new HeliconeInstrumentor({ sdkModule: HeliconeHelpers });
  const second = new HeliconeInstrumentor({ sdkModule: HeliconeHelpers });
  await first.activate();
  const patchedSendLog = prototype.sendLog;
  await second.activate();
  assert.notStrictEqual(patchedSendLog, originalSendLog);
  assert.strictEqual(prototype.sendLog, patchedSendLog);

  first.deactivate();
  await createLogger().sendLog(
    { model: "still-active", messages: [{ role: "user", content: "hi" }] },
    { choices: [{ message: { role: "assistant", content: "hello" } }] },
    { startTime: Date.now() - 1, endTime: Date.now(), status: 200 },
  );
  assert.equal(captureState.spans.length, 1);
  assert.strictEqual(prototype.sendLog, patchedSendLog);

  second.deactivate();
  assert.strictEqual(prototype.sendLog, originalSendLog);
  await createLogger().sendLog(
    { model: "inactive", messages: [{ role: "user", content: "hi" }] },
    { choices: [{ message: { role: "assistant", content: "hello" } }] },
    { startTime: Date.now() - 1, endTime: Date.now(), status: 200 },
  );
  assert.equal(captureState.spans.length, 1);
});

test("activation rejects incompatible helper module surfaces", async () => {
  const instrumentor = new HeliconeInstrumentor({
    sdkModule: { HeliconeManualLogger: class UnsupportedLogger {} },
  });
  await assert.rejects(
    instrumentor.activate(),
    /HeliconeManualLogger\.sendLog/,
  );
  assert.equal(instrumentor.isActive(), false);
  await assert.rejects(
    instrumentHelicone({
      sdkModule: { HeliconeManualLogger: class UnsupportedConvenienceLogger {} },
    }),
    /HeliconeManualLogger\.sendLog/,
  );
});

function responseToolCall(id, name, args) {
  return {
    id,
    type: "function",
    function: { name, arguments: JSON.stringify(args) },
  };
}

function jsonLineStream(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(`${JSON.stringify(chunk)}\n`));
      }
      controller.close();
    },
  });
}

function heliconeValueStream(values) {
  const create = () => ({
    async *[Symbol.asyncIterator]() {
      for (const value of values) yield value;
    },
    tee() {
      return [create(), create()];
    },
    toReadableStream() {
      return new ReadableStream({
        start(controller) {
          for (const value of values) controller.enqueue(value);
          controller.close();
        },
      });
    },
  });
  return create();
}
