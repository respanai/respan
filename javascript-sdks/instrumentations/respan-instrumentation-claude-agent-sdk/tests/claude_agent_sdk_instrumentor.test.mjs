import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import { ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS } from "@opentelemetry/semantic-conventions/incubating";

import { ClaudeAgentSDKInstrumentor } from "../dist/index.js";

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

function createFakeSdk({
  emitAssistant = true,
  emitPartialMessages = true,
  emitUserToolResult = false,
  resultError = false,
  resultUsage,
  toolFailure = false,
  toolName = "get_weather",
  usePostToolBatchOnly = false,
} = {}) {
  const calls = [];

  return {
    calls,
    async query(args) {
      calls.push(args);
      const hooks = args.options?.hooks ?? {};

      async function runHooks(name, input, toolUseId) {
        const groups = Array.isArray(hooks[name]) ? hooks[name] : [];
        for (const group of groups) {
          const callbacks = Array.isArray(group?.hooks) ? group.hooks : [];
          for (const callback of callbacks) {
            if (typeof callback === "function") {
              await callback(input, toolUseId);
            }
          }
        }
      }

      return (async function*() {
        await runHooks("UserPromptSubmit", {
          session_id: "sess-123",
          prompt: args.prompt,
        });

        await runHooks(
          "PreToolUse",
          {
            session_id: "sess-123",
            tool_use_id: "toolu_123",
            tool_name: toolName,
            tool_input: { city: "Tokyo" },
          },
          "toolu_123",
        );

        yield {
          type: "system",
          subtype: "init",
          session_id: "sess-123",
          model: "claude-sonnet-4-5",
          tools: [toolName],
        };

        if (emitPartialMessages) {
          yield* streamAssistantEvents(toolName);
        }

        if (emitAssistant) {
          yield {
            type: "assistant",
            session_id: "sess-123",
            message: {
              model: "claude-sonnet-4-5",
              content: [
                {
                  type: "thinking",
                  thinking: "Need current weather.",
                },
                {
                  type: "tool_use",
                  id: "toolu_123",
                  name: toolName,
                  input: { city: "Tokyo" },
                },
                {
                  type: "text",
                  text: "Tokyo is sunny.",
                },
              ],
            },
          };
        }

        if (emitUserToolResult) {
          yield {
            type: "user",
            session_id: "sess-123",
            message: {
              role: "user",
              content: [
                {
                  type: "tool_result",
                  tool_use_id: "toolu_123",
                  content: [{ type: "text", text: "sunny" }],
                },
              ],
            },
          };
        }

        if (toolFailure) {
          await runHooks(
            "PostToolUseFailure",
            {
              session_id: "sess-123",
              tool_use_id: "toolu_123",
              tool_name: toolName,
              tool_input: { city: "Tokyo" },
              error: "Tool execution failed",
            },
            "toolu_123",
          );
        } else if (usePostToolBatchOnly) {
          await runHooks("PostToolBatch", {
            session_id: "sess-123",
            tool_calls: [
              {
                tool_use_id: "toolu_123",
                tool_name: toolName,
                tool_input: { city: "Tokyo" },
                tool_response: { forecast: "sunny" },
              },
            ],
          });
        } else {
          await runHooks(
            "PostToolUse",
            {
              session_id: "sess-123",
              tool_use_id: "toolu_123",
              tool_name: toolName,
              tool_response: { forecast: "sunny" },
            },
            "toolu_123",
          );
          await runHooks("PostToolBatch", {
            session_id: "sess-123",
            tool_calls: [
              {
                tool_use_id: "toolu_123",
                tool_name: toolName,
                tool_input: { city: "Tokyo" },
                tool_response: { forecast: "sunny" },
              },
            ],
          });
        }

        if (resultError) {
          yield {
            type: "result",
            subtype: "error_during_execution",
            session_id: "sess-123",
            is_error: true,
            api_error_status: 429,
            errors: ["rate limited"],
            total_cost_usd: 0.0123,
            usage:
              resultUsage ?? {
                inputTokens: 21,
                outputTokens: 4,
                cacheReadInputTokens: 3,
                cacheCreationInputTokens: 1,
              },
            modelUsage: {
              "claude-sonnet-4-5": {
                inputTokens: 17,
                outputTokens: 4,
                cacheReadInputTokens: 3,
                cacheCreationInputTokens: 1,
              },
            },
          };
          return;
        }

        yield {
          type: "result",
          subtype: "success",
          session_id: "sess-123",
          is_error: false,
          result: "Tokyo is sunny.",
          total_cost_usd: 0.04241955,
          usage:
            resultUsage ?? {
              input_tokens: 19,
              output_tokens: 7,
              cache_read_input_tokens: 2,
              cache_creation_input_tokens: 1,
            },
        };
      })();
    },
  };
}

function* streamAssistantEvents(toolName) {
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "message_start",
      message: {
        model: "claude-sonnet-4-5",
        usage: { input_tokens: 3, output_tokens: 0 },
      },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "content_block_start",
      index: 0,
      content_block: { type: "thinking", thinking: "" },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "content_block_delta",
      index: 0,
      delta: { type: "thinking_delta", thinking: "Need current weather." },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: { type: "content_block_stop", index: 0 },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "content_block_start",
      index: 1,
      content_block: {
        type: "tool_use",
        id: "toolu_123",
        name: toolName,
        input: {},
      },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: "{\"city\":\"Tokyo\"}" },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: { type: "content_block_stop", index: 1 },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "content_block_start",
      index: 2,
      content_block: { type: "text", text: "" },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "content_block_delta",
      index: 2,
      delta: { type: "text_delta", text: "Tokyo is " },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: {
      type: "content_block_delta",
      index: 2,
      delta: { type: "text_delta", text: "sunny." },
    },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: { type: "content_block_stop", index: 2 },
  };
  yield {
    type: "stream_event",
    session_id: "sess-123",
    event: { type: "message_stop" },
  };
}

function spanByLogType(logType) {
  return captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === logType,
  );
}

function parseAttr(span, key) {
  return JSON.parse(span.attributes[key]);
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
    "prompt_cache_hit_tokens",
    "prompt_cache_creation_tokens",
    "cost",
  ]) {
    assert.equal(attrs[key], undefined, `${key} should not be emitted`);
  }
}

test("instrumentor patches query, merges hooks, and emits canonical tool/agent/chat spans", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk({ emitUserToolResult: true });
  const existingHook = async () => ({ ok: true });
  const originalQuery = sdk.query;

  const instrumentor = new ClaudeAgentSDKInstrumentor({
    sdkModule: sdk,
    agentName: "weather_agent",
  });

  await instrumentor.activate();

  assert.notEqual(sdk.query, originalQuery);

  const iterator = await sdk.query({
    prompt: "What is the weather in Tokyo?",
    options: {
      hooks: {
        Stop: [{ hooks: [existingHook] }],
      },
      tools: [{ name: "get_weather", input_schema: { type: "object" } }],
    },
  });

  const yielded = [];
  for await (const item of iterator) {
    yielded.push(item.type);
  }

  assert.deepEqual(yielded, [
    "system",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "stream_event",
    "assistant",
    "user",
    "result",
  ]);
  assert.equal(sdk.calls.length, 1);
  assert.equal(sdk.calls[0].options.hooks.Stop[0].hooks[0], existingHook);
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.UserPromptSubmit));
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.PreToolUse));
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.PostToolUse));
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.PostToolUseFailure));
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.PostToolBatch));

  assert.equal(captureState.spans.length, 3);

  const toolSpan = spanByLogType("tool");
  const agentSpan = spanByLogType("agent");
  const chatSpan = spanByLogType("chat");

  assert.ok(toolSpan);
  assert.ok(agentSpan);
  assert.ok(chatSpan);
  assert.equal(toolSpan.instrumentationScope?.name, "@respan/instrumentation-claude-agent-sdk");
  assert.equal(agentSpan.instrumentationScope?.name, "@respan/instrumentation-claude-agent-sdk");
  assert.equal(chatSpan.instrumentationScope?.name, "@respan/instrumentation-claude-agent-sdk");

  assert.equal(toolSpan.attributes["traceloop.entity.name"], "get_weather");
  assert.deepEqual(parseAttr(toolSpan, "traceloop.entity.input"), {
    name: "get_weather",
    arguments: { city: "Tokyo" },
  });
  assert.deepEqual(parseAttr(toolSpan, "traceloop.entity.output"), [{ type: "text", text: "sunny" }]);
  assertNoOffContractAliases(toolSpan.attributes);

  assert.equal(agentSpan.attributes["traceloop.entity.name"], "weather_agent");
  assert.equal(agentSpan.attributes["gen_ai.request.model"], undefined);
  assert.equal(agentSpan.attributes["llm.request.type"], undefined);
  assert.equal(agentSpan.attributes["traceloop.entity.output"], "Tokyo is sunny.");
  assertNoOffContractAliases(agentSpan.attributes);

  assert.equal(chatSpan.attributes["traceloop.entity.name"], "weather_agent.chat");
  assert.equal(chatSpan.attributes["gen_ai.system"], "anthropic");
  assert.equal(chatSpan.attributes["llm.request.type"], "chat");
  assert.equal(chatSpan.attributes["gen_ai.request.model"], "claude-sonnet-4-5");
  assert.equal(chatSpan.attributes["gen_ai.usage.input_tokens"], 16);
  assert.equal(chatSpan.attributes["gen_ai.usage.output_tokens"], 7);
  assert.equal(chatSpan.attributes["gen_ai.usage.prompt_tokens"], 16);
  assert.equal(chatSpan.attributes["gen_ai.usage.completion_tokens"], 7);
  assert.equal(chatSpan.attributes["llm.usage.total_tokens"], 26);
  assert.equal(chatSpan.attributes["llm.usage.cache_read_input_tokens"], 2);
  assert.equal(
    chatSpan.attributes[ATTR_GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS],
    1,
  );
  assert.deepEqual(parseAttr(chatSpan, "respan.metadata"), {
    response_cost: 0.04241955,
  });
  assert.equal(
    chatSpan.attributes["respan.sessions.session_identifier"],
    "sess-123",
  );
  assert.deepEqual(parseAttr(chatSpan, "llm.request.functions"), [
    {
      type: "function",
      function: { name: "get_weather", parameters: { type: "object" } },
    },
  ]);
  assert.equal(chatSpan.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(
    chatSpan.attributes["gen_ai.prompt.0.content"],
    "What is the weather in Tokyo?",
  );
  assert.equal(chatSpan.attributes["gen_ai.prompt.1.role"], "tool");
  assert.deepEqual(JSON.parse(chatSpan.attributes["gen_ai.prompt.1.content"]), [
    { type: "text", text: "sunny" },
  ]);
  assert.equal(chatSpan.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(chatSpan.attributes["gen_ai.completion.0.content"], "Tokyo is sunny.");
  assert.deepEqual(parseAttr(chatSpan, "gen_ai.completion.0.tool_calls"), [
    {
      id: "toolu_123",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(
    chatSpan.attributes["gen_ai.completion.0.tool_calls.0.function.name"],
    undefined,
  );
  assertNoOffContractAliases(chatSpan.attributes);
  assert.equal(toolSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);
  assert.equal(chatSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);
  assert.equal(toolSpan.spanContext().traceId, agentSpan.spanContext().traceId);
  assert.equal(chatSpan.spanContext().traceId, agentSpan.spanContext().traceId);

  instrumentor.deactivate();

  assert.equal(sdk.query, originalQuery);
});

test("instrumentor emits errored tool spans for PostToolUseFailure", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk({ toolFailure: true });

  const instrumentor = new ClaudeAgentSDKInstrumentor({
    sdkModule: sdk,
    agentName: "weather_agent",
  });

  await instrumentor.activate();

  const iterator = await sdk.query({
    prompt: "What is the weather in Tokyo?",
    options: {},
  });

  for await (const _item of iterator) {
    // Drain the stream so spans are emitted.
  }

  const toolSpan = spanByLogType("tool");
  const agentSpan = spanByLogType("agent");

  assert.ok(toolSpan);
  assert.ok(agentSpan);
  assert.equal(toolSpan.status.code, 2);
  assert.equal(toolSpan.status.message, "Tool execution failed");
  assert.equal(
    parseAttr(toolSpan, "traceloop.entity.output"),
    "Tool execution failed",
  );
  assert.equal(toolSpan.parentSpanContext?.spanId, agentSpan.spanContext().spanId);
  assertNoOffContractAliases(toolSpan.attributes);

  instrumentor.deactivate();
});

test("instrumentor derives usage from legacy prompt token details", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk({
    resultUsage: {
      prompt_tokens: 19,
      completion_tokens: 7,
      prompt_tokens_details: {
        cached_tokens: 2,
        cache_creation_tokens: 1,
      },
    },
  });

  const instrumentor = new ClaudeAgentSDKInstrumentor({
    sdkModule: sdk,
    agentName: "weather_agent",
  });

  await instrumentor.activate();

  const iterator = await sdk.query({
    prompt: "What is the weather in Tokyo?",
    options: {},
  });

  for await (const _item of iterator) {
    // Drain the stream so spans are emitted.
  }

  const chatSpan = spanByLogType("chat");

  assert.ok(chatSpan);
  assert.equal(chatSpan.attributes["gen_ai.usage.input_tokens"], 16);
  assert.equal(chatSpan.attributes["gen_ai.usage.output_tokens"], 7);
  assert.equal(chatSpan.attributes["gen_ai.usage.prompt_tokens"], 16);
  assert.equal(chatSpan.attributes["gen_ai.usage.completion_tokens"], 7);
  assert.equal(chatSpan.attributes["llm.usage.total_tokens"], 26);
  assert.equal(chatSpan.attributes["llm.usage.cache_read_input_tokens"], 2);
  assertNoOffContractAliases(chatSpan.attributes);

  instrumentor.deactivate();
});

test("instrumentor handles streaming-only assistant output, result errors, and PostToolBatch completion", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk({
    emitAssistant: false,
    resultError: true,
    usePostToolBatchOnly: true,
  });

  const instrumentor = new ClaudeAgentSDKInstrumentor({
    sdkModule: sdk,
    agentName: "weather_agent",
  });

  await instrumentor.activate();

  const iterator = await sdk.query({
    prompt: "What is the weather in Tokyo?",
    options: {},
  });

  for await (const _item of iterator) {
    // Drain the stream so spans are emitted.
  }

  const toolSpan = spanByLogType("tool");
  const agentSpan = spanByLogType("agent");
  const chatSpan = spanByLogType("chat");

  assert.ok(toolSpan);
  assert.ok(agentSpan);
  assert.ok(chatSpan);
  assert.deepEqual(parseAttr(toolSpan, "traceloop.entity.output"), { forecast: "sunny" });
  assert.equal(chatSpan.attributes["gen_ai.completion.0.content"], "Tokyo is sunny.");
  assert.deepEqual(parseAttr(chatSpan, "gen_ai.completion.0.tool_calls"), [
    {
      id: "toolu_123",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(chatSpan.attributes["gen_ai.usage.input_tokens"], 17);
  assert.equal(chatSpan.attributes["gen_ai.usage.output_tokens"], 4);
  assert.equal(chatSpan.attributes["llm.usage.total_tokens"], 25);
  assert.equal(chatSpan.attributes["llm.usage.cache_read_input_tokens"], 3);
  assert.equal(chatSpan.status.code, 2);
  assert.equal(chatSpan.status.message, "rate limited");
  assert.equal(agentSpan.status.code, 2);
  assertNoOffContractAliases(chatSpan.attributes);

  instrumentor.deactivate();
});

test("instrumentor extracts SDK MCP server tool definitions", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk({ toolName: "mcp__demo__get_weather" });
  const weatherInputSchema = {
    vendor: "zod",
    _internalValidator: () => true,
    toJSONSchema() {
      return {
        type: "object",
        properties: {
          city: { type: "string" },
        },
      };
    },
  };

  const instrumentor = new ClaudeAgentSDKInstrumentor({
    sdkModule: sdk,
    agentName: "weather_agent",
  });

  await instrumentor.activate();

  const iterator = await sdk.query({
    prompt: "What is the weather in Tokyo?",
    options: {
      mcpServers: {
        demo: {
          type: "sdk",
          instance: {
            _registeredTools: {
              get_weather: {
                description: "Get weather",
                inputSchema: weatherInputSchema,
              },
            },
          },
        },
      },
    },
  });

  for await (const _item of iterator) {
    // Drain the stream so spans are emitted.
  }

  const chatSpan = spanByLogType("chat");

  assert.ok(chatSpan);
  assert.deepEqual(parseAttr(chatSpan, "llm.request.functions"), [
    {
      type: "function",
      function: {
        name: "mcp__demo__get_weather",
        description: "Get weather",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
          },
        },
      },
    },
  ]);
  assert.deepEqual(parseAttr(chatSpan, "gen_ai.completion.0.tool_calls"), [
    {
      id: "toolu_123",
      type: "function",
      function: {
        name: "mcp__demo__get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assertNoOffContractAliases(chatSpan.attributes);
});

test("instrumentor normalizes Zod-like MCP server schemas", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk({ toolName: "mcp__demo__get_weather" });
  const zodLikeInputSchema = {
    "~standard": {
      vendor: "zod",
      version: 1,
    },
    def: {
      type: "object",
      shape: {
        city: {
          def: { type: "string" },
          type: "string",
        },
        unit: {
          def: {
            type: "optional",
            innerType: {
              def: { type: "string" },
              type: "string",
            },
          },
          type: "optional",
        },
      },
    },
  };

  const instrumentor = new ClaudeAgentSDKInstrumentor({
    sdkModule: sdk,
    agentName: "weather_agent",
  });

  await instrumentor.activate();

  const iterator = await sdk.query({
    prompt: "What is the weather in Tokyo?",
    options: {
      mcpServers: {
        demo: {
          type: "sdk",
          instance: {
            _registeredTools: {
              get_weather: {
                description: "Get weather",
                inputSchema: zodLikeInputSchema,
              },
            },
          },
        },
      },
    },
  });

  for await (const _item of iterator) {
    // Drain the stream so spans are emitted.
  }

  const chatSpan = spanByLogType("chat");

  assert.ok(chatSpan);
  assert.deepEqual(parseAttr(chatSpan, "llm.request.functions"), [
    {
      type: "function",
      function: {
        name: "mcp__demo__get_weather",
        description: "Get weather",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string" },
            unit: { type: "string" },
          },
          required: ["city"],
        },
      },
    },
  ]);
  assertNoOffContractAliases(chatSpan.attributes);
});

test("native Query remains synchronous and preserves current SDK controls and option passthrough", async () => {
  captureState.spans = [];
  const messages = [
    { type: "system", subtype: "task_notification", task_id: "scheduled-1", origin: { type: "task_notification", fireReason: "scheduled" } },
    { type: "rate_limit_event", rate_limit_info: { status: "rejected", resetsAt: 123 } },
    { type: "result", subtype: "success", session_id: "session-latest", result: "", structured_output: { ok: true }, usage: { input_tokens: 2, output_tokens: 3 } },
  ];
  class NativeQuery {
    #index = 0;
    #calls = [];
    [Symbol.asyncIterator]() { return this; }
    async next() { return this.#index < messages.length ? { value: messages[this.#index++], done: false } : { done: true }; }
    async interrupt() { this.#calls.push("interrupt"); return this.#calls.length; }
    async setModel(model) { this.#calls.push(model); }
    async readMcpResource(server, uri) { return { server, uri, calls: this.#calls }; }
    async mcpServerStatus() { return [{ tools: [{ _meta: { ui: { resourceUri: "ui://latest" } } }] }]; }
    async askSideQuestion(question) { return { question, seen: this.#index }; }
    close() { this.#calls.push("close"); }
  }
  let received;
  const sdk = { query(args) { received = args; return new NativeQuery(); } };
  const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  const hooks = { Stop: [{ hooks: [async () => ({})] }] };
  const options = { hooks, verbatimPrompts: true, resume: "session-latest", includePartialMessages: true, outputFormat: { type: "json_schema", schema: { type: "object" } } };
  const query = sdk.query({ prompt: "@literal /literal", options });
  assert.equal(query.then, undefined);
  assert.ok(query instanceof NativeQuery);
  assert.equal(query[Symbol.asyncIterator](), query);
  assert.equal(await query.interrupt(), 1);
  await query.setModel("claude-opus-5-5");
  assert.deepEqual(await query.readMcpResource("apps", "ui://latest"), { server: "apps", uri: "ui://latest", calls: ["interrupt", "claude-opus-5-5"] });
  assert.equal((await query.mcpServerStatus())[0].tools[0]._meta.ui.resourceUri, "ui://latest");
  assert.deepEqual(await query.askSideQuestion("status"), { question: "status", seen: 0 });
  const output = [];
  for await (const message of query) output.push(message);
  assert.deepEqual(output, messages);
  assert.equal(received.options.verbatimPrompts, true);
  assert.equal(received.options.resume, "session-latest");
  assert.equal(received.options.outputFormat, options.outputFormat);
  assert.equal(options.hooks, hooks);
  query.close();
  assert.equal(captureState.spans.length, 2);
  assert.ok(captureState.spans.some(span => String(span.attributes["traceloop.entity.output"]).includes('"ok":true')));
  instrumentor.deactivate();
});

test("early return finalizes once, closes source, and keeps a query usable after deactivation", async () => {
  captureState.spans = [];
  let closed = 0;
  const sdk = { query() { return (async function* () { try { yield { type: "assistant", message: { content: [{ type: "text", text: "partial" }] } }; yield {}; } finally { closed++; } })(); } };
  const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  const stream = sdk.query({ prompt: "early" });
  instrumentor.deactivate();
  for await (const message of stream) { assert.equal(message.type, "assistant"); break; }
  await stream.return();
  assert.equal(closed, 1);
  assert.equal(captureState.spans.length, 2);
});

test("synchronous query errors retain their synchronous throw contract", async () => {
  captureState.spans = [];
  const sdk = { query() { throw new Error("native synchronous failure"); } };
  const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  assert.throws(() => sdk.query({ prompt: "bad" }), /native synchronous failure/);
  assert.equal(captureState.spans.length, 2);
  instrumentor.deactivate();
});

test("SDK result reconciles missing tool hooks and ignores duplicate late hooks", async () => {
  captureState.spans = [];
  const sdk = { query({ options }) { return (async function* () {
    for (const group of options.hooks.PreToolUse) for (const hook of group.hooks) await hook({ tool_name: "Read", tool_input: { path: "fixture" } }, "tool-latest");
    yield { type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "tool-latest", content: "denied by policy", is_error: true }] } };
    for (const group of options.hooks.PostToolUseFailure) for (const hook of group.hooks) await hook({ tool_name: "Read", error: "denied" }, "tool-latest");
    yield { type: "result", subtype: "success", result: "handled" };
  })(); } };
  const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  for await (const _ of sdk.query({ prompt: "tool failure" })) {}
  const tools = captureState.spans.filter(span => span.attributes["respan.entity.log_type"] === "tool");
  assert.equal(tools.length, 1);
  assert.equal(tools[0].attributes["traceloop.entity.output"], '"denied by policy"');
  assert.equal(tools[0].status.code, 2);
  instrumentor.deactivate();
});

test("terminal task_updated closes only its matched background invocation", async () => {
  captureState.spans = [];
  const sdk = { query({ options }) { return (async function* () {
    for (const group of options.hooks.PreToolUse) for (const hook of group.hooks) await hook({ tool_name: "Agent", tool_input: { prompt: "inspect fixture" } }, "background-tool");
    yield { type: "system", subtype: "task_started", task_id: "background-task", tool_use_id: "background-tool" };
    yield { type: "system", subtype: "task_updated", task_id: "background-task", patch: { status: "running" } };
    yield { type: "system", subtype: "task_updated", task_id: "background-task", patch: { status: "killed" } };
    yield { type: "system", subtype: "task_notification", task_id: "background-task", tool_use_id: "background-tool", status: "stopped" };
    yield { type: "result", subtype: "success", structured_output: { stopped: true } };
  })(); } };
  const instrumentor = new ClaudeAgentSDKInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  for await (const _ of sdk.query({ prompt: "background" })) {}
  const tools = captureState.spans.filter(span => span.attributes["respan.entity.log_type"] === "tool");
  assert.equal(tools.length, 1);
  assert.equal(JSON.parse(tools[0].attributes["traceloop.entity.output"]).status, "killed");
  assert.equal(tools[0].status.code, 2);
  instrumentor.deactivate();
});
