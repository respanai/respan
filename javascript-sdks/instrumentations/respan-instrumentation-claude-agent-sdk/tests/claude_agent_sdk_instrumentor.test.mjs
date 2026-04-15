import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

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

function createFakeSdk() {
  const calls = [];

  return {
    calls,
    async query(args) {
      calls.push(args);
      const hooks = args.options?.hooks ?? {};

      function firstHook(name) {
        const groups = Array.isArray(hooks[name]) ? hooks[name] : [];
        for (const group of groups) {
          if (Array.isArray(group?.hooks) && typeof group.hooks[0] === "function") {
            return group.hooks[0];
          }
        }
        return undefined;
      }

      return (async function*() {
        await firstHook("UserPromptSubmit")?.({
          session_id: "sess-123",
          prompt: args.prompt,
        });

        await firstHook("PreToolUse")?.({
          session_id: "sess-123",
          tool_use_id: "toolu_123",
          tool_name: "get_weather",
          tool_input: { city: "Tokyo" },
        });

        yield {
          type: "system",
          data: {
            session_id: "sess-123",
          },
        };

        yield {
          type: "assistant",
          message: {
            model: "claude-sonnet-4-5",
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
          },
        };

        await firstHook("PostToolUse")?.({
          session_id: "sess-123",
          tool_use_id: "toolu_123",
          tool_name: "get_weather",
          tool_response: { forecast: "sunny" },
        });

        yield {
          type: "result",
          subtype: "success",
          session_id: "sess-123",
          result: "Tokyo is sunny.",
          total_cost_usd: 0.04241955,
          usage: {
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

test("instrumentor patches query, merges hooks, and emits tool + agent spans", async () => {
  captureState.spans = [];
  const sdk = createFakeSdk();
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

  assert.deepEqual(yielded, ["system", "assistant", "result"]);
  assert.equal(sdk.calls.length, 1);
  assert.equal(sdk.calls[0].options.hooks.Stop[0].hooks[0], existingHook);
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.UserPromptSubmit));
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.PreToolUse));
  assert.ok(Array.isArray(sdk.calls[0].options.hooks.PostToolUse));

  assert.equal(captureState.spans.length, 2);

  const toolSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "tool",
  );
  const agentSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "agent",
  );

  assert.ok(toolSpan);
  assert.ok(agentSpan);
  assert.equal(toolSpan.attributes["traceloop.entity.name"], "get_weather");
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.input"]), {
    city: "Tokyo",
  });
  assert.deepEqual(JSON.parse(toolSpan.attributes["traceloop.entity.output"]), {
    forecast: "sunny",
  });

  assert.equal(agentSpan.attributes["traceloop.entity.name"], "weather_agent");
  assert.equal(agentSpan.attributes["gen_ai.request.model"], "claude-sonnet-4-5");
  assert.equal(agentSpan.attributes.model, "claude-sonnet-4-5");
  assert.equal(agentSpan.attributes.prompt_tokens, 16);
  assert.equal(agentSpan.attributes.completion_tokens, 7);
  assert.equal(agentSpan.attributes.total_request_tokens, 23);
  assert.equal(agentSpan.attributes.prompt_cache_hit_tokens, 2);
  assert.equal(agentSpan.attributes.prompt_cache_creation_tokens, 1);
  assert.equal(agentSpan.attributes.cost, 0.04241955);
  assert.equal(
    agentSpan.attributes["respan.sessions.session_identifier"],
    "sess-123",
  );
  assert.deepEqual(
    JSON.parse(agentSpan.attributes["respan.span.tools"]),
    [{ type: "function", function: { name: "get_weather", parameters: { type: "object" } } }],
  );
  assert.deepEqual(
    JSON.parse(agentSpan.attributes["respan.span.tool_calls"]),
    [
      {
        id: "toolu_123",
        type: "function",
        function: {
          name: "get_weather",
          arguments: "{\"city\":\"Tokyo\"}",
        },
      },
    ],
  );
  assert.deepEqual(
    JSON.parse(agentSpan.attributes["traceloop.entity.input"]),
    [{ role: "user", content: "What is the weather in Tokyo?" }],
  );
  assert.deepEqual(
    agentSpan.attributes["traceloop.entity.output"],
    "Tokyo is sunny.",
  );
  assert.equal(toolSpan.parentSpanId, agentSpan.spanContext().spanId);
  assert.equal(toolSpan.spanContext().traceId, agentSpan.spanContext().traceId);

  instrumentor.deactivate();

  assert.equal(sdk.query, originalQuery);
});
