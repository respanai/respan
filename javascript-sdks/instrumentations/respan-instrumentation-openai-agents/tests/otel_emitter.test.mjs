import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

import { emitSdkItem } from "../dist/_otel_emitter.js";

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

function emitAndCapture(item) {
  captureState.spans = [];
  emitSdkItem(item);
  assert.equal(captureState.spans.length, 1);
  return captureState.spans[0].attributes;
}

function emitAndCaptureSpan(item) {
  captureState.spans = [];
  emitSdkItem(item);
  assert.equal(captureState.spans.length, 1);
  return captureState.spans[0];
}

function makeBaseSpanData(spanData) {
  return {
    traceId: "trace_test_123",
    spanId: "span_test_456",
    parentId: "parent_test_789",
    started_at: new Date().toISOString(),
    ended_at: new Date().toISOString(),
    error: null,
    spanData,
  };
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
    assert.equal(attrs[key], undefined, `${key} should not be emitted`);
  }
}

test("emit trace stores SDK trace metadata on workflow span", () => {
  const span = emitAndCaptureSpan({
    traceId: "trace_test_123",
    name: "openai_agents_gateway_basic.workflow",
    groupId: "openai-agents-ts-123",
    metadata: {
      run_id: "openai-agents-ts-123",
      example: "openai-agents-sdk",
    },
  });
  const attrs = span.attributes;

  assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], "workflow");
  assert.equal(span.instrumentationLibrary.version, "1.0.6");
  assert.equal(
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME],
    "openai_agents_gateway_basic.workflow",
  );
  assert.equal(
    attrs[RespanSpanAttributes.RESPAN_TRACE_GROUP_ID],
    "openai-agents-ts-123",
  );
  assert.deepEqual(JSON.parse(attrs[RespanSpanAttributes.RESPAN_METADATA]), {
    group_id: "openai-agents-ts-123",
    run_id: "openai-agents-ts-123",
    example: "openai-agents-sdk",
  });
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit response stores canonical LLM tool, message, and usage attrs", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "response",
      _input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Tell me everything about Tokyo" }],
        },
        {
          type: "function_call",
          call_id: "call_weather",
          name: "get_weather",
          arguments: "{\"city\":\"Tokyo\"}",
        },
        {
          type: "function_call_output",
          call_id: "call_weather",
          output: "Sunny, 22°C in Tokyo",
        },
      ],
      _response: {
        model: "gpt-4o",
        output: [
          {
            type: "function_call",
            call_id: "call_stats",
            name: "get_city_stats",
            arguments: "{\"city\":\"Tokyo\"}",
          },
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Here is Tokyo info" }],
          },
        ],
        tools: [
          {
            type: "function",
            name: "get_weather",
            description: "Get weather",
            parameters: { type: "object" },
          },
        ],
        usage: {
          input_tokens: 10,
          output_tokens: 3,
          total_tokens: 13,
          input_tokens_details: {
            cached_tokens: 4,
          },
        },
      },
    }),
  );

  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Tell me everything about Tokyo" },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_weather",
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
      content: "Sunny, 22°C in Tokyo",
      tool_call_id: "call_weather",
    },
  ]);
  assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], "chat");
  assert.equal(attrs[SpanAttributes.LLM_REQUEST_TYPE], "chat");
  assert.equal(attrs[SpanAttributes.LLM_SYSTEM], "openai");
  assert.equal(attrs[SpanAttributes.LLM_REQUEST_MODEL], "gpt-4o");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Tell me everything about Tokyo");
  assert.equal(attrs["gen_ai.prompt.1.role"], "assistant");
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.1.tool_calls"]), [
    {
      id: "call_weather",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(attrs["gen_ai.prompt.2.role"], "tool");
  assert.equal(attrs["gen_ai.prompt.2.content"], "Sunny, 22°C in Tokyo");
  assert.equal(attrs["traceloop.entity.output"], "Here is Tokyo info");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Here is Tokyo info");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_stats",
      type: "function",
      function: {
        name: "get_city_stats",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]), [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather",
        parameters: { type: "object" },
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 3);
  assert.equal(attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS], 10);
  assert.equal(attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS], 3);
  assert.equal(attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS], 13);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 4);
  assert.ok(!attrs["traceloop.entity.input"].includes("[object Object]"));
  assertNoOffContractAliases(attrs);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit generation extracts canonical attrs from raw chat completions output", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "generation",
      input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Use the tool" }],
        },
      ],
      output: [
        {
          id: "chatcmpl_123",
          object: "chat.completion",
          model: "gpt-4o",
          choices: [
            {
              message: {
                role: "assistant",
                content: "Searching the docs.",
                tool_calls: [
                  {
                    id: "call_docs",
                    type: "function",
                    function: {
                      name: "search_docs",
                      arguments: "{\"query\":\"otel\"}",
                    },
                  },
                ],
              },
            },
          ],
          usage: {
            prompt_tokens: 8,
            completion_tokens: 2,
            total_tokens: 10,
            prompt_tokens_details: {
              cached_tokens: 1,
            },
          },
        },
      ],
    }),
  );

  assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], "chat");
  assert.equal(attrs[SpanAttributes.LLM_REQUEST_TYPE], "chat");
  assert.equal(attrs[SpanAttributes.LLM_SYSTEM], "openai");
  assert.equal(attrs[SpanAttributes.LLM_REQUEST_MODEL], "gpt-4o");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Use the tool" },
  ]);
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Use the tool");
  assert.equal(attrs["traceloop.entity.output"], "Searching the docs.");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Searching the docs.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_docs",
      type: "function",
      function: {
        name: "search_docs",
        arguments: "{\"query\":\"otel\"}",
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 8);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 2);
  assert.equal(attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS], 8);
  assert.equal(attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS], 2);
  assert.equal(attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS], 10);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 1);
  assert.ok(!attrs["traceloop.entity.input"].includes("[object Object]"));
  assertNoOffContractAliases(attrs);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit function serializes wrapped text tool output", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "function",
      name: "get_weather",
      input: { city: "Tokyo" },
      output: { type: "text", text: "Sunny, 22°C in Tokyo" },
    }),
  );

  assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], "tool");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), {
    name: "get_weather",
    arguments: { city: "Tokyo" },
  });
  assert.equal(JSON.parse(attrs["traceloop.entity.output"]), "Sunny, 22°C in Tokyo");
  assertNoOffContractAliases(attrs);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit generation preserves boolean false output", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "generation",
      model: "gpt-4o",
      input: "Return false",
      output: false,
      usage: {
        prompt_tokens: 2,
        completion_tokens: 1,
      },
    }),
  );

  assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], "chat");
  assert.equal(attrs[SpanAttributes.LLM_REQUEST_TYPE], "chat");
  assert.equal(attrs["traceloop.entity.output"], "false");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "false");
  assertNoOffContractAliases(attrs);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit response preserves chat completions tool call messages", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "response",
      _input: [
        { role: "user", content: "Check Tokyo weather" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "call_weather_chat",
              type: "function",
              function: {
                name: "get_weather",
                arguments: "{\"city\":\"Tokyo\"}",
              },
            },
          ],
        },
      ],
      _response: {
        model: "gpt-4o",
        output: "Done",
        usage: {
          input_tokens: 5,
          output_tokens: 1,
        },
      },
    }),
  );

  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), [
    { role: "user", content: "Check Tokyo weather" },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_weather_chat",
          type: "function",
          function: {
            name: "get_weather",
            arguments: "{\"city\":\"Tokyo\"}",
          },
        },
      ],
    },
  ]);
  assert.equal(attrs["traceloop.entity.output"], "Done");
  assert.equal(attrs["gen_ai.prompt.1.role"], "assistant");
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.1.tool_calls"]), [
    {
      id: "call_weather_chat",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);
  assert.equal(attrs["gen_ai.completion.0.content"], "Done");
  assert.equal(attrs["gen_ai.completion.0.tool_calls"], undefined);
  assertNoOffContractAliases(attrs);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit response handles modern agents item and content variants", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "response",
      _input: [
        {
          type: "message",
          role: "user",
          content: [
            { type: "input_text", text: "Find tracing docs" },
            { type: "input_image", image: "https://example.test/diagram.png" },
            { type: "audio", audio: "base64-audio", format: "wav" },
          ],
        },
        {
          type: "tool_search_call",
          callId: "search_1",
          arguments: { query: "otel" },
          status: "completed",
        },
        {
          type: "tool_search_output",
          callId: "search_1",
          tools: [{ type: "tool_reference", functionName: "lookup_docs" }],
        },
      ],
      _response: {
        model: "gpt-5-mini",
        output: [
          {
            type: "hosted_tool_call",
            id: "hosted_1",
            name: "file_search_call",
            arguments: "{\"query\":\"agents\"}",
            status: "completed",
          },
          {
            type: "message",
            role: "assistant",
            content: [
              { type: "refusal", refusal: "I cannot share internal files." },
              { type: "output_text", text: "Here is a public summary." },
              { type: "image", image: "image-output" },
            ],
          },
        ],
        tools: [
          {
            type: "namespace",
            name: "docs",
            description: "Documentation tools",
            tools: [
              {
                type: "function",
                name: "lookup_docs",
                parameters: { type: "object" },
              },
            ],
          },
        ],
        usage: {
          input_tokens: 12,
          output_tokens: 6,
        },
      },
    }),
  );

  assert.equal(attrs["gen_ai.prompt.0.content"], "Find tracing docs\n[image]\n[audio]");
  assert.deepEqual(JSON.parse(attrs["gen_ai.prompt.1.tool_calls"]), [
    {
      id: "search_1",
      type: "function",
      function: {
        name: "tool_search_call",
        arguments: "{\"query\":\"otel\"}",
      },
      status: "completed",
      openai_agents_type: "tool_search_call",
    },
  ]);
  assert.equal(attrs["gen_ai.prompt.2.role"], "tool");
  assert.equal(attrs["traceloop.entity.output"], "I cannot share internal files.\nHere is a public summary.\n[image]");
  assert.equal(attrs["gen_ai.completion.0.content"], "I cannot share internal files.\nHere is a public summary.\n[image]");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "hosted_1",
      type: "function",
      function: {
        name: "file_search_call",
        arguments: "{\"query\":\"agents\"}",
      },
      status: "completed",
      openai_agents_type: "hosted_tool_call",
    },
  ]);
  assert.deepEqual(JSON.parse(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]), [
    {
      type: "namespace",
      name: "docs",
      description: "Documentation tools",
      tools: [
        {
          type: "function",
          function: {
            name: "lookup_docs",
            parameters: { type: "object" },
          },
        },
      ],
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 6);
  assertNoOffContractAliases(attrs);
});

test("emit agent omits tool and handoff aliases", () => {
  const attrs = emitAndCapture(
    makeBaseSpanData({
      type: "agent",
      name: "Router",
      tools: ["lookup_docs"],
      handoffs: ["Support"],
      output_type: "text",
    }),
  );

  assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], "agent");
  assert.equal(attrs[SpanAttributes.TRACELOOP_ENTITY_NAME], "Router");
  assert.equal(attrs[RespanSpanAttributes.RESPAN_METADATA_AGENT_NAME], "Router");
  assertNoOffContractAliases(attrs);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("emit handoff, guardrail, custom, and mcp tools use common contract attrs", () => {
  const handoff = emitAndCapture(
    makeBaseSpanData({
      type: "handoff",
      from_agent: "Router",
      to_agent: "Support",
    }),
  );
  assert.equal(handoff[RespanSpanAttributes.RESPAN_LOG_TYPE], "task");
  assert.deepEqual(JSON.parse(handoff[SpanAttributes.TRACELOOP_ENTITY_INPUT]), {
    from_agent: "Router",
  });
  assert.deepEqual(JSON.parse(handoff[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]), {
    to_agent: "Support",
  });
  assertNoOffContractAliases(handoff);

  const guardrail = emitAndCapture(
    makeBaseSpanData({
      type: "guardrail",
      name: "PII check",
      triggered: true,
    }),
  );
  assert.equal(guardrail[RespanSpanAttributes.RESPAN_LOG_TYPE], "guardrail");
  assert.deepEqual(JSON.parse(guardrail[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]), {
    triggered: true,
  });
  assert.equal(guardrail[RespanSpanAttributes.RESPAN_METADATA_GUARDRAIL_NAME], "PII check");
  assertNoOffContractAliases(guardrail);

  const custom = emitAndCapture(
    makeBaseSpanData({
      type: "custom",
      name: "rank_candidates",
      data: {
        input: { candidates: 3 },
        output: { selected: 1 },
        phase: "rerank",
      },
    }),
  );
  assert.equal(custom[RespanSpanAttributes.RESPAN_LOG_TYPE], "task");
  assert.deepEqual(JSON.parse(custom[SpanAttributes.TRACELOOP_ENTITY_INPUT]), {
    candidates: 3,
  });
  assert.deepEqual(JSON.parse(custom[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]), {
    selected: 1,
  });
  assert.deepEqual(JSON.parse(custom[RespanSpanAttributes.RESPAN_METADATA]), {
    phase: "rerank",
  });
  assertNoOffContractAliases(custom);

  const mcp = emitAndCapture(
    makeBaseSpanData({
      type: "mcp_tools",
      server: "docs",
      result: ["lookup_docs"],
    }),
  );
  assert.equal(mcp[RespanSpanAttributes.RESPAN_LOG_TYPE], "tool");
  assert.deepEqual(JSON.parse(mcp[SpanAttributes.TRACELOOP_ENTITY_INPUT]), {
    server: "docs",
  });
  assert.deepEqual(JSON.parse(mcp[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]), [
    "lookup_docs",
  ]);
  assertNoOffContractAliases(mcp);
});

test("emit span without ended_at defaults end time to start time", () => {
  const span = emitAndCaptureSpan({
    traceId: "trace_test_123",
    spanId: "span_test_456",
    parentId: "parent_test_789",
    started_at: "2026-03-30T00:00:00.000Z",
    error: null,
    spanData: {
      type: "generation",
      model: "gpt-4o",
      input: "hello",
      output: "world",
      usage: {
        prompt_tokens: 1,
        completion_tokens: 1,
      },
    },
  });

  assert.deepEqual(span.startTime, [1774828800, 0]);
  assert.deepEqual(span.endTime, [1774828800, 0]);
  assert.deepEqual(span.duration, [0, 0]);
});
