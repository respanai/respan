import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

import {
  clearSdkTrace,
  emitSdkItem,
  registerSdkTrace,
} from "../dist/_otel_emitter.js";

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);
const packageRequire = createRequire(import.meta.url);
const { version: packageVersion } = packageRequire("../package.json");

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
  assert.equal(span.instrumentationScope.version, packageVersion);
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

test("emit agent preserves tool, handoff, and output configuration in canonical metadata", () => {
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
  assert.equal(
    attrs["respan.metadata.openai_agents.agent_configuration.tools"],
    JSON.stringify(["lookup_docs"]),
  );
  assert.equal(
    attrs["respan.metadata.openai_agents.agent_configuration.handoffs"],
    JSON.stringify(["Support"]),
  );
  assert.equal(
    attrs["respan.metadata.openai_agents.agent_configuration.output_type"],
    "text",
  );
  assertNoOffContractAliases(attrs);
  assert.equal(attrs["traceloop.span.kind"], undefined);
});

test("noncanonical SDK IDs map to deterministic full-width nonzero OTel IDs", () => {
  const seenTraceIds = new Set();
  const seenSpanIds = new Set();

  for (let index = 0; index < 4096; index += 1) {
    const item = {
      ...makeBaseSpanData({
        type: "agent",
        name: `Agent ${index}`,
      }),
      traceId: `trace_test_${index}`,
      spanId: `span_test_${index}`,
      parentId: `parent_test_${index}`,
    };
    const span = emitAndCaptureSpan(item);
    const first = span.spanContext();

    const repeat = emitAndCaptureSpan(item).spanContext();

    assert.equal(first.traceId, repeat.traceId);
    assert.equal(first.spanId, repeat.spanId);
    assert.match(first.traceId, /^(?!0{32}$)[0-9a-f]{32}$/);
    assert.match(first.spanId, /^(?!0{16}$)[0-9a-f]{16}$/);
    assert.notEqual(first.traceId.slice(0, 8), first.traceId.slice(8, 16));
    assert.notEqual(first.spanId.slice(0, 8), first.spanId.slice(8, 16));
    seenTraceIds.add(first.traceId);
    seenSpanIds.add(first.spanId);
  }

  assert.equal(seenTraceIds.size, 4096);
  assert.equal(seenSpanIds.size, 4096);
});

test("registered trace marker and grouping propagate to every emitted child", () => {
  const traceId = "trace_marker_context";
  registerSdkTrace({
    traceId,
    name: "marker-workflow",
    groupId: "marker-group",
    metadata: {
      custom_identifier: "otel2-fix-marker",
      run_id: "otel2-fix-marker",
    },
  });

  try {
    const attrs = emitAndCapture({
      ...makeBaseSpanData({ type: "agent", name: "Marker Agent" }),
      traceId,
    });
    assert.equal(
      attrs[RespanSpanAttributes.RESPAN_SPAN_CUSTOM_ID],
      "otel2-fix-marker",
    );
    assert.equal(
      attrs[RespanSpanAttributes.RESPAN_TRACE_GROUP_ID],
      "marker-group",
    );
    assert.equal(
      attrs["respan.metadata.custom_identifier"],
      "otel2-fix-marker",
    );
    assert.equal(attrs["respan.metadata.run_id"], "otel2-fix-marker");
  } finally {
    clearSdkTrace(traceId);
  }
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

test("real latest SDK Runner retains task and turn parents without double counting", async () => {
  const { Agent, Runner, Usage, withGenerationSpan, withTrace } = await import("@openai/agents");
  const { OpenAIAgentsInstrumentor } = await import("../dist/index.js");
  const instrumentor = new OpenAIAgentsInstrumentor();
  instrumentor.activate();
  captureState.spans = [];
  const model = {
    async getResponse(request) {
      const output = [{ type: "message", role: "assistant", status: "completed",
        content: [{ type: "output_text", text: "Verified" }] }];
      return withGenerationSpan(async () => ({ output,
        usage: new Usage({ inputTokens: 7, outputTokens: 2, totalTokens: 9 }),
        responseId: "test_response",
      }), { data: { input: request.input, output, model: "test-model",
        usage: { input_tokens: 7, output_tokens: 2 } } });
    },
    async *getStreamedResponse() { throw new Error("not used"); },
  };
  const result = await withTrace("latest-runner", () => new Runner().run(
    new Agent({ name: "Current SDK", model }), "Verify",
  ));
  assert.equal(result.finalOutput, "Verified");
  const spans = captureState.spans;
  assert.equal(spans.length, 5);
  const ids = new Set(spans.map(s => s.spanContext().spanId));
  assert.equal(spans.filter(s => !s.parentSpanContext).length, 1);
  for (const span of spans) {
    if (span.parentSpanContext) assert.ok(ids.has(span.parentSpanContext.spanId));
    assertNoOffContractAliases(span.attributes);
  }
  assert.equal(spans.reduce((sum, s) => sum + (s.attributes["gen_ai.usage.input_tokens"] ?? 0), 0), 7);
  const task = spans.find(s => s.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE] === "workflow" && s.parentSpanContext);
  assert.equal(JSON.parse(task.attributes["respan.metadata.openai_agents.usage"]).total_tokens, 9);
  instrumentor.deactivate();
});

test("native SDK voice spans preserve audio payloads and error messages", async () => {
  const { Span } = await import("@openai/agents");
  const processor = { onSpanStart() {}, onSpanEnd() {} };
  for (const [type, input, output, logType] of [
    ["transcription", { data: "YXVkaW8=", format: "pcm" }, "Hello", "transcription"],
    ["speech", "Hello", { data: "YXVkaW8=", format: "pcm" }, "speech"],
    ["speech_group", "Hello", undefined, "task"],
  ]) {
    const item = new Span({ traceId: "trace_voice", spanId: `span_${type}`,
      data: { type, input, output }, startedAt: "2026-09-23T00:00:00Z", endedAt: "2026-09-23T00:00:01Z",
    }, processor);
    const attrs = emitAndCapture(item);
    assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], logType);
    assert.deepEqual(JSON.parse(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]), input);
    if (output !== undefined) assert.deepEqual(JSON.parse(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]), output);
  }
  const item = new Span({ traceId: "trace_error", data: { type: "turn", turn: 1, agent_name: "Failure" } }, processor);
  item.setError({ message: "Expected SDK error", data: { status_code: 429 } });
  assert.equal(emitAndCaptureSpan(item).status.message, "Expected SDK error");
});

test("stream cancellation preserves context and overlapping instrumentor lifetime", async () => {
  const { OpenAIChatCompletionsModel, withTrace, withCustomSpan } = await import('@openai/agents');
  const { OpenAIAgentsInstrumentor } = await import('../dist/index.js');
  const { isStreaming } = await import('../dist/_streaming.js');
  const prototype = OpenAIChatCompletionsModel.prototype;
  const original = prototype.getStreamedResponse;
  let closed = false;
  const source = async function* () {
    try { assert.equal(isStreaming(), true); yield 'first'; }
    finally { assert.equal(isStreaming(), true); closed = true; }
  };
  prototype.getStreamedResponse = source;
  const first = new OpenAIAgentsInstrumentor();
  const second = new OpenAIAgentsInstrumentor();
  try {
    first.activate(); first.activate(); second.activate(); first.deactivate();
    assert.notEqual(prototype.getStreamedResponse, source);
    const stream = prototype.getStreamedResponse.call({});
    assert.deepEqual(await stream.next(), {value:'first', done:false});
    assert.equal(isStreaming(), false);
    await stream.return(); assert.equal(closed, true);
    second.deactivate(); assert.equal(prototype.getStreamedResponse, source);
    captureState.spans = [];
    await withTrace('disabled', () => withCustomSpan(async()=>{}, {data:{name:'disabled',data:{}}}));
    assert.equal(captureState.spans.length, 0);
  } finally { first.deactivate(); second.deactivate(); prototype.getStreamedResponse = original; }
});

test("tool-only Chat Completions never become JSON assistant text", () => {
  for (const content of [null, '']) {
    const attrs = emitAndCapture(makeBaseSpanData({type:'generation', model:'gpt-4o-mini', output:[{
      object:'chat.completion',choices:[{message:{role:'assistant',content,
        tool_calls:[{id:'call_forecast',type:'function',function:{name:'forecast',arguments:'{"city":"Paris"}'}}]}}],
    }]}));
    assert.equal(attrs['gen_ai.completion.0.content'], '');
    assert.equal(JSON.parse(attrs['gen_ai.completion.0.tool_calls'])[0].function.name,'forecast');
  }
});

test('Respan content opt-out removes MCP, audio and LLM payloads', () => {
  const previous = process.env.RESPAN_TRACE_CONTENT;
  process.env.RESPAN_TRACE_CONTENT = 'false';
  try {
    for (const data of [
      {type:'mcp_tools',server:'weather',result:['PRIVATE_CONTENT']},
      {type:'speech',input:'PRIVATE_CONTENT',output:{data:'PRIVATE_CONTENT',format:'pcm'},model:'tts-1'},
      {type:'generation',input:[{role:'user',content:'PRIVATE_CONTENT'}],output:[{role:'assistant',content:'PRIVATE_CONTENT'}],model:'test'},
    ]) {
      const attrs=emitAndCapture(makeBaseSpanData(data));
      assert.ok(!JSON.stringify(attrs).includes('PRIVATE_CONTENT'));
      assert.equal(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT],undefined);
      assert.equal(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT],undefined);
    }
  } finally {
    if(previous===undefined)delete process.env.RESPAN_TRACE_CONTENT;
    else process.env.RESPAN_TRACE_CONTENT=previous;
  }
});

test('raw Responses hosted calls survive output and conversation history', () => {
  for(const type of ['web_search_call','file_search_call','image_generation_call','mcp_call']) {
    const call={type,id:'hosted_123',action:{type:'search',query:'weather'}};
    const attrs=emitAndCapture(makeBaseSpanData({type:'response',_input:[call],
      _response:{model:'gpt-4o-mini',tools:[],output:[call]}}));
    assert.equal(attrs['gen_ai.completion.0.content'],'');
    const emitted=JSON.parse(attrs['gen_ai.completion.0.tool_calls'])[0];
    assert.equal(emitted.id,'hosted_123');assert.equal(emitted.function.name,type);
    assert.equal(JSON.parse(attrs['gen_ai.prompt.0.tool_calls'])[0].id,'hosted_123');
  }
});

test('native Responses no-data tracing cannot export SDK private payload fields', async () => {
  const { OpenAIResponsesModel, withTrace } = await import('@openai/agents');
  const { OpenAIAgentsInstrumentor } = await import('../dist/index.js');
  const inst = new OpenAIAgentsInstrumentor();inst.activate();captureState.spans=[];
  try {
    const model = new OpenAIResponsesModel({baseURL:'https://api.openai.com/v1'},'gpt-4o');
    model._fetchResponse=async()=>({id:'response_private',object:'response',created_at:1,status:'completed',
      model:'gpt-4o',output:[{id:'message',type:'message',role:'assistant',status:'completed',
        content:[{type:'output_text',text:'PRIVATE_OUTPUT',annotations:[]}]}],
      usage:{input_tokens:1,output_tokens:1,total_tokens:2,input_tokens_details:{cached_tokens:0},output_tokens_details:{reasoning_tokens:0}},
    });
    await withTrace('private-response',()=>model.getResponse({input:[{type:'message',role:'user',content:'PRIVATE_INPUT'}],
      modelSettings:{},tools:[],handoffs:[],outputType:'text',tracing:'enabled_without_data'}));
    const modelSpan=captureState.spans.find(s=>s.attributes[RespanSpanAttributes.RESPAN_LOG_TYPE]==='chat');
    assert.ok(modelSpan);
    assert.ok(!JSON.stringify(modelSpan.attributes).includes('PRIVATE_'));
    assert.equal(modelSpan.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT],undefined);
    assert.equal(modelSpan.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT],undefined);
  } finally {inst.deactivate();}
});

test('hosted file search, interpreter, and image payloads retain arguments and results',()=>{
  for(const [call,args,result] of [
    [{type:'file_search_call',id:'fs',queries:['hello']},{queries:['hello']},undefined],
    [{type:'code_interpreter_call',id:'ci',code:'print(42)',container_id:'c',outputs:[{type:'logs',logs:'42'}]},{code:'print(42)',container_id:'c'},[{type:'logs',logs:'42'}]],
    [{type:'image_generation_call',id:'img',revised_prompt:'cat',result:'BASE64'},{revised_prompt:'cat'},'BASE64'],
  ]) {
    const attrs=emitAndCapture(makeBaseSpanData({type:'response',_response:{output:[call]}}));
    const mapped=JSON.parse(attrs['gen_ai.completion.0.tool_calls'])[0];
    assert.deepEqual(JSON.parse(mapped.function.arguments),args);
    assert.deepEqual(mapped.output,result);
  }
});
