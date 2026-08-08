import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { BraintrustInstrumentor } from "../dist/index.js";

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

test("exports Braintrust LLM rows with canonical Respan attrs", () => {
  captureState.spans = [];
  const instrumentor = new BraintrustInstrumentor();

  const exported = instrumentor.exportRecord({
    id: "braintrust-llm-row",
    project_id: "project-1",
    log_id: "g",
    span_id: "braintrust-llm-span",
    root_span_id: "braintrust-root",
    span_parents: ["braintrust-parent"],
    created: "2026-06-18T00:00:00.000Z",
    span_attributes: {
      type: "llm",
      name: "braintrust.llm.chat",
      model: "openai/gpt-4.1-nano",
      tools: [
        {
          name: "lookup_weather",
          description: "Look up weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      ],
    },
    input: {
      messages: [
        { role: "system", content: "Be concise." },
        { role: "user", content: "Weather in Tokyo?" },
      ],
    },
    output: {
      content: "Tokyo is sunny.",
      tool_calls: [
        { id: "call_1", name: "lookup_weather", arguments: { city: "Tokyo" } },
      ],
    },
    metadata: { example: "braintrust" },
    metrics: {
      prompt_tokens: 12,
      completion_tokens: 8,
      cache_read_input_tokens: 2,
    },
    scores: { helpfulness: 1 },
    tags: ["typescript", "braintrust"],
  });

  assert.equal(exported, true);
  assert.equal(captureState.spans.length, 1);

  const span = captureState.spans[0];
  const attrs = span.attributes;
  assert.equal(span.instrumentationScope.name, "@respan/instrumentation-braintrust");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["traceloop.entity.name"], "braintrust.llm.chat");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.system"], "openai");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4.1-nano");
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Be concise.");
  assert.equal(attrs["gen_ai.prompt.1.role"], "user");
  assert.equal(attrs["gen_ai.prompt.1.content"], "Weather in Tokyo?");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Tokyo is sunny.");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 8);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 8);
  assert.equal(attrs["llm.usage.total_tokens"], 20);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 2);
  assert.equal(attrs["respan.metadata.example"], "braintrust");
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "lookup_weather",
        description: "Look up weather",
        parameters: { type: "object", properties: { city: { type: "string" } } },
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: { name: "lookup_weather", arguments: JSON.stringify({ city: "Tokyo" }) },
    },
  ]);

  for (const banned of [
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
    assert.equal(attrs[banned], undefined, `${banned} should not be emitted`);
  }
});

test("merges Braintrust updates and preserves original span identity", () => {
  captureState.spans = [];
  const instrumentor = new BraintrustInstrumentor();

  const exported = instrumentor.exportRecords([
    {
      id: "merge-row",
      project_id: "project-1",
      log_id: "g",
      span_id: "original-span",
      root_span_id: "root-span",
      span_attributes: { type: "task", name: "braintrust.task" },
      tags: ["initial"],
      input: { step: "start" },
      metrics: { start: 1_780_000_000 },
    },
    {
      id: "merge-row",
      project_id: "project-1",
      log_id: "g",
      span_id: "updated-span",
      root_span_id: "updated-root",
      _is_merge: true,
      tags: ["initial", "merged"],
      output: { step: "done" },
      metrics: { end: 1_780_000_001 },
    },
  ]);

  assert.equal(exported, 1);
  assert.equal(captureState.spans.length, 1);

  const span = captureState.spans[0];
  const attrs = span.attributes;
  assert.equal(span.spanContext().spanId, "6c5009c66c5009c6");
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.input"]), { step: "start" });
  assert.deepEqual(JSON.parse(attrs["traceloop.entity.output"]), { step: "done" });
  assert.deepEqual(JSON.parse(attrs["respan.metadata.braintrust_tags"]), ["initial", "merged"]);
});

test("exports error rows and captured propagated attributes", () => {
  captureState.spans = [];
  const instrumentor = new BraintrustInstrumentor();
  const propagated = {
    custom_identifier: "braintrust-run-123",
    trace_group_identifier: "Braintrust TypeScript Example.workflow",
    metadata: { example: "propagated" },
  };

  const exported = instrumentor.exportRecord(
    {
      id: "error-row",
      project_id: "project-1",
      log_id: "g",
      span_id: "error-span",
      root_span_id: "error-root",
      span_attributes: { type: "task", name: "braintrust.failing_task" },
      input: { step: "postprocess" },
      error: { message: "tool failed" },
    },
    { propagatedAttributes: propagated },
  );

  assert.equal(exported, true);
  assert.equal(captureState.spans.length, 1);

  const span = captureState.spans[0];
  const attrs = span.attributes;
  assert.equal(attrs["respan.entity.log_type"], "task");
  assert.equal(attrs["error.message"], "tool failed");
  assert.equal(attrs.status_code, 500);
  assert.equal(attrs["respan.span_params.custom_identifier"], "braintrust-run-123");
  assert.equal(attrs["respan.trace.trace_group_identifier"], "Braintrust TypeScript Example.workflow");
  assert.equal(attrs["traceloop.workflow.name"], "Braintrust TypeScript Example.workflow");
  assert.equal(attrs["respan.metadata.example"], "propagated");
});

test("background logger bridge resolves lazy Braintrust records", async () => {
  captureState.spans = [];
  const instrumentor = new BraintrustInstrumentor();
  const braintrust = await import("braintrust");
  const state = braintrust._internalGetGlobalState();

  try {
    await instrumentor.activate();
    const logger = state.bgLogger();
    logger.log([
      {
        get: async () => ({
            id: "lazy-row",
            project_id: "project-1",
            log_id: "g",
            span_id: "lazy-span",
            root_span_id: "lazy-root",
            span_attributes: { type: "task", name: "braintrust.lazy_task" },
            output: "done",
          }),
      },
    ]);

    await logger.flush();
    assert.equal(captureState.spans.length, 1);
    assert.equal(captureState.spans[0].attributes["traceloop.entity.name"], "braintrust.lazy_task");
  } finally {
    await instrumentor.deactivate();
  }
});
