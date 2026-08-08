import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { ArizeInstrumentor } from "../dist/index.js";

const OI_SPAN_KIND = "openinference.span.kind";

function makeSpan({
  name = "test-span",
  attributes = {},
  resourceAttributes = {},
  instrumentationScopeName = "@arizeai/phoenix-otel",
} = {}) {
  return {
    name,
    attributes: { ...attributes },
    _attributes: { ...attributes },
    resource: { attributes: { ...resourceAttributes } },
    instrumentationScope: {
      name: instrumentationScopeName,
      version: "1.0.2",
    },
    instrumentationScope: {
      name: instrumentationScopeName,
      version: "1.0.2",
    },
  };
}

function resetTracerProvider(provider) {
  if (typeof trace.disable === "function") {
    trace.disable();
  }
  if (provider) {
    trace.setGlobalTracerProvider(provider);
  }
}

function createFakeTracerProvider(processor) {
  return {
    activeSpanProcessor: processor,
    getTracer() {
      return {
        startSpan() {
          throw new Error("startSpan should not be called in this test");
        },
      };
    },
  };
}

test("ArizeInstrumentor translates Phoenix OpenInference spans and sanitizes the export clone", () => {
  const processorSeen = [];
  const exportedSpans = [];
  const manager = {
    onEnd(span) {
      exportedSpans.push(span);
    },
  };
  const processor = {
    getProcessorManager() {
      return manager;
    },
    onEnd(span) {
      processorSeen.push(span);
      manager.onEnd(span);
    },
  };
  resetTracerProvider(createFakeTracerProvider(processor));

  const instrumentor = new ArizeInstrumentor();
  try {
    instrumentor.activate();

    const span = makeSpan({
      name: "synthetic-llm",
      attributes: {
        [OI_SPAN_KIND]: "LLM",
        "input.value": "How do I trace TypeScript apps?",
        "output.value": "Use Phoenix helpers with Respan.",
        "llm.model_name": "gpt-4o-mini",
        "llm.provider": "openai",
        "llm.token_count.prompt": 12,
        "llm.token_count.completion": 24,
        "llm.token_count.total": 36,
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "How do I trace TypeScript apps?",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "Use Phoenix helpers with Respan.",
        "traceloop.entity.path": "arize-demo.workflow",
      },
    });

    processor.onEnd(span);

    assert.equal(processorSeen.length, 1);
    assert.strictEqual(processorSeen[0], span);
    assert.equal(span.attributes["respan.entity.log_type"], "chat");
    assert.equal(span.attributes["llm.request.type"], "chat");
    assert.equal(span.attributes["traceloop.entity.name"], "synthetic-llm");
    assert.equal(span.attributes["traceloop.entity.input"], "How do I trace TypeScript apps?");
    assert.equal(span.attributes["traceloop.entity.output"], "Use Phoenix helpers with Respan.");
    assert.equal(span.attributes["gen_ai.request.model"], "gpt-4o-mini");
    assert.equal(span.attributes["gen_ai.system"], "openai");
    assert.equal(span.attributes["gen_ai.prompt.0.role"], "user");
    assert.equal(span.attributes["gen_ai.completion.0.role"], "assistant");
    assert.equal(span.attributes["traceloop.workflow.name"], "arize-demo.workflow");
    assert.equal(span.attributes[OI_SPAN_KIND], "LLM");

    assert.equal(exportedSpans.length, 1);
    assert.notStrictEqual(exportedSpans[0], span);
    assert.equal(exportedSpans[0].attributes["respan.entity.log_type"], "chat");
    assert.equal(exportedSpans[0].attributes["traceloop.workflow.name"], "arize-demo.workflow");
    assert.equal(exportedSpans[0].attributes[OI_SPAN_KIND], undefined);
    assert.equal(exportedSpans[0].attributes["llm.input_messages.0.message.role"], undefined);
    assert.equal(span.attributes[OI_SPAN_KIND], "LLM");
  } finally {
    instrumentor.deactivate();
  }
});

test("ArizeInstrumentor sanitizes directly when no processor manager is present", () => {
  const exportedSpans = [];
  const processor = {
    onEnd(span) {
      exportedSpans.push(span);
    },
  };
  resetTracerProvider(createFakeTracerProvider(processor));

  const instrumentor = new ArizeInstrumentor();
  try {
    instrumentor.activate();

    const span = makeSpan({
      attributes: {
        [OI_SPAN_KIND]: "CHAIN",
        "input.value": "raw input",
        "output.value": "raw output",
      },
    });
    processor.onEnd(span);

    assert.equal(span.attributes["respan.entity.log_type"], "workflow");
    assert.equal(exportedSpans.length, 1);
    assert.notStrictEqual(exportedSpans[0], span);
    assert.equal(exportedSpans[0].attributes[OI_SPAN_KIND], undefined);
    assert.equal(exportedSpans[0].attributes["traceloop.entity.input"], "raw input");
  } finally {
    instrumentor.deactivate();
  }
});

test("ArizeInstrumentor keeps the translator hook until all active instances deactivate", () => {
  const processor = {
    onEnd() {},
  };
  resetTracerProvider(createFakeTracerProvider(processor));

  const originalOnEnd = processor.onEnd;
  const first = new ArizeInstrumentor({ name: "arize-first" });
  const second = new ArizeInstrumentor({ name: "arize-second" });

  first.activate();
  const patchedOnEnd = processor.onEnd;
  assert.notStrictEqual(patchedOnEnd, originalOnEnd);

  second.activate();
  assert.strictEqual(processor.onEnd, patchedOnEnd);

  first.deactivate();
  assert.strictEqual(processor.onEnd, patchedOnEnd);

  second.deactivate();
  assert.strictEqual(processor.onEnd, originalOnEnd);
});
