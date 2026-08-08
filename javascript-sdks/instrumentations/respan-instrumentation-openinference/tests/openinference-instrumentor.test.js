import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  OpenInferenceInstrumentor,
  OpenInferenceTranslator,
  prepareOpenInferenceSpanForExport,
} from "../dist/index.js";

const OI_SPAN_KIND = "openinference.span.kind";
const CLAUDE_AGENT_SDK_SCOPE_NAME =
  "@arizeai/openinference-instrumentation-claude-agent-sdk";

function makeSpan({
  name = "test-span",
  attributes = {},
  resourceAttributes = {},
  instrumentationScopeName = "test-scope",
} = {}) {
  return {
    name,
    attributes: { ...attributes },
    _attributes: { ...attributes },
    resource: { attributes: { ...resourceAttributes } },
    instrumentationScope: {
      name: instrumentationScopeName,
      version: "1.0.0",
    },
    instrumentationScope: {
      name: instrumentationScopeName,
      version: "1.0.0",
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

class FakeInstrumentor {
  setTracerProvider(tracerProvider) {
    this.tracerProvider = tracerProvider;
  }

  instrument({ tracerProvider } = {}) {
    this.instrumentedWith = tracerProvider;
  }

  uninstrument() {
    this.uninstrumented = true;
  }
}

test("translator promotes Claude Agent SDK token fields without mutating original OI attrs", () => {
  const translator = new OpenInferenceTranslator();
  const span = makeSpan({
    name: "ClaudeAgent.query",
    instrumentationScopeName: CLAUDE_AGENT_SDK_SCOPE_NAME,
    attributes: {
      [OI_SPAN_KIND]: "AGENT",
      "input.value": "Explain tracing",
      "input.mime_type": "text/plain",
      "output.value": "Tracing adds spans",
      "output.mime_type": "text/plain",
      "llm.model_name": "claude-sonnet-4-6",
      "llm.provider": "Anthropic",
      "llm.system": "Anthropic",
      "llm.token_count.prompt": 3,
      "llm.token_count.completion": 97,
      "llm.token_count.total": 100,
      "process.pid": 1234,
    },
  });

  translator.onEnd(span);

  assert.equal(span.attributes["respan.entity.log_type"], "agent");
  assert.equal(span.attributes["llm.request.type"], "chat");
  assert.equal(span.attributes["gen_ai.request.model"], "claude-sonnet-4-6");
  assert.equal(span.attributes["gen_ai.system"], "anthropic");
  assert.equal(span.attributes["gen_ai.provider.name"], "anthropic");
  assert.equal(span.attributes.model, "claude-sonnet-4-6");
  assert.equal(span.attributes.prompt_tokens, 3);
  assert.equal(span.attributes.completion_tokens, 97);
  assert.equal(span.attributes.total_request_tokens, 100);
  assert.equal(span.attributes["traceloop.entity.input"], "Explain tracing");
  assert.equal(span.attributes["traceloop.entity.output"], "Tracing adds spans");
  assert.equal(span.attributes[OI_SPAN_KIND], "AGENT");
  assert.equal(span.attributes["llm.token_count.prompt"], 3);
  assert.equal(span.attributes["input.mime_type"], "text/plain");
  assert.equal(span.attributes["process.pid"], 1234);
  assert.notStrictEqual(span._attributes, span.attributes);

  const exportedSpan = prepareOpenInferenceSpanForExport(span);
  assert.notStrictEqual(exportedSpan, span);
  assert.equal(exportedSpan.attributes[OI_SPAN_KIND], undefined);
  assert.equal(exportedSpan.attributes["llm.token_count.prompt"], undefined);
  assert.equal(exportedSpan.attributes["input.mime_type"], undefined);
  assert.equal(exportedSpan.attributes["process.pid"], undefined);
  assert.strictEqual(exportedSpan._attributes, exportedSpan.attributes);
  assert.equal(span.attributes[OI_SPAN_KIND], "AGENT");
});

test("translator rebuilds LLM message content from text blocks and export cleanup removes duplicated OI message attrs", () => {
  const translator = new OpenInferenceTranslator();
  const span = makeSpan({
    attributes: {
      [OI_SPAN_KIND]: "LLM",
      "llm.input_messages.0.message.role": "user",
      "llm.input_messages.0.message.contents.0.message_content.text": "Hello",
      "llm.input_messages.0.message.contents.1.message_content.text": "world",
      "llm.output_messages.0.message.role": "assistant",
      "llm.output_messages.0.message.contents.0.message_content.text": "Hi there",
    },
  });

  translator.onEnd(span);

  assert.equal(span.attributes["gen_ai.prompt.0.role"], "user");
  assert.equal(span.attributes["gen_ai.prompt.0.content"], "Hello\nworld");
  assert.equal(span.attributes["gen_ai.completion.0.role"], "assistant");
  assert.equal(span.attributes["gen_ai.completion.0.content"], "Hi there");
  assert.equal(
    span.attributes["llm.input_messages.0.message.role"],
    "user",
  );

  const exportedSpan = prepareOpenInferenceSpanForExport(span);
  assert.equal(
    exportedSpan.attributes["llm.input_messages.0.message.role"],
    undefined,
  );
  assert.equal(
    exportedSpan.attributes["llm.output_messages.0.message.role"],
    undefined,
  );
});

test("translator falls back to JSON when content blocks are not plain text", () => {
  const translator = new OpenInferenceTranslator();
  const span = makeSpan({
    attributes: {
      [OI_SPAN_KIND]: "LLM",
      "llm.input_messages.0.message.contents.0.message_content.type": "image",
    },
  });

  translator.onEnd(span);

  assert.equal(
    span.attributes["gen_ai.prompt.0.content"],
    JSON.stringify([{ type: "image" }]),
  );
});

test("export cleanup preserves untranslated OI message and tool attrs on non-LLM spans", () => {
  const translator = new OpenInferenceTranslator();
  const span = makeSpan({
    attributes: {
      [OI_SPAN_KIND]: "AGENT",
      "llm.input_messages.0.message.role": "user",
      "llm.input_messages.0.message.content": "keep me",
      "llm.tools": JSON.stringify([{ name: "weather" }]),
      "llm.invocation_parameters": JSON.stringify({ temperature: 0.1 }),
    },
  });

  translator.onEnd(span);
  const exportedSpan = prepareOpenInferenceSpanForExport(span);

  assert.equal(
    exportedSpan.attributes["llm.input_messages.0.message.role"],
    "user",
  );
  assert.equal(
    exportedSpan.attributes["llm.input_messages.0.message.content"],
    "keep me",
  );
  assert.equal(
    exportedSpan.attributes["llm.tools"],
    JSON.stringify([{ name: "weather" }]),
  );
  assert.equal(
    exportedSpan.attributes["llm.invocation_parameters"],
    JSON.stringify({ temperature: 0.1 }),
  );
});

test("instrumentor hook preserves processor-visible span state and sanitizes export clone", () => {
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
  const provider = createFakeTracerProvider(processor);
  resetTracerProvider(provider);

  let instrumentor;
  try {
    instrumentor = new OpenInferenceInstrumentor(FakeInstrumentor);
    instrumentor.activate();

    const oiSpan = makeSpan({
      name: "ClaudeAgent.query",
      instrumentationScopeName: CLAUDE_AGENT_SDK_SCOPE_NAME,
      resourceAttributes: {
        "service.name": "respan-test",
        "process.pid": 42,
        "host.name": "local-dev",
        "custom.attr": "kept",
      },
      attributes: {
        [OI_SPAN_KIND]: "AGENT",
        "llm.model_name": "claude-sonnet-4-6",
        "llm.token_count.prompt": 3,
        "llm.token_count.completion": 97,
        "llm.token_count.total": 100,
      },
    });

    processor.onEnd(oiSpan);

    assert.equal(processorSeen.length, 1);
    assert.strictEqual(processorSeen[0], oiSpan);
    assert.equal(processorSeen[0].attributes[OI_SPAN_KIND], "AGENT");
    assert.equal(processorSeen[0].attributes["llm.token_count.prompt"], 3);

    assert.equal(exportedSpans.length, 1);
    assert.notStrictEqual(exportedSpans[0], oiSpan);
    assert.equal(exportedSpans[0].attributes[OI_SPAN_KIND], undefined);
    assert.equal(exportedSpans[0].attributes["llm.token_count.prompt"], undefined);
    assert.deepEqual(exportedSpans[0].resource.attributes, {
      "service.name": "respan-test",
      "custom.attr": "kept",
    });
    assert.equal(exportedSpans[0].instrumentationScope.name, "");
    assert.equal(exportedSpans[0].instrumentationScope.name, "");
    assert.equal(oiSpan.resource.attributes["process.pid"], 42);
    assert.equal(
      oiSpan.instrumentationScope.name,
      CLAUDE_AGENT_SDK_SCOPE_NAME,
    );

    instrumentor.deactivate();

    const rawSpan = makeSpan({
      attributes: {
        custom: "value",
      },
      resourceAttributes: {
        "service.name": "respan-test",
        "process.pid": 99,
      },
    });

    processor.onEnd(rawSpan);

    assert.equal(processorSeen.length, 2);
    assert.strictEqual(processorSeen[1], rawSpan);
    assert.equal(exportedSpans.length, 2);
    assert.strictEqual(exportedSpans[1], rawSpan);
  } finally {
    try {
      instrumentor?.deactivate();
    } catch {}
    resetTracerProvider();
  }
});

test("instrumentor hook leaves non-openinference spans unsanitized while active", () => {
  const capturedSpans = [];
  const processor = {
    onEnd(span) {
      capturedSpans.push(span);
    },
  };
  const provider = createFakeTracerProvider(processor);
  resetTracerProvider(provider);

  let instrumentor;
  try {
    instrumentor = new OpenInferenceInstrumentor(FakeInstrumentor);
    instrumentor.activate();

    const nonOiSpan = makeSpan({
      attributes: {
        custom: "value",
      },
      resourceAttributes: {
        "service.name": "respan-test",
        "process.pid": 77,
      },
    });

    processor.onEnd(nonOiSpan);

    assert.equal(capturedSpans.length, 1);
    assert.strictEqual(capturedSpans[0], nonOiSpan);
  } finally {
    try {
      instrumentor?.deactivate();
    } catch {}
    resetTracerProvider();
  }
});
