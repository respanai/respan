import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { CohereInstrumentor } from "../dist/index.js";

const captureState = { spans: [] };
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);

class FakeV2Client {
  async chat(_request) {
    return {
      id: "chat-v2",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Hello from Cohere v2." }],
        toolCalls: [
          {
            id: "call_1",
            type: "function",
            function: {
              name: "lookup_docs",
              arguments: JSON.stringify({ topic: "respan" }),
            },
          },
        ],
      },
      usage: {
        tokens: { inputTokens: 12, outputTokens: 5 },
        cachedTokens: 2,
      },
    };
  }

  async chatStream(_request) {
    return streamFrom([
      { type: "content-delta", delta: { message: { content: { text: "stream" } } } },
      {
        type: "message-end",
        delta: {
          message: {
            role: "assistant",
            content: [{ type: "text", text: "stream done" }],
          },
          usage: { tokens: { inputTokens: 3, outputTokens: 2 } },
        },
      },
    ]);
  }

  async embed(request) {
    return {
      id: "embed-v2",
      embeddings: { float: [[0.1, 0.2, 0.3]] },
      texts: request.texts,
      meta: { billedUnits: { inputTokens: 4 } },
    };
  }

  async rerank(_request) {
    return {
      id: "rerank-v2",
      results: [{ index: 1, relevanceScore: 0.97 }],
      meta: { billedUnits: { searchUnits: 1 } },
    };
  }
}

class FakeCohereClient {
  constructor() {
    this._v2 = new FakeV2Client();
  }

  get v2() {
    return this._v2;
  }

  async chat(_request) {
    return {
      text: "Hello from Cohere v1.",
      toolCalls: [{ name: "lookup_docs", parameters: { topic: "respan" } }],
      meta: { tokens: { inputTokens: 9, outputTokens: 4 } },
    };
  }

  async generate(_request) {
    return {
      generations: [{ text: "Generated response." }],
      meta: { billedUnits: { inputTokens: 6, outputTokens: 3 } },
    };
  }

  async generateStream(_request) {
    return streamFrom([
      { eventType: "text-generation", text: "Generated " },
      {
        eventType: "stream-end",
        response: {
          generations: [{ text: "Generated stream response." }],
          meta: { billedUnits: { inputTokens: 7, outputTokens: 4 } },
        },
      },
    ]);
  }

  async embed(request) {
    return {
      responseType: "embeddings_floats",
      embeddings: [[0.4, 0.5]],
      texts: request.texts,
      meta: { billedUnits: { inputTokens: 3 } },
    };
  }

  async rerank(_request) {
    return {
      results: [{ index: 0, relevanceScore: 0.91 }],
      meta: { billedUnits: { searchUnits: 2 } },
    };
  }
}

function streamFrom(events) {
  return {
    async *[Symbol.asyncIterator]() {
      for (const event of events) {
        yield event;
      }
    },
  };
}

function fakeModule() {
  return { CohereClient: FakeCohereClient };
}

async function flushMicrotasks() {
  await Promise.resolve();
  await new Promise((resolve) => setImmediate(resolve));
}

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

test.beforeEach(() => {
  captureState.spans = [];
});

test("does not activate when the installed SDK has no supported methods", async () => {
  class UnsupportedCohereClient {}

  const instrumentor = new CohereInstrumentor({
    sdkModule: { CohereClient: UnsupportedCohereClient },
  });
  await instrumentor.activate();

  assert.equal(instrumentor.isActive(), false);
  instrumentor.deactivate();
});

test("captures v2 chat tools with canonical Cohere LLM attributes", async () => {
  const module = fakeModule();
  const instrumentor = new CohereInstrumentor({ sdkModule: module });
  await instrumentor.activate();

  const client = new module.CohereClient();
  await client.v2.chat({
    model: "command-a-03-2025",
    messages: [{ role: "user", content: "Use the docs." }],
    tools: [
      {
        type: "function",
        function: {
          name: "lookup_docs",
          description: "Search documentation.",
          parameters: {
            type: "object",
            properties: { topic: { type: "string" } },
          },
        },
      },
    ],
  });
  await flushMicrotasks();

  assert.equal(captureState.spans.length, 1);
  const [span] = captureState.spans;
  const attrs = span.attributes;

  assert.equal(span.instrumentationScope?.name, "@respan/instrumentation-cohere");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["gen_ai.system"], "cohere");
  assert.equal(attrs["gen_ai.request.model"], "command-a-03-2025");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Use the docs.");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Hello from Cohere v2.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: {
        name: "lookup_docs",
        arguments: JSON.stringify({ topic: "respan" }),
      },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "lookup_docs",
        description: "Search documentation.",
        parameters: {
          type: "object",
          properties: { topic: { type: "string" } },
        },
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 5);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 12);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 5);
  assert.equal(attrs["llm.usage.total_tokens"], 17);
  assert.equal(attrs["llm.usage.cache_read_input_tokens"], 2);
  assert.equal(attrs["llm.system"], undefined);
  assert.equal(attrs.tools, undefined);
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs["respan.span.tools"], undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);

  instrumentor.deactivate();
});

test("captures embeddings and rerank without off-contract aliases", async () => {
  const module = fakeModule();
  const instrumentor = new CohereInstrumentor({ sdkModule: module });
  await instrumentor.activate();

  const client = new module.CohereClient();
  await client.v2.embed({
    model: "embed-v4.0",
    texts: ["hello"],
    inputType: "classification",
    embeddingTypes: ["float"],
  });
  await client.v2.rerank({
    model: "rerank-v4.0",
    query: "capital",
    documents: ["Nevada", "Washington, DC"],
    topN: 1,
  });
  await flushMicrotasks();

  assert.equal(captureState.spans.length, 2);
  const embedSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "embedding",
  );
  const rerankSpan = captureState.spans.find(
    (span) => span.attributes["respan.entity.log_type"] === "task",
  );

  assert.ok(embedSpan);
  assert.ok(rerankSpan);
  assert.equal(embedSpan.attributes["llm.request.type"], "embedding");
  assert.equal(embedSpan.attributes["gen_ai.request.model"], "embed-v4.0");
  assert.deepEqual(JSON.parse(embedSpan.attributes["traceloop.entity.input"]), ["hello"]);
  assert.deepEqual(JSON.parse(embedSpan.attributes["traceloop.entity.output"]), {
    float: [[0.1, 0.2, 0.3]],
  });
  assert.equal(embedSpan.attributes["gen_ai.usage.input_tokens"], 4);
  assert.equal(embedSpan.attributes["gen_ai.usage.prompt_tokens"], 4);
  assert.equal(embedSpan.attributes.tools, undefined);
  assert.equal(embedSpan.attributes.tool_calls, undefined);

  assert.equal(rerankSpan.attributes["llm.request.type"], "rerank");
  assert.equal(rerankSpan.attributes["gen_ai.request.model"], "rerank-v4.0");
  assert.equal(rerankSpan.attributes["gen_ai.prompt.0.content"], "capital");
  assert.deepEqual(JSON.parse(rerankSpan.attributes["traceloop.entity.output"]), [
    { index: 1, relevanceScore: 0.97 },
  ]);
  assert.equal(rerankSpan.attributes["llm.usage.total_tokens"], 1);
  assert.equal(rerankSpan.attributes["respan.span.tool_calls"], undefined);

  instrumentor.deactivate();
});

test("captures streaming generation when the async iterator completes", async () => {
  const module = fakeModule();
  const instrumentor = new CohereInstrumentor({ sdkModule: module });
  await instrumentor.activate();

  const client = new module.CohereClient();
  const stream = await client.generateStream({
    model: "command",
    prompt: "Write one sentence.",
  });
  for await (const _event of stream) {
    // consume stream
  }
  await flushMicrotasks();

  assert.equal(captureState.spans.length, 1);
  const [span] = captureState.spans;
  const attrs = span.attributes;
  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "completion");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Write one sentence.");
  assert.equal(attrs["gen_ai.completion.0.content"], "Generated stream response.");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 4);
  assert.equal(attrs["llm.usage.total_tokens"], 11);

  instrumentor.deactivate();
});

test("records failed Cohere calls as error spans", async () => {
  class ErrorClient extends FakeCohereClient {
    async chat(_request) {
      throw new Error("cohere unavailable");
    }
  }

  const module = { CohereClient: ErrorClient };
  const instrumentor = new CohereInstrumentor({ sdkModule: module });
  await instrumentor.activate();

  const client = new module.CohereClient();
  await assert.rejects(
    () => client.chat({ model: "command", message: "fail" }),
    /cohere unavailable/,
  );
  await flushMicrotasks();

  assert.equal(captureState.spans.length, 1);
  const [span] = captureState.spans;
  assert.equal(span.attributes["respan.entity.log_type"], "chat");
  assert.equal(span.attributes["status_code"], 500);
  assert.equal(span.attributes["error.message"], "cohere unavailable");
  assert.equal(span.status.code, 2);

  instrumentor.deactivate();
});
