import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { trace } from "@opentelemetry/api";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { AzureOpenAIInstrumentor } from "../dist/index.js";
const sdk = createRequire(import.meta.url)(
  process.env.AZURE_BUDGET_MODULE || "openai",
);
const spans = [];
trace.setGlobalTracerProvider(
  new BasicTracerProvider({
    spanProcessors: [
      {
        onStart() {},
        onEnd(span) {
          spans.push(span);
        },
        forceFlush: async () => {},
        shutdown: async () => {},
      },
    ],
  }),
);
const instrumentor = new AzureOpenAIInstrumentor({ openAIModule: sdk });
await instrumentor.activate();
try {
  const choices = Array.from({ length: 76 }, (_, index) => ({
    index,
    message: { role: "assistant", content: `complete output ${index}` },
    finish_reason: "stop",
  }));
  const client = new sdk.AzureOpenAI({
    apiKey: "fixture",
    endpoint: "https://fixture.openai.azure.com",
    apiVersion: "2024-10-21",
    deployment: "deployment",
    maxRetries: 0,
    fetch: async () =>
      new Response(
        JSON.stringify({
          id: "native-budget",
          object: "chat.completion",
          created: 1,
          model: "native-model",
          choices,
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        }),
        { status: 201, headers: { "content-type": "application/json" } },
      ),
  });
  await client.chat.completions.create({
    model: "deployment",
    messages: Array.from({ length: 76 }, (_, i) => ({
      role: "user",
      content: `complete input ${i}`,
    })),
    tools: Array.from({ length: 76 }, (_, i) => ({
      type: "function",
      function: {
        name: `tool${i}`,
        parameters: {
          type: "object",
          properties: { flag: { type: "boolean", default: false } },
        },
      },
    })),
    extraAttributes: { "respan.metadata": '{"budget":"default128"}' },
  });
  const attrs = spans[0].attributes;
  const result = {
    attributes: Object.keys(attrs).length,
    input: attrs["traceloop.entity.input"]
      ? JSON.parse(attrs["traceloop.entity.input"]).length
      : null,
    tools: attrs["llm.request.functions"]
      ? JSON.parse(attrs["llm.request.functions"]).length
      : null,
    output: attrs["traceloop.entity.output"]
      ? JSON.parse(attrs["traceloop.entity.output"]).length
      : null,
    usage: attrs["gen_ai.usage.input_tokens"] ?? null,
    metadata: attrs["respan.metadata"] ?? null,
    httpStatus: attrs["http.response.status_code"] ?? null,
    dropped: spans[0].droppedAttributesCount,
  };
  console.log(JSON.stringify(result));
  assert.equal(result.attributes, 128);
  assert.equal(result.input, 76);
  assert.equal(result.tools, 76);
  assert.equal(result.output, 76);
  assert.equal(result.usage, 0);
  assert.equal(result.httpStatus, 201);
  assert.equal(JSON.parse(result.metadata).budget, "default128");
  assert(result.dropped > 0);
} finally {
  instrumentor.deactivate();
}
