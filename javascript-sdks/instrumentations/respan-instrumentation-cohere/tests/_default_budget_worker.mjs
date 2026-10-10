import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { CohereInstrumentor } from "../dist/index.js";
import { fixture } from "./_native_fixture.mjs";
const require = createRequire(import.meta.url),
  sdk = require(process.env.COHERE_NATIVE_MODULE || "cohere-ai");
delete process.env.OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT;
delete process.env.OTEL_ATTRIBUTE_COUNT_LIMIT;
const spans = [];
const provider = new NodeTracerProvider({
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
});
provider.register();
const server = await fixture(),
  plugin = new CohereInstrumentor({ sdkModule: sdk });
await plugin.activate();
try {
  const options = { token: "fixture", environment: server.url, maxRetries: 0 };
  const tools = Array.from({ length: 80 }, (_, index) => ({
    type: "function",
    function: {
      name: `tool_${index}`,
      parameters: {
        type: "object",
        properties: { value: { type: "number", default: 0 } },
      },
    },
  }));
  const request = sdk.CohereClientV2
    ? {
        model: "command",
        messages: Array.from({ length: 80 }, (_, i) => ({
          role: "user",
          content: `message-${i}`,
        })),
        tools,
      }
    : {
        model: "command",
        message: "last-message",
        chatHistory: Array.from({ length: 80 }, (_, i) => ({
          role: "USER",
          message: `message-${i}`,
        })),
      };
  const client = sdk.CohereClientV2
    ? new sdk.CohereClientV2(options)
    : new sdk.CohereClient(options);
  await client.chat(request);
  await new Promise((r) => setImmediate(r));
  assert.equal(spans.length, 1);
  const attrs = spans[0].attributes;
  assert.equal(
    JSON.parse(attrs["traceloop.entity.input"]).length,
    sdk.CohereClientV2 ? 80 : 81,
  );
  assert.ok(attrs["traceloop.entity.output"]);
  assert.equal(
    JSON.parse(attrs["traceloop.entity.output"])[0].content[0]?.text ??
      JSON.parse(attrs["traceloop.entity.output"])[0].content,
    "fixture-answer",
  );
  if (sdk.CohereClientV2)
    assert.equal(JSON.parse(attrs["llm.request.functions"]).length, 80);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 0);
  assert.ok(spans[0].droppedAttributesCount > 0);
  console.log(
    JSON.stringify({
      canonicalPayloads: true,
      droppedOptionalIndexedAttributes: spans[0].droppedAttributesCount,
    }),
  );
} finally {
  plugin.deactivate();
  await server.close();
  await provider.shutdown();
}
