import assert from "node:assert/strict";
import { trace } from "@opentelemetry/api";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import * as SDK from "@aws-sdk/client-bedrock-runtime";
import { AWSBedrockInstrumentor } from "../dist/index.js";
import { handler } from "./native-fixture.mjs";
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
const instrumentor = new AWSBedrockInstrumentor({ sdkModule: SDK });
await instrumentor.activate();
const response = {
  output: {
    message: {
      role: "assistant",
      content: Array.from({ length: 80 }, (_, i) => ({
        text: `complete output ${i}`,
      })),
    },
  },
  usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
  stopReason: "end_turn",
};
const client = new SDK.BedrockRuntimeClient({
  region: "us-east-1",
  credentials: { accessKeyId: "fixture", secretAccessKey: "fixture" },
  maxAttempts: 1,
  requestHandler: handler({ response, status: 201 }),
});
try {
  await client.send(
    new SDK.ConverseCommand({
      modelId: "fixture-model",
      messages: Array.from({ length: 76 }, (_, i) => ({
        role: "user",
        content: [{ text: `complete input ${i}` }],
      })),
      toolConfig: {
        tools: Array.from({ length: 76 }, (_, i) => ({
          toolSpec: {
            name: `tool${i}`,
            inputSchema: {
              json: {
                type: "object",
                properties: { flag: { type: "boolean", default: false } },
              },
            },
          },
        })),
      },
    }),
  );
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
      ? JSON.parse(attrs["traceloop.entity.output"]).output.message.content
          .length
      : null,
    usage: attrs["gen_ai.usage.input_tokens"] ?? null,
    metadata: attrs["respan.metadata"] ?? null,
    httpStatus: attrs["http.response.status_code"] ?? null,
    dropped: spans[0].droppedAttributesCount,
  };
  console.log(
    JSON.stringify({
      ...result,
      metadata: result.metadata ? "complete" : null,
    }),
  );
  assert.equal(result.attributes, 128);
  assert.equal(result.input, 76);
  assert.equal(result.tools, 76);
  assert.equal(result.output, 80);
  assert.equal(result.usage, 0);
  assert.equal(result.httpStatus, 201);
  assert.equal(JSON.parse(result.metadata).request.toolConfig.tools.length, 76);
  assert(result.dropped > 0);
} finally {
  instrumentor.deactivate();
  client.destroy();
}
