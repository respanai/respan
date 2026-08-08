import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import {
  AWSBedrockInstrumentor,
  buildBedrockAttrs,
} from "../dist/index.js";

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

class ConverseCommand {
  constructor(input) {
    this.input = input;
  }
}

class ConverseStreamCommand {
  constructor(input) {
    this.input = input;
  }
}

class InvokeModelCommand {
  constructor(input) {
    this.input = input;
  }
}

class InvokeModelWithResponseStreamCommand {
  constructor(input) {
    this.input = input;
  }
}

function encodedJson(value) {
  return new TextEncoder().encode(JSON.stringify(value));
}

function createFakeModule({ fail = false } = {}) {
  class BedrockRuntimeClient {
    constructor() {
      this.calls = [];
    }

    async send(command) {
      this.calls.push(command);
      if (fail) {
        const error = new Error("bedrock unavailable");
        error.$metadata = { httpStatusCode: 429 };
        throw error;
      }

      if (command instanceof ConverseCommand) {
        return {
          $metadata: { httpStatusCode: 200 },
          output: {
            message: {
              role: "assistant",
              content: [
                { toolUse: { toolUseId: "toolu_1", name: "lookup", input: { city: "Tokyo" } } },
                { text: "Tokyo is sunny." },
              ],
            },
          },
          usage: {
            inputTokens: 11,
            outputTokens: 5,
            totalTokens: 16,
          },
        };
      }

      if (command instanceof InvokeModelCommand) {
        return {
          $metadata: { httpStatusCode: 200 },
          body: encodedJson({
            content: [{ type: "text", text: "Hello from invoke." }],
            role: "assistant",
            usage: {
              input_tokens: 7,
              output_tokens: 4,
            },
          }),
        };
      }

      if (command instanceof ConverseStreamCommand) {
        return {
          $metadata: { httpStatusCode: 200 },
          stream: (async function*() {
            yield {
              contentBlockStart: {
                start: {
                  toolUse: {
                    toolUseId: "toolu_stream",
                    name: "lookup",
                    input: { city: "Paris" },
                  },
                },
              },
            };
            yield { contentBlockDelta: { delta: { text: "Bonjour" } } };
            yield { contentBlockDelta: { delta: { text: " Paris" } } };
            yield {
              metadata: {
                usage: {
                  inputTokens: 9,
                  outputTokens: 3,
                  totalTokens: 12,
                },
              },
            };
          })(),
        };
      }

      if (command instanceof InvokeModelWithResponseStreamCommand) {
        return {
          $metadata: { httpStatusCode: 200 },
          body: (async function*() {
            yield {
              chunk: {
                bytes: encodedJson({
                  type: "content_block_delta",
                  delta: { text: "streamed " },
                }),
              },
            };
            yield {
              chunk: {
                bytes: encodedJson({
                  type: "content_block_delta",
                  delta: { text: "invoke" },
                }),
              },
            };
            yield {
              chunk: {
                bytes: encodedJson({
                  type: "message_delta",
                  usage: { output_tokens: 2 },
                }),
              },
            };
          })(),
        };
      }

      return {};
    }
  }

  return { BedrockRuntimeClient };
}

test("buildBedrockAttrs maps Converse request, tools, response, usage, and tool calls", () => {
  const attrs = buildBedrockAttrs({
    operationName: "Converse",
    apiParams: {
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      system: [{ text: "Be brief." }],
      messages: [
        {
          role: "user",
          content: [{ text: "Weather in Tokyo?" }],
        },
      ],
      toolConfig: {
        tools: [
          {
            toolSpec: {
              name: "lookup",
              description: "Lookup weather",
              inputSchema: {
                json: {
                  type: "object",
                  properties: { city: { type: "string" } },
                },
              },
            },
          },
        ],
      },
    },
    responsePayload: {
      output: {
        message: {
          role: "assistant",
          content: [
            { toolUse: { toolUseId: "toolu_1", name: "lookup", input: { city: "Tokyo" } } },
            { text: "Tokyo is sunny." },
          ],
        },
      },
      usage: {
        inputTokens: 11,
        outputTokens: 5,
        totalTokens: 16,
      },
    },
  });

  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["gen_ai.system"], "bedrock");
  assert.equal(attrs["llm.system"], undefined);
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "anthropic.claude-3-haiku-20240307-v1:0");
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.1.role"], "user");
  assert.equal(attrs["gen_ai.prompt.1.content"], "Weather in Tokyo?");
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "lookup",
        description: "Lookup weather",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
        },
      },
    },
  ]);
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Tokyo is sunny.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "toolu_1",
      type: "function",
      function: {
        name: "lookup",
        arguments: JSON.stringify({ city: "Tokyo" }),
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 11);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 5);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 11);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 5);
  assert.equal(attrs["llm.usage.total_tokens"], 16);
  assert.equal(attrs["respan.span.tools"], undefined);
  assert.equal(attrs.tools, undefined);
  assert.equal(attrs.tool_calls, undefined);
  assert.equal(attrs.model, undefined);
  assert.equal(attrs.prompt_tokens, undefined);
});

test("instrumentor patches BedrockRuntimeClient send and emits Converse spans", async () => {
  captureState.spans = [];
  const sdkModule = createFakeModule();
  const originalSend = sdkModule.BedrockRuntimeClient.prototype.send;
  const instrumentor = new AWSBedrockInstrumentor({ sdkModule });
  await instrumentor.activate();

  assert.notEqual(sdkModule.BedrockRuntimeClient.prototype.send, originalSend);

  const client = new sdkModule.BedrockRuntimeClient();
  await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [{ role: "user", content: [{ text: "Weather?" }] }],
    }),
  );

  assert.equal(captureState.spans.length, 1);
  const span = captureState.spans[0];
  assert.equal(span.name, "aws_bedrock.chat");
  assert.equal(span.instrumentationScope?.name, "@respan/instrumentation-aws-bedrock");
  assert.equal(span.attributes["respan.entity.log_type"], "chat");
  assert.equal(span.attributes["gen_ai.system"], "bedrock");
  assert.equal(span.attributes["gen_ai.completion.0.content"], "Tokyo is sunny.");
  assert.equal(span.attributes["error.message"], undefined);

  instrumentor.deactivate();
  assert.equal(sdkModule.BedrockRuntimeClient.prototype.send, originalSend);
});

test("instrumentor captures InvokeModel body payload without off-contract aliases", async () => {
  captureState.spans = [];
  const sdkModule = createFakeModule();
  const instrumentor = new AWSBedrockInstrumentor({ sdkModule });
  await instrumentor.activate();

  const client = new sdkModule.BedrockRuntimeClient();
  const response = await client.send(
    new InvokeModelCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        messages: [{ role: "user", content: "Hello" }],
      }),
    }),
  );

  assert.ok(response.body instanceof Uint8Array);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["gen_ai.prompt.0.content"], "Hello");
  assert.equal(attrs["gen_ai.completion.0.content"], "Hello from invoke.");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 7);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 4);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs.total_request_tokens, undefined);

  instrumentor.deactivate();
});

test("streaming spans are emitted after stream consumption", async () => {
  captureState.spans = [];
  const sdkModule = createFakeModule();
  const instrumentor = new AWSBedrockInstrumentor({ sdkModule });
  await instrumentor.activate();

  const client = new sdkModule.BedrockRuntimeClient();
  const response = await client.send(
    new ConverseStreamCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [{ role: "user", content: [{ text: "Weather in Paris?" }] }],
    }),
  );

  assert.equal(captureState.spans.length, 0);
  const eventTypes = [];
  for await (const event of response.stream) {
    eventTypes.push(Object.keys(event)[0]);
  }

  assert.deepEqual(eventTypes, [
    "contentBlockStart",
    "contentBlockDelta",
    "contentBlockDelta",
    "metadata",
  ]);
  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["gen_ai.completion.0.content"], "Bonjour Paris");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 9);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 3);
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "toolu_stream",
      type: "function",
      function: {
        name: "lookup",
        arguments: JSON.stringify({ city: "Paris" }),
      },
    },
  ]);

  instrumentor.deactivate();
});

test("InvokeModelWithResponseStream chunks are normalized", async () => {
  captureState.spans = [];
  const sdkModule = createFakeModule();
  const instrumentor = new AWSBedrockInstrumentor({ sdkModule });
  await instrumentor.activate();

  const client = new sdkModule.BedrockRuntimeClient();
  const response = await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        messages: [{ role: "user", content: "Stream" }],
      }),
    }),
  );

  for await (const _event of response.body) {
    // Consume stream to trigger span emission.
  }

  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["gen_ai.completion.0.content"], "streamed invoke");
  assert.equal(attrs["gen_ai.usage.output_tokens"], 2);

  instrumentor.deactivate();
});

test("failed Bedrock calls emit error spans", async () => {
  captureState.spans = [];
  const sdkModule = createFakeModule({ fail: true });
  const instrumentor = new AWSBedrockInstrumentor({ sdkModule });
  await instrumentor.activate();

  const client = new sdkModule.BedrockRuntimeClient();
  await assert.rejects(
    client.send(
      new ConverseCommand({
        modelId: "anthropic.claude-3-haiku-20240307-v1:0",
        messages: [{ role: "user", content: [{ text: "Fail" }] }],
      }),
    ),
    /bedrock unavailable/,
  );

  assert.equal(captureState.spans.length, 1);
  const span = captureState.spans[0];
  assert.equal(span.status.code, 2);
  assert.equal(span.attributes["error.message"], "bedrock unavailable");
  assert.equal(span.attributes.status_code, 429);
  assert.equal(span.attributes["gen_ai.prompt.0.content"], "Fail");

  instrumentor.deactivate();
});
