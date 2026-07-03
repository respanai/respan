import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";

import { AzureOpenAIInstrumentor } from "../dist/index.js";

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

function createModernOpenAIModule() {
  class ChatCompletions {
    constructor(client) {
      this._client = client;
    }

    async create(params) {
      assert.equal(params.extraAttributes, undefined);
      if (params.stream) {
        return (async function* stream() {
          yield {
            model: "gpt-4o-mini",
            choices: [
              {
                delta: {
                  role: "assistant",
                  content: "Paris",
                  tool_calls: [
                    {
                      index: 0,
                      id: "call_1",
                      type: "function",
                      function: { name: "lookup_city", arguments: "{\"city\"" },
                    },
                  ],
                },
              },
            ],
          };
          yield {
            model: "gpt-4o-mini",
            usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 },
            choices: [
              {
                delta: {
                  content: " is walkable.",
                  tool_calls: [
                    {
                      index: 0,
                      function: { arguments: ":\"Paris\"}" },
                    },
                  ],
                },
              },
            ],
          };
        })();
      }

      return {
        model: params.model,
        choices: [
          {
            message: {
              role: "assistant",
              content: "Paris is compact and transit friendly.",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: {
                    name: "lookup_city",
                    arguments: "{\"city\":\"Paris\"}",
                  },
                },
              ],
            },
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 8,
          total_tokens: 18,
          prompt_tokens_details: { cached_tokens: 2 },
        },
      };
    }
  }

  class Completions {
    constructor(client) {
      this._client = client;
    }

    async create(params) {
      return {
        model: params.model,
        choices: [{ text: "A concise text completion." }],
        usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
      };
    }
  }

  class Embeddings {
    constructor(client) {
      this._client = client;
    }

    async create(params) {
      return {
        model: params.model,
        data: [{ embedding: [0.1, 0.2, 0.3] }],
        usage: { prompt_tokens: 4, total_tokens: 4 },
      };
    }
  }

  class AzureOpenAI {
    static Chat = { Completions: ChatCompletions };
    static Completions = Completions;
    static Embeddings = Embeddings;

    constructor() {
      this.apiVersion = "2024-10-21";
      this.deploymentName = "gpt-4o-mini";
      this.baseURL = "https://example.openai.azure.com/openai";
      this.chat = { completions: new ChatCompletions(this) };
      this.completions = new Completions(this);
      this.embeddings = new Embeddings(this);
    }
  }

  return { AzureOpenAI };
}

test("patches modern AzureOpenAI chat completions with canonical attrs", async () => {
  captureState.spans = [];
  const openAIModule = createModernOpenAIModule();
  const instrumentor = new AzureOpenAIInstrumentor({ openAIModule });
  await instrumentor.activate();

  const client = new openAIModule.AzureOpenAI();
  const result = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Summarize Paris." },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "lookup_city",
          description: "Lookup city notes.",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      },
    ],
    extraAttributes: {
      "respan.metadata.example": "azure-openai-test",
      "custom.nested": { ok: true },
    },
  });

  assert.equal(result.choices[0].message.content, "Paris is compact and transit friendly.");
  assert.equal(captureState.spans.length, 1);

  const [span] = captureState.spans;
  const attrs = span.attributes;
  assert.equal(span.instrumentationLibrary.name, "@respan/instrumentation-azure-openai");
  assert.equal(attrs["respan.entity.log_method"], "ts_tracing");
  assert.equal(attrs["respan.entity.log_type"], "chat");
  assert.equal(attrs["gen_ai.system"], "azure");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-4o-mini");
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.1.content"], "Summarize Paris.");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "Paris is compact and transit friendly.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: { name: "lookup_city", arguments: "{\"city\":\"Paris\"}" },
    },
  ]);
  assert.deepEqual(JSON.parse(attrs["llm.request.functions"]), [
    {
      type: "function",
      function: {
        name: "lookup_city",
        description: "Lookup city notes.",
        parameters: { type: "object", properties: { city: { type: "string" } } },
      },
    },
  ]);
  assert.equal(attrs["gen_ai.usage.input_tokens"], 10);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 8);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 10);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 8);
  assert.equal(attrs["llm.usage.total_tokens"], 18);
  assert.equal(attrs["respan.metadata.example"], "azure-openai-test");
  assert.equal(attrs["respan.span.tools"], undefined);
  assert.equal(attrs["respan.span.tool_calls"], undefined);
  assert.equal(attrs.tools, undefined);
  assert.equal(attrs.tool_calls, undefined);

  instrumentor.deactivate();
});

test("patches modern AzureOpenAI text completions with parser-compatible attrs", async () => {
  captureState.spans = [];
  const openAIModule = createModernOpenAIModule();
  const instrumentor = new AzureOpenAIInstrumentor({ openAIModule });
  await instrumentor.activate();

  const client = new openAIModule.AzureOpenAI();
  const result = await client.completions.create({
    model: "gpt-35-turbo-instruct",
    prompt: "Write one sentence.",
    extraAttributes: {
      "respan.metadata.example": "azure-openai-completion-test",
    },
  });

  assert.equal(result.choices[0].text, "A concise text completion.");
  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gpt-35-turbo-instruct");
  assert.equal(attrs["gen_ai.prompt.0.role"], "user");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Write one sentence.");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.completion.0.content"], "A concise text completion.");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 3);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 5);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 3);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 5);
  assert.equal(attrs["llm.usage.total_tokens"], 8);
  assert.equal(attrs.prompt_tokens, undefined);
  assert.equal(attrs.completion_tokens, undefined);

  instrumentor.deactivate();
});

test("aggregates modern AzureOpenAI streaming tool calls", async () => {
  captureState.spans = [];
  const openAIModule = createModernOpenAIModule();
  const instrumentor = new AzureOpenAIInstrumentor({ openAIModule });
  await instrumentor.activate();

  const client = new openAIModule.AzureOpenAI();
  const stream = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Use a tool." }],
    stream: true,
  });

  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }

  assert.equal(chunks.length, 2);
  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["gen_ai.completion.0.content"], "Paris is walkable.");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: { name: "lookup_city", arguments: "{\"city\":\"Paris\"}" },
    },
  ]);

  instrumentor.deactivate();
});

test("captures embeddings without exporting vectors", async () => {
  captureState.spans = [];
  const openAIModule = createModernOpenAIModule();
  const instrumentor = new AzureOpenAIInstrumentor({ openAIModule });
  await instrumentor.activate();

  const client = new openAIModule.AzureOpenAI();
  const result = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: ["alpha", "beta"],
  });

  assert.deepEqual(result.data[0].embedding, [0.1, 0.2, 0.3]);
  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], "embedding");
  assert.equal(attrs["llm.request.type"], "embedding");
  assert.equal(attrs["gen_ai.request.model"], "text-embedding-3-small");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 4);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 4);
  assert.equal(attrs["ai.embedding"], undefined);
  assert.equal(attrs["ai.embeddings"], undefined);
  assert.match(attrs["traceloop.entity.output"], /embedding_count/);
  assert.doesNotMatch(attrs["traceloop.entity.output"], /0\\.1/);

  instrumentor.deactivate();
});

test("patches legacy @azure/openai OpenAIClient methods", async () => {
  captureState.spans = [];

  class OpenAIClient {
    async getCompletions(deployment, prompt, options) {
      assert.equal(options.extraAttributes, undefined);
      return {
        choices: [{ text: `Completion for ${prompt}` }],
        usage: { promptTokens: 6, completionTokens: 7, totalTokens: 13 },
        model: deployment,
      };
    }
  }

  const instrumentor = new AzureOpenAIInstrumentor({
    openAIModule: {},
    azureOpenAIModule: { OpenAIClient },
  });
  await instrumentor.activate();

  const client = new OpenAIClient();
  const result = await client.getCompletions("legacy-deployment", "legacy prompt", {
    extraAttributes: { "respan.metadata.legacy": true },
  });

  assert.equal(result.choices[0].text, "Completion for legacy prompt");
  assert.equal(captureState.spans.length, 1);
  const attrs = captureState.spans[0].attributes;
  assert.equal(attrs["respan.entity.log_type"], "text");
  assert.equal(attrs["gen_ai.request.model"], "legacy-deployment");
  assert.equal(attrs["gen_ai.prompt.0.content"], "legacy prompt");
  assert.equal(attrs["gen_ai.completion.0.content"], "Completion for legacy prompt");
  assert.equal(attrs["respan.metadata.legacy"], true);

  instrumentor.deactivate();
});
