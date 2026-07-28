import assert from "node:assert/strict";
import test from "node:test";

import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { RespanSpanAttributes } from "@respan/respan-sdk";
import { WORKFLOW_NAME_KEY } from "@respan/tracing";
import { SpanAttributes } from "@traceloop/ai-semantic-conventions";

import {
  VertexAIInstrumentor,
  buildGenerateContentAttrs,
  extractUsage,
  requestPayloadFromCall,
} from "../dist/index.js";

const capturedSpans = [];
const originalGetTracerProvider = trace.getTracerProvider.bind(trace);
const contextManager = new AsyncLocalStorageContextManager().enable();
context.setGlobalContextManager(contextManager);

function makeUsage(promptTokens = 3, completionTokens = 4) {
  return {
    promptTokenCount: promptTokens,
    candidatesTokenCount: completionTokens,
    totalTokenCount: promptTokens + completionTokens,
  };
}

function makeResponse(text = "Hello", opts = {}) {
  return {
    candidates: [
      {
        content: {
          role: "model",
          parts: opts.parts ?? [{ text }],
        },
      },
    ],
    usageMetadata: opts.usage ?? makeUsage(),
  };
}

function makeTool(name = "get_weather") {
  return {
    functionDeclarations: [
      {
        name,
        description: "Get weather",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
        },
      },
    ],
  };
}

function createFakeVertexAIModule() {
  class GenerativeModel {
    constructor(model = "gemini-2.0-flash", opts = {}) {
      this.model = model;
      this.systemInstruction = opts.systemInstruction;
      this.tools = opts.tools;
      this.generationConfig = opts.generationConfig;
    }

    generateContent(request) {
      return Promise.resolve({
        response: makeResponse(`generate: ${JSON.stringify(request)}`, {
          usage: makeUsage(5, 6),
        }),
      });
    }

    generateContentStream(_request) {
      return Promise.resolve({
        stream: (async function* streamChunks() {
          yield makeResponse("stream ");
          yield makeResponse("done", { usage: makeUsage(7, 8) });
        })(),
        response: Promise.resolve(makeResponse("stream done", { usage: makeUsage(7, 8) })),
      });
    }

    startChat() {
      return new ChatSession(this);
    }
  }

  class ChatSession {
    constructor(model) {
      this.model = model;
    }

    sendMessage(content) {
      return Promise.resolve({
        response: makeResponse(`chat: ${content}`, { usage: makeUsage(9, 10) }),
      });
    }

    sendMessageStream(content) {
      return Promise.resolve({
        response: Promise.resolve(
          makeResponse(`chat stream: ${content}`, { usage: makeUsage(11, 12) }),
        ),
      });
    }
  }

  return { GenerativeModel, ChatSession };
}

function resetTraceCapture() {
  capturedSpans.length = 0;
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value() {
      return {
        activeSpanProcessor: {
          onEnd(span) {
            capturedSpans.push(span);
          },
        },
      };
    },
  });
}

test.after(() => {
  context.disable();
  Object.defineProperty(trace, "getTracerProvider", {
    configurable: true,
    writable: true,
    value: originalGetTracerProvider,
  });
});

test("buildGenerateContentAttrs emits canonical chat fields without banned aliases", () => {
  const response = makeResponse("", {
    parts: [
      {
        functionCall: {
          id: "call_1",
          name: "get_weather",
          args: { city: "Tokyo" },
        },
      },
    ],
    usage: makeUsage(13, 14),
  });

  const attrs = buildGenerateContentAttrs({
    requestPayload: {
      model: "gemini-2.0-flash",
      contents: [{ role: "user", parts: [{ text: "Weather in Tokyo?" }] }],
      systemInstruction: "Be concise",
      tools: [makeTool()],
      generationConfig: {
        maxOutputTokens: 128,
        temperature: 0.2,
        topP: 0.9,
        topK: 32,
      },
    },
    responseOrChunks: response,
  });

  assert.equal(attrs[RespanSpanAttributes.RESPAN_LOG_TYPE], "chat");
  assert.equal(attrs["gen_ai.system"], "google");
  assert.equal(attrs["llm.request.type"], "chat");
  assert.equal(attrs["gen_ai.request.model"], "gemini-2.0-flash");
  assert.equal(attrs["gen_ai.request.max_tokens"], 128);
  assert.equal(attrs["gen_ai.prompt.0.role"], "system");
  assert.equal(attrs["gen_ai.prompt.0.content"], "Be concise");
  assert.equal(attrs["gen_ai.prompt.1.role"], "user");
  assert.equal(attrs["gen_ai.prompt.1.content"], "Weather in Tokyo?");
  assert.equal(attrs["gen_ai.completion.0.role"], "assistant");
  assert.equal(attrs["gen_ai.usage.input_tokens"], 13);
  assert.equal(attrs["gen_ai.usage.output_tokens"], 14);
  assert.equal(attrs["gen_ai.usage.prompt_tokens"], 13);
  assert.equal(attrs["gen_ai.usage.completion_tokens"], 14);
  assert.equal(attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS], 27);

  assert.equal(JSON.parse(attrs["llm.request.functions"])[0].function.name, "get_weather");
  assert.deepEqual(JSON.parse(attrs["gen_ai.completion.0.tool_calls"]), [
    {
      id: "call_1",
      type: "function",
      function: {
        name: "get_weather",
        arguments: "{\"city\":\"Tokyo\"}",
      },
    },
  ]);

  for (const bannedKey of [
    "respan.span.tools",
    "respan.span.tool_calls",
    "tools",
    "tool_calls",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_request_tokens",
  ]) {
    assert.equal(attrs[bannedKey], undefined, `${bannedKey} should not be emitted`);
  }
});

test("patches generate, stream, and chat methods and emits spans", async () => {
  resetTraceCapture();
  const sdkModule = createFakeVertexAIModule();
  const instrumentor = new VertexAIInstrumentor({ sdkModule });
  await instrumentor.activate();

  const model = new sdkModule.GenerativeModel("gemini-2.0-flash", {
    systemInstruction: "Use short answers",
    tools: [makeTool("lookup_city")],
  });

  const result = await model.generateContent({
    contents: [{ role: "user", parts: [{ text: "Hello" }] }],
  });
  assert.match(result.response.candidates[0].content.parts[0].text, /^generate:/);

  const streamResult = await model.generateContentStream({
    contents: [{ role: "user", parts: [{ text: "Stream" }] }],
  });
  await streamResult.response;

  const chat = model.startChat();
  await chat.sendMessage("Continue");
  const chatStream = await chat.sendMessageStream("Continue streaming");
  await chatStream.response;

  assert.equal(capturedSpans.length, 4);
  assert.equal(capturedSpans[0].name, "vertexai.generate_content");
  assert.equal(capturedSpans[0].instrumentationLibrary.name, "@respan/instrumentation-vertexai");
  assert.equal(capturedSpans[0].attributes["gen_ai.request.model"], "gemini-2.0-flash");
  assert.equal(capturedSpans[0].attributes["gen_ai.usage.input_tokens"], 5);
  assert.equal(capturedSpans[1].attributes["gen_ai.completion.0.content"], "stream done");
  assert.equal(capturedSpans[2].name, "vertexai.chat.send_message");
  assert.equal(capturedSpans[2].attributes["gen_ai.prompt.1.content"], "Continue");
  assert.equal(capturedSpans[3].attributes["gen_ai.usage.completion_tokens"], 12);

  instrumentor.deactivate();
});

test("requestPayloadFromCall reads model defaults and chat input", () => {
  const sdkModule = createFakeVertexAIModule();
  const tool = makeTool("lookup");
  const model = new sdkModule.GenerativeModel("gemini-2.0-flash", {
    systemInstruction: "Use short answers",
    tools: [tool],
    generationConfig: { temperature: 0.1 },
  });
  const chat = model.startChat();

  const payload = requestPayloadFromCall(chat, ["Hello"], { isChatMethod: true });

  assert.equal(payload.model, "gemini-2.0-flash");
  assert.equal(payload.contents, "Hello");
  assert.equal(payload.systemInstruction, "Use short answers");
  assert.deepEqual(payload.tools, [tool]);
  assert.deepEqual(payload.generationConfig, undefined);
});

test("active workflow name is attached to injected Vertex AI spans", () => {
  const workflowContext = context.active().setValue(
    WORKFLOW_NAME_KEY,
    "typescript_vertexai_generate_content_example",
  );

  const attrs = context.with(workflowContext, () =>
    buildGenerateContentAttrs({
      requestPayload: {
        model: "gemini-2.0-flash",
        contents: "Hello",
      },
      responseOrChunks: makeResponse("Hi"),
    }),
  );

  assert.equal(
    attrs[SpanAttributes.TRACELOOP_WORKFLOW_NAME],
    "typescript_vertexai_generate_content_example",
  );
});

test("records failed generation as an error span", async () => {
  resetTraceCapture();

  class ErrorGenerativeModel {
    constructor(model = "gemini-2.0-flash") {
      this.model = model;
    }

    generateContent() {
      throw new Error("boom");
    }
  }

  const sdkModule = { GenerativeModel: ErrorGenerativeModel };
  const instrumentor = new VertexAIInstrumentor({ sdkModule });
  await instrumentor.activate();

  assert.throws(() => new sdkModule.GenerativeModel().generateContent("fail"), /boom/);

  assert.equal(capturedSpans.length, 1);
  assert.equal(capturedSpans[0].status.code, 2);
  assert.equal(capturedSpans[0].status.message, "boom");
  assert.equal(capturedSpans[0].attributes["error.message"], "boom");
  assert.equal(capturedSpans[0].attributes.status_code, 500);
  assert.equal(capturedSpans[0].attributes["gen_ai.request.model"], "gemini-2.0-flash");

  instrumentor.deactivate();
});

// Gemini reports thinking tokens separately but bills them at the output rate. Left out
// of the completion count the span contradicts itself: prompt + completion stops
// reconciling against the total the API returned, and the thinking tokens land on no
// attribute at all, so anything costing off the span under-reports output.
test("thinking tokens fold into the output count", () => {
  const result = extractUsage({
    usageMetadata: {
      promptTokenCount: 100,
      candidatesTokenCount: 50,
      thoughtsTokenCount: 800,
      totalTokenCount: 950,
    },
  });

  assert.equal(result.promptTokenCount, 100);
  assert.equal(result.candidatesTokenCount, 850);
  assert.equal(result.totalTokenCount, 950);
  assert.equal(
    result.promptTokenCount + result.candidatesTokenCount,
    result.totalTokenCount,
  );
});

test("thinking tokens fold in when the payload uses snake_case", () => {
  const result = extractUsage({
    usage_metadata: {
      prompt_token_count: 100,
      candidates_token_count: 50,
      thoughts_token_count: 800,
      total_token_count: 950,
    },
  });

  assert.equal(result.candidatesTokenCount, 850);
});

// Control: no thoughts field at all, which is every non-thinking model. This is why the
// defect went unnoticed, since the existing fixtures all look like this.
test("usage is unchanged when the model does not think", () => {
  const result = extractUsage({ usageMetadata: makeUsage(100, 50) });

  assert.equal(result.candidatesTokenCount, 50);
  assert.equal(result.totalTokenCount, 150);
});

test("zero thinking tokens leave the output count alone", () => {
  const result = extractUsage({
    usageMetadata: {
      promptTokenCount: 100,
      candidatesTokenCount: 50,
      thoughtsTokenCount: 0,
      totalTokenCount: 150,
    },
  });

  assert.equal(result.candidatesTokenCount, 50);
});
