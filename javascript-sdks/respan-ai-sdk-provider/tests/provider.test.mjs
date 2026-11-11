import assert from "node:assert/strict";
import { test } from "node:test";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { generateText, streamText } from "ai";
import { createRespan, gatewayModelId, wireFormat } from "../dist/index.js";

const BASE = "https://gateway.test";

// Records each request and fails it, so a test can inspect exactly what the
// provider would send without a network call or a format-specific response.
function recorder() {
  const calls = [];
  const fetch = async (url, init) => {
    calls.push({
      url: String(url),
      headers: Object.fromEntries(new Headers(init?.headers).entries()),
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    });
    return new Response(JSON.stringify({ error: { message: "recorded" } }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  };
  return { calls, fetch };
}

async function send(modelId, extra = {}) {
  const { calls, fetch } = recorder();
  const respan = createRespan({ apiKey: "test-key", baseURL: BASE, fetch });
  await generateText({ model: respan(modelId), prompt: "Hi", maxRetries: 0, ...extra }).catch(() => {});
  assert.equal(calls.length, 1, `one request for ${modelId}`);
  return calls[0];
}

test("Claude models use Anthropic Messages with x-api-key", async () => {
  for (const id of ["anthropic/claude-sonnet-4.5", "claude-sonnet-4-5-20250929"]) {
    const call = await send(id);
    assert.equal(call.url, `${BASE}/v1/messages`, id);
    assert.equal(call.headers["x-api-key"], "test-key", id);
  }
});

test("Gemini models use generateContent with x-goog-api-key", async () => {
  const call = await send("google/gemini-3.5-flash");
  assert.match(call.url, new RegExp(`^${BASE}/v1beta/models/[^/]+:generateContent$`));
  assert.equal(call.headers["x-goog-api-key"], "test-key");
});

test("OpenAI models use the Responses API with a bearer key", async () => {
  for (const id of ["openai/gpt-5.5", "gpt-5.5"]) {
    const call = await send(id);
    assert.equal(call.url, `${BASE}/v1/responses`, id);
    assert.equal(call.headers.authorization, "Bearer test-key", id);
  }
});

test("routes and other providers use Chat Completions with the ID unchanged", async () => {
  for (const id of ["smart", "azure/gpt-5.5", "xai/grok-4", "openrouter/openai/gpt-oss-120b"]) {
    const call = await send(id);
    assert.equal(call.url, `${BASE}/v1/chat/completions`, id);
    assert.equal(call.body.model, id, id);
  }
});

test("Anthropic provider options reach the request body", async () => {
  const call = await send("anthropic/claude-sonnet-4.5", {
    providerOptions: { anthropic: { thinking: { type: "enabled", budgetTokens: 2048 } } },
  });
  assert.deepEqual(call.body.thinking, { type: "enabled", budget_tokens: 2048 });
});

test("string model IDs resolve through AI_SDK_DEFAULT_PROVIDER", async () => {
  const { calls, fetch } = recorder();
  const previous = globalThis.AI_SDK_DEFAULT_PROVIDER;
  globalThis.AI_SDK_DEFAULT_PROVIDER = createRespan({ apiKey: "test-key", baseURL: BASE, fetch });
  try {
    await generateText({ model: "anthropic/claude-sonnet-4.5", prompt: "Hi", maxRetries: 0 }).catch(() => {});
  } finally {
    globalThis.AI_SDK_DEFAULT_PROVIDER = previous;
  }
  assert.equal(calls[0]?.url, `${BASE}/v1/messages`);
});

test("a missing API key fails when the model is used, not at import", () => {
  const saved = process.env.RESPAN_API_KEY;
  delete process.env.RESPAN_API_KEY;
  try {
    const respan = createRespan({ baseURL: BASE });
    assert.throws(() => respan("gpt-5.5"), /RESPAN_API_KEY/);
  } finally {
    if (saved !== undefined) process.env.RESPAN_API_KEY = saved;
  }
});

test("maker prefixes are dropped so the gateway can fall back across providers", async () => {
  assert.equal((await send("anthropic/claude-sonnet-4.5")).body.model, "claude-sonnet-4-5");
  assert.equal((await send("openai/gpt-5.5")).body.model, "gpt-5.5");
  assert.match((await send("google/gemini-3.5-flash")).url, /\/models\/gemini-3\.5-flash:generateContent$/);
  assert.equal(gatewayModelId("azure/gpt-5.5"), "azure/gpt-5.5");
  assert.equal(gatewayModelId("claude-haiku-4-5"), "claude-haiku-4-5");
});

test("Claude version dots become dashes so the Anthropic provider applies the model's limits", async () => {
  for (const id of ["anthropic/claude-sonnet-4.5", "claude-sonnet-4.5", "claude-sonnet-4-5"]) {
    const call = await send(id);
    assert.equal(call.body.model, "claude-sonnet-4-5", id);
    assert.equal(call.body.max_tokens, 64000, id);
  }
  assert.equal(gatewayModelId("anthropic/claude-opus-4.1"), "claude-opus-4-1");
  assert.equal(gatewayModelId("google/gemini-3.5-flash"), "gemini-3.5-flash");
  assert.equal(gatewayModelId("openai/gpt-5.5"), "gpt-5.5");
  assert.equal(gatewayModelId("bedrock/claude-sonnet-4.5"), "bedrock/claude-sonnet-4.5");
});

test("wireFormat picks the format from the prefix or the model name", () => {
  assert.equal(wireFormat("anthropic/claude-sonnet-4.5"), "anthropic");
  assert.equal(wireFormat("claude-haiku-4-5"), "anthropic");
  assert.equal(wireFormat("gemini/gemini-3.5-flash"), "google");
  assert.equal(wireFormat("gemini-3.5-flash"), "google");
  assert.equal(wireFormat("openai/gpt-5.5"), "openai");
  assert.equal(wireFormat("o4-mini"), "openai");
  assert.equal(wireFormat("bedrock/claude-sonnet-4-5"), "chat");
  assert.equal(wireFormat("smart"), "chat");
});

// --- Gateway log link -------------------------------------------------------
// The gateway logs every call it serves and returns that row's id in
// x-respan-log-id. The AI SDK's model-call span names the row instead of
// counting the call a second time.

const LOG_ID = "4f1c1b0e5c8a4d3e9b7a6f5e4d3c2b1a";
const contextManager = new AsyncLocalStorageContextManager().enable();
context.setGlobalContextManager(contextManager);
const tracer = new BasicTracerProvider().getTracer("test");

function gatewayFetch({ stream = false, logId = LOG_ID } = {}) {
  return async () => {
    const headers = {
      "content-type": stream ? "text/event-stream" : "application/json",
      ...(logId ? { "x-respan-log-id": logId } : {}),
    };
    if (!stream) {
      return new Response(JSON.stringify({
        id: "chatcmpl-1", object: "chat.completion", created: 1, model: "azure/gpt-4.1-mini",
        choices: [{ index: 0, message: { role: "assistant", content: "Hi!" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      }), { status: 200, headers });
    }
    const chunks = [
      { id: "chatcmpl-1", object: "chat.completion.chunk", created: 1, model: "azure/gpt-4.1-mini",
        choices: [{ index: 0, delta: { role: "assistant", content: "Hi!" }, finish_reason: null }] },
      { id: "chatcmpl-1", object: "chat.completion.chunk", created: 1, model: "azure/gpt-4.1-mini",
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } },
    ];
    const body = chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";
    return new Response(body, { status: 200, headers });
  };
}

// Stands in for the AI SDK's model-call span, which is the active span while
// the provider calls the gateway (AI SDK 7 names it `chat <model>`).
async function inSpan(attributes, run) {
  const span = tracer.startSpan("chat azure/gpt-4.1-mini", { attributes });
  await context.with(trace.setSpan(context.active(), span), run);
  span.end();
  return span.attributes;
}

const MODEL_CALL = { "gen_ai.operation.name": "chat" };

test("generateText links the model-call span to the gateway log", async () => {
  const respan = createRespan({ apiKey: "k", baseURL: BASE, fetch: gatewayFetch() });
  const attrs = await inSpan(MODEL_CALL, () =>
    generateText({ model: respan("azure/gpt-4.1-mini"), prompt: "Hi", maxRetries: 0 }));
  assert.equal(attrs["respan.gateway.log_id"], LOG_ID);
});

test("streamText links the model-call span to the gateway log", async () => {
  const respan = createRespan({ apiKey: "k", baseURL: BASE, fetch: gatewayFetch({ stream: true }) });
  const attrs = await inSpan(MODEL_CALL, async () => {
    const result = streamText({ model: respan("azure/gpt-4.1-mini"), prompt: "Hi", maxRetries: 0 });
    assert.equal(await result.text, "Hi!");
  });
  assert.equal(attrs["respan.gateway.log_id"], LOG_ID);
});

test("legacy AI SDK model-call spans are linked too", async () => {
  const respan = createRespan({ apiKey: "k", baseURL: BASE, fetch: gatewayFetch() });
  const attrs = await inSpan({ "ai.operationId": "ai.generateText.doGenerate" }, () =>
    generateText({ model: respan("azure/gpt-4.1-mini"), prompt: "Hi", maxRetries: 0 }));
  assert.equal(attrs["respan.gateway.log_id"], LOG_ID);
});

test("a span that is not an AI SDK model call is never linked", async () => {
  const respan = createRespan({ apiKey: "k", baseURL: BASE, fetch: gatewayFetch() });
  const attrs = await inSpan({ "traceloop.span.kind": "task" }, () =>
    generateText({ model: respan("azure/gpt-4.1-mini"), prompt: "Hi", maxRetries: 0 }));
  assert.equal(attrs["respan.gateway.log_id"], undefined);
});

test("no header, no link", async () => {
  const respan = createRespan({ apiKey: "k", baseURL: BASE, fetch: gatewayFetch({ logId: "" }) });
  const attrs = await inSpan(MODEL_CALL, () =>
    generateText({ model: respan("azure/gpt-4.1-mini"), prompt: "Hi", maxRetries: 0 }));
  assert.equal(attrs["respan.gateway.log_id"], undefined);
});
