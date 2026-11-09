import assert from "node:assert/strict";
import { test } from "node:test";
import { generateText } from "ai";
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
  assert.equal((await send("anthropic/claude-sonnet-4.5")).body.model, "claude-sonnet-4.5");
  assert.equal((await send("openai/gpt-5.5")).body.model, "gpt-5.5");
  assert.match((await send("google/gemini-3.5-flash")).url, /\/models\/gemini-3\.5-flash:generateContent$/);
  assert.equal(gatewayModelId("azure/gpt-5.5"), "azure/gpt-5.5");
  assert.equal(gatewayModelId("claude-haiku-4-5"), "claude-haiku-4-5");
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
