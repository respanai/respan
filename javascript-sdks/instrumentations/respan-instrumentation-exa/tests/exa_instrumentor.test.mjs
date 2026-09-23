import assert from "node:assert/strict";
import test from "node:test";

import { trace } from "@opentelemetry/api";
import { ATTR_ERROR_MESSAGE } from "@opentelemetry/semantic-conventions/incubating";
import { RespanSpanAttributes } from "@respan/respan-sdk";

import {
  ExaInstrumentor,
} from "../dist/index.js";
import {
  EXA_METADATA_NAMESPACE,
  METADATA_CITATIONS,
  METADATA_OPERATION,
  METADATA_RESEARCH_LEGACY,
  METADATA_REQUEST_ID,
  METADATA_STREAM_COMPLETED,
  OFF_CONTRACT_ALIASES,
} from "../dist/_constants.js";
import {
  CANONICAL_ATTRS,
  buildStartAttributes,
  buildSuccessAttributes,
} from "../dist/_translator.js";
import { safeJson } from "../dist/_serialization.js";

const capturedSpans = [];
let spanCounter = 1;
let metadataOnStart;

class FakeSpan {
  constructor(name, attributes = {}) {
    this.name = name;
    this.attributes = { ...attributes };
    this.ended = false;
    this.exceptions = [];
    this.status = undefined;
    this.id = spanCounter++;
  }

  setAttribute(key, value) {
    this.attributes[key] = value;
  }

  recordException(error) {
    this.exceptions.push(error);
  }

  setStatus(status) {
    this.status = status;
  }

  spanContext() {
    return {
      traceId: this.id.toString(16).padStart(32, "0"),
      spanId: this.id.toString(16).padStart(16, "0"),
      traceFlags: 1,
      isRemote: false,
      isValid: true,
    };
  }

  end() {
    if (this.ended) throw new Error(`span ${this.name} ended twice`);
    this.ended = true;
    capturedSpans.push(this);
  }
}

function resetTracer() {
  capturedSpans.length = 0;
  spanCounter = 1;
  metadataOnStart = undefined;
  trace.disable?.();
  trace.setGlobalTracerProvider({
    getTracer() {
      return {
        startSpan(name, options = {}) {
          const span = new FakeSpan(name, options.attributes);
          if (metadataOnStart) {
            const key = RespanSpanAttributes.RESPAN_METADATA;
            const existing = JSON.parse(span.attributes[key]);
            span.setAttribute(
              key,
              JSON.stringify({ ...metadataOnStart, ...existing }),
            );
          }
          return span;
        },
      };
    },
  });
}

class FakeExa {
  async search(query, options = {}) {
    if (query === "fail") throw new Error("deterministic Exa failure");
    if (query === "rate-limit") {
      const error = new Error("deterministic Exa rate limit");
      error.statusCode = 429;
      throw error;
    }
    return {
      results: [{ url: "https://example.com", text: "result text" }],
      requestId: "req-search",
      resolvedSearchType: options.type ?? "auto",
      costDollars: { total: 0.007 },
    };
  }

  async *streamSearch() {
    yield { content: "fresh ", citations: [] };
    yield { content: "result", citations: [{ url: "https://example.com" }] };
  }

  async searchAndContents(query, options = {}) {
    return await this.search(query, options);
  }

  async getContents(urls) {
    return { results: [{ url: urls[0], text: "page body" }] };
  }

  async findSimilar(url) {
    return { results: [{ url: `${url}/similar` }] };
  }

  async findSimilarAndContents(url) {
    return await this.findSimilar(url);
  }

  async answer() {
    return {
      answer: "Grounded answer",
      citations: [{ url: "https://example.com/source" }],
      costDollars: { total: 0.005 },
    };
  }

  async *streamAnswer() {
    yield { content: "Grounded ", citations: [] };
    yield { content: "stream", citations: [] };
  }
}

class FakeAgentRunEventsClient {
  async list(runId) {
    return { data: [{ type: "run.completed", runId }] };
  }
}

class FakeAgentBetaRunEventsClient extends FakeAgentRunEventsClient {
  async list(runId, options = {}) {
    return await super.list(runId, options);
  }
}

class FakeAgentRunsClient {
  async create(params) {
    if (params.stream) {
      return (async function* () {
        yield { type: "run.started", query: params.query };
        yield { type: "run.completed", output: "agent result" };
      })();
    }
    return { id: "run-1", status: "queued", query: params.query };
  }

  async get(runId) {
    return { id: runId, status: "completed", output: "agent result" };
  }

  async list() {
    return { data: [{ id: "run-1" }] };
  }

  async cancel(runId) {
    return { id: runId, status: "cancelled" };
  }

  async delete(runId) {
    return { id: runId, deleted: true };
  }

  async pollUntilFinished(runId) {
    return await this.get(runId);
  }

  async createAndWait(params) {
    const created = await this.create(params);
    return await this.get(created.id);
  }
}

class FakeAgentBetaRunsClient extends FakeAgentRunsClient {
  async create(params) {
    return await super.create(params);
  }

  async get(runId, options = {}) {
    return await super.get(runId, options);
  }

  async list(options = {}) {
    return await super.list(options);
  }

  async cancel(runId, options = {}) {
    return await super.cancel(runId, options);
  }

  async stop(runId) {
    return { id: runId, status: "stopped" };
  }

  async delete(runId, options = {}) {
    return await super.delete(runId, options);
  }

  async pollUntilFinished(runId, options = {}) {
    return await super.pollUntilFinished(runId, options);
  }

  async createAndWait(params, options = {}) {
    return await super.createAndWait(params, options);
  }
}

class FakeResearchClient {
  async create(params) {
    return { researchId: "research-1", status: "pending", ...params };
  }

  async get(researchId, options = {}) {
    if (options.stream) {
      return (async function* () {
        yield { type: "research.started" };
        yield { type: "research.completed", output: "report" };
      })();
    }
    return { researchId, status: "completed", output: "report" };
  }

  async list() {
    return { data: [{ researchId: "research-1" }] };
  }

  async pollUntilFinished(researchId) {
    return await this.get(researchId);
  }
}

function fakeSdk() {
  return {
    Exa: FakeExa,
    default: FakeExa,
    AgentRunsClient: FakeAgentRunsClient,
    AgentBetaRunsClient: FakeAgentBetaRunsClient,
    AgentRunEventsClient: FakeAgentRunEventsClient,
    AgentBetaRunEventsClient: FakeAgentBetaRunEventsClient,
    ResearchClient: FakeResearchClient,
  };
}

function span(name) {
  return capturedSpans.find((item) => item.name === name);
}

function exaMetadata(attributes) {
  return JSON.parse(attributes[RespanSpanAttributes.RESPAN_METADATA])[
    EXA_METADATA_NAMESPACE
  ];
}

test.after(() => {
  trace.disable?.();
});

test.afterEach(() => {
  for (const item of capturedSpans) {
    assert.equal(
      Object.keys(item.attributes).some((key) => key.startsWith("exa.")),
      false,
    );
    assert.equal(item.attributes["llm.is_streaming"], undefined);
  }
});

test("pure translators emit canonical tool and answer contracts", () => {
  const tool = buildStartAttributes({
    config: { entityName: "search", family: "tool", operation: "search" },
    input: { query: "fresh AI", apiKey: "secret" },
    captureContent: true,
    streaming: false,
    hasParent: false,
  });
  assert.equal(tool[RespanSpanAttributes.RESPAN_LOG_TYPE], "tool");
  assert.equal(tool[CANONICAL_ATTRS.entityPath], "");
  assert.deepEqual(JSON.parse(tool[CANONICAL_ATTRS.entityInput]), {
    name: "search",
    arguments: { query: "fresh AI", apiKey: "<redacted>" },
  });
  assert.deepEqual(exaMetadata(tool), {
    language: "typescript",
    operation: "search",
    stream: false,
  });
  assert.equal(Object.keys(tool).some((key) => key.startsWith("exa.")), false);
  for (const alias of OFF_CONTRACT_ALIASES) assert.equal(tool[alias], undefined);

  const chatConfig = { entityName: "answer", family: "chat", operation: "answer" };
  const defaultChat = buildStartAttributes({
    config: chatConfig,
    input: { query: "What changed?" },
    captureContent: true,
    streaming: false,
    hasParent: false,
  });
  const chat = buildStartAttributes({
    config: chatConfig,
    input: { query: "What changed?", systemPrompt: "Cite sources", model: "exa-pro" },
    captureContent: true,
    streaming: false,
    hasParent: true,
  });
  const completion = buildSuccessAttributes({
    config: chatConfig,
    input: { query: "What changed?", model: "exa-pro" },
    result: {
      answer: "A grounded answer",
      citations: [{ url: "https://example.com/source" }],
      model: "exa-pro",
    },
    captureContent: true,
    streaming: false,
  });
  assert.equal(chat["gen_ai.system"], "exa");
  assert.equal(defaultChat["gen_ai.request.model"], undefined);
  assert.equal(defaultChat["llm.is_streaming"], undefined);
  assert.equal(exaMetadata(defaultChat).stream, false);
  assert.equal(chat["gen_ai.request.model"], "exa-pro");
  assert.equal(completion["gen_ai.request.model"], "exa-pro");
  assert.equal(chat[CANONICAL_ATTRS.entityName], "answer");
  assert.equal(chat["gen_ai.prompt.0.role"], "system");
  assert.equal(chat["gen_ai.prompt.1.content"], "What changed?");
  assert.equal(completion["gen_ai.completion.0.content"], "A grounded answer");
  assert.deepEqual(exaMetadata(completion)[METADATA_CITATIONS], [
    { url: "https://example.com/source" },
  ]);
  for (const attrs of [defaultChat, chat, completion]) {
    assert.equal(Object.keys(attrs).some((key) => key.startsWith("exa.")), false);
  }
});

test("traces core success, errors, and content privacy", async () => {
  resetTracer();
  const instrumentor = new ExaInstrumentor({ sdkModule: fakeSdk() });
  await instrumentor.activate();
  try {
    metadataOnStart = { run_id: "exa-sop-run" };
    const result = await new FakeExa().search("fresh AI", { type: "auto" });
    metadataOnStart = undefined;
    assert.equal(result.results.length, 1);
    await new FakeExa().searchAndContents("fresh contents");
    await new FakeExa().getContents(["https://example.com"]);
    await new FakeExa().findSimilar("https://example.com");
    await new FakeExa().findSimilarAndContents("https://example.com");
    await assert.rejects(() => new FakeExa().search("fail"), /deterministic Exa failure/);
    await assert.rejects(
      () => new FakeExa().search("rate-limit"),
      /deterministic Exa rate limit/,
    );
  } finally {
    instrumentor.deactivate();
  }

  const searches = capturedSpans.filter((item) => item.name === "search");
  assert.equal(searches.length, 3);
  const success = searches.find((item) => item.status.code === 1);
  const failures = searches.filter((item) => item.status.code === 2);
  assert.equal(exaMetadata(success.attributes)[METADATA_REQUEST_ID], "req-search");
  assert.equal(
    JSON.parse(success.attributes[RespanSpanAttributes.RESPAN_METADATA]).run_id,
    "exa-sop-run",
  );
  assert.equal(
    JSON.parse(success.attributes[CANONICAL_ATTRS.entityOutput]).results.length,
    1,
  );
  assert.equal(success.attributes.status_code, 200);
  assert.deepEqual(
    new Set(failures.map((item) => item.attributes.status_code)),
    new Set([429, 500]),
  );
  assert.ok(
    failures.some((item) =>
      /deterministic Exa failure/.test(item.attributes[ATTR_ERROR_MESSAGE]),
    ),
  );
  const semanticNames = new Map(
    capturedSpans.map((item) => [exaMetadata(item.attributes)[METADATA_OPERATION], item.name]),
  );
  assert.equal(semanticNames.get("searchAndContents"), "search_and_contents");
  assert.equal(semanticNames.get("getContents"), "get_contents");
  assert.equal(semanticNames.get("findSimilar"), "find_similar");
  assert.equal(
    semanticNames.get("findSimilarAndContents"),
    "find_similar_and_contents",
  );
  for (const item of capturedSpans) {
    assert.equal(
      Object.keys(item.attributes).some((key) => key.startsWith("exa.")),
      false,
    );
  }

  resetTracer();
  const privateInstrumentor = new ExaInstrumentor({
    sdkModule: fakeSdk(),
    captureContent: false,
  });
  await privateInstrumentor.activate();
  try {
    await new FakeExa().answer("private question", { systemPrompt: "private prompt" });
  } finally {
    privateInstrumentor.deactivate();
  }
  const attrs = span("answer").attributes;
  assert.equal(attrs[CANONICAL_ATTRS.entityInput], undefined);
  assert.equal(attrs[CANONICAL_ATTRS.entityOutput], undefined);
  assert.equal(Object.keys(attrs).some((key) => key.startsWith("gen_ai.prompt.")), false);
  assert.equal(Object.keys(attrs).some((key) => key.startsWith("gen_ai.completion.")), false);
});

test("stream spans finish on exhaustion and early return", async () => {
  resetTracer();
  const instrumentor = new ExaInstrumentor({ sdkModule: fakeSdk() });
  await instrumentor.activate();
  try {
    const content = [];
    for await (const chunk of new FakeExa().streamSearch("query")) {
      content.push(chunk.content);
    }
    assert.equal(content.join(""), "fresh result");

    for await (const chunk of new FakeExa().streamAnswer("answer")) {
      assert.equal(chunk.content, "Grounded ");
      break;
    }
  } finally {
    instrumentor.deactivate();
  }

  const search = span("search");
  assert.equal(exaMetadata(search.attributes)[METADATA_STREAM_COMPLETED], true);
  assert.equal(exaMetadata(search.attributes).stream, true);
  assert.equal(search.attributes["llm.is_streaming"], undefined);
  assert.equal(
    JSON.parse(search.attributes[CANONICAL_ATTRS.entityOutput]).content,
    "fresh result",
  );
  assert.equal(
    exaMetadata(span("answer").attributes)[METADATA_STREAM_COMPLETED],
    false,
  );
});

test("covers Agent and legacy Research core surfaces", async () => {
  resetTracer();
  const instrumentor = new ExaInstrumentor({ sdkModule: fakeSdk() });
  await instrumentor.activate();
  try {
    const run = await new FakeAgentRunsClient().createAndWait({ query: "research company" });
    assert.equal(run.status, "completed");
    const stream = await new FakeAgentRunsClient().create({
      query: "stream agent",
      stream: true,
    });
    const events = [];
    for await (const event of stream) events.push(event);
    assert.equal(events.at(-1).type, "run.completed");
    const research = await new FakeResearchClient().pollUntilFinished("research-1");
    assert.equal(research.status, "completed");
    assert.equal(
      (await new FakeAgentRunsClient().get("standalone-run")).status,
      "completed",
    );
    assert.equal(
      (await new FakeResearchClient().get("standalone-research")).status,
      "completed",
    );
  } finally {
    instrumentor.deactivate();
  }

  const operations = capturedSpans.map(
    (item) => exaMetadata(item.attributes)[METADATA_OPERATION],
  );
  assert.equal(
    operations.filter((operation) => operation === "agent.runs.createAndWait").length,
    1,
  );
  assert.equal(
    operations.filter((operation) => operation === "agent.runs.create").length,
    1,
  );
  assert.equal(
    operations.filter((operation) => operation === "agent.runs.get").length,
    1,
  );
  assert.equal(
    operations.filter((operation) => operation === "research.pollUntilFinished").length,
    1,
  );
  assert.equal(
    operations.filter((operation) => operation === "research.get").length,
    1,
  );
  assert.ok(
    capturedSpans.some(
      (item) => exaMetadata(item.attributes)[METADATA_RESEARCH_LEGACY],
    ),
  );
  assert.ok(
    capturedSpans.some(
      (item) =>
        exaMetadata(item.attributes)[METADATA_OPERATION] ===
          "agent.runs.createAndWait" && item.name === "run",
    ),
  );
  assert.ok(
    capturedSpans.some(
      (item) =>
        exaMetadata(item.attributes)[METADATA_OPERATION] ===
          "research.pollUntilFinished" && item.name === "research",
    ),
  );
  assert.ok(
    capturedSpans.some(
      (item) =>
        exaMetadata(item.attributes)[METADATA_OPERATION] === "agent.runs.get" &&
        item.name === "run.get",
    ),
  );
  assert.ok(
    capturedSpans.some(
      (item) =>
        exaMetadata(item.attributes)[METADATA_OPERATION] === "research.get" &&
        item.name === "research.get",
    ),
  );
});

test("keeps shared patches until the last instrumentor deactivates", async () => {
  resetTracer();
  const sdk = fakeSdk();
  const original = FakeExa.prototype.search;
  const first = new ExaInstrumentor({ sdkModule: sdk });
  const second = new ExaInstrumentor({ sdkModule: sdk });
  await first.activate();
  const wrapped = FakeExa.prototype.search;
  await second.activate();
  assert.equal(FakeExa.prototype.search, wrapped);
  first.deactivate();
  assert.equal(FakeExa.prototype.search, wrapped);
  second.deactivate();
  assert.equal(FakeExa.prototype.search, original);
});

test("matches the exact npm-stable exa-js 2.19 surface", async () => {
  const sdk = await import("exa-js");
  const { Exa } = sdk;
  const packageJson = await import("exa-js/package.json", { with: { type: "json" } });
  assert.equal(packageJson.default.version, "2.19.0");
  for (const method of [
    "search",
    "streamSearch",
    "getContents",
    "answer",
    "streamAnswer",
  ]) {
    assert.equal(typeof Exa.prototype[method], "function");
  }
  const client = new Exa("not-used");
  assert.equal(typeof client.tools.webSearch, "function");
  assert.equal(client.tools.getContents, undefined);
  assert.equal(typeof client.agent.runs.createAndWait, "function");
  assert.equal(typeof client.research.pollUntilFinished, "function");

  const researchPrototype = Object.getPrototypeOf(client.research);
  const betaRunsPrototype = Object.getPrototypeOf(client.beta.agent.runs);
  const originalResearchCreate = researchPrototype.create;
  const originalBetaCreateAndWait = betaRunsPrototype.createAndWait;
  const instrumentor = new ExaInstrumentor({ sdkModule: sdk });
  await instrumentor.activate();
  try {
    assert.notEqual(researchPrototype.create, originalResearchCreate);
    assert.notEqual(betaRunsPrototype.createAndWait, originalBetaCreateAndWait);
  } finally {
    instrumentor.deactivate();
  }
  assert.equal(researchPrototype.create, originalResearchCreate);
  assert.equal(betaRunsPrototype.createAndWait, originalBetaCreateAndWait);
});

test("redacts nested credentials during serialization", () => {
  assert.deepEqual(
    JSON.parse(
      safeJson({
        query: "safe",
        headers: { Authorization: "Bearer secret", "x-api-key": "secret" },
      }),
    ),
    {
      query: "safe",
      headers: { Authorization: "<redacted>", "x-api-key": "<redacted>" },
    },
  );
});
