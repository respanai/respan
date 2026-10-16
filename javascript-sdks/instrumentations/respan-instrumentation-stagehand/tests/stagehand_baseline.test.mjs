// Baseline wire contract: what a customer configures must reach every span, in
// the shape Respan ingestion reads. These drive the REAL Stagehand v4 SDK and
// its Chrome extension (headless local Chrome, a local HTTP page, a scripted
// client model) through the real Respan pipeline: RespanTelemetry's span
// processor and filters, then the exporter. Assertions read what the exporter
// would send.
//
// Ingestion collects ONLY flat `respan.metadata.<key>` attributes; a
// JSON-string `respan.metadata` is dropped (and @respan/tracing 1.6 writes
// propagated metadata as exactly that). Thread / session / trace group are
// what group a Stagehand session's spans.
import assert from "node:assert/strict";
import test from "node:test";
import { context } from "@opentelemetry/api";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { RespanTelemetry, propagateAttributes, withWorkflow } from "@respan/tracing";
import { z } from "zod";
import { StagehandInstrumentor, withModelName } from "../dist/index.js";
import { launchBrowser, scriptedModel, servePage, USAGE } from "./fixtures.mjs";

const exporter = new InMemorySpanExporter();
const telemetry = new RespanTelemetry({
  apiKey: "test-key",
  baseURL: "http://127.0.0.1:9",
  exporter,
  silenceInitializationMessage: true,
});
await telemetry.initialize();
// Imported after the Respan provider is registered, so Stagehand's own
// OpenTelemetry RPC spans are live: they must still stay out of exports.
const stagehandModule = await import("@browserbasehq/stagehand");
const { Stagehand } = stagehandModule;

const instrumentor = new StagehandInstrumentor({ stagehandModule, clientModelName: "scripted-1" });
await instrumentor.activate();
const site = await servePage();
test.after(async () => {
  instrumentor.deactivate();
  await site.close();
  await telemetry.shutdown();
});

const CUSTOMER_ID = "respan.customer_params.customer_identifier";
const THREAD = "respan.threads.thread_identifier";
const SESSION = "respan.sessions.session_identifier";
const GROUP = "respan.trace.trace_group_identifier";
const LOG_TYPE = "respan.entity.log_type";
const ENTITY = "traceloop.entity.name";
const INPUT = "traceloop.entity.input";
const OUTPUT = "traceloop.entity.output";
const ENTITY_NAME_KEY = Symbol.for("traceloop.entity.name");
const WORKFLOW_NAME_KEY = Symbol.for("traceloop.workflow.name");
const CUSTOMER_METADATA = { agent_id: "a1", role: "planner" };

async function exported() {
  await telemetry.flush();
  const spans = exporter.getFinishedSpans();
  exporter.reset();
  return spans;
}

/** One Stagehand session: create, the flow, close. Returns the exported spans. */
async function session(flow, { model = scriptedModel(), modelName } = {}) {
  const browser = await launchBrowser();
  try {
    const generate = modelName ? withModelName(model.generate, modelName) : model.generate;
    const stagehand = await Stagehand.create({ browser, model: { generate }, logging: { level: "off" } });
    try {
      const [page] = await browser.context.pages();
      await page.goto(site.url);
      await flow(stagehand, page, browser);
    } finally {
      await stagehand.close();
    }
  } finally {
    await browser.close();
  }
}

const byEntity = (spans, name) => spans.filter((s) => s.attributes[ENTITY] === name);
const sessionId = (spans) => byEntity(spans, "stagehand.session")[0].attributes["stagehand.session.id"];

async function standardFlow(stagehand) {
  await stagehand.extract("get the heading", z.object({ extraction: z.string() }));
  await stagehand.act("click the sign in button");
}

function assertFullSession(spans) {
  const types = spans.map((s) => s.attributes[LOG_TYPE]);
  assert.equal(types.filter((t) => t === "agent").length, 1, types.join());
  for (const type of ["task", "chat", "tool"]) assert.ok(types.includes(type), `${type} in ${types.join()}`);
  // Every exported span is ours: Stagehand's native RPC spans never leak.
  for (const span of spans) assert.ok(span.attributes[LOG_TYPE], span.name);
}

test("baseline: span tree is session -> operation -> llm / browser action", async () => {
  await session(standardFlow);
  const spans = await exported();
  assertFullSession(spans);
  const [root] = byEntity(spans, "stagehand.session");
  const extract = byEntity(spans, "stagehand.extract")[0];
  const act = byEntity(spans, "stagehand.act")[0];
  const id = (s) => s.spanContext().spanId;
  const parent = (s) => s.parentSpanContext?.spanId;
  assert.equal(parent(root), undefined);
  assert.equal(parent(extract), id(root));
  assert.equal(parent(act), id(root));
  const llm = byEntity(spans, "stagehand.llm");
  // extract makes two inferences (extraction + completion check), act one.
  assert.deepEqual(llm.map(parent).sort(), [id(extract), id(extract), id(act)].sort());
  for (const chat of llm) {
    assert.equal(chat.attributes["gen_ai.request.model"], "scripted-1");
    assert.equal(chat.attributes["gen_ai.usage.prompt_tokens"], USAGE.inputTokens);
    assert.equal(chat.attributes["gen_ai.usage.completion_tokens"], USAGE.outputTokens);
    assert.equal(chat.attributes["gen_ai.usage.cache_read_input_tokens"], USAGE.cachedInputTokens);
    assert.equal(chat.attributes["gen_ai.prompt.0.role"], "system");
    assert.ok(chat.attributes["gen_ai.completion.0.content"]);
  }
  const goto = byEntity(spans, "page.goto")[0];
  assert.equal(parent(goto), id(root));
  assert.match(goto.attributes[OUTPUT], /"status":200/);
  // Operation results, Stagehand's cache status and the post-act screenshot.
  assert.equal(JSON.parse(extract.attributes[OUTPUT]).data.extraction, "Hello browser agent");
  assert.equal(extract.attributes["stagehand.cache.status"], "DISABLED");
  const actOutput = JSON.parse(act.attributes[OUTPUT]);
  assert.equal(actOutput.data.success, true);
  assert.match(actOutput.screenshot.image_url.url, /^data:image\/jpeg;base64,/);
  assert.equal(JSON.parse(root.attributes[OUTPUT]).operations, 2);
});

test("baseline: custom metadata is flat respan.metadata.<key> on every span", async () => {
  await propagateAttributes({ metadata: CUSTOMER_METADATA }, () => session(standardFlow));
  const spans = await exported();
  assertFullSession(spans);
  for (const span of spans) {
    for (const [key, value] of Object.entries(CUSTOMER_METADATA))
      assert.equal(span.attributes[`respan.metadata.${key}`], value, `${span.name} ${key}`);
    assert.equal(span.attributes["respan.metadata"], undefined, span.name);
  }
});

test("baseline: customer identifier on every span", async () => {
  await propagateAttributes({ customer_identifier: "customer-1" }, () => session(standardFlow));
  const spans = await exported();
  assertFullSession(spans);
  for (const span of spans) assert.equal(span.attributes[CUSTOMER_ID], "customer-1", span.name);
});

test("baseline: thread, session and trace group default to the Stagehand session, one trace", async () => {
  await session(standardFlow);
  const spans = await exported();
  assertFullSession(spans);
  const id = sessionId(spans);
  assert.ok(id);
  for (const span of spans)
    for (const key of [THREAD, SESSION, GROUP]) assert.equal(span.attributes[key], id, `${span.name} ${key}`);
  assert.equal(new Set(spans.map((s) => s.spanContext().traceId)).size, 1);
});

test("baseline: customer thread and trace group win over the defaults", async () => {
  await propagateAttributes({ thread_identifier: "my-thread", group_identifier: "my-group" }, () => session(standardFlow));
  const spans = await exported();
  assertFullSession(spans);
  const id = sessionId(spans);
  for (const span of spans) {
    assert.equal(span.attributes[THREAD], "my-thread", span.name);
    assert.equal(span.attributes[GROUP], "my-group", span.name);
    assert.equal(span.attributes[SESSION], id, span.name);
  }
});

test("baseline: parallel sessions do not share traces or identifiers", async () => {
  await Promise.all(
    // Session b names its own client model; a falls back to the instrumentor's.
    ["a", "b"].map((name) =>
      propagateAttributes({ customer_identifier: `customer-${name}` }, () =>
        session(standardFlow, name === "b" ? { modelName: "scripted-2" } : {}),
      ),
    ),
  );
  const spans = await exported();
  const roots = byEntity(spans, "stagehand.session");
  assert.equal(roots.length, 2);
  const traceOf = new Map(roots.map((r) => [r.attributes[THREAD], r.spanContext().traceId]));
  assert.equal(traceOf.size, 2);
  for (const span of spans) assert.equal(span.spanContext().traceId, traceOf.get(span.attributes[THREAD]), span.name);
  for (const root of roots) {
    const mine = spans.filter((s) => s.attributes[THREAD] === root.attributes[THREAD]);
    assert.equal(new Set(mine.map((s) => s.attributes[CUSTOMER_ID])).size, 1);
    const chats = mine.filter((s) => s.attributes[ENTITY] === "stagehand.llm");
    assert.equal(chats.length, 3);
    const customer = mine[0].attributes[CUSTOMER_ID];
    const model = customer === "customer-b" ? "scripted-2" : "scripted-1";
    for (const chat of chats) assert.equal(chat.attributes["gen_ai.request.model"], model, customer);
  }
});

test("baseline: concurrent operations in one session keep their own LLM calls", async () => {
  const other = await servePage("<title>Second</title><h1>Second page heading</h1>");
  // The extraction echoes the heading the model was shown, so each LLM span's
  // prompt says which page (and therefore which operation) it belongs to.
  const model = scriptedModel({
    extraction: (params) => ({ extraction: /Second page heading/.test(JSON.stringify(params.messages)) ? "second" : "first" }),
  });
  try {
    await session(
      async (stagehand, page, browser) => {
        const second = await browser.context.newPage(other.url);
        await Promise.all([
          stagehand.extract("read the heading of the sign-in page", z.object({ extraction: z.string() }), { page }),
          stagehand.extract("read the heading of the second page", z.object({ extraction: z.string() }), { page: second }),
        ]);
      },
      { model },
    );
  } finally {
    await other.close();
  }
  const spans = await exported();
  const extracts = byEntity(spans, "stagehand.extract");
  assert.equal(extracts.length, 2);
  for (const op of extracts) {
    const answer = JSON.parse(op.attributes[OUTPUT]).data.extraction;
    const calls = byEntity(spans, "stagehand.llm").filter((s) => s.parentSpanContext?.spanId === op.spanContext().spanId);
    assert.equal(calls.length, 2, answer);
    const sawSecond = calls.some((c) => /Second page heading/.test(c.attributes[INPUT]));
    assert.equal(sawSecond, answer === "second", answer);
  }
});

test("baseline: content off keeps identifiers and drops content", async () => {
  const quiet = new StagehandInstrumentor({ stagehandModule, captureContent: false });
  instrumentor.deactivate();
  await quiet.activate();
  try {
    await propagateAttributes({ customer_identifier: "customer-1", metadata: CUSTOMER_METADATA }, () =>
      session(async (stagehand) => {
        await stagehand.extract("A private instruction", z.object({ extraction: z.string() }));
        await stagehand.act("click the sign in button");
      }),
    );
  } finally {
    quiet.deactivate();
    await instrumentor.activate();
  }
  const spans = await exported();
  assertFullSession(spans);
  for (const span of spans) {
    assert.equal(span.attributes[CUSTOMER_ID], "customer-1", span.name);
    assert.equal(span.attributes["respan.metadata.agent_id"], "a1", span.name);
    assert.ok(span.attributes[THREAD], span.name);
    const dumped = JSON.stringify(span.attributes);
    assert.ok(!dumped.includes("A private instruction"), span.name);
    assert.ok(!dumped.includes(site.url), span.name);
    assert.ok(!dumped.includes("Hello browser agent"), span.name);
    assert.ok(!dumped.includes("base64"), span.name);
  }
  // Structure and tokens survive.
  const chat = byEntity(spans, "stagehand.llm");
  assert.equal(chat.length, 3);
  for (const span of chat) assert.equal(span.attributes["gen_ai.usage.prompt_tokens"], USAGE.inputTokens);
});

test("baseline: act variables and typed credentials never reach a span", async () => {
  const secret = "s3cr3t-value-7781";
  const typed = "typed-password-9931";
  const model = scriptedModel({ target: /textbox/, method: "fill", args: [`%password%`] });
  await session(
    async (stagehand, page) => {
      await stagehand.act("type %password% into the password field", {
        variables: { password: { value: secret, description: "the password" } },
      });
      // A credential field: the typed text is redacted, even when the selector
      // is an XPath (what observe() returns) that says nothing about the field.
      await page.locator("#password").fill(typed);
      await page.locator("xpath=/html[1]/body[1]/form[1]/label[2]/input[1]").fill(typed);
      // An ordinary field keeps its text.
      await page.locator("xpath=/html[1]/body[1]/form[1]/label[1]/input[1]").fill("ada@example.com");
      // An error that echoes the secret is masked too.
      await page.goto(`not-a-url-${secret}`).catch(() => {});
    },
    { model },
  );
  const spans = await exported();
  assertFullSession(spans);
  for (const span of spans) {
    const dumped = JSON.stringify([span.attributes, span.status, span.events]);
    assert.ok(!dumped.includes(secret), span.name);
    assert.ok(!dumped.includes(typed), span.name);
  }
  const all = JSON.stringify(spans.map((s) => [s.attributes, s.status]));
  assert.ok(all.includes("%password%"));
  const fills = byEntity(spans, "locator.fill").map((s) => JSON.parse(s.attributes[INPUT]).text);
  assert.deepEqual(fills, ["[redacted]", "[redacted]", "ada@example.com"]);
});

test("baseline: errors carry `Type: message` and the span is marked failed", async () => {
  await session(async (stagehand, page) => {
    await assert.rejects(page.locator("#missing").click({ timeout: 300 }));
  });
  const spans = await exported();
  const [click] = byEntity(spans, "locator.click");
  assert.equal(click.status.code, 2);
  assert.match(click.status.message, /^\w*Error: /);
});

test("baseline: inside a Respan workflow the session nests and native RPC spans stay out", async () => {
  // Stagehand opens one native OpenTelemetry span per RPC in the context of
  // the call. Respan exports any span whose context carries a workflow's
  // entity path, so Stagehand must run without one.
  const entityPaths = [];
  await withWorkflow({ name: "checkout" }, () =>
    session(async (stagehand, page) => {
      const send = stagehand.rpcClient.send;
      stagehand.rpcClient.send = function (...args) {
        const active = context.active();
        entityPaths.push(active.getValue(ENTITY_NAME_KEY) ?? active.getValue(WORKFLOW_NAME_KEY));
        return send.apply(this, args);
      };
      await standardFlow(stagehand);
      await page.locator("#email").fill("a@b.c");
    }),
  );
  assert.ok(entityPaths.length >= 3);
  assert.deepEqual([...new Set(entityPaths)], [undefined]);
  const spans = await exported();
  const [workflow] = spans.filter((s) => s.attributes["traceloop.span.kind"] === "workflow");
  assert.ok(workflow);
  const [root] = byEntity(spans, "stagehand.session");
  assert.equal(root.parentSpanContext?.spanId, workflow.spanContext().spanId);
  for (const span of spans) assert.equal(span.attributes["rpc.system.name"], undefined, span.name);
});
