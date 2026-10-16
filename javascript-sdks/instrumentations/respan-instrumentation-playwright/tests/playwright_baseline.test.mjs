// Baseline wire contract: what a customer configures must reach every span, in
// the shape Respan ingestion reads. These drive a real headless Chrome through
// playwright-core against a local sign-in page, and capture spans through the
// real Respan pipeline (RespanTelemetry's span processor and filters, then the
// exporter). A scripted mini-agent (a fake LLM whose tool calls map to
// Playwright actions) stands in for the agent a customer wraps around Playwright.
//
// Ingestion collects ONLY flat `respan.metadata.<key>` attributes; a
// JSON-string `respan.metadata` is dropped (and @respan/tracing 1.6 writes
// propagated metadata as exactly that).
//
// One browser is launched with no span active and shared by every test, the
// way a test suite or an agent server keeps one browser: Playwright fires
// callbacks (route handlers, `page.on` listeners) in the context the browser
// was launched in, so these tests also prove callbacks are re-parented.
import assert from "node:assert/strict";
import test from "node:test";
import { context, trace } from "@opentelemetry/api";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { RespanTelemetry, propagateAttributes, withWorkflow } from "@respan/tracing";
import { expect } from "@playwright/test";
import { chromium } from "playwright-core";
import { PlaywrightInstrumentor, withSecrets } from "../dist/index.js";
import { PASSWORD, USERNAME, chromePath, launchOptions, runAgent, serveSite } from "./fixtures.mjs";

const skip = chromePath() ? false : "needs a local Chrome/Chromium";

const exporter = new InMemorySpanExporter();
const telemetry = new RespanTelemetry({
  apiKey: "test-key",
  baseURL: "http://127.0.0.1:9",
  exporter,
  silenceInitializationMessage: true,
});
await telemetry.initialize();
let instrumentor = new PlaywrightInstrumentor();
await instrumentor.activate();
const site = await serveSite();
const browser = skip ? undefined : await chromium.launch(launchOptions());
test.after(async () => {
  await browser?.close();
  instrumentor.deactivate();
  await site.close();
  await telemetry.shutdown();
});

const CUSTOMER_ID = "respan.customer_params.customer_identifier";
const THREAD = "respan.threads.thread_identifier";
const GROUP = "respan.trace.trace_group_identifier";
const LOG_TYPE = "respan.entity.log_type";
const SPAN_KIND = "traceloop.span.kind";
const INPUT = "traceloop.entity.input";
const OUTPUT = "traceloop.entity.output";
const PAGE_URL = "playwright.page.url";
const PAGE_TITLE = "playwright.page.title";
const API_NAME = "playwright.api.name";
const CUSTOMER_METADATA = { agent_id: "a1", role: "browser" };
const API_TOKEN = "tok-5531-secret";
const OTHER_PASSWORD = "never-typed-anywhere-9921";
const KEYBOARD_PASSWORD = "typed-on-the-keyboard-4410";

async function exported() {
  await telemetry.flush();
  const spans = exporter.getFinishedSpans();
  exporter.reset();
  return spans;
}

/** A new context and page on the shared browser; the flow, then close. */
async function session(flow, { route = false } = {}) {
  const browserContext = await browser.newContext();
  try {
    const page = await browserContext.newPage();
    if (route) {
      await page.route("**/done.html?*", (r) => r.fulfill({ body: "<title>Mocked</title>ok", contentType: "text/html" }));
    }
    await flow(page);
  } finally {
    await browserContext.close();
  }
}

const login = (options) => session((page) => runAgent(page, site.url), options);
const calls = (spans) => spans.filter((s) => s.attributes[API_NAME]);
const roots = (spans) => spans.filter((s) => !s.parentSpanContext?.spanId);
const named = (spans, name) => spans.filter((s) => s.attributes[API_NAME] === name);
const dumped = (span) => JSON.stringify([span.attributes, span.status, span.events]);
const input = (span) => JSON.parse(span.attributes[INPUT]);

function assertNestedUnderOneRoot(spans) {
  const rootSpans = roots(spans);
  assert.equal(rootSpans.length, 1, rootSpans.map((s) => s.name).join());
  const [root] = rootSpans;
  const names = new Set(calls(spans).map((s) => s.attributes[API_NAME]));
  for (const name of ["page.goto", "locator.fill", "page.click", "page.waitForURL"]) assert.ok(names.has(name), [...names].join());
  for (const span of calls(spans)) {
    assert.equal(span.spanContext().traceId, root.spanContext().traceId, span.name);
    assert.equal(span.parentSpanContext?.spanId, root.spanContext().spanId, span.name);
    // Exportable: Respan drops spans without a kind.
    assert.equal(span.attributes[LOG_TYPE], "tool", span.name);
    assert.equal(span.attributes[SPAN_KIND], "tool", span.name);
  }
  return root;
}

// -- parentage and exportability -----------------------------------------------

test("baseline: calls nest under the active span, one tool span per public call", { skip }, async () => {
  await withWorkflow({ name: "login" }, () => login());
  const spans = await exported();
  assertNestedUnderOneRoot(spans);
  // Local calls (page.locator) and our own probes (title, password check) are not spans.
  const names = calls(spans).map((s) => s.attributes[API_NAME]);
  assert.deepEqual(
    names.filter((n) => !n.startsWith("browser")),
    ["page.goto", "locator.fill", "locator.fill", "page.click", "page.waitForURL"],
  );
  const [goto] = named(spans, "page.goto");
  assert.equal(input(goto).url, `${site.url}/index.html`);
  assert.equal(JSON.parse(goto.attributes[OUTPUT]).status, 200);
  assert.equal(goto.attributes[PAGE_TITLE], "Sign in");
  assert.equal(goto.attributes[PAGE_URL], `${site.url}/index.html`);
});

test("baseline: route handlers and listeners keep the caller's parent and customer context", { skip }, async () => {
  // Playwright fires these from its connection, in the context the browser was
  // launched in (here: none). Without re-parenting, Route.fulfill and
  // Dialog.accept have no parent, so they are not traced at all.
  await propagateAttributes({ customer_identifier: "customer-1" }, () =>
    withWorkflow({ name: "login" }, () =>
      session(
        async (page) => {
          page.on("dialog", (dialog) => dialog.accept());
          await runAgent(page, site.url);
          await page.evaluate(() => alert("hi"));
        },
        { route: true },
      ),
    ),
  );
  const spans = await exported();
  const root = assertNestedUnderOneRoot(spans);
  for (const name of ["route.fulfill", "dialog.accept"]) {
    const [span] = named(spans, name);
    assert.ok(span, name);
    assert.equal(span.parentSpanContext?.spanId, root.spanContext().spanId, name);
    assert.equal(span.attributes[CUSTOMER_ID], "customer-1", name);
  }
});

test("baseline: calls under a customer's own agent span nest under it", { skip }, async () => {
  const agent = trace.getTracer("my-agent").startSpan("my_agent", {
    attributes: { [LOG_TYPE]: "agent", [SPAN_KIND]: "agent", "traceloop.entity.name": "my_agent" },
  });
  await context.with(trace.setSpan(context.active(), agent), () => login());
  agent.end();
  assertNestedUnderOneRoot(await exported());
});

test("baseline: no active span means no spans", { skip }, async () => {
  // Standalone scripts and test suites are not flooded with one-span traces.
  await login({ route: true });
  assert.deepEqual(await exported(), []);
});

// -- customer context -----------------------------------------------------------

test("baseline: customer metadata is flat respan.metadata.<key>, with the identifier, on every span", { skip }, async () => {
  await propagateAttributes({ customer_identifier: "customer-1", metadata: CUSTOMER_METADATA }, () =>
    withWorkflow({ name: "login" }, () => login({ route: true })),
  );
  const spans = await exported();
  assertNestedUnderOneRoot(spans);
  for (const span of spans) {
    assert.equal(span.attributes[CUSTOMER_ID], "customer-1", span.name);
    for (const [key, value] of Object.entries(CUSTOMER_METADATA))
      assert.equal(span.attributes[`respan.metadata.${key}`], value, `${span.name} ${key}`);
    assert.equal(span.attributes["respan.metadata"], undefined, span.name);
  }
});

test("baseline: customer thread and trace group on every span", { skip }, async () => {
  // Playwright has no run id of its own: its spans join the caller's thread and group.
  await propagateAttributes({ thread_identifier: "my-thread", group_identifier: "my-group" }, () =>
    withWorkflow({ name: "login" }, () => login({ route: true })),
  );
  const spans = await exported();
  assertNestedUnderOneRoot(spans);
  for (const span of spans) {
    assert.equal(span.attributes[THREAD], "my-thread", span.name);
    assert.equal(span.attributes[GROUP], "my-group", span.name);
  }
});

test("baseline: parallel sessions do not share traces or identifiers", { skip }, async () => {
  await Promise.all(
    ["a", "b"].map((name) =>
      propagateAttributes({ customer_identifier: `customer-${name}` }, () =>
        withWorkflow({ name: `login-${name}` }, () => login({ route: true })),
      ),
    ),
  );
  const spans = await exported();
  const traceOf = new Map(roots(spans).map((r) => [r.attributes[CUSTOMER_ID], r.spanContext().traceId]));
  assert.deepEqual([...traceOf.keys()].sort(), ["customer-a", "customer-b"]);
  assert.equal(new Set(traceOf.values()).size, 2);
  for (const span of spans) assert.equal(span.spanContext().traceId, traceOf.get(span.attributes[CUSTOMER_ID]), span.name);
  for (const name of ["a", "b"]) {
    const mine = spans.filter((s) => s.attributes[CUSTOMER_ID] === `customer-${name}`);
    assert.equal(named(mine, "route.fulfill").length, 1, name);
  }
});

// -- content, secrets, errors ------------------------------------------------------

test("baseline: password text never reaches a span", { skip }, async () => {
  // Typed into a password field: masked in the call, and wherever it comes back
  // later (this form submits by GET, so the password lands in the URL).
  await withWorkflow({ name: "login" }, () =>
    session(async (page) => {
      await runAgent(page, site.url);
      await page.goto(`${site.url}/index.html`);
      await page.locator("#password").focus();
      await page.keyboard.type(KEYBOARD_PASSWORD);
      await page.locator("#user").focus();
      await page.keyboard.type("plain-text-ok");
    }),
  );
  const spans = await exported();
  assertNestedUnderOneRoot(spans);
  for (const span of spans) {
    assert.ok(!dumped(span).includes(PASSWORD), span.name);
    assert.ok(!dumped(span).includes(KEYBOARD_PASSWORD), span.name);
  }
  const fills = new Map(named(spans, "locator.fill").map((s) => [input(s).selector, input(s).value]));
  assert.equal(fills.get("#password"), "<masked>");
  assert.equal(fills.get("#user"), USERNAME); // ordinary text is kept
  assert.deepEqual(named(spans, "keyboard.type").map((s) => input(s).text), ["<masked>", "plain-text-ok"]);
  assert.ok(spans.some((s) => (s.attributes[PAGE_URL] ?? "").includes("<secret>password</secret>")));
});

test("baseline: declared secrets never reach a span", { skip }, async () => {
  await withWorkflow({ name: "with_token" }, () =>
    withSecrets({ api_token: API_TOKEN }, () =>
      session(async (page) => {
        await page.goto(`${site.url}/index.html?token=${API_TOKEN}`);
        await page.evaluate((t) => t.length, API_TOKEN);
        await assert.rejects(page.click(`#${API_TOKEN}`, { timeout: 300 }), new RegExp(API_TOKEN));
      }),
    ),
  );
  const spans = await exported();
  for (const span of spans) assert.ok(!dumped(span).includes(API_TOKEN), span.name);
  assert.ok(spans.some((s) => dumped(s).includes("<secret>api_token</secret>")));
});

test("baseline: a failed text entry is masked", { skip }, async () => {
  // A fill that fails never reached a field we could inspect: fail closed.
  await withWorkflow({ name: "missing_field" }, () =>
    session(async (page) => {
      await page.goto(`${site.url}/index.html`);
      await assert.rejects(page.fill("#no-such-field", OTHER_PASSWORD, { timeout: 300 }), { name: "TimeoutError" });
    }),
  );
  const spans = await exported();
  const [fill] = named(spans, "page.fill");
  assert.equal(fill.status.code, 2);
  assert.match(fill.status.message, /^TimeoutError: page\.fill: Timeout 300ms exceeded/);
  for (const span of spans) assert.ok(!dumped(span).includes(OTHER_PASSWORD), span.name);
});

test("baseline: screenshots are exported as compact JPEG image parts, at most 1280 px wide", { skip }, async () => {
  // A photo-like page: its PNG screenshot is large, so it is re-encoded.
  const noisy = `<canvas id=c width=1600 height=900></canvas><script>
    const g = document.getElementById("c").getContext("2d"), img = g.createImageData(1600, 900);
    for (let i = 0; i < img.data.length; i++) img.data[i] = i % 4 === 3 ? 255 : Math.random() * 256;
    g.putImageData(img, 0, 0);</script>`;
  await withWorkflow({ name: "shots" }, () =>
    session(async (page) => {
      await page.setViewportSize({ width: 1600, height: 900 });
      await page.setContent(noisy);
      await page.screenshot();
      await page.locator("#c").screenshot();
    }),
  );
  const spans = await exported();
  for (const name of ["page.screenshot", "locator.screenshot"]) {
    const [span] = named(spans, name);
    const url = JSON.parse(span.attributes[OUTPUT]).screenshot.image_url.url;
    assert.match(url, /^data:image\/jpeg;base64,/, name);
    assert.equal(jpegWidth(Buffer.from(url.split(",")[1], "base64")), 1280, name);
  }
});

/** Width from a JPEG's start-of-frame marker. */
function jpegWidth(data) {
  for (let i = 2; i < data.length - 9; ) {
    const marker = data[i + 1];
    if (marker >= 0xc0 && marker <= 0xc3) return data.readUInt16BE(i + 7);
    i += 2 + data.readUInt16BE(i + 2);
  }
  return undefined;
}

test("baseline: content off keeps identifiers and drops content", { skip }, async () => {
  instrumentor.deactivate();
  const quiet = new PlaywrightInstrumentor({ captureContent: false });
  await quiet.activate();
  try {
    await propagateAttributes({ customer_identifier: "customer-1", metadata: CUSTOMER_METADATA }, () =>
      withWorkflow({ name: "private" }, () =>
        session(async (page) => {
          await runAgent(page, site.url);
          await page.screenshot();
          await assert.rejects(page.click("#secret-selector-xyz", { timeout: 300 }));
        }),
      ),
    );
  } finally {
    quiet.deactivate();
    instrumentor = new PlaywrightInstrumentor();
    await instrumentor.activate();
  }
  const spans = await exported();
  const playwright = calls(spans);
  for (const name of ["page.goto", "page.screenshot", "page.click"]) assert.ok(named(playwright, name).length, name);
  for (const span of playwright) {
    assert.equal(span.attributes[CUSTOMER_ID], "customer-1", span.name);
    assert.equal(span.attributes["respan.metadata.agent_id"], "a1", span.name);
    for (const key of [INPUT, OUTPUT, PAGE_URL, PAGE_TITLE]) assert.equal(span.attributes[key], undefined, `${span.name} ${key}`);
    for (const secret of [site.url, USERNAME, PASSWORD, "secret-selector-xyz", "image_url"])
      assert.ok(!dumped(span).includes(secret), `${span.name} ${secret}`);
  }
  const failed = playwright.filter((s) => s.status.code === 2);
  assert.deepEqual(failed.map((s) => s.status.message), ["TimeoutError"]);
});

test("baseline: errors carry `Type: message` and the span is marked failed", { skip }, async () => {
  await withWorkflow({ name: "missing" }, () =>
    session(async (page) => {
      await page.goto(`${site.url}/index.html`);
      await assert.rejects(page.locator("#missing").click({ timeout: 300 }));
    }),
  );
  const [click] = named(await exported(), "locator.click");
  assert.equal(click.status.code, 2);
  assert.match(click.status.message, /^TimeoutError: locator\.click: Timeout 300ms exceeded/);
});

test("baseline: expect() matchers are spans, a mismatch is an AssertionError span", { skip }, async () => {
  await withWorkflow({ name: "assertions" }, () =>
    session(async (page) => {
      await page.goto(`${site.url}/index.html`);
      await expect(page).toHaveTitle("Sign in");
      await expect(page.locator("#user")).toBeVisible();
      await assert.rejects(expect(page.locator("#submit")).toHaveText("Something else", { timeout: 300 }));
      await expect(page.locator("#submit")).not.toHaveText("Something else");
    }),
  );
  const spans = await exported();
  const [root] = roots(spans);
  const expects = calls(spans).filter((s) => s.attributes[API_NAME].startsWith("expect."));
  assert.deepEqual(
    expects.map((s) => [s.attributes[API_NAME], s.status.code]),
    [["expect.toHaveTitle", 0], ["expect.toBeVisible", 0], ["expect.toHaveText", 2], ["expect.not.toHaveText", 0]],
  );
  for (const span of expects) assert.equal(span.parentSpanContext?.spanId, root.spanContext().spanId);
  const [failed] = expects.filter((s) => s.status.code === 2);
  assert.match(failed.status.message, /^AssertionError: /);
  assert.match(failed.status.message, /Sign in/); // what the page showed
});
