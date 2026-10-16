// Baseline wire contract: what a customer configures must reach every span, in
// the shape Respan ingestion reads. These drive a real headless Chrome through
// puppeteer-core against a local sign-in page, and capture spans through the
// real Respan pipeline (RespanTelemetry's span processor and filters, then the
// exporter). A scripted mini-agent (a fake LLM whose tool calls map to
// Puppeteer actions) stands in for the agent a customer wraps around Puppeteer.
//
// Ingestion collects ONLY flat `respan.metadata.<key>` attributes; a
// JSON-string `respan.metadata` is dropped (and @respan/tracing 1.6 writes
// propagated metadata as exactly that).
//
// One browser is launched with no span active and shared by every test, the
// way a test suite or an agent server keeps one browser: Puppeteer fires
// listeners (request interception, dialogs) and exposed functions from its
// connection, in the context the browser was launched in, so these tests also
// prove callbacks are re-parented.
import assert from "node:assert/strict";
import test from "node:test";
import { context, trace } from "@opentelemetry/api";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { RespanTelemetry, propagateAttributes, withWorkflow } from "@respan/tracing";
import puppeteer from "puppeteer-core";
import { PuppeteerInstrumentor, withSecrets } from "../dist/index.js";
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
let instrumentor = new PuppeteerInstrumentor();
await instrumentor.activate();
const site = await serveSite();
const browser = skip ? undefined : await puppeteer.launch(launchOptions());
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
const PAGE_URL = "puppeteer.page.url";
const PAGE_TITLE = "puppeteer.page.title";
const API_NAME = "puppeteer.api.name";
const CUSTOMER_METADATA = { agent_id: "a1", role: "browser" };
const API_TOKEN = "tok-5531-secret";
const OTHER_PASSWORD = "never-typed-anywhere-9921";
const KEYBOARD_PASSWORD = "typed-on-the-keyboard-4410";
const HANDLE_PASSWORD = "typed-into-a-handle-7781";
const LOCATOR_PASSWORD = "filled-by-a-locator-3392";
const HEADER_TOKEN = "Bearer hdr-0042-secret";
const COOKIE_VALUE = "cookie-7709-secret";
/** The calls a session's browser context makes around the flow. */
const SESSION_CALLS = new Set(["browser.createBrowserContext", "browserContext.newPage", "browserContext.close"]);

async function exported() {
  await telemetry.flush();
  const spans = exporter.getFinishedSpans();
  exporter.reset();
  return spans;
}

const intercept = (request) =>
  request.url().includes("/done.html")
    ? request.respond({ status: 200, contentType: "text/html", body: "<title>Mocked</title>ok" })
    : request.continue();

/** A new browser context and page on the shared browser; the flow, then close. */
async function session(flow, { route = false, on = browser } = {}) {
  const browserContext = await on.createBrowserContext();
  try {
    const page = await browserContext.newPage();
    if (route) {
      await page.setRequestInterception(true);
      page.on("request", intercept);
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
const byStart = (spans) => [...spans].sort((a, b) => a.startTime[0] - b.startTime[0] || a.startTime[1] - b.startTime[1]);
const flowNames = (spans) => byStart(calls(spans)).map((s) => s.attributes[API_NAME]).filter((n) => !SESSION_CALLS.has(n));
/** The login flow's calls; page.click and page.waitForNavigation start together (Promise.all). */
const LOGIN_CALLS = ["locator.fill", "page.click", "page.goto", "page.type", "page.waitForNavigation"];

function assertNestedUnderOneRoot(spans) {
  const rootSpans = roots(spans);
  assert.equal(rootSpans.length, 1, rootSpans.map((s) => s.name).join());
  const [root] = rootSpans;
  const names = new Set(calls(spans).map((s) => s.attributes[API_NAME]));
  for (const name of ["page.goto", "locator.fill", "page.type", "page.click", "page.waitForNavigation"]) assert.ok(names.has(name), [...names].join());
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

test("baseline: calls nest under the active span, one tool span per outermost public call", { skip }, async () => {
  await withWorkflow({ name: "login" }, () => login());
  const spans = await exported();
  assertNestedUnderOneRoot(spans);
  // page.type -> frame.type -> elementHandle.type -> keyboard.type, page.click ->
  // ... -> mouse.click: only the outermost call is a span; our probes are none.
  assert.deepEqual(flowNames(spans).sort(), LOGIN_CALLS);
  for (const name of SESSION_CALLS) assert.equal(named(spans, name).length, 1, name);
  const [goto] = named(spans, "page.goto");
  assert.equal(input(goto).url, `${site.url}/index.html`);
  assert.deepEqual(JSON.parse(goto.attributes[OUTPUT]), { type: "HTTPResponse", url: `${site.url}/index.html`, status: 200 });
  assert.equal(goto.attributes[PAGE_TITLE], "Sign in");
  assert.equal(goto.attributes[PAGE_URL], `${site.url}/index.html`);
  assert.deepEqual(input(named(spans, "locator.fill")[0]), { locator: "#user", value: USERNAME });
});

test("baseline: the calls Puppeteer makes inside a call are spans only when called directly", { skip }, async () => {
  await withWorkflow({ name: "direct" }, () =>
    session(async (page) => {
      await page.goto(`${site.url}/index.html`);
      await page.mainFrame().click("#user");
      const user = await page.$("#user");
      await user.type("x1");
      await page.keyboard.press("Tab");
      await page.mouse.click(5, 5);
    }),
  );
  const spans = await exported();
  assert.deepEqual(flowNames(spans), ["page.goto", "frame.click", "page.$", "elementHandle.type", "keyboard.press", "mouse.click"]);
});

test("baseline: listeners, interception handlers and exposed functions keep the caller's parent and customer context", { skip }, async () => {
  // Puppeteer fires these from its connection, in the context the browser was
  // launched in (here: none). Without re-parenting, request.respond,
  // dialog.accept and the exposed function's page.title have no parent, so
  // they are not traced at all.
  await propagateAttributes({ customer_identifier: "customer-1" }, () =>
    withWorkflow({ name: "login" }, () =>
      session(
        async (page) => {
          page.on("dialog", (dialog) => dialog.accept());
          await page.exposeFunction("currentTitle", () => page.title());
          await runAgent(page, site.url);
          await page.evaluate(async () => {
            await window.currentTitle();
            alert("hi");
          });
        },
        { route: true },
      ),
    ),
  );
  const spans = await exported();
  const root = assertNestedUnderOneRoot(spans);
  for (const name of ["request.respond", "request.continue", "dialog.accept", "page.title"]) {
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

test("baseline: WebDriver BiDi pages are traced the same way", { skip }, async () => {
  const bidi = await puppeteer.launch(launchOptions({ protocol: "webDriverBiDi" }));
  try {
    await withWorkflow({ name: "login_bidi" }, () =>
      session(
        async (page) => {
          await runAgent(page, site.url);
          // Defined only on the BiDi classes (BidiKeyboard, BidiPage), not on the abstract ones.
          await page.keyboard.press("Tab");
          await page.reload();
        },
        { on: bidi },
      ),
    );
  } finally {
    await bidi.close();
  }
  const spans = await exported();
  assertNestedUnderOneRoot(spans);
  assert.deepEqual(flowNames(spans).sort(), [...LOGIN_CALLS, "keyboard.press", "page.reload"].sort());
  assert.equal(input(named(spans, "page.type")[0]).text, "<masked>");
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
  // Puppeteer has no run id of its own: its spans join the caller's thread and group.
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
    assert.equal(named(mine, "request.respond").length, 1, name);
    assert.equal(named(mine, "page.type").length, 1, name);
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
      await page.locator("#password").fill(LOCATOR_PASSWORD);
      const field = await page.$("#password");
      await field.type(HANDLE_PASSWORD);
      await page.focus("#password");
      await page.keyboard.type(KEYBOARD_PASSWORD);
      await page.focus("#user");
      await page.keyboard.type("plain-text-ok");
      await page.setContent('<select id="plan"><option>free</option><option>pro</option></select>');
      await page.locator("#plan").fill("pro");
    }),
  );
  const spans = await exported();
  assertNestedUnderOneRoot(spans);
  for (const span of spans) {
    for (const secret of [PASSWORD, LOCATOR_PASSWORD, HANDLE_PASSWORD, KEYBOARD_PASSWORD]) assert.ok(!dumped(span).includes(secret), `${span.name} ${secret}`);
  }
  assert.deepEqual(named(spans, "page.type").map((s) => input(s).text), ["<masked>"]);
  const fills = byStart(named(spans, "locator.fill")).map((s) => [input(s).locator, input(s).value]);
  assert.deepEqual(fills, [["#user", USERNAME], ["#password", "<masked>"], ["#plan", "pro"]]); // ordinary values are kept
  assert.deepEqual(named(spans, "elementHandle.type").map((s) => input(s).text), ["<masked>"]);
  assert.deepEqual(byStart(named(spans, "keyboard.type")).map((s) => input(s).text), ["<masked>", "plain-text-ok"]);
  assert.ok(spans.some((s) => (s.attributes[PAGE_URL] ?? "").includes("<secret>password</secret>")));
});

test("baseline: declared secrets never reach a span", { skip }, async () => {
  await withWorkflow({ name: "with_token" }, () =>
    withSecrets({ api_token: API_TOKEN }, () =>
      session(async (page) => {
        await page.goto(`${site.url}/index.html?token=${API_TOKEN}`);
        await page.evaluate((t) => t.length, API_TOKEN);
        await assert.rejects(page.locator(`#${API_TOKEN}`).setTimeout(300).click(), { name: "TimeoutError" });
        await assert.rejects(page.click(`#${API_TOKEN}`), new RegExp(API_TOKEN));
      }),
    ),
  );
  const spans = await exported();
  for (const span of spans) assert.ok(!dumped(span).includes(API_TOKEN), span.name);
  assert.ok(spans.some((s) => dumped(s).includes("<secret>api_token</secret>")));
});

test("baseline: credentials in headers, cookies and HTTP auth never reach a span", { skip }, async () => {
  await withWorkflow({ name: "credentials" }, () =>
    session(async (page) => {
      await page.setExtraHTTPHeaders({ authorization: HEADER_TOKEN });
      await page.authenticate({ username: "ada", password: OTHER_PASSWORD });
      await page.goto(`${site.url}/index.html`);
      await page.setCookie({ name: "sid", value: COOKIE_VALUE, url: site.url });
      await page.cookies();
    }),
  );
  const spans = await exported();
  for (const name of ["page.setExtraHTTPHeaders", "page.authenticate", "page.setCookie", "page.cookies"]) assert.ok(named(spans, name).length, name);
  for (const span of spans) {
    for (const secret of [HEADER_TOKEN, OTHER_PASSWORD, COOKIE_VALUE]) assert.ok(!dumped(span).includes(secret), `${span.name} ${secret}`);
  }
  assert.equal(input(named(spans, "page.setCookie")[0]).cookies[0].name, "sid"); // the rest is kept
});

test("baseline: a failed text entry is masked", { skip }, async () => {
  // A type that fails never reached a field we could inspect: fail closed.
  await withWorkflow({ name: "missing_field" }, () =>
    session(async (page) => {
      await page.goto(`${site.url}/index.html`);
      await assert.rejects(page.type("#no-such-field", OTHER_PASSWORD), /No element found/);
      await assert.rejects(page.locator("#no-such-field").setTimeout(300).fill(OTHER_PASSWORD), { name: "TimeoutError" });
    }),
  );
  const spans = await exported();
  const [type] = named(spans, "page.type");
  assert.equal(type.status.code, 2);
  assert.match(type.status.message, /^Error: No element found for selector: #no-such-field/);
  assert.equal(input(type).text, "<masked>");
  assert.equal(named(spans, "locator.fill")[0].status.code, 2);
  for (const span of spans) assert.ok(!dumped(span).includes(OTHER_PASSWORD), span.name);
});

test("baseline: screenshots are exported as compact JPEG image parts, at most 1280 px wide", { skip }, async () => {
  // A photo-like page: its PNG screenshot is large, so it is re-encoded.
  const noisy = `<canvas id=c width=1600 height=900></canvas><script>
    const g = document.getElementById("c").getContext("2d"), img = g.createImageData(1600, 900);
    for (let i = 0; i < img.data.length; i++) img.data[i] = i % 4 === 3 ? 255 : Math.random() * 256;
    g.putImageData(img, 0, 0);</script><style>body{margin:0}</style>`;
  await withWorkflow({ name: "shots" }, () =>
    session(async (page) => {
      await page.setViewport({ width: 1600, height: 900 });
      await page.setContent(noisy);
      await page.screenshot();
      await page.screenshot({ encoding: "base64" });
      await (await page.$("#c")).screenshot();
    }),
  );
  const spans = await exported();
  const shots = [...named(spans, "page.screenshot"), ...named(spans, "elementHandle.screenshot")];
  assert.equal(shots.length, 3);
  for (const span of shots) {
    const url = JSON.parse(span.attributes[OUTPUT]).screenshot.image_url.url;
    assert.match(url, /^data:image\/jpeg;base64,/, span.name);
    assert.equal(jpegWidth(Buffer.from(url.split(",")[1], "base64")), 1280, span.name);
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
  const quiet = new PuppeteerInstrumentor({ captureContent: false });
  await quiet.activate();
  try {
    await propagateAttributes({ customer_identifier: "customer-1", metadata: CUSTOMER_METADATA }, () =>
      withWorkflow({ name: "private" }, () =>
        session(async (page) => {
          await runAgent(page, site.url);
          await page.screenshot();
          await assert.rejects(page.locator("#secret-selector-xyz").setTimeout(300).click());
        }),
      ),
    );
  } finally {
    quiet.deactivate();
    instrumentor = new PuppeteerInstrumentor();
    await instrumentor.activate();
  }
  const spans = await exported();
  const puppeteerSpans = calls(spans);
  for (const name of ["page.goto", "page.screenshot", "locator.click", "page.type"]) assert.ok(named(puppeteerSpans, name).length, name);
  for (const span of puppeteerSpans) {
    assert.equal(span.attributes[CUSTOMER_ID], "customer-1", span.name);
    assert.equal(span.attributes["respan.metadata.agent_id"], "a1", span.name);
    for (const key of [INPUT, OUTPUT, PAGE_URL, PAGE_TITLE]) assert.equal(span.attributes[key], undefined, `${span.name} ${key}`);
    for (const secret of [site.url, USERNAME, PASSWORD, "secret-selector-xyz", "image_url"])
      assert.ok(!dumped(span).includes(secret), `${span.name} ${secret}`);
  }
  const failed = puppeteerSpans.filter((s) => s.status.code === 2);
  assert.deepEqual(failed.map((s) => s.status.message), ["TimeoutError"]);
});

test("baseline: errors carry `Type: message` and the span is marked failed", { skip }, async () => {
  await withWorkflow({ name: "missing" }, () =>
    session(async (page) => {
      await page.goto(`${site.url}/index.html`);
      await assert.rejects(page.locator("#missing").setTimeout(300).click());
      await assert.rejects(page.waitForSelector("#missing", { timeout: 300 }));
    }),
  );
  const spans = await exported();
  const [click] = named(spans, "locator.click");
  assert.equal(click.status.code, 2);
  assert.match(click.status.message, /^TimeoutError: Timed out after waiting 300ms/);
  const [wait] = named(spans, "page.waitForSelector");
  assert.match(wait.status.message, /^TimeoutError: Waiting for selector `#missing` failed/);
  assert.equal(wait.events[0].attributes["exception.type"], "TimeoutError");
});
