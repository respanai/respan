# @respan/instrumentation-puppeteer

Respan tracing for [Puppeteer](https://pptr.dev/) in JavaScript and TypeScript
(`puppeteer` and `puppeteer-core`).

```bash
npm install @respan/respan @respan/instrumentation-puppeteer puppeteer
```

```ts
import puppeteer from "puppeteer";
import { Respan } from "@respan/respan";
import { withWorkflow } from "@respan/tracing";
import { PuppeteerInstrumentor } from "@respan/instrumentation-puppeteer";

const respan = new Respan({ instrumentations: [new PuppeteerInstrumentor()] }); // RESPAN_API_KEY
await respan.initialize();

await withWorkflow({ name: "check_docs" }, async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.goto("https://pptr.dev");
  await page.locator("::-p-text(Get started)").click();
  await browser.close();
});

await respan.flush(); // short scripts: send spans before the process exits
```

## What is traced

Puppeteer is what an agent, a test or a script drives, so it adds no root span
of its own. Every public call (from a curated list, below) becomes one `tool`
span under the span that is active when it runs:

```
workflow / your agent span
├─ puppeteer.launch        tool
├─ browser.newPage         tool
├─ page.goto               tool   url in; status, final URL and page title out
├─ openai.chat             chat   (your LLM call, from its own instrumentation)
├─ locator.fill            tool   the locator's selector + value in (password fields masked)
├─ page.click              tool
└─ page.screenshot         tool   the image, as a compact JPEG
```

- **One span per outermost call.** Puppeteer's public methods call each other
  (`page.type` calls `frame.type`, which calls `elementHandle.type`, which
  calls `keyboard.type`). Only the call your code made is a span; the calls it
  makes inside are not. Call `frame.click` or `keyboard.press` yourself and
  those are spans.
- **What is a call.** `Page`, `Frame`, `ElementHandle`, `Locator`, `Keyboard`,
  `Mouse`, `HTTPRequest` (`continue` / `respond` / `abort`), `Dialog`,
  `Browser`, `BrowserContext` and screen recordings: navigation, waiting,
  evaluating, querying, clicking / typing / selecting, screenshots and PDFs,
  emulation, cookies and headers, `exposeFunction`, and `puppeteer.launch` /
  `puppeteer.connect`. The full list is `TRACED_METHODS` in `src/_constants.ts`.
  Local, synchronous methods (`page.url()`, `page.locator()`) are not spans.
  Both protocols are covered: CDP and WebDriver BiDi.
- **Only under an active span.** A call made with no active span (a plain
  script, a test suite without a Respan workflow) is not traced, so standalone
  runs do not turn into thousands of one-span traces. Wrap the code in
  `withWorkflow` / `withAgent`, or your agent framework's own span.
- **Callbacks nest where you registered them.** Puppeteer fires `page.on(...)`
  listeners (including request-interception handlers) and `exposeFunction`
  callbacks from its connection, in the context the browser was launched in.
  They run in the context they were registered in instead, so a
  `request.respond()` inside a `page.on("request")` handler nests under your
  workflow and keeps your customer identifier and metadata, even when one
  browser is shared.
- **Page context.** Each call on a page records the page URL; navigations
  (`goto`, `reload`, `goBack`, `goForward`, `setContent`, `waitForNavigation`)
  also record the title.
- **Screenshots.** `page.screenshot()` / `elementHandle.screenshot()` output
  (bytes or `encoding: "base64"`) is attached as an `image_url` part,
  re-encoded as JPEG (max 1280 px wide) when that is smaller. Respan stores it
  as log media. With `screenshotOn`, the instrumentation also takes a screenshot
  after navigations or actions, for a flip-book of the run.
- **Recordings.** `page.record({ path })` and `page.screencast({ path })` record
  the video's path as `puppeteer.recording.path` on their span.
- **Errors.** A failed call is an error span whose message is Puppeteer's own,
  e.g. `TimeoutError: Waiting for selector \`#missing\` failed`.
- **Passwords.** Text typed into a password field (`page.type`, `frame.type`,
  `elementHandle.type`, `locator.fill`, `keyboard.type` into a focused password
  field) is exported as `<masked>`. Once seen, the same text is masked
  everywhere else in that trace (a GET form puts it in the next URL). If the
  instrumentation cannot tell where the text went, or the call failed, it masks
  it. `password` / `token` / `secret` / `authorization` / `cookie` keys (e.g.
  `page.authenticate`, `setExtraHTTPHeaders`) and cookie values are masked too.
- **Your secrets.** Values passed to `withSecrets({...}, fn)` never reach a span:

  ```ts
  import { withSecrets } from "@respan/instrumentation-puppeteer";

  await withSecrets({ api_token: token }, () =>
    page.goto(`https://example.com/?token=${token}`), // exported as <secret>api_token</secret>
  );
  ```
- **Customer context.** `customer_identifier`, `metadata`, `thread_identifier`
  and the other fields set with `propagateAttributes(...)` reach every span, with
  metadata as flat `respan.metadata.<key>` attributes. Puppeteer has no session
  id of its own, so its spans join your trace and thread.

## Options

| Option | Default | Effect |
|---|---|---|
| `captureContent` | `true` | `false` keeps span names, timings, errors (type only) and customer context, and drops arguments, results, page URLs and titles, and screenshots. `RESPAN_TRACE_CONTENT=false` or `TRACELOOP_TRACE_CONTENT=false` do the same. |
| `captureScreenshots` | `true` | `false` drops all images; screenshot calls report only their byte size. |
| `screenshotOn` | `"never"` | `"navigation"`: a page screenshot after each navigation. `"action"`: also after each click, type, fill, press, select, ... Each one is an extra browser round trip. |
| `puppeteer` | `puppeteer-core` | The Puppeteer module to trace. `puppeteer` loads `puppeteer-core`; pass yours if your app resolves a different copy than this package does. |

Supported: `puppeteer-core` 25 (and the `puppeteer` 25 releases built on it),
Node.js 22.12+.

## Customer metadata with @respan/tracing 1.5 - 1.6

Those versions write propagated metadata as one JSON `respan.metadata`
attribute, which Respan ingestion does not read. While active, this package
expands it into flat `respan.metadata.<key>` attributes on every exported span
(through `@respan/tracing`'s span-transformer hook), so your metadata reaches
Respan on the workflow span as well as the Puppeteer spans.

## Not yet covered

- Session replay (an rrweb recording synced to the span timeline). See `docs/GAPS.md`.
- A per-browser `puppeteer.session` span. See `docs/GAPS.md`.
