# @respan/instrumentation-playwright

Respan tracing for [Playwright](https://playwright.dev/) in JavaScript and
TypeScript (`playwright`, `playwright-core`, and `@playwright/test` used as a
library).

```bash
npm install @respan/respan @respan/instrumentation-playwright playwright
```

```ts
import { chromium } from "playwright";
import { Respan } from "@respan/respan";
import { withWorkflow } from "@respan/tracing";
import { PlaywrightInstrumentor } from "@respan/instrumentation-playwright";

const respan = new Respan({ instrumentations: [new PlaywrightInstrumentor()] }); // RESPAN_API_KEY
await respan.initialize();

await withWorkflow({ name: "check_docs" }, async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("https://playwright.dev");
  await page.getByRole("link", { name: "Get started" }).click();
  await browser.close();
});

await respan.flush(); // short scripts: send spans before the process exits
```

## What is traced

Playwright is what an agent, a test or a script drives, so it adds no root span
of its own. Every public API call that reaches the browser becomes one `tool`
span under the span that is active when it runs:

```
workflow / your agent span
├─ browserType.launch      tool
├─ page.goto               tool   url in; status, final URL and page title out
├─ openai.chat             chat   (your LLM call, from its own instrumentation)
├─ locator.fill            tool   selector + value in (password fields masked)
├─ locator.click           tool
└─ page.screenshot         tool   the image, as a compact JPEG
```

- **One span per public call**, named the way Playwright's own trace viewer
  names it (`page.goto`, `locator.click`, `expect.toHaveText`,
  `apiRequestContext.get`, ...). Local calls that never reach the browser
  (`page.locator()`, `page.getByRole()`) are not spans, and neither are the
  calls Playwright makes internally.
- **Only under an active span.** A call made with no active span (a plain
  script, a test suite without a Respan workflow) is not traced, so standalone
  runs do not turn into thousands of one-span traces. Wrap the code in
  `withWorkflow` / `withAgent`, or your agent framework's own span.
- **Callbacks nest where you registered them.** Playwright fires `page.on(...)`
  listeners, `page.route(...)` handlers, `exposeFunction` callbacks and locator
  handlers from its connection, in the context the browser was launched in.
  They run in the context they were registered in instead, so a
  `route.fulfill()` inside a route handler nests under your workflow and keeps
  your customer identifier and metadata, even when one browser is shared.
- **Page context.** Each call on a page records the page URL; navigations
  (`goto`, `reload`, `goBack`, `goForward`, `setContent`) also record the title.
- **Screenshots.** `page.screenshot()` / `locator.screenshot()` output is
  attached as an `image_url` part, re-encoded as JPEG (max 1280 px wide) when
  that is smaller. Respan stores it as log media. With `screenshotOn`, the
  instrumentation also takes a screenshot after navigations or actions, for a
  flip-book of the run.
- **Errors.** A failed call is an error span whose message is Playwright's own,
  e.g. `TimeoutError: locator.click: Timeout 30000ms exceeded.` A failed
  `expect(...)` matcher is an `AssertionError` span.
- **Passwords.** Text typed into a password field (`fill`, `type`,
  `pressSequentially`, `keyboard.type` / `insertText` into a focused password
  field) is exported as `<masked>`. Once seen, the same text is masked everywhere
  else in that trace (a GET form puts it in the next URL). If the
  instrumentation cannot tell where the text went, or the call failed, it masks
  it. `password` / `token` / `secret` option keys (e.g. `httpCredentials`) are
  masked too.
- **Your secrets.** Values passed to `withSecrets({...}, fn)` never reach a span:

  ```ts
  import { withSecrets } from "@respan/instrumentation-playwright";

  await withSecrets({ api_token: token }, () =>
    page.goto(`https://example.com/?token=${token}`), // exported as <secret>api_token</secret>
  );
  ```
- **Customer context.** `customer_identifier`, `metadata`, `thread_identifier`
  and the other fields set with `propagateAttributes(...)` reach every span, with
  metadata as flat `respan.metadata.<key>` attributes. Playwright has no session
  id of its own, so its spans join your trace and thread.
- **Playwright's own trace stays clean.** The extra calls the instrumentation
  makes (page title, `screenshotOn` screenshots, the password-field check) are
  internal calls, so they do not show up in a Playwright trace.zip either.

## Options

| Option | Default | Effect |
|---|---|---|
| `captureContent` | `true` | `false` keeps span names, timings, errors (type only) and customer context, and drops parameters, results, page URLs and titles, and screenshots. `RESPAN_TRACE_CONTENT=false` or `TRACELOOP_TRACE_CONTENT=false` do the same. |
| `captureScreenshots` | `true` | `false` drops all images; screenshot calls report only their byte size. |
| `screenshotOn` | `"never"` | `"navigation"`: a page screenshot after each navigation. `"action"`: also after each click, fill, check, press, ... Each one is an extra browser round trip. |
| `playwright` | `playwright-core` | The Playwright module to trace. `playwright` and `@playwright/test` load `playwright-core`; pass yours if your app resolves a different copy than this package does. |

Supported: `playwright-core` 1.55 to 1.64 (and the `playwright` /
`@playwright/test` releases built on them), Node.js 20+.

## Customer metadata with @respan/tracing 1.5 - 1.6

Those versions write propagated metadata as one JSON `respan.metadata`
attribute, which Respan ingestion does not read. While active, this package
expands it into flat `respan.metadata.<key>` attributes on every exported span
(through `@respan/tracing`'s span-transformer hook), so your metadata reaches
Respan on the workflow span as well as the Playwright spans.

## Not yet covered

- Session replay (an rrweb recording synced to the span timeline). See `docs/GAPS.md`.
- A per-browser `playwright.session` span. See `docs/GAPS.md`.
