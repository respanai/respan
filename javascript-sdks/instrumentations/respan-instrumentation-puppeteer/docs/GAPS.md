# Puppeteer observability: coverage, parity with Laminar, known limitations

## What Laminar captures

Laminar's Puppeteer integration (`instrumentModules: { puppeteer }`, JS/TS
only, `puppeteer` / `puppeteer-core` >= 19) patches `launch` / `connect` (on the
module, `PuppeteerNode.prototype`, `Puppeteer.prototype` and `default`) and
`browser.createBrowserContext`. Each page gets one `puppeteer.page` span and an
injected rrweb recorder; the recording is shipped keyed to the active trace. It
emits **no per-call spans**. Its value is a session replay linked to whatever
span was active.

## What this package captures

| | Laminar (JS) | Respan |
|---|---|---|
| One span per Puppeteer call | no (one span per page) | **yes**, the outermost public call only |
| Call arguments / results / errors | no | yes (`captureContent`) |
| Page URL and title per call | no | yes |
| Screenshots | via replay | `page.screenshot()` / `elementHandle.screenshot()` output inline as compact JPEG; optional `screenshotOn` flip-book |
| `page.record()` / `page.screencast()` | no | the video's path on the span (`puppeteer.recording.path`) |
| Password / declared-secret masking | rrweb input masking | typed passwords masked, also where they reappear; `withSecrets({...})`; credential keys, auth headers, cookie values |
| Callbacks (listeners, request interception, exposed functions) nested in the registering trace | n/a | yes |
| CDP and WebDriver BiDi | yes | yes |
| Customer id / metadata (flat) / thread on every span | - | yes |
| Session replay (rrweb) | **yes** | no, see G2 |

## Seams

- A curated list of public methods (`TRACED_METHODS` in `src/_constants.ts`) on
  Puppeteer's abstract `api/` classes (`Page`, `Frame`, `ElementHandle`,
  `Locator`, `Keyboard`, `Mouse`, `HTTPRequest`, `Dialog`, `Browser`,
  `BrowserContext`, `ScreenRecording`) and on their CDP (`CdpPage`, ...) and
  WebDriver BiDi (`BidiPage`, ...) implementations, wherever a class defines
  the method itself (`goto` is on `Page` but also on `CdpFrame` / `BidiFrame`;
  `keyboard.type` exists only on `CdpKeyboard` / `BidiKeyboard`). The BiDi
  classes are imported from `puppeteer-core/internal/bidi/bidi.js` at
  activation (Puppeteer loads them lazily).
- Puppeteer has no single API-call boundary like Playwright's API zone, so a
  reentrancy guard in the OpenTelemetry context makes only the outermost call a
  span: while a traced call runs, every wrapped method it reaches runs untraced.
- `BrowserLauncher.prototype.launch` (and `ChromeLauncher` / `FirefoxLauncher`)
  and `Puppeteer.prototype.connect`: the browser's start. `puppeteer.launch` is
  an instance-bound function, so the launcher is the patchable boundary.
- `EventEmitter.prototype.on` / `off` and `Page.prototype.on` / `off` (Page wraps
  `request` listeners in an interception queue that runs them outside the
  emitter): listeners run in the context they were registered in.
  `exposeFunction`'s callback runs in the caller's context.
- `Page.locator` / `Frame.locator` / `ElementHandle.asLocator` and
  `Locator.copyOptions`: what a locator was made from, shown on its spans.

All of these are public classes, but the CDP/BiDi implementation classes and
the BiDi entry point are internal. The pin is `puppeteer-core >=25.0.2 <26`
(tested on 25.0.2 and 25.14.0); widen it after testing each major.

## Open gaps

| # | Gap | Impact | Next step |
|---|---|---|---|
| G1 | **Shared helpers are copied, not shared.** `src/_shared.ts` is the third copy of the browser-instrumentation helpers (after `@respan/instrumentation-playwright` and, in part, `@respan/instrumentation-stagehand`): `safely`, `flattenMetadata`, `secretReplacements`, `mask`, `maskSpan`, `setError`, `recordError`, `isImage`, `compactImage` (with `downscale`), `imagePart`; and, in `src/index.ts`, the same `withSecrets` / learned-password / listener-binding / `withDeadline` / content-policy patterns as the Playwright package. | three places to fix a masking or flattening bug | **move to @respan/tracing** (one browser-capture module the three import). Blocked: the published npm `@respan/tracing` (1.6.x) and its source in this repo (1.1.x) have diverged, so a new export cannot be released yet. Lead decision. |
| G2 | **Session replay** (rrweb recording synced to the span timeline) is not implemented. This is a known gap across every Respan browser integration; it needs SDK, ingestion and UI support together. | no sub-call replay (scrolls, hovers, typing) | interim: `screenshotOn: "action"`, and `page.record({ path })` alongside (its path is on the span) |
| G3 | No `puppeteer.session` span from launch to close (the spec's design). Calls nest under the active span instead, so an LLM call and the browser calls it caused stay in order in the trace; `puppeteer.launch` and `browser.close` are spans of their own. Same choice as the Playwright JS package. | the browser's lifetime is not one span | add an opt-in session span if customers ask |
| G4 | Calls made with no active span are not traced (deliberate, see README). | a bare script or test shows nothing | wrap it in `withWorkflow` / `withAgent` |
| G5 | Listeners run in the context they were registered in, per emitter (the first registration of a function on one page wins; the same function on two pages binds to each page's own registration). Objects created before activation keep their listeners unbound. Other callbacks Puppeteer calls from its connection (`waitForRequest` / `waitForResponse` predicates, `waitForTarget` predicates) are not re-parented; Puppeteer calls made inside them are traced only if a span is active there. | rare mis-parenting | activate Respan before launching browsers |
| G6 | Password detection checks `type="password"` on the element being typed into, before typing (`elementHandle.focus` / `type` / `press` / `select` inside the call), or on the focused element for `keyboard.type` / `sendCharacter`. `locator.fill` whose value is already in the field types nothing, so the check never runs and its text is masked (fail closed). Typed text shorter than 4 characters is masked in its own call but not learned for masking elsewhere. Text typed into an ordinary field that is still sensitive is exported unless declared with `withSecrets(...)` or `captureContent: false`. `keyboard.type` on a page the instrumentation has not seen a call on is masked (it cannot find the focused field). | over-masking in edge cases; sensitive text in ordinary fields | declare it with `withSecrets(...)` |
| G7 | Only the curated method list is traced. Not traced: `page.waitForRequest` / `waitForResponse` / `waitForFrame`, `Target.page()`, `CDPSession.send` (raw CDP; too noisy, the spec suggests opt-in), `page.coverage` / `page.tracing`, `page.accessibility.snapshot()`, `Touchscreen`, `FileChooser.accept`. | no span for those calls | extend `TRACED_METHODS` |
| G8 | `puppeteer.launch` is traced through the launcher (`ChromeLauncher.launch`), so it is named `puppeteer.launch` whichever entry point was used. `connect` is traced through `Puppeteer.prototype.connect`, which every `connect` reaches. | none known | - |
| G9 | The page title is recorded for navigations on `page` / `frame` (`goto`, `reload`, `goBack`, `goForward`, `setContent`, `waitForNavigation`). Clicks that navigate record the URL only. | no title on a navigating click | - |
| G10 | `screenshotOn: "navigation" / "action"` costs one screenshot round trip per matching call, and the title lookup one round trip per navigation. Screenshots are re-encoded with `pngjs` + `jpeg-js` (dependencies of this package); a PNG over 2 MB of base64 after compaction is summarised. | slower runs with screenshots on | default is `"never"` |
| G11 | Return values are capped at 4000 characters per string (`page.content()` is truncated). Page functions are exported as their source text (also capped). | long HTML is cut | - |
| G12 | The video of `page.record()` / `page.screencast()` is not uploaded to Respan; only its local path is recorded. `page.screencast()` needs ffmpeg (not installed on the dev laptop, so only `page.record()` was run live). | two places to look | link them when replay lands |
| G13 | `@respan/tracing` 1.5 / 1.6 write propagated metadata as one JSON attribute that ingestion drops. This package registers a span transformer that flattens it on every exported span while active (and flattens its own spans if the transformer cannot register). The same process-wide workaround lives in the Playwright and Stagehand packages. | none while active | remove all three once `@respan/tracing` writes flat keys |
| G14 | A call that returns synchronously is not given post-call probes (page URL, title, screenshot); every call in the list returns a promise in Puppeteer 25. | none today | - |

## Baseline

`tests/puppeteer_baseline.test.mjs` is the wire contract (CI runs it with
`npm test`). It drives a real headless Chrome (the runner's `google-chrome`,
found the same way as the Playwright JS baseline) through puppeteer-core
against a local sign-in page, through Respan's real span processor and
exporter, with one browser launched outside any span and shared by every test
(so callbacks are re-parented, not merely inherited from the launch). It checks
every exported span for:
- parentage and exportability: every call nests under the active span (a Respan
  workflow, or a customer's own agent span) with a log type and span kind;
- reentrancy: `page.type` / `page.click` / `locator.fill` produce one span each,
  not the `frame` / `elementHandle` / `keyboard` / `mouse` calls inside them;
  those are spans when called directly;
- callbacks: a request-interception handler's `request.respond` /
  `request.continue`, a `page.on("dialog")` listener's `dialog.accept` and an
  exposed function's `page.title` nest under the workflow and keep the customer
  id;
- no spans without an active parent span;
- WebDriver BiDi: the same flow over `protocol: "webDriverBiDi"`, including
  methods only the BiDi classes define (`keyboard.press`, `page.reload`);
- customer metadata as flat `respan.metadata.<key>` (never a JSON blob) and the
  customer identifier, thread and trace group on every span, the workflow root
  included;
- parallel sessions on one browser (sharing one listener function): separate
  traces, no shared identifiers;
- password text: masked in `page.type`, `locator.fill`, `elementHandle.type`
  and `keyboard.type` into a focused password field, and in the URL the form
  submits to; ordinary text kept;
- declared secrets: masked in URLs, `evaluate` arguments and error messages;
- credentials: `authorization` headers, `page.authenticate` passwords and
  cookie values never exported;
- a failed text entry is masked (fail closed);
- screenshots: `page.screenshot` (bytes and base64) and
  `elementHandle.screenshot` exported as JPEG at most 1280 px wide;
- content off: identifiers kept; arguments, results, URLs, titles, screenshots
  and error messages dropped;
- errors carry `Type: message`.

Each mechanism was disabled in turn (in the compiled `dist/`) and the suite
rerun; every one makes at least one baseline test fail:

| Disabled | Fails |
|---|---|
| reentrancy guard (inner calls become spans) | 10 of 16 |
| "only under an active span" | nesting test, no-parent test, BiDi test |
| span kind on call spans | 7 of 16 |
| listener binding (`EventEmitter.on`) | callbacks test, parallel sessions test |
| `Page.on` binding (interception handlers) | callbacks test, parallel sessions test |
| per-emitter bindings (first registration anywhere wins) | no-parent, metadata, thread/group, parallel sessions tests |
| `exposeFunction` callback binding | callbacks test |
| element password check | password test, BiDi test |
| focused-field check (`keyboard.type`) | password test |
| learning typed passwords for later masking | password test |
| fail-closed masking | failed-text-entry test |
| `captureContent` | content-off test |
| `withSecrets` | declared-secrets test |
| credential keys (headers, HTTP auth) | credentials test |
| cookie value masking | credentials test |
| metadata flattening transformer | metadata test |
| JPEG compaction | screenshot test |
| base64 screenshot decoding | screenshot test |
| BiDi implementation classes | BiDi test |
| locator descriptions | nesting test, password test |
| describing Puppeteer objects in results | nesting test |

## Verification log

- `npm test` (16 baseline tests), real headless Chrome 155, on puppeteer-core
  25.14.0 and 25.0.2.
- Live (2026-10-16): `javascript-sdks/examples/puppeteer`, 15 scenarios
  mirroring pptr.dev's guides plus an LLM tool-calling agent through the Respan
  gateway and a WebDriver BiDi run, against Respan staging (customer
  `puppeteer-scenarios`), verified by reading every trace back from the logs API
  (`verify_live.py`, built on `respan_tracing.testing.live`): **15/15 PASS**
  with defaults, **15/15** with `--screenshot-on action`, **15/15** with
  `--no-content` (no page URL, typed text, title or image in any Puppeteer row).
