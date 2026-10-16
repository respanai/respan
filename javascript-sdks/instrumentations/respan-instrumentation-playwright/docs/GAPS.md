# Playwright (JS) observability: coverage, parity with Laminar, known limitations

## What Laminar captures

Laminar's Playwright integration (`instrumentModules: { playwright: { chromium } }`)
wraps `BrowserType.launch` / `connect` / `connectOverCDP` /
`launchPersistentContext` and `Browser.newContext` / `newPage`. Its main job is
to inject an rrweb recorder into every page and ship the recording keyed to the
active trace; the JS SDK also adds one `playwright.page` span per page. It emits
**no per-call spans**. Its value is a session replay linked to whatever span was
active.

## What this package captures

| | Laminar (JS) | Respan |
|---|---|---|
| One span per Playwright API call | no (one span per page) | **yes**, at the granularity of Playwright's own trace viewer |
| Call parameters / results / errors | no | yes (`captureContent`) |
| `expect(...)` matchers, failures as error spans | no | yes |
| Page URL and title per call | no | yes |
| Screenshots | via replay | `page.screenshot()` / `locator.screenshot()` output inline as compact JPEG; optional `screenshotOn` flip-book |
| Password / declared-secret masking | rrweb input masking | typed passwords masked, also where they reappear; `withSecrets({...})`; credential option keys |
| Callbacks (route handlers, listeners, exposed functions, locator handlers) nested in the registering trace | n/a | yes |
| Customer id / metadata (flat) / thread on every span | - | yes |
| Session replay (rrweb) | **yes** | no, see L1 |

## Seams

- `playwright._instrumentation.addListener({ onApiCallBegin, onApiCallEnd })`:
  the client instrumentation hook `@playwright/test` uses for its steps. One
  begin/end per public call, named by Playwright (`page.goto`), internal calls
  excluded.
- `Connection.prototype.sendMessageToServer`: the call's object, protocol
  parameters and result; post-call probes run here, before the call returns.
- `Connection.prototype.dispatch`: callback events (`route`, `bindingCall`,
  `locatorHandlerTriggered`, `webSocketRoute`) run in the context their
  registration ran in.
- `on` / `once` / `addListener` / `prependListener` / `off` / `removeListener`
  on Playwright's emitter base classes: listeners run in the context they were
  registered in.

All four are private to playwright-core. The pin is `playwright-core
>=1.55,<1.65` (tested on 1.55.0 and 1.64.1); widen it after testing each minor.
Before 1.55, Playwright ran route handlers inside the (finished) API zone of the
`page.route()` call that registered them, so their calls never reached any
instrumentation listener, Playwright's own included.

## Known limitations

| # | Limitation | Impact | Next step |
|---|---|---|---|
| L1 | **Session replay** (rrweb recording synced to the span timeline) is not implemented. This is a known gap across every Respan browser integration; it needs SDK, ingestion and UI support together. | no sub-call replay (scrolls, hovers, typing) | interim: `screenshotOn: "action"`, and Playwright's own `context.tracing` trace.zip or `recordVideo` alongside |
| L2 | No `playwright.session` span from launch to close. Calls nest under the active span instead, so an LLM call and the browser calls it caused stay in order in the trace. | the browser's lifetime is not one span | add an opt-in session span if customers ask |
| L3 | Calls made with no active span are not traced (deliberate, see README). | a bare script or test shows nothing | wrap it in `withWorkflow` / `withAgent` |
| L4 | Listeners run in the context they were registered in (first registration of a function wins); route / exposed-function / locator handlers in the context the page or browser context **last** registered one in. A handler registered in one workflow and fired during another (a shared page) nests under the first. Objects created before activation (a browser launched before `respan.initialize()`) keep their listeners unbound. | rare mis-parenting on shared pages | activate Respan before launching browsers |
| L5 | Password detection checks `type="password"` on the target after the call (an internal, non-waiting query). Typed text shorter than 4 characters is masked in its own call but not learned for masking elsewhere. Text typed into a non-password field that is still sensitive (an API key in a text box) is exported unless declared with `withSecrets({...})` or `captureContent: false`. | sensitive text in ordinary fields | declare it with `withSecrets(...)` |
| L6 | Calls Playwright marks internal are not spans: `download.saveAs`, `video.saveAs`, `apiResponse.body()` / `json()` / `text()`, `apiRequestContext.dispose()`. Playwright's trace viewer does not show them either. The Python package traces them (it wraps the public classes). | no span for saving a download or reading an API body | wrap them in `withTool` if needed |
| L7 | The page title is recorded for navigations that make a browser call (`goto`, `reload`, `goBack`, `goForward`, `setContent`). `waitForURL` waits on the client side, so it records the URL only. | no title on `waitForURL` spans | - |
| L8 | `screenshotOn: "navigation" / "action"` costs one screenshot round trip per matching call, and the title lookup one round trip per navigation. Screenshots are re-encoded with playwright-core's bundled codecs; if those are unavailable, PNGs are kept (or summarised above 2 MB of base64). | slower runs with screenshots on | default is `"never"` |
| L9 | Return values are capped at 4000 characters per string (`page.content()` is truncated). | long HTML is cut | - |
| L10 | Playwright's own trace viewer data (trace.zip) and videos are not uploaded to Respan. | two places to look | link them when replay lands |
| L11 | The Playwright MCP server (`@playwright/mcp`) runs Playwright in its own process; it is traced only if this instrumentation is activated in that process. | MCP-driven browsers show only the MCP tool calls | an MCP-server integration |
| L12 | `@respan/tracing` 1.5 / 1.6 write propagated metadata as one JSON attribute that ingestion drops. This package registers a span transformer that flattens it on every exported span while active (and flattens its own spans if the transformer cannot register). The same workaround lives in `@respan/instrumentation-stagehand`. | none while active | remove both once `@respan/tracing` writes flat keys |

## Baseline

`tests/playwright_baseline.test.mjs` is the wire contract (CI runs it with
`npm test`). It drives a real headless Chrome through playwright-core against a
local sign-in page, through Respan's real span processor and exporter, with one
browser launched outside any span and shared by every test (so callbacks are
re-parented, not merely inherited from the launch). It checks every exported
span for:
- parentage and exportability: every call nests under the active span (a Respan
  workflow, or a customer's own agent span) with a log type and span kind; local
  calls and the instrumentation's own probes are not spans;
- callbacks: a route handler's `route.fulfill` and a `page.on("dialog")`
  listener's `dialog.accept` nest under the workflow and keep the customer id;
- no spans without an active parent span;
- customer metadata as flat `respan.metadata.<key>` (never a JSON blob) and the
  customer identifier, thread and trace group on every span, the workflow root
  included;
- parallel sessions on one browser: separate traces, no shared identifiers;
- password text: masked in `fill` and in `keyboard.type` into a focused password
  field, and in the URL the form submits to; ordinary text kept;
- declared secrets: masked in URLs, `evaluate` arguments and error messages;
- a failed text entry is masked (fail closed);
- screenshots: `page.screenshot` and `locator.screenshot` exported as JPEG at
  most 1280 px wide;
- content off: identifiers kept; parameters, results, URLs, titles, screenshots
  and error messages dropped;
- errors carry `Type: message`; `expect` matchers are spans and a mismatch is an
  `AssertionError` span.

Each mechanism was disabled in turn and the suite rerun; every one makes at
least one baseline test fail:

| Disabled | Fails |
|---|---|
| callback re-parenting at dispatch (route handlers) | callbacks test, parallel sessions test |
| listener binding (`page.on`) | callbacks test |
| "only under an active span" | nesting test, no-parent test |
| span kind on call spans | 6 of 14 |
| password-field check | password test |
| focused-field check (`keyboard.type`) | password test |
| learning typed passwords for later masking | password test |
| fail-closed masking on failed calls | failed-text-entry test |
| `captureContent` | content-off test |
| `withSecrets` | declared-secrets test |
| metadata flattening transformer | metadata test |
| JPEG compaction | screenshot test |
| `locator.screenshot` image adoption | screenshot test |
| describing Playwright objects in results | nesting test |
| `expect` mismatch detection | assertions test |

The parallel-sessions test also guards trace isolation (it passes with the
mechanisms above that do not concern it). Flattening the instrumentation's own
spans is a fallback for when the transformer cannot register; with it the
transformer covers every span, so disabling only the fallback fails nothing.

## Verification log

- `npm test` (14 baseline tests), real headless Chrome, on playwright-core
  1.64.1 and 1.55.0.
- Live (2026-10-16): `javascript-sdks/examples/playwright`, 14 scenarios
  mirroring Playwright's docs plus an LLM tool-calling agent through the Respan
  gateway, verified against a Respan environment by reading every trace back
  from the logs API (`verify_live.py`, built on `respan_tracing.testing.live`):
  **14/14 PASS** with defaults, **14/14** with `--screenshot-on action`
  (navigations carry a stored screenshot), **14/14** with `--no-content` (no page
  URL, typed text, title or image in any Playwright row).
