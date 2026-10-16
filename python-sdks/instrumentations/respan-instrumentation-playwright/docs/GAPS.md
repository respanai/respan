# Playwright observability: coverage, parity with Laminar, known limitations

## What Laminar captures

Laminar's Playwright integration (Python and JS) wraps `BrowserType.launch` /
`connect` / `connect_over_cdp` / `launch_persistent_context` and
`Browser.new_context` / `new_page`. In Python its only job is to inject an rrweb
recorder into every page and ship the recording keyed to the active trace; it
emits **no per-call spans**. The JS SDK adds one `playwright.page` span per page.
Its value is a session replay linked to whatever span was active.

## What this package captures

| | Laminar (Python) | Respan |
|---|---|---|
| One span per Playwright API call (sync + async) | no | **yes**, at the same granularity as Playwright's trace viewer |
| Call arguments / return values / errors | no | yes (`capture_content`) |
| Page URL and title per call | no | yes |
| Screenshots | via replay | `page.screenshot()` output inline; optional `screenshot_on` flip-book |
| Password / declared-secret masking | rrweb input masking | typed passwords masked, also where they reappear; `secrets({...})` |
| Callback calls nested in the right trace (sync greenlets) | n/a | yes |
| Customer id / metadata / thread on every span | - | yes |
| Session replay (rrweb) | **yes** | no, see below |

## Seams

- `playwright.async_api._generated.<Class>.<method>` for every public
  `async def` method: exactly the calls that round-trip to the browser (417 in
  Playwright 1.64).
- `playwright.sync_api._generated.<Class>.<method>` for the same names.
- `AsyncBase._wrap_handler` / `SyncBase._wrap_handler`: every user callback.

These are Playwright's generated public classes and two private helpers.
The pin is `playwright >=1.50,<1.65` (tested on 1.50.0 and 1.64.0); widen it
after testing each minor.

## Known limitations

| # | Limitation | Impact | Next step |
|---|---|---|---|
| L1 | **Session replay** (rrweb recording synced to the span timeline) is not implemented. This is a known gap across every Respan browser integration; it needs SDK, ingestion and UI support together. | no sub-call replay (scrolls, hovers, typing) | interim: `screenshot_on="action"`, and Playwright's own `context.tracing` trace.zip or `record_video_dir` alongside |
| L2 | No `playwright.session` span from launch to close. Calls nest under the active span instead, so an LLM call and the browser calls it caused stay in order in the trace. | the browser's lifetime is not one span | add an opt-in session span if customers ask |
| L3 | Calls made with no active span are not traced (deliberate, see README). | a bare script shows nothing | wrap it in `@workflow` / `@agent` |
| L4 | Callbacks run in the context they were **registered** in. A route handler registered inside one workflow and fired during another (a shared page) nests under the first. | rare mis-parenting on shared pages | none planned |
| L5 | Password detection checks `type="password"` on the target after the call (an internal, non-waiting query). Text typed into a non-password field that is still sensitive (an API key in a text box) is exported unless declared with `secrets({...})` or `capture_content=False`. | sensitive text in ordinary fields | declare it with `secrets(...)` |
| L6 | `screenshot_on="navigation"/"action"` costs one screenshot round trip per matching call, and the page-title lookup one round trip per navigation. | slower runs with screenshots on | default is `"never"` |
| L7 | Return values are capped at 4000 characters per string (`page.content()` is truncated). | long HTML is cut | - |
| L8 | Playwright's own trace viewer data (trace.zip) and videos are not uploaded to Respan. | two places to look | link them when replay lands |
| L9 | The Playwright MCP server (`@playwright/mcp`, Node) is not instrumented by this package; its calls happen in a Node process. | MCP-driven browsers show only the MCP tool calls | the JS instrumentation |

## Baseline

`tests/test_baseline.py` is the wire contract. It drives a real headless Chrome
through the sync and async APIs against a local sign-in page, inside a Respan
workflow (and a customer agent span, and a worker thread), and checks every
exported span for:
- parentage: every call nests under the active span, in both APIs;
- exportability: every call span has a log type and a span kind (Respan drops
  spans without one);
- the sync API's greenlet hop: a `page.route` handler's `Route.fulfill` nests under
  the workflow and keeps the propagated customer identifier;
- no spans without an active parent span;
- customer metadata as flat `respan.metadata.<key>`, `customer_identifier`,
  thread and trace group on every span;
- parallel sessions: separate traces, no shared identifiers;
- password text: masked in the call, and in the URL the form submits to;
- declared secrets: masked in URLs, `evaluate` arguments and error messages;
- a failed text entry is masked (fail closed);
- content off: identifiers kept; arguments, URLs, titles, screenshots and error
  messages dropped.

Each mechanism was disabled in turn and the suite rerun; every one makes at least
one baseline test fail:

| Disabled | Fails |
|---|---|
| callback context carry-over | sync route handler test |
| callbacks use the caller's context (not the finished call span's) | both route handler tests, metadata and thread tests |
| span kind on call spans | 11 of 14 |
| "only under an active span" | no-parent test |
| password-field check | password test |
| learning typed passwords for later masking | password test |
| fail-closed masking on failed calls | failed-text-entry test |
| `capture_content` | content-off test |
| `secrets(...)` | declared-secrets test |

The parallel-sessions test passes either way (the async API keeps context per
task on its own); it stays as a regression guard.

## Verification log

- `pytest tests` (23 tests: 14 baseline + 9 unit), real headless Chrome, on
  Playwright 1.64.0 and 1.50.0.
- Live (2026-10-15): `examples/playwright/run_all.py`, 14 scenarios mirroring
  Playwright for Python's docs, verified live against a Respan environment by
  reading every trace back from the logs API: **14/14 PASS**. Also with
  `--screenshot-on action` (screenshots stored on action spans) and
  `--no-content` (no URL, typed text or title in any stored row).
- LLM calls made through the Respan gateway are logged by the gateway; the
  OpenAI instrumentation's span links to that row instead of repeating its usage,
  so the live check reads tokens and cost from the linked row.
