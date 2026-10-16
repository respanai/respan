# Playwright + Respan (JavaScript / TypeScript)

Trace [Playwright](https://playwright.dev/) in Respan with
`@respan/instrumentation-playwright`. Every Playwright call your code makes
(`page.goto`, `locator.click`, `expect(...).toHaveText`, ...) becomes a span
under your workflow or agent span, with the page URL and title, screenshots,
and Playwright's own error messages. Typed passwords are masked, and callbacks
(route handlers, `page.on` listeners, exposed functions) nest under the
workflow that registered them.

## Setup

```bash
cd javascript-sdks/examples/playwright
npm install                 # Node.js 20+
cp .env.example .env        # then set RESPAN_API_KEY
```

The scenarios drive the installed Google Chrome (`channel: "chrome"`). Without
Chrome, run `npx playwright install chromium` and set `PLAYWRIGHT_CHANNEL=`
(empty) in `.env` to use Playwright's own Chromium.

`.npmrc` sets `install-links=true`, so the instrumentation is installed from this
repo as a real package. After changing it, rebuild it (`npx tsc` in its
directory) and run `npm install` here again. Outside this repo:
`npm install @respan/instrumentation-playwright`.

## Run

```bash
npx tsx run_all.ts --list                       # all scenarios
npx tsx run_all.ts                              # run them all
npx tsx run_all.ts forms llm_agent --headed     # some of them, with the browser visible
npx tsx run_all.ts --screenshot-on action       # a page screenshot after every action
npx tsx run_all.ts --no-content                 # captureContent: false
```

`run_all.ts` writes `runs/<run id>.json`. Then check every trace in Respan:

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python verify_live.py runs/<run id>.json
```

Every scenario runs inside a Respan workflow span under customer
`playwright-js-scenarios`, with `metadata.scenario=<name>` and
`metadata.run_id=<run>`. The verifier reads each trace back from the Respan logs
API and prints PASS/FAIL per scenario with its trace ids and a link to its root
span, and writes the same report to `runs/<run id>.md`.

Every trace must have: every span the SDK exported, and exactly one root; the
customer id and both metadata keys on every span; one thread id; every
Playwright call the SDK traced stored as a `tool` span, call for call; tokens
and cost on every LLM call; and error spans only where the scenario expects
them. Scenarios add their own checks (which calls must appear, which must fail,
which carry a screenshot, which strings must never appear). With
`--no-content`, no Playwright row may contain a page URL, typed text, title or
image; with `--screenshot-on action`, navigations carry a screenshot.

Most scenarios use a small local site served from `site/`, so they need no
accounts and give the same result every time. Screenshots, downloads, the saved
storage state and the trace.zip land in `artifacts/`.

| # | Scenario | Mirrors Playwright's | What to look for in Respan |
|---|---|---|---|
| 01 | `library_quickstart` | Library | workflow -> `browserType.launch`, `page.goto` with URL + title, `page.screenshot` with the image |
| 02 | `forms` | Actions / input | `locator.fill` shows the email; the password fill shows `<masked>`; after the GET submit the URL shows `<secret>password</secret>` |
| 03 | `assertions` | Assertions | `@playwright/test`'s `expect`: a failing `expect.toHaveText` is an `AssertionError` span with what the page showed |
| 04 | `multiple_pages` | Pages | each call's page URL says which tab it acted on, including the new tab |
| 05 | `network_route` | Network / mock APIs | `route.fulfill` from the route handler nests under the workflow |
| 06 | `evaluate` | Evaluating JavaScript | expressions and arguments in, JSON results out |
| 07 | `downloads` | Downloads | the click and `page.waitForEvent` for the download |
| 08 | `storage_state` | Authentication | save the state, reuse it in a new context |
| 09 | `tracing_zip` | Trace viewer | `tracing.start/stop`; `npx playwright show-trace artifacts/trace.zip` shows the same calls |
| 10 | `api_request` | API testing | `apiRequestContext.get` without a browser |
| 11 | `connect_over_cdp` | `browserType.connectOverCDP` | attach to a running Chrome; its calls nest the same way |
| 12 | `parallel_contexts` | Browser contexts | one shared browser, 2 traces, each with its own thread id and only its own calls |
| 13 | `callbacks` | Events, dialogs | a browser launched outside the workflow: `dialog.accept`, `route.fulfill` and the `page.title` an exposed function calls all nest under the workflow |
| 14 | `llm_agent` | (LLM in the loop) | an agent span: OpenAI-compatible chat calls through the Respan gateway interleaved with the Playwright calls each tool call made |

LLM calls made through the Respan gateway are logged by the gateway itself; the
traced span links to that log, so `verify_live.py` checks tokens and cost on the
linked log.

Not covered here: the Playwright MCP server (`@playwright/mcp`) and the
Playwright test runner's own reporting (a test without a Respan span is not
traced); `launchPersistentContext` with your own profile; remote browser vendors
that need accounts.
