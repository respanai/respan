# Playwright + Respan

Trace [Playwright for Python](https://playwright.dev/python/) in Respan with
`respan-instrumentation-playwright`. Every Playwright call your code makes
(`Page.goto`, `Locator.click`, `expect(...)`, ...) becomes a span under your
workflow or agent span, with the page URL and title, screenshots, and Playwright's
own error messages. Typed passwords are masked.

## Setup

```bash
cd python-sdks/examples/playwright
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11-3.13
pip install -r requirements.txt
cp .env.example .env                                   # then set RESPAN_API_KEY
```

The scenarios drive the installed Google Chrome (`channel="chrome"`). Without
Chrome, run `playwright install chromium` and set `PLAYWRIGHT_CHANNEL=` (empty) in
`.env` to use Playwright's own Chromium.

`requirements.txt` installs the instrumentation from this repo (editable), so
edits under `../../instrumentations/respan-instrumentation-playwright` take effect
on the next run. Outside this repo: `pip install respan-instrumentation-playwright`.

## Run

```bash
python scenarios/03_forms.py                  # any single scenario, then its live check
python run_all.py --list                      # all scenarios
python run_all.py                             # run them all, check every trace in Respan
python run_all.py forms llm_agent --headed
python run_all.py --screenshot-on action      # a page screenshot after every action
python run_all.py --no-content                # capture_content=False
```

Every scenario runs inside a Respan workflow span under customer
`playwright-scenarios`, with `metadata.scenario=<name>` and `metadata.run_id=<run>`.
After the runs, `run_all.py` reads each trace back from the Respan logs API and
checks it, printing PASS/FAIL with the trace ids and a link to each root span. It
also writes `runs/<run id>.md`.

Every trace must have: every span the SDK exported, and exactly one root; the
customer id and both metadata keys on every span; one thread id; every Playwright
call the SDK traced stored as a `tool` span, call for call; tokens and cost on
every LLM call; and error spans only where the scenario expects them. Scenarios
add their own checks (which calls must appear, which must fail, which carry a
screenshot, which strings must never appear).

Most scenarios use a small local site served from `site/`, so they need no
accounts and give the same result every time. Screenshots, downloads, the saved
storage state and the trace.zip land in `artifacts/`.

| # | Scenario | Mirrors Playwright for Python's | What to look for in Respan |
|---|---|---|---|
| 01 | `library_quickstart` | Library (sync) | workflow -> `BrowserType.launch`, `Page.goto` with URL + title, `Page.screenshot` with the image |
| 02 | `async_quickstart` | Library (async) | the same tree from the async API |
| 03 | `forms` | Actions / input | `Locator.fill` shows the email; the password fill shows `<masked>`; after the GET submit the URL shows `<secret>password</secret>` |
| 04 | `assertions` | Assertions | a failing `LocatorAssertions.to_have_text` is an error span with Playwright's message |
| 05 | `multiple_pages` | Pages | each call's page URL says which tab it acted on, including the new tab |
| 06 | `network_route` | Network / mock APIs | `Route.fulfill` from the route handler nests under the workflow |
| 07 | `evaluate` | Evaluating JavaScript | expressions in, JSON results out |
| 08 | `downloads` | Downloads | `Download.save_as` with its path |
| 09 | `storage_state` | Authentication | save the state, reuse it in a new context |
| 10 | `tracing_zip` | Trace viewer | `Tracing.start/stop`; `playwright show-trace artifacts/trace.zip` shows the same calls |
| 11 | `api_request` | API testing | `APIRequestContext.get` without a browser |
| 12 | `connect_over_cdp` | `BrowserType.connect_over_cdp` | attach to a running Chrome; its calls nest the same way |
| 13 | `parallel_contexts` | Browser contexts | 2 traces, each with its own thread id and only its own calls |
| 14 | `llm_agent` | (LLM in the loop) | an `@agent` span: OpenAI-compatible chat calls through the Respan gateway interleaved with the Playwright calls each tool call made |

LLM calls made through the Respan gateway are logged by the gateway itself; the
traced span links to that log, so `run_all.py` checks tokens and cost on the
linked log.

Not covered here: the Playwright MCP server (`@playwright/mcp`) runs in Node;
`launch_persistent_context` with your own profile; remote browser vendors that
need accounts.
