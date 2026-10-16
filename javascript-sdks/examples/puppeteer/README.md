# Puppeteer + Respan (JavaScript / TypeScript)

Trace [Puppeteer](https://pptr.dev/) in Respan with
`@respan/instrumentation-puppeteer`. Every Puppeteer call your code makes
(`page.goto`, `locator.click`, `elementHandle.type`, ...) becomes a span under
your workflow or agent span, with the page URL and title, screenshots, and
Puppeteer's own error messages. Typed passwords are masked, and callbacks
(`page.on` listeners, request-interception handlers, exposed functions) nest
under the workflow that registered them.

## Setup

```bash
cd javascript-sdks/examples/puppeteer
npm install                 # Node.js 22.12+
cp .env.example .env        # then set RESPAN_API_KEY
```

The scenarios use `puppeteer-core` with the installed Google Chrome (or
`CHROME_PATH` in `.env`). With the full `puppeteer` package instead, Puppeteer
downloads its own Chrome.

`.npmrc` sets `install-links=true`, so the instrumentation is installed from this
repo as a real package. After changing it, rebuild it (`npx tsc` in its
directory), then `rm -rf node_modules/@respan/instrumentation-puppeteer && npm install`
here (the version does not change, so a plain `npm install` keeps the old copy).
Outside this repo:
`npm install @respan/instrumentation-puppeteer`.

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
`puppeteer-scenarios`, with `metadata.scenario=<name>` and
`metadata.run_id=<run>`. The verifier reads each trace back from the Respan logs
API and prints PASS/FAIL per scenario with its trace ids and a link to its root
span, and writes the same report to `runs/<run id>.md`.

Every trace must have: every span the SDK exported, and exactly one root; the
customer id and both metadata keys on every span; one thread id; every
Puppeteer call the SDK traced stored as a `tool` span, call for call; tokens and
cost on every LLM call; and error spans only where the scenario expects them.
Scenarios add their own checks (which calls must appear, which must fail, which
carry a screenshot or a recording path, which strings must never appear). With
`--no-content`, no Puppeteer row may contain a page URL, typed text, title or
image; with `--screenshot-on action`, navigations carry a screenshot.

Most scenarios use a small local site served from `site/`, so they need no
accounts and give the same result every time. Screenshots, the PDF, the
download and the recording land in `artifacts/`.

| # | Scenario | Mirrors pptr.dev's | What to look for in Respan |
|---|---|---|---|
| 01 | `screenshot` | Screenshots | workflow -> `puppeteer.launch`, `page.goto` with URL + title, `page.screenshot` with the image |
| 02 | `pdf` | PDF generation | `page.pdf` with its options in and the byte size out |
| 03 | `forms` | Page interactions | `page.type` shows the email; the password shows `<masked>`; after the GET submit the URL shows `<secret>password</secret>` |
| 04 | `locators` | Locators | `locator.click` / `locator.fill` show the selector the locator came from (`::-p-text(...)`, `::-p-aria(...)`) |
| 05 | `interception` | Request interception | `request.respond` / `request.continue` from the `page.on("request")` handler nest under the workflow |
| 06 | `evaluate` | JavaScript execution | page functions as source text with their arguments in; JSON results out |
| 07 | `new_tab` | Pages / targets | each call's page URL says which tab it acted on, including the new tab |
| 08 | `download` | Downloads | `browser.createBrowserContext` with the download folder, then the click |
| 09 | `connect` | Connecting to a running browser | `puppeteer.connect`; the attached browser's calls nest the same way |
| 10 | `screen_recording` | Screen recording | `page.record` carries the video's path (`puppeteer.recording.path`); `screenRecorder.stop` ends it |
| 11 | `errors` | Timeouts | error spans with Puppeteer's own messages (`TimeoutError: Waiting for selector ...`) |
| 12 | `llm_agent` | (LLM in the loop) | an agent span: OpenAI-compatible chat calls through the Respan gateway interleaved with the Puppeteer calls each tool call made |
| 13 | `parallel_pages` | Browser contexts | one shared browser, 2 traces, each with its own thread id and only its own calls |
| 14 | `callbacks` | Events, dialogs, exposeFunction | a browser launched outside the workflow: `dialog.accept` and the `page.title` an exposed function calls nest under the workflow |
| 15 | `webdriver_bidi` | WebDriver BiDi | the same spans over BiDi as over CDP |

LLM calls made through the Respan gateway are logged by the gateway itself; the
traced span links to that log, so `verify_live.py` checks tokens and cost on the
linked log. In `llm_agent`, a `page.evaluate` that runs while a click navigates
can fail ("Execution context was destroyed"); the agent sees the error and
continues, and the verifier lists it as an incidental error span.

Not covered here: `page.screencast()` (it needs ffmpeg; `page.record()` is
the built-in equivalent), Chrome extensions, and remote browser vendors that
need accounts.
