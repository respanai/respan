# respan-instrumentation-playwright

Respan tracing for [Playwright for Python](https://playwright.dev/python/), sync and async APIs.

```bash
pip install respan-ai respan-instrumentation-playwright
```

```python
from playwright.sync_api import sync_playwright
from respan import Respan, workflow
from respan_instrumentation_playwright import PlaywrightInstrumentor

respan = Respan(instrumentations=[PlaywrightInstrumentor()])  # RESPAN_API_KEY


@workflow(name="check_docs")
def check_docs():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://playwright.dev/python")
        page.get_by_role("link", name="Get started").click()
        browser.close()


check_docs()
respan.flush()  # short scripts: send spans before the process exits
```

With `respan-tracing` alone, call `PlaywrightInstrumentor().activate()` after
`RespanTelemetry(...)`.

## What is traced

Playwright is what an agent, a test or a script drives, so it adds no root span
of its own. Every public API call that reaches the browser becomes one `tool`
span under the span that is active when it runs:

```
workflow / your agent span
├─ BrowserType.launch      tool
├─ Page.goto               tool   url in; status, final URL and page title out
├─ llm.gpt-4.1-mini        chat   (your LLM call, from its own instrumentation)
├─ Locator.fill            tool   locator + value in (password fields masked)
├─ Locator.click           tool
└─ Page.screenshot         tool   the image, as a compact JPEG
```

- **One span per public call**, named the way Playwright's own trace viewer names
  it (`Page.goto`, `Locator.click`, `LocatorAssertions.to_have_text`,
  `APIRequestContext.get`, ...). Local calls that never reach the browser
  (`page.locator()`, `page.get_by_role()`) are not spans.
- **Only under an active span.** A call made with no active span (a plain script,
  a test suite without a Respan workflow) is not traced, so standalone runs do not
  turn into thousands of one-span traces. Wrap the code in `@workflow`, `@agent`,
  or your agent framework's own span.
- **Sync and async.** Both APIs produce the same spans. Callbacks you register
  (`page.route`, `page.on`, `expose_function`, ...) run in the context they were
  registered in, so calls inside them nest correctly and keep your customer
  identifier and metadata. (The sync API runs callbacks in its own greenlet, which
  otherwise loses both.)
- **Page context.** Each call on a page records the page URL; navigations
  (`goto`, `reload`, `go_back`, `go_forward`, `set_content`, `wait_for_url`)
  also record the title.
- **Screenshots.** `page.screenshot()` / `locator.screenshot()` output is attached
  as an `image_url` part, re-encoded as JPEG (max 1280 px wide). Respan stores it as
  log media. With `screenshot_on`, the instrumentation also takes a screenshot after
  navigations or actions, for a flip-book of the run.
- **Errors.** A failed call is an error span whose message is Playwright's own,
  e.g. `TimeoutError: Locator.click: Timeout 30000ms exceeded.`
- **Passwords.** Text typed into a password field (`fill`, `type`,
  `press_sequentially`, `keyboard.type` / `insert_text` into a focused password
  field) is exported as `<masked>`. Once seen, the same text is masked everywhere
  else in that run (a GET form puts it in the next URL). If the instrumentation
  cannot tell where the text went, or the call failed, it masks it.
- **Your secrets.** Values passed to `secrets({...})` never reach a span:

  ```python
  from respan_instrumentation_playwright import secrets

  with secrets({"api_token": token}):
      page.goto(f"https://example.com/?token={token}")  # exported as <secret>api_token</secret>
  ```
- **Customer context.** `customer_identifier`, `metadata`, `thread_identifier` and
  the other fields set on `Respan(...)` or `propagate_attributes(...)` reach every
  span. Playwright has no session id of its own, so its spans join your trace and
  thread.
- **Playwright's own trace stays clean.** The extra calls the instrumentation makes
  (page title, `screenshot_on` screenshots, the password-field check) are marked
  internal, so they do not show up as your calls in a Playwright trace.zip.

## Options

| Option | Default | Effect |
|---|---|---|
| `capture_content` | `True` | `False` keeps span names, timings, errors (type only) and customer context, and drops arguments, return values, page URLs and titles, and screenshots. `TRACELOOP_TRACE_CONTENT=false` does the same. |
| `capture_screenshots` | `True` | `False` drops all images; screenshot calls report only their byte size. |
| `screenshot_on` | `"never"` | `"navigation"`: a page screenshot after each navigation. `"action"`: also after each click, fill, check, press, ... Each one is an extra browser round trip. |

## Not yet covered

- Session replay (an rrweb recording synced to the span timeline). See `docs/GAPS.md`.
- A per-browser `playwright.session` span. See `docs/GAPS.md`.
