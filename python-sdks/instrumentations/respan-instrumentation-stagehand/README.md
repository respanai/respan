# respan-instrumentation-stagehand

Respan tracing for [Stagehand](https://github.com/browserbase/stagehand) v4
browser agents in Python (the `stagehand` package).

```bash
pip install respan-ai respan-instrumentation-stagehand stagehand
```

```python
import asyncio

from respan import Respan
from respan_instrumentation_stagehand import StagehandInstrumentor
from stagehand import Stagehand, local_browser

respan = Respan(instrumentations=[StagehandInstrumentor()])  # RESPAN_API_KEY


async def main():
    browser = await local_browser.launch()
    stagehand = await Stagehand.create(
        browser=browser, model="openai/gpt-5.4-mini", model_api_key="sk-..."
    )
    try:
        [page] = await browser.context.pages()
        await page.goto("https://news.ycombinator.com")
        await stagehand.act("click the 'new' link")
    finally:
        await stagehand.close()
        await browser.close()


asyncio.run(main())
respan.flush()  # short scripts: send spans before the process exits
```

Requires Stagehand **4.2** or later (v4's `Stagehand.create` API) and Python 3.11+.
The older v0-v3 Python client (`Stagehand(config)` + `init()`, which talked to
a hosted API) is not covered.

## What is traced

```
stagehand.session        agent   Stagehand.create() ... close(): settings in; operations, failures, token totals out
├─ stagehand.act         task    instruction, options (variable names only), result, cache status, screenshot after the act
│  └─ stagehand.llm      chat    every inference: prompt, completion, model, tokens
├─ stagehand.extract     task    instruction, JSON schema, extracted data
├─ stagehand.observe     task    instruction, suggested actions with their selectors
└─ page.goto, locator.fill, page.screenshot, context.new_page, ...   tool
```

- **One trace per session.** Every span of a Stagehand instance shares its trace,
  and its thread, session and trace-group identifiers default to the session id
  (the Browserbase session id when there is one). Values set with
  `respan.propagate_attributes(thread_identifier=..., group_identifier=...)` win.
  `customer_identifier` and metadata reach every span.
- **LLM calls.** Stagehand v4 runs its inference inside its Chrome extension.
  - With a **client model** (`model=<async callable>`) every inference comes
    back to your process, so each one is a full `chat` span: messages (with
    images), response format or tools, completion, and tokens. Name the model
    with `with_model_name(generate, "gpt-4.1-mini")` (or
    `StagehandInstrumentor(client_model_name=...)`) so Respan can price it. A
    client model is also how you route Stagehand through the Respan gateway or
    any OpenAI-compatible endpoint, since v4's built-in models take no base URL.
  - With a **built-in model** (`model="openai/..."`), Stagehand only reports
    each operation's token totals. Each operation then gets one content-free
    `chat` span with those tokens and the model name, so usage and cost still
    show up.
- **Concurrent operations.** Operations running at once on one session (for
  example `extract` on two pages with `asyncio.gather`) keep their own LLM
  calls. Stagehand gives the callback no operation id, so the prompt is matched
  against each running operation's instruction; two identical instructions
  running at the same time on one session cannot be told apart.
- **Screenshots.** After each `act`, the page is captured as a JPEG and attached
  to the act span's output. Respan stores it as log media and shows it in the
  log view. That is one extra screenshot round trip per `act` (bounded to a few
  seconds); `capture_screenshots=False` skips it. `page.screenshot()` calls show
  their image too.
- **Secrets.** Values passed as `variables` to `act` / `observe` are replaced by
  their `%name%` placeholder in every span of the session, including errors.
  Text typed with `locator.fill` / `locator.type` / `page.type` is exported as
  `[redacted]` when the target field is a credential (a password input, or a
  name, id or autocomplete that says password, token, one-time code, card
  number, and so on). The page is asked for the field's attributes, since
  selectors from `observe()` say nothing about the field. When the field cannot
  be identified, the text is redacted.
- **Errors.** Failed operations and browser actions are error spans with
  `Type: message`. An `act` that Stagehand reports as unsuccessful is an error
  span too.
- **No duplicate spans.** Provider instrumentations (for example Respan's
  OpenAI one) are suppressed inside a client model's call, since the `chat`
  span already carries it.

## Options

| Option | Default | Effect |
|---|---|---|
| `capture_content` | `True` | `False` keeps span structure, timings, models, operation names and token counts but drops instructions, prompts, completions, results, page URLs, typed text and exception messages (only the exception type is kept). `RESPAN_TRACE_CONTENT=false` or `TRACELOOP_TRACE_CONTENT=false` do the same. |
| `capture_screenshots` | `True` | `False` skips the screenshot after each `act` and replaces images in prompts with a size placeholder. |
| `client_model_name` | `None` | The model a client model calls, reported on its `chat` spans. `with_model_name()` sets it per model instead. |

## Not yet covered

- Session replay (a browser recording synced to the timeline). See `docs/GAPS.md`.
- The pre-v4 Python client and the Browserbase-hosted API mode.
- `experimental_batch()` callbacks run inside the extension and are not traced
  beyond their own operations.

An example with runnable scenarios lives in `python-sdks/examples/stagehand`.
