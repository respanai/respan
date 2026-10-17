# respan-instrumentation-kernel

Respan tracing for [Kernel](https://www.kernel.sh) (`pip install kernel`): cloud
browsers, the Computer Controls, Process and Playwright Execution APIs, replays,
and Kernel app actions. Sync (`Kernel`) and async (`AsyncKernel`) clients.

```bash
pip install respan-ai respan-instrumentation-kernel
```

```python
import kernel
from kernel import Kernel
from playwright.async_api import async_playwright
from respan import Respan
from respan_instrumentation_kernel import KernelInstrumentor

# Start Respan before your actions are defined: @app.action handlers are traced
# when they are registered.
respan = Respan(instrumentations=[KernelInstrumentor()])  # RESPAN_API_KEY

client = Kernel()  # KERNEL_API_KEY
app = kernel.App("python-basic")


@app.action("get-page-title")
async def get_page_title(ctx: kernel.KernelContext, input_data: dict) -> dict:
    kernel_browser = client.browsers.create(invocation_id=ctx.invocation_id)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(kernel_browser.cdp_ws_url)
        page = await (await browser.new_context()).new_page()
        try:
            await page.goto(input_data["url"])
            return {"title": await page.title()}
        finally:
            await browser.close()
            client.browsers.delete_by_id(kernel_browser.session_id)
```

With `respan-tracing` alone, call `KernelInstrumentor().activate()` after
`RespanTelemetry(...)`.

## What is traced

Kernel hosts the browser; your agent drives it. This package traces the Kernel
side. Pair it with the instrumentation of what drives the browser:
[browser-use](../respan-instrumentation-browser-use),
[Playwright](../respan-instrumentation-playwright), or the LLM SDK of a
computer-use loop (OpenAI, Anthropic, Google).

```
agent.get-page-title                 agent   a Kernel app action (input payload, output)
├─ tool.kernel_browsers_create       tool    browsers.create: session id, live view link
├─ browser_use agent / Playwright    ...     from their own instrumentations
├─ tool.kernel_computer_capture_screenshot  tool    the screenshot, as a compact JPEG
├─ tool.kernel_computer_click_mouse  tool    x, y
├─ tool.kernel_computer_type_text    tool    text (password fields masked, see below)
├─ tool.kernel_process_exec          tool    command in; decoded stdout/stderr, exit code out
├─ tool.kernel_playwright_execute    tool    Playwright code in, result out
├─ tool.kernel_replays_start         tool    replays.start: replay id + view link
└─ tool.kernel_browsers_delete_by_id tool
```

- **One `tool` span per Kernel API call**, named `kernel.<group>.<method>`
  (`kernel.browsers.create`, `kernel.computer.click_mouse`, `kernel.process.exec`,
  `kernel.playwright.execute`, `kernel.replays.start`, ...; shown as
  `tool.kernel_<group>_<method>`), with
  `kernel.api.name` and `kernel.session.id` attributes. Covered:
  `browsers.create / retrieve / update / list / delete_by_id / load_extensions /
  curl / repl`, every `browsers.computer.*` call (including `batch`), every
  `browsers.process.*` call, `browsers.playwright.execute`, and
  `browsers.replays.start / stop / list / download`. Calls are traced whether or
  not a span is active (each Kernel call is a remote, billable operation).
- **Kernel app actions.** A handler registered with `@app.action(name)` is an
  `agent` span `kernel.action.<name>` (shown as `agent.<name>`) with the payload as
  input, the return value as output and the invocation id. Spans are flushed when
  the handler returns, since a deployed action's process may stop right after.
- **Screenshots.** `computer.capture_screenshot` output is attached as an
  `image_url` part, re-encoded as JPEG (max 1280 px wide); Respan stores it as log
  media. The response you get back is untouched: the bytes are read from the
  already-loaded response, never from the stream.
- **Process output.** `stdout_b64` / `stderr_b64` are decoded to text (capped at
  4000 characters per stream); the exit code and process id are attributes.
- **Replays.** `replays.start` puts `kernel.replay.id` and `kernel.replay.view_url`
  on its span and on the action's root span, so the recording is one click from
  the trace.
- **Errors.** A failed call is an error span whose message is the client's own,
  e.g. `NotFoundError: Error code: 404 - {...}`. A `playwright.execute` run that
  fails (Kernel answers `success: false`) is an error span
  `PlaywrightExecutionError: <error>`.
- **Grouping.** The Kernel browser session id is the default thread, session and
  trace group of each call's span. Inside an action, the first browser the action
  creates groups the whole run, including the spans of browser-use, Playwright and
  LLM calls started after it, so one action run is one trace with one thread (the
  invocation id is used when the action creates no browser). Values you set on
  `Respan(...)` or with `propagate_attributes(...)` always win.
- **Customer context.** `customer_identifier`, `metadata` and the other fields set
  on `Respan(...)` or `propagate_attributes(...)` reach every span.

## Secrets

- **Your Kernel API key** never reaches a span (exported as
  `<secret>kernel_api_key</secret>` if it shows up in code, commands or errors).
- **Browser tokens.** The `jwt` in `cdp_ws_url`, `webdriver_ws_url`, live view and
  replay URLs grants control of the browser; it is exported as `jwt=<masked>`.
- **Process environment.** `env` values of `process.exec` / `spawn` are shown as
  `<masked>`, and (when 8+ characters) masked in the command's output too.
- **Declared secrets.** Values passed to `secrets({...})` never reach a span:

  ```python
  from respan_instrumentation_kernel import secrets

  with secrets({"password": password}):
      client.browsers.computer.type_text(session_id, text=password)  # <secret>password</secret>
  ```
- **Typing into password fields.** The Computer API types at the OS level, so
  Kernel cannot say which field the text went to. With
  `probe_password_fields=True`, each `type_text` first asks the browser (one
  `playwright.execute` call, not traced) whether the focused element is a password
  field, and exports the text as `<masked>` if it is, if the check fails, or if the
  focus is inside an iframe. Typing in a `computer.batch` is always masked with this
  option on, since a batch can move focus. Off by default because it costs one
  extra Kernel call per `type_text`; without it, declare the value with
  `secrets(...)`.

## Options

| Option | Default | Effect |
|---|---|---|
| `capture_content` | `True` | `False` keeps span names, timings, session/replay/process ids, exit codes, errors (type only) and customer context, and drops arguments, responses, screenshots, process output and URLs. `RESPAN_TRACE_CONTENT=false` does the same. |
| `capture_screenshots` | `True` | `False` drops screenshot images; screenshot calls report only their byte size. |
| `probe_password_fields` | `False` | Mask text typed into password fields (see above). One extra `playwright.execute` per `type_text`. |

## Not yet covered

- Session replay synced to the span timeline (rrweb). The interim is Kernel's own
  replay link on the trace. See `docs/GAPS.md`.
- `process.stdout_stream` events, `browsers.fs`, `browsers.logs`, `telemetry`,
  browser pools and profiles. See `docs/GAPS.md`.
- Live verification against real Kernel browsers is pending a `KERNEL_API_KEY`;
  everything here is verified against a local fake Kernel API. See `docs/GAPS.md`.
