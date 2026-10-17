# Kernel observability: coverage, parity with Laminar, known limitations

## Status

**Live verification against real Kernel browsers is pending a `KERNEL_API_KEY`.**
Kernel has no local emulator: its browsers are cloud VMs. Everything below is
verified against `tests/fake_kernel.py`, a local stand-in for the Kernel API
whose browsers are a real headless Chrome (real CDP, real screenshots, real
clicks and typing, real process execution), driven through the real `kernel`
client. The Respan half was verified live: the example scenarios ran against the
fake Kernel API and exported to a Respan environment, and every trace was read
back from the logs API (see the verification log).

## What Laminar captures

Laminar's `KernelInstrumentor` (lmnr 0.7.67) patches from outside:
`BrowsersResource.create / retrieve / list / delete / delete_by_id /
load_extensions`, `ComputerResource.capture_screenshot` (output replaced by a
`<BINARY_BLOB_SCREENSHOT>` placeholder) and the mouse/keyboard calls,
`PlaywrightResource.execute`, `ProcessResource.exec / kill / spawn / status /
stdin / stdout_stream` (base64 output decoded), and `KernelApp.action` (span
`action.<name>`, then a flush). Its session recording comes from the Playwright
instrumentation's rrweb recorder on the CDP connection.

## What this package captures

| | Laminar | Respan |
|---|---|---|
| Browser lifecycle (`create`, `retrieve`, `update`, `list`, `delete_by_id`, `load_extensions`, `curl`, `repl`) | most | yes, `tool` spans |
| Computer Controls (every `computer.*`, incl. `batch`) | yes | yes |
| Screenshot output | placeholder only | **the image**, as a compact JPEG `image_url` (Respan media) |
| Process API | yes, decoded | yes, decoded (capped), exit code, process id |
| Playwright Execution | yes | yes; `success: false` is an error span |
| Replays (`start / stop / list / download`) | no | yes; replay id + view link on the span **and the action root** |
| `KernelApp.action` handlers | `action.<name>` + flush | `agent` span `kernel.action.<name>` (payload in, result out, invocation id) + flush |
| Grouping by Kernel session | no | thread / session / trace group = browser session id; the action's browser groups every span of the run, including browser-use / Playwright / LLM spans |
| Secret masking | no | API key, browser `jwt` tokens, process `env` values, `secrets({...})`, optional password-field probe |
| `capture_content=False` (structure only) | - | yes |
| Sync and async clients | yes | yes |
| Session replay (rrweb) synced to the timeline | **yes** (via Playwright) | no, see L1 |

## Seams

Kernel's Stainless-generated resource classes, patched from outside (no fork),
sync and async: `kernel.resources.browsers.browsers.(Async)BrowsersResource`,
`...computer.(Async)ComputerResource`, `...process.(Async)ProcessResource`,
`...playwright[.playwright].(Async)PlaywrightResource` (a module in older
releases, a package in newer ones; both are handled), `...replays.(Async)ReplaysResource`,
and `kernel.app_framework.KernelApp.action`. A resource a Kernel release lacks
is skipped, not fatal. `with_raw_response` / `with_streaming_response` wrappers
call the patched methods, so they are traced too (their output is the byte size).

The pin is `kernel >=0.100.0,<1.0.0`; the suite passes on 0.100.0, 0.121.0
(the newest that resolves next to `browser-use` 0.13.x, which pins `anyio`
below what `kernel` 0.127 needs) and 0.127.0.

## Known limitations

| # | Limitation | Impact | Next step |
|---|---|---|---|
| L1 | **Session replay** (an rrweb recording synced to the span timeline) is not implemented. This is the known gap across every Respan browser integration; it needs SDK, ingestion and UI work together. | no in-trace replay | interim: Kernel's own replays (`replays.start`), whose id and view link land on the action root span; screenshots on Computer API spans |
| L2 | **Not verified against real Kernel yet** (no `KERNEL_API_KEY`). The fake API follows the `kernel` client's routes and response models, but real Kernel may differ in: direct-to-VM routing (`base_url` + `jwt`; the fake returns no `base_url`), screenshot sizes and formats, `playwright.execute` semantics (the fake runs code with a small `page` shim, not real Playwright), replay availability (may need a headful browser). | unknown live surprises | run `examples/kernel/run_all.py` with a key (see the example README) |
| L3 | `type_text` masking by field type needs `probe_password_fields=True` (one extra `playwright.execute` per call). Off by default, so without it a password typed through the Computer API is exported unless declared with `secrets(...)`. With it on, a probe that fails, an iframe focus, and any `batch` typing are masked (fail closed). | passwords in computer-use loops | decide whether to default it on once the per-call cost is measured live |
| L4 | `process.stdout_stream` is traced as the call that opens the stream; the streamed events (stdout/stderr chunks, exit) are not recorded. | streamed process output not in the trace | wrap the stream iterator if customers ask |
| L5 | Not traced: `browsers.fs.*`, `browsers.logs.*`, `browsers.telemetry.*`, `browsers.webmcp.*`, `browser_pools`, `profiles`, `extensions`, `deployments`, `invocations`, `apps`. | those calls are invisible | add the ones customers use |
| L6 | Grouping outside a Kernel action: each call's span uses its own browser's session id, but spans of other instrumentations (browser-use, Playwright, LLM) in the same trace keep their own defaults, so a trace can show two thread ids. Inside an action this is handled (the action's first browser is propagated to every later span). | split threads for code not run as a Kernel action | wrap the run in `propagate_attributes(thread_identifier=browser.session_id)` |
| L7 | `@app.action` handlers registered **before** Respan starts are not wrapped (registration is the seam, as in Laminar). | an untraced action root | start Respan before defining the app (README) |
| L8 | Masking learned values: env values and typed passwords are masked across the call only when 8+ characters long (shorter ones would mangle unrelated text); the call's input always shows them masked. | a short env value echoed in output is exported | declare it with `secrets(...)` |
| L9 | Kernel's TS SDK (`@onkernel/sdk`) is not instrumented. | TS Kernel apps untraced | `@respan/instrumentation-kernel`, after this package |

## Baseline

`tests/test_baseline.py` is the wire contract. It drives the real `kernel` client
(sync and async) against the fake Kernel API, with a real headless Chrome behind
it, and Playwright connected over CDP the way Kernel's templates connect. A
scripted computer-use model (fixed tool calls) stands in for the agent. Every
exported span is checked through the real Respan pipeline
(`respan_tracing.testing.capture_exported_spans`):

- the sample-app action is one trace: an `agent` root with payload, output and
  invocation id; `browsers.create` / `delete_by_id` nested under it;
- every Kernel span is exportable (log type + span kind);
- thread / session / trace group = the Kernel session on every span; an action
  with no browser falls back to its invocation id;
- other instrumentations' spans inside an action join the action's thread;
- the action flushes when the handler returns;
- the computer-use loop, sync and async: screenshots exported as JPEG while the
  caller still reads the full PNG; clicks and typing land in the real page;
- customer metadata (flat `respan.metadata.<key>`) and identifier on every span;
- customer thread / trace group win over the Kernel default;
- parallel actions: separate traces, sessions and identifiers;
- process output decoded, exit code recorded, env values masked in input and output;
- `playwright.execute` code and result; a failed run is a `PlaywrightExecutionError`
  span; replay id and masked view link on the replay span and the action root;
- errors are `Type: message`;
- the API key and browser `jwt` tokens never reach a span;
- `secrets({...})` values never reach a span (typed text, process output, errors);
- with `probe_password_fields`, password-field typing (and batch typing) is masked,
  ordinary typing is kept, the probe itself is not a span; a failed probe masks;
- content off: identifiers, session/replay ids and exit codes kept; inputs,
  outputs, URLs, screenshots and error messages dropped.

Each mechanism was disabled in turn and the baseline rerun; every one makes at
least one test fail:

| Disabled | Fails |
|---|---|
| async resources patched | async computer loop |
| span kind / log type on Kernel spans | action trace, sync and async computer loops |
| browser session as the default grouping | action trace, both computer loops, parallel actions |
| first browser groups the action run | action trace, parallel actions |
| action propagates its session to other spans | other instrumentations join the thread |
| `KernelApp.action` wrapped | 7 tests (action trace, flush, no-browser action, metadata, customer grouping, parallel, replay link) |
| flush when the action returns | flush test |
| screenshot re-encoded as compact JPEG | both computer loops |
| process output decoded | process test |
| env values masked in output | process test |
| env values masked in input | process test |
| API key masked | API key / token test |
| URL `jwt` tokens masked | API key / token test, Playwright + replays test |
| `secrets(...)` | declared-secrets test |
| password-field probe | probe test, failed-probe test |
| fail closed when the probe fails | failed-probe test |
| `capture_content` | content-off test |
| `playwright.execute` failure as an error span | Playwright + replays test |
| replay link on the action root | replay-link test |
| error message on error spans | errors test |
| customer grouping wins | customer thread / trace group test |

The customer-metadata test only fails when actions are not wrapped (metadata is
copied by Respan's span processor); it stays as a regression guard.

## Verification log

- `pytest tests` (25 tests: 18 baseline + 7 unit), real headless Chrome behind the
  fake Kernel API, on `kernel` 0.100.0, 0.121.0 and 0.127.0.
- 2026-10-16, `examples/kernel/run_all.py --fake`: 6 scenarios mirroring Kernel's
  Python templates and API docs, run against the fake Kernel API + local Chrome,
  exported to Respan **staging** and read back from the logs API: **6/6 PASS**
  (sample_app `86471428bee230a202b14504fc98de49`, browser_use
  `e9a3da3d58bd45bffc3e4c53c9c6b7ae`, computer_api
  `5a2186b50b99d4d93254d456feb98505`, process_exec
  `107141394b0eaa738361c3b52fac884f`, replays `26e4049f83b145293b1b815b7a9158e1`,
  playwright_execute `5682262f691027d49a02f11f9e4a8f06`). Also `--no-content`
  (all 6; no Kernel span stores input or output) and `--probe-passwords` (the 5
  scenarios without an LLM): all pass.
- **Pending:** the same suite with a `KERNEL_API_KEY` on real Kernel browsers.
