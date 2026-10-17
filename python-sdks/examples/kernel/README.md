# Kernel example (Python)

Kernel's Python app templates and API guides, traced with
[`respan-instrumentation-kernel`](../../instrumentations/respan-instrumentation-kernel),
plus a runner that reads every trace back from Respan and checks it.

> **Live status: pending a `KERNEL_API_KEY`.** These scenarios are written to run on
> real Kernel cloud browsers, but have so far only run against the package's local
> fake Kernel API (`--fake`: a real headless Chrome behind Kernel's HTTP routes),
> exporting to a real Respan environment. See the package's `docs/GAPS.md`.

| # | Scenario | Mirrors | What to look for in Respan |
|---|---|---|---|
| 01 | `sample_app` | template `sample-app` (Playwright over CDP) | `agent.get-page-title` -> `browsers.create` / `delete_by_id`; thread = the Kernel session |
| 02 | `browser_use` | template `browser-use` | the browser-use agent, steps, LLM calls and actions inside the Kernel action, one thread |
| 03 | `computer_api` | docs: Computer Controls | screenshots as JPEG, click / type / batch spans; the password as `<secret>password</secret>` |
| 04 | `process_exec` | docs: Process API | decoded stdout/stderr, exit code; `env` as `<masked>` |
| 05 | `replays` | docs: Replays | `kernel.replay.id` / `kernel.replay.view_url` on the replay span and the action root |
| 06 | `playwright_execute` | docs: Playwright Execution | code in, result out; the failing run as an error span |

Each scenario is a Kernel app (`kernel.App`, `@app.action`), as in Kernel's
templates. The runner invokes the action locally the way the Kernel runtime does
(`handler(KernelContext, payload)`); the browsers are Kernel's. Deploying the
apps with `kernel deploy` is not covered here (the scenarios import the runner's
harness); in your own app, start `Respan(...)` at the top of the deployed file.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # RESPAN_API_KEY, KERNEL_API_KEY

python run_all.py           # every scenario on Kernel cloud browsers, then check the traces
python run_all.py sample_app replays
python run_all.py --no-content --probe-passwords
python scenarios/03_computer_api.py   # one scenario
```

Without a Kernel account, `python run_all.py --fake` (or `KERNEL_FAKE=1 python
scenarios/<file>.py`) runs everything against the fake Kernel API from
`../../instrumentations/respan-instrumentation-kernel/tests/fake_kernel.py`, with a
local Chrome (set `CHROME_PATH` if it is not in the usual place). The browser-use
scenario still makes real model calls, through the Respan gateway.

Live runs open a public page (`KERNEL_TARGET_URL`, default `https://example.com`):
a Kernel cloud browser cannot reach a server on your machine. Fake runs serve
`site/` locally.

## What the runner checks

Generic checks come from `respan_tracing.testing.live` (`trace_problems`): every
exported span stored, one root, customer id and metadata on every span, one thread
id per trace, tokens and cost on chat spans, no forbidden strings. Kernel-specific
checks:

- one trace per action run, rooted at the `agent.<action>` span;
- every Kernel call the SDK traced is stored, call for call, as a `tool` span;
- the trace's thread is the Kernel session created in it;
- no unmasked browser token (`jwt=`) and no Kernel API key anywhere;
- expected calls, error calls and screenshots per scenario; with `--no-content`,
  no Kernel span stores input or output.

## Recommended stacks

Kernel is the browser; trace what drives it too:

- browser-use on Kernel: `KernelInstrumentor()` + `BrowserUseInstrumentor()` (this example).
- Playwright over CDP: `KernelInstrumentor()` + `PlaywrightInstrumentor()`.
- Computer-use loops (Anthropic / OpenAI / Gemini): `KernelInstrumentor()` + the
  provider SDK's instrumentation.
