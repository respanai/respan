# respan-instrumentation-browser-use

Respan tracing for [browser-use](https://github.com/browser-use/browser-use) agents.

```bash
pip install respan-ai respan-instrumentation-browser-use
```

```python
from browser_use import Agent, ChatOpenAI
from respan import Respan
from respan_instrumentation_browser_use import BrowserUseInstrumentor

respan = Respan(instrumentations=[BrowserUseInstrumentor()])  # RESPAN_API_KEY

agent = Agent(task="Find the top story on Hacker News", llm=ChatOpenAI(model="gpt-4.1-mini"))
await agent.run()
respan.flush()  # short scripts: send spans before the process exits
```

With `respan-tracing` alone, call `BrowserUseInstrumentor().activate()` after
`RespanTelemetry(...)`.

## What is traced

```
browser_use.agent          agent   task in; final result, success, steps, usage out
└─ browser_use.step        task    one per agent step: page URL/title, reasoning, actions, screenshot
   ├─ browser_use.llm      chat    every model call, any browser-use provider; tokens + cache tokens
   └─ <action name>        tool    one per browser action (navigate, click, input, extract, done, ...)
```

- **Screenshots.** Each step's output carries that step's page screenshot as an
  `image_url` part, re-encoded as JPEG (max 1280 px wide) to keep exports small. Respan stores it as log media and shows it in the log view.
  LLM prompts keep the screenshot the model actually saw.
- **Any LLM provider.** Model calls are traced at browser-use's own chat-model
  layer, so OpenAI, Anthropic, Google, Browser Use Cloud, OpenRouter, Ollama and
  custom models all produce the same `chat` span. Provider SDK instrumentations
  that honour OpenTelemetry's language-model suppression key do not log the same
  call twice.
- **Action failures.** browser-use returns action errors instead of raising them;
  those `tool` spans are marked as errors.
- **Anthropic cache tokens.** browser-use's native Anthropic clients leave cache
  writes out of `prompt_tokens`; they are added back so Respan prices the full prompt.
- **Activation order.** Models registered by an Agent built before activation are
  still traced.
- **Secrets.** Action parameters are captured before browser-use substitutes
  `sensitive_data`, so secrets stay as their `<secret>name</secret>` placeholders.
  Any value that comes back with a real secret in it (a page URL, an error such as
  "Navigation to https://...?token=... blocked") is masked the same way before export.
- **Grouping.** Every span of a run carries the Agent's session id as its thread,
  session and trace-group identifier, so follow-up tasks on the same `Agent`
  (`add_new_task` + `run()`) group together. Values set through
  `respan.propagate_attributes(thread_identifier=..., group_identifier=...)` win.
- **Replays.** `Agent.rerun_history()` is its own trace under a
  `browser_use.rerun` agent span, in the same thread as the original run.
- **Customer context.** `customer_identifier`, `metadata` and the other fields
  set on `Respan(...)` or `propagate_attributes(...)` reach every span.

## Options

| Option | Default | Effect |
|---|---|---|
| `capture_content` | `True` | `False` keeps span structure, timings, models, action names and token counts but drops the task, prompts, completions, action params and results, page URLs and titles, and exception messages (only the exception type is kept). `TRACELOOP_TRACE_CONTENT=false` does the same. |
| `capture_screenshots` | `True` | `False` drops step screenshots and replaces inline images in prompts with a size placeholder. |
| `record_session` | `RESPAN_RECORD_SESSION` env, else `False` | **Preview.** Records each run's pages with [rrweb](https://github.com/rrweb-io/rrweb) and puts a replay link on every span of the run (`metadata.session_replay_url`, `metadata.session_replay_id`); each link ends in `#t=<ms>` and opens the replay at the moment that span started. Against a backend without link reservation, only the agent span is linked. Every input is masked and `sensitive_data` values never leave the process. Nothing is recorded when content capture is off. If the upload fails the run is unaffected and a local `.html` player is written instead (its path is logged). |

## Session replay (preview)

Record what the browser did and watch it back from the trace:

```python
respan = Respan(instrumentations=[BrowserUseInstrumentor(record_session=True)])
```

or, without code changes, `RESPAN_RECORD_SESSION=1 python your_agent.py`.

- **What is recorded.** The pages the agent drives, with
  [rrweb](https://github.com/rrweb-io/rrweb) (DOM snapshots + changes, not video),
  over browser-use's own CDP connection. Nothing to install in the browser.
- **Where to watch it.** Every span of a recorded run (agent, steps, actions, LLM
  calls) carries `metadata.session_replay_url`. Open it from the log's metadata:
  the replay opens paused at the moment that span started (`#t=<ms>`), with every
  step, action and LLM call listed beside it to jump between. `metadata.session_replay_id`
  is the same on every span of the run.
- **Privacy.** Every input value is masked in the page, `sensitive_data` values never
  leave the process, and nothing is recorded with `capture_content=False`.
- **Failure is harmless.** The recording is uploaded once, when the run ends; if
  that fails the run is unaffected and a local `.html` player is written instead
  (its path is logged). The links are time-limited; `expires_at` comes back with the
  upload.
- **Limits.** One stream that follows the agent's focused tab; canvas and video do not
  replay; images load from the original site at playback. Full list in
  [`docs/GAPS.md`](docs/GAPS.md).

`examples/browser-use/replay_demo.py` runs a short recorded agent and prints the
replay links.

## Not yet covered

- Session replay is a preview: one stream that follows the agent's focused tab, not
  yet synced to the step timeline in the Respan UI. See `docs/GAPS.md`.
- `browser_use.beta` agents.
