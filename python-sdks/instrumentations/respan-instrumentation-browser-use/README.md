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
  `image_url` part. Respan stores it as log media and shows it in the log view.
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

## Not yet covered

- Session replay (rrweb recording synced to the step timeline). See `docs/GAPS.md`.
- `browser_use.beta` agents.
