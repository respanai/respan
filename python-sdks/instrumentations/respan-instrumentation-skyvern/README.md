# respan-instrumentation-skyvern

Trace [Skyvern](https://github.com/Skyvern-AI/skyvern) browser agents in Respan.
Each Skyvern run becomes one trace:

```
agent.skyvern_task                 Skyvern.run_task (or run_workflow / login / download_files)
├── task  (skyvern.step)           one per agent step: page URL, actions + results, screenshot
│   ├── llm.<model>                extract-actions: the model call that chose the actions
│   ├── tool.click                 one per browser action (click, input_text, select_option, ...)
│   └── tool.input_text
├── llm.<model>                    check-user-goal, the speculative next-step plan, extraction
└── tool.complete                  the action that finished the run
```

`page.act / page.extract / page.validate / page.ai_*` calls on a Skyvern browser page
are traced the same way, one `agent.<ai_action>` trace per call.

## Install

```bash
pip install respan-ai respan-instrumentation-skyvern
pip install "skyvern[local]"         # the in-process engine, for Skyvern.local()
python -m playwright install chromium
```

## Use

```python
from respan import Respan
from respan_instrumentation_skyvern import SkyvernInstrumentor

respan = Respan(api_key="...", instrumentations=[SkyvernInstrumentor()])

from skyvern import Skyvern

skyvern = Skyvern.local()
result = await skyvern.run_task(
    prompt="Find the top post on Hacker News", url="https://news.ycombinator.com",
    wait_for_completion=True,
)
```

Skyvern's model can go through the Respan gateway (OpenAI-compatible), so one key
covers the model and the traces:

```python
from skyvern.schemas.llm import LLMConfig

skyvern = Skyvern.local(llm_config=LLMConfig(
    model_name="openai/gpt-4.1-mini", required_env_vars=[], supports_vision=True,
    add_assistant_prefix=False,
    litellm_params={"api_base": "https://api.respan.ai/api", "api_key": RESPAN_API_KEY},
))
```

A full runnable example with live checks is in
[`python-sdks/examples/skyvern`](../../examples/skyvern).

## What is captured

| Span | From | Carries |
|---|---|---|
| `agent.skyvern_task` / `_workflow` / `_login` / `_download_files` | `Skyvern.run_task`, `run_workflow`, `login`, `download_files` | prompt, URL, engine, schema, max_steps in; status, output, failure_reason, step count, recording URL out. A failed / terminated / timed-out run is an error span with Skyvern's reason |
| `agent.<ai_action>` | `page.act`, `page.extract`, `page.validate`, `page.ai_*` (`run_sdk_action`) | the action and its result |
| `task` (`skyvern.step`) | each agent step | step number, page URL, every action with its results, the screenshot the model saw (JPEG), token counts and cost Skyvern recorded |
| `task` (`skyvern.block.<type>`) | each workflow block | block label and type, status, output, failure reason |
| `tool.<action>` | each browser action | the action (element, text, reasoning); a failed action is an error span with Skyvern's exception |
| `llm.<model>` | every model call Skyvern makes (all of its prompt handlers end in litellm) | Skyvern's prompt name (`skyvern.llm.prompt_name`), messages with screenshots, completion, prompt / completion / cached tokens |

- **Local and cloud.** With `Skyvern.local()` the whole tree is captured. Against
  Skyvern Cloud (`Skyvern(api_key=...)`) the engine runs remotely, so you get the run
  span with its status, output and recording link.
- **Grouping.** Thread, session and trace group default to the Skyvern run id, so a
  run's spans group together; values you set with `propagate_attributes` win.
- **Secrets.** Values Skyvern resolves as secrets during a run (credential and secret
  workflow parameters, one-time codes) are replaced by `<secret>name</secret>` in every
  span.
- **Content off.** `SkyvernInstrumentor(capture_content=False)` (or Respan's
  content-tracing switch) keeps the tree, ids, statuses and token counts and drops
  prompts, outputs, URLs and screenshots. `capture_screenshots=False` drops only the
  images.
- **No double logging.** Skyvern's model calls go through litellm; while a Skyvern
  chat span is open, the LiteLLM and OpenAI instrumentations are suppressed.

Skyvern's own OpenTelemetry spans (tracer `skyvern`) are not exported; Respan spans
are parented to each other across them. See [docs/GAPS.md](docs/GAPS.md) for
known limitations.

## Tests

`tests/test_baseline.py` drives Skyvern's real engine and a headless Chromium on a
local page with a scripted model, and checks every exported span (needs
`tests/requirements.txt` and `python -m playwright install chromium`):

```bash
pip install -e . -r tests/requirements.txt pytest pytest-asyncio
python -m pytest tests -q
```
