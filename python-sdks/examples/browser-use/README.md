# browser-use + Respan

Trace [browser-use](https://github.com/browser-use/browser-use) agents in Respan with
`respan-instrumentation-browser-use`. Each run becomes one trace: the agent, every
step with its page screenshot, every LLM call with tokens and cost, and every
browser action.

## Setup

```bash
cd python-sdks/examples/browser-use
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11-3.13
pip install -r requirements.txt
cp .env.example .env                                   # then set RESPAN_API_KEY
```

browser-use drives an installed Google Chrome or Chromium; `browser-use --doctor`
checks that it can find and control one.

Working on the instrumentation itself? Install it from this repo instead:
`pip install -e ../../instrumentations/respan-instrumentation-browser-use`.

## Run

```bash
python hello_world.py                  # watch Chrome read Hacker News
python feature_tour.py                 # list scenarios
python feature_tour.py structured      # one scenario
python feature_tour.py all --headless  # everything, no window
```

| Scenario | What it shows in Respan |
|---|---|
| `structured` | `output_model_schema`: the `done` tool span carries a typed result |
| `custom-tool` | your own `@tools.action` function as its own `tool` span |
| `secrets` | `sensitive_data`: spans show `<secret>demo_pass</secret>`, never the password |
| `failure` | an action that fails (`navigate` to a dead domain) is an error span |

Flags for any scenario: `--no-content` (`capture_content=False`: keeps the span
tree, timings, models and tokens, drops prompts, results, URLs and error messages)
and `--no-screenshots` (`capture_screenshots=False`). Each run prints its trace id.

## What a trace looks like

```
agent.browser_use              the task in; final result, success, steps, usage out
├─ tool.navigate               browser-use opens a URL found in the task before step 1
├─ task  (step 1)              page URL/title, the model's reasoning, actions, screenshot
│  ├─ llm.<model>              the step's model call: messages, tokens, cost
│  ├─ tool.click / tool.input  one span per browser action
│  └─ tool.extract
│     └─ llm.<model>           extract runs its own model call over the page
├─ task  (step 2) ...
└─ llm.<model>                 the judge: browser-use grades the run (use_judge=True)
```

## browser-use in five minutes

browser-use runs a loop: screenshot + simplified DOM of the page -> LLM picks
actions -> actions run in a real Chrome over CDP -> repeat until the model calls
`done` or `max_steps` runs out.

```python
from browser_use import Agent, Browser, ChatOpenAI, Tools

agent = Agent(
    task="...",                          # plain-English goal
    llm=ChatOpenAI(model="gpt-4.1-mini"),  # also ChatAnthropic, ChatGoogle, ChatOllama, ...
    browser=Browser(headless=False),     # the Chrome it drives
)
history = await agent.run(max_steps=20)
```

The `Agent` options you will reach for most:

| Option | What it does |
|---|---|
| `tools=Tools()` | add your own actions with `@tools.action("description")`; args come from the model, `browser_session` / `page_url` are injected if you ask for them |
| `output_model_schema=MyModel` | the final answer must match a Pydantic model; read it with `history.get_structured_output(MyModel)` |
| `sensitive_data={"name": "value"}` | the model writes `name`; browser-use types `value`. Requires `Browser(allowed_domains=[...])` |
| `use_vision` | send screenshots to the model (default on) |
| `flash_mode=True` | faster, cheaper steps: no thinking/evaluation fields |
| `use_judge` | grade the run with one extra LLM call at the end (default on) |
| `max_failures` | consecutive failed steps before giving up |
| `initial_actions=[...]` | actions to run before the model's first step |
| `extend_system_message` | add your own rules to the system prompt |

`Browser(...)` options: `headless`, `allowed_domains`, `user_data_dir` (reuse a
logged-in profile), `cdp_url` (attach to a Chrome that is already running),
`keep_alive`, `window_size`.

`run()` returns an `AgentHistoryList`: `final_result()`, `is_successful()`,
`is_done()`, `judgement()`, `errors()`, `urls()`, `action_names()`,
`number_of_steps()`, `total_duration_seconds()`, `screenshot_paths()`.

Hooks: `agent.run(on_step_start=fn, on_step_end=fn)` get the `Agent` each step;
`feature_tour.py` uses one to print the trace id.

## Enabling the instrumentation

```python
from respan import Respan
from respan_instrumentation_browser_use import BrowserUseInstrumentor

respan = Respan(instrumentations=[BrowserUseInstrumentor()])
...
respan.flush()   # before a short script exits
```

The model calls go to whichever provider you configure. These examples point
browser-use's `ChatOpenAI` at the Respan gateway so one key covers both; the
gateway also keeps its own request log for each call.
