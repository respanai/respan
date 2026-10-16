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
cp .env.example .env                                   # then set RESPAN_API_KEY (+ staging URLs)
```

browser-use drives an installed Google Chrome or Chromium; `browser-use --doctor`
checks that it can find and control one.

`requirements.txt` installs the instrumentation from this repo (editable), so
edits under `../../instrumentations/respan-instrumentation-browser-use` take effect
on the next run. Outside this repo: `pip install respan-instrumentation-browser-use`.

## Run

```bash
python hello_world.py                         # one agent, watch Chrome read Hacker News
python scenarios/11_sensitive_data.py         # any single scenario, then its live check
python run_all.py --list                      # all scenarios
python run_all.py                             # run them all, check every trace in Respan
python run_all.py judge custom_tools --headless
```

Every scenario runs under customer `browser-use-scenarios` with
`metadata.scenario=<name>` and `metadata.run_id=<run>`, so you can filter for it in
Respan. After the runs, `run_all.py` reads each trace back from the Respan logs API
and checks it, printing PASS/FAIL with a link to each agent span. It also writes
`runs/<run id>.md`. Flags: `--headless`, `--no-content`
(`capture_content=False`), `--no-screenshots` (`capture_screenshots=False`).

Every trace must have: one root span; the customer id and both metadata keys on
every span; every span's thread id equal to its agent's browser-use session id;
tokens and cost on every successful LLM span; and error spans only where the
scenario expects them. Scenarios add their own checks, listed below.

| # | Scenario | Mirrors browser-use's | What to look for in Respan |
|---|---|---|---|
| 01 | `basic_search` | getting_started/01_basic_search | agent -> steps -> LLM call + search/navigate; a screenshot on every step |
| 02 | `form_filling` | getting_started/02_form_filling | `tool.input` spans with typed values, `tool.click` submits |
| 03 | `data_extraction` | getting_started/03, features/custom_output | `tool.done` carries the typed `output_model_schema` object |
| 04 | `multi_step` | getting_started/04_multi_step_task | several steps, page URL/title + screenshot per step |
| 05 | `fast_agent` | getting_started/05_fast_agent | `flash_mode`: shorter prompts/completions |
| 06 | `custom_system_prompt` | features/custom_system_prompt | the extra rule at the end of every system message |
| 07 | `initial_actions` | features/initial_actions | `tool.navigate` under the agent, before step 1 (no LLM) |
| 08 | `follow_up_tasks` | features/follow_up_tasks | 2 traces, same thread id: the Threads view groups them |
| 09 | `multi_tab` | features/multi_tab | `navigate` with `new_tab`, `switch`; step URLs show the active tab |
| 10 | `parallel_agents` | features/parallel_agents | 2 traces, each with its own thread id, no cross-over |
| 11 | `sensitive_data` | features/sensitive_data | params show `<secret>demo_pass</secret>`; the password is in no span |
| 12 | `restrict_urls` | features/restrict_urls | the blocked `tool.navigate` is an error span |
| 13 | `blocked_domains` | features/blocked_domains | same, with `prohibited_domains` |
| 14 | `fallback_model` | features/fallback_model | the primary model's call fails (`ModelProviderError: Connection error.`), the fallback's call follows. browser-use's judge uses the PRIMARY model and does not fall back, so the scenario sets `judge_llm` |
| 15 | `extraction_model` | features/small_model_for_extraction | `tool.extract` holds an LLM span on the small model |
| 16 | `judge` | features/judge_trace | the judge's LLM span after the steps; `judge_verdict` on the agent |
| 17 | `custom_tools` | custom-functions/ | your `@tools.action` functions as `tool.save_note`, `tool.current_page_title` |
| 18 | `action_filters` | custom-functions/action_filters | a tool offered only on matching domains |
| 19 | `file_system` | file_system/file_system | `tool.write_file` / `tool.read_file` |
| 20 | `save_as_pdf` | features/save_as_pdf | `tool.save_as_pdf` |
| 21 | `scrolling` | features/scrolling_page | `tool.scroll`; screenshots move down the page |
| 22 | `hooks` | `run(on_step_start=, on_step_end=)` | hook code runs inside the agent span |
| 23 | `stop_externally` | features/stop_externally | the run stops after 2 steps and the agent span still closes |
| 24 | `max_steps` | `run(max_steps=2)` | ends unfinished, `is_done: false` |
| 25 | `no_vision` | `use_vision=False` | no image parts in any LLM prompt |
| 26 | `rerun_history` | features/rerun_history | the replay is its own trace with the same tool spans |
| 27 | `action_failure` | (failure handling) | `tool.navigate` to a dead domain is an error span |

Not covered here, because they need accounts or setup this example can't assume:
- Browser Use Cloud: cloud browser, sandbox, skills, `ChatBrowserUse`.
- Your own Chrome profile: `real_browser`, `save_cookies`, CDP and Playwright attach.
- Third-party services: 2FA, Gmail, Slack, Discord, AgentMail, CAPTCHAs, shopping and job applications.
- `browser_use.beta`: needs the Browser Use terminal binary.
- Other model providers: the instrumentation traces all of browser-use's model
  classes the same way, but this example only has the Respan gateway.

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

Hooks: `agent.run(on_step_start=fn, on_step_end=fn)` get the `Agent` each step
(scenario 22).

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
