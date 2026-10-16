# Skyvern + Respan

Trace [Skyvern](https://github.com/Skyvern-AI/skyvern) browser agents in Respan with
`respan-instrumentation-skyvern`. Each run becomes one trace: the task, every step
with its page screenshot, every LLM call Skyvern makes with tokens and cost, and every
browser action.

Skyvern runs in local mode (`Skyvern.local()`): the engine runs in your process with
an in-memory database, and its model calls go through the Respan gateway, so one
Respan key covers the model and the traces. No Skyvern account is needed.

## Setup

```bash
cd python-sdks/examples/skyvern
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11-3.13
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env                                   # then set RESPAN_API_KEY
```

`requirements.txt` installs the instrumentation from this repo (editable), so edits
under `../../instrumentations/respan-instrumentation-skyvern` take effect on the next
run. Outside this repo: `pip install respan-instrumentation-skyvern`.

## Run

```bash
python hello_world.py                         # one task: read the top Hacker News post
python scenarios/04_form_filling.py           # any single scenario, then its live check
python run_all.py --list                      # all scenarios
python run_all.py --headless                  # run them all, check every trace in Respan
python run_all.py quickstart max_steps
```

Every scenario runs under customer `skyvern-scenarios` with `metadata.scenario=<name>`
and `metadata.run_id=<run>`, so you can filter for it in Respan. After the runs,
`run_all.py` reads each trace back from the Respan logs API and checks it, printing
PASS/FAIL with a link to each run. It also writes `runs/<run id>.md`. Flags:
`--headless`, `--no-content` (`capture_content=False`), `--no-screenshots`
(`capture_screenshots=False`).

Every trace must have: every exported span stored; one root span; the customer id and
both metadata keys on every span; every span's thread id equal to the Skyvern run id;
tokens and cost on every successful LLM span; a Skyvern prompt name on every LLM
span; one tool span per action Skyvern recorded on each step, plus the LLM call that
chose them; and Skyvern's own run status. Scenarios add their own checks, listed below.

| # | Scenario | Mirrors Skyvern's | What to look for in Respan |
|---|---|---|---|
| 01 | `quickstart` | getting-started quickstart (`run_task`) | `agent.skyvern_task` -> step -> `extract-actions` LLM call + actions; screenshot per step |
| 02 | `data_extraction` | `run_task(data_extraction_schema=...)` | the typed output on the run span |
| 03 | `multi_step` | `run_task` navigation | several steps, each with its page URL, screenshot and `tool.click` |
| 04 | `form_filling` | `run_task` on a form | `tool.input_text` spans with the typed values, then the submit click |
| 05 | `max_steps` | `run_task(max_steps=1)` | the run span is an error carrying Skyvern's failure reason |
| 06 | `impossible_task` | terminate / `failure_reason` | `tool.terminate`; the run span is an error with the reason |
| 07 | `page_agent_task` | `page.agent.run_task` on a local browser | one run span with the final status, steps driving the page you opened |
| 08 | `engine_v3` | `run_task(engine=skyvern_v3)` | the v3 agent loop's LLM calls (`taskv3-agent-loop`) with their tool calls |

Not covered here, because they need Skyvern's persistent local mode (Postgres) or a
Skyvern Cloud account:
- Skyvern's in-memory mode fails inside Skyvern itself for: `run_workflow` (workflow
  blocks, for-loops, login blocks), `login` with stored credentials, `download_files`,
  the `skyvern_v2` engine, and `page.act / page.extract / page.validate / page.ai_*`.
  The instrumentation traces all of them; its tests cover them against a fake server.
- Skyvern Cloud: cloud browsers, recordings, the credentials vault.
- CUA engines (`openai_cua`, `anthropic_cua`): need provider keys.
