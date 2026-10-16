# Stagehand (Python) + Respan

Trace [Stagehand](https://github.com/browserbase/stagehand) v4 browser agents in
Respan with `respan-instrumentation-stagehand`. Each Stagehand session becomes
one trace: the session, every `act` / `extract` / `observe` with its LLM calls
(tokens and cost), every browser action, and a screenshot after each `act`.

Everything runs on a local Chrome (no Browserbase account). The model calls go
through the Respan gateway via Stagehand v4's client-model callback
(`Stagehand.create(model=<async callable>)`, see `respan_model()` in
`scenarios/_harness.py`), so one Respan key covers both traces and models.

## Setup

```bash
cd python-sdks/examples/stagehand
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11-3.13
pip install -r requirements.txt                        # Google Chrome must be installed
cp .env.example .env                                   # then set RESPAN_API_KEY
```

`requirements.txt` installs the instrumentation from this repo (editable), so
edits under `../../instrumentations/respan-instrumentation-stagehand` take effect
on the next run. Outside this repo: `pip install respan-instrumentation-stagehand`.

## Run

```bash
python hello_world.py                        # one session: act + extract on Hacker News
python scenarios/05_act_variables.py         # any single scenario, then its live check
python run_all.py --list                     # all scenarios
python run_all.py                            # run them all, check every trace in Respan
python run_all.py persisted_login multi_tab --headless
```

Every scenario runs under customer `stagehand-py-scenarios` with
`metadata.scenario=<name>` and `metadata.run_id=<run>`. After the runs,
`run_all.py` reads each trace back from the Respan logs API and prints
PASS/FAIL per scenario with its trace ids and links; it also writes
`runs/<run id>.md`. Flags: `--headless`, `--no-content`
(`capture_content=False`), `--no-screenshots` (`capture_screenshots=False`).

The generic checks come from `respan_tracing.testing.live`: every exported span
is stored, one root span, customer id and metadata on every span, one thread id,
tokens and cost on every LLM span, no secrets. The Stagehand checks: one session
span per trace, the thread id equals the session id, the session's operation
count matches its task spans, and each operation's LLM spans add up to the token
usage Stagehand itself reports for it. The harness also turns on Respan's OpenAI
instrumentation, so that last check fails if the gateway call inside the client
model were logged a second time.

| # | Scenario | Mirrors Stagehand's | What to look for in Respan |
|---|---|---|---|
| 01 | `hello` | README quickstart | session -> act / extract, each with its LLM calls; a screenshot on the act |
| 02 | `persisted_login` | cookbooks/persisted-login | 2 traces on one profile; observe returns selectors; the password `locator.fill` shows `[redacted]`, the email does not |
| 03 | `paginated_catalog` | cookbooks/paginated-catalog | 3 extracts + 2 acts in order, 4 products per page |
| 04 | `extract_schema` | showcase (extract with a schema) | the extract's input shows the JSON schema; output is the typed list |
| 05 | `act_variables` | docs: act variables | `%pin%` everywhere, the PIN nowhere |
| 06 | `multi_tab` | docs: multiple pages | `context.new_page`; two extracts in parallel, each with its own LLM calls |
| 07 | `screenshot_files` | cookbooks/files-to-bucket | the `page.screenshot` span shows the image |
| 08 | `approval_flow` | cookbooks/approve-form-submission | observe -> approve -> act on the observed action (no LLM call) |
| 09 | `extract_screenshot` | docs: extract `screenshot` | the extract's LLM prompt shows the page image |
| 10 | `second_model` | docs: model configuration | a session on a second model |
| 11 | `failed_act` | docs: timeouts and errors | `locator.click` error span with `Type: message`; the impossible act reports failure |
| 12 | `saucedemo_checkout` | showcase/qa-checkout-flow | a dozen acts with screenshots; the final extract confirms the order |
| 13 | `pricing_monitor` | showcase/saas-pricing-monitor | the extract lists the plans with prices |
| 14 | `parallel_sessions` | docs: multiple instances | 2 traces, each with its own thread id |
| 15 | `system_prompt` | docs: `system_prompt` | the session input and every LLM system message carry it |

Not mirrored: Browserbase-only examples (session recordings, the hosted API,
showcase flows that need a Browserbase account) and the agent integrations
(Stagehand as tools for other agent frameworks), which are traced by those
frameworks' own Respan instrumentations.
