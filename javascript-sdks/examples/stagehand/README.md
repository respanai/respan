# Stagehand + Respan

Trace [Stagehand](https://github.com/browserbase/stagehand) v4 browser agents in
Respan with `@respan/instrumentation-stagehand`. Each Stagehand session becomes
one trace: the session, every `act` / `extract` / `observe` with its LLM calls
(tokens and cost), every browser action, and a screenshot after each `act`.

Everything runs on a local Chrome (no Browserbase account). The model calls go
through the Respan gateway via Stagehand v4's client-model callback
(`model: { generate }`, see `respanModel()` in `harness.ts`), so one Respan key
covers both traces and models.

## Setup

```bash
cd javascript-sdks/examples/stagehand
npm install                 # Node.js 22.18+; Google Chrome must be installed
cp .env.example .env        # then set RESPAN_API_KEY
```

`.npmrc` sets `install-links=true`, so the instrumentation is installed from this
repo as a real package. After changing it, rebuild it (`npx tsc` in its
directory) and run `npm install` here again. Outside this repo:
`npm install @respan/instrumentation-stagehand`.

## Run

```bash
npx tsx hello_world.ts                       # one session: act + extract on Hacker News
npx tsx run_all.ts --list                    # all scenarios
npx tsx run_all.ts                           # run them all (add --headless to hide Chrome)
npx tsx run_all.ts persisted_login multi_tab # some of them
```

`run_all.ts` writes `runs/<run id>.json`. Then check every trace in Respan:

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python verify_live.py runs/<run id>.json
```

Every scenario runs under customer `stagehand-scenarios` with
`metadata.scenario=<name>` and `metadata.run_id=<run>`. The verifier reads each
trace back from the Respan logs API and prints PASS/FAIL per scenario with links.
Generic checks come from `respan_tracing.testing.live`: every exported span is
stored, one root span, customer id and metadata on every span, one thread id,
tokens and cost on every LLM span, no secrets. The Stagehand checks: one session
span per trace, the thread id equals the session id, the session's operation
count matches its task spans, and each operation's LLM spans add up to the
token usage Stagehand itself reports for it.

| # | Scenario | Mirrors Stagehand's | What to look for in Respan |
|---|---|---|---|
| 01 | `hello` | README quickstart | session -> act / extract, each with its LLM calls; a screenshot on the act |
| 02 | `persisted_login` | cookbooks/persisted-login | 2 traces on one profile; observe returns selectors; the password `locator.fill` shows `[redacted]`, the email does not |
| 03 | `paginated_catalog` | cookbooks/paginated-catalog | 3 extracts + 2 acts in order, 4 products per page |
| 04 | `extract_schema` | showcase (extract with a schema) | the extract's input shows the JSON schema; output is the typed list |
| 05 | `act_variables` | docs: act variables | `%pin%` everywhere, the PIN nowhere |
| 06 | `multi_tab` | docs: multiple pages | `context.newPage`; two extracts in parallel, each with its own LLM calls |
| 07 | `screenshot_files` | cookbooks/files-to-bucket | the `page.screenshot` span shows the image |
| 08 | `approval_flow` | cookbooks/approve-form-submission | observe -> approve -> act on the observed action (no LLM call) |
| 09 | `extract_screenshot` | docs: extract `screenshot` | the extract's LLM prompt shows the page image |
| 10 | `second_model` | docs: model configuration | a session on a second model |
| 11 | `failed_act` | docs: timeouts and errors | `locator.click` error span with `Type: message`; the impossible act reports failure |
| 12 | `saucedemo_checkout` | showcase/qa-checkout-flow | a dozen acts with screenshots; the final extract confirms the order |
| 13 | `pricing_monitor` | showcase/saas-pricing-monitor | the extract lists the plans with prices |
| 14 | `parallel_sessions` | docs: multiple instances | 2 traces, each with its own thread id |
| 15 | `system_prompt` | docs: `systemPrompt` | the session input and every LLM system message carry it |

Not mirrored: Browserbase-only examples (session recordings, the hosted API,
showcase flows that need a Browserbase account) and the agent integrations
(Stagehand as tools for Vercel AI SDK, Mastra, Claude Agent SDK, ...), which
are traced by those frameworks' own Respan instrumentations.
