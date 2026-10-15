# browser-use observability: plan, parity with Laminar, open gaps

Goal: Respan traces for browser-use agents that match what
[Laminar's browser-use integration](https://laminar.sh/docs/integrations/browser-use#browser-use-observability)
offers: agent steps and a browser session recording, synced on one timeline.

## How Laminar does it

- browser-use ships `browser_use/observability.py`: `observe` / `observe_debug`
  decorators that emit spans **only when the `lmnr` package is installed**. So
  Laminar's step spans are built into browser-use itself (`agent.run`,
  `agent.step`, `act`, ...).
- `Laminar.initialize()` also instruments the browser (Playwright, or CDP for
  browser-use >= 0.6) and injects an rrweb recorder into each page. Recording
  chunks are shipped to Laminar's backend, and their UI plays the recording
  with the active step highlighted.

Respan can't use the `lmnr` hook (it imports `lmnr` by name), so this package
wraps the same seams directly.

## Phases

| Phase | What | Where | Status |
|---|---|---|---|
| 1 | Span tree: agent run -> step -> LLM call / browser action, tokens, cache tokens, judge verdict, action errors | SDK (this package) | **done**, soaked against staging |
| 1b | Per-step screenshot on the step span | SDK. No backend change: inline `data:` images >= 1000 chars are moved to log media storage by the CH writer (`utils/data_processing/media_processing.py`) and the log view renders `image_url` parts (`LogMessage.tsx` -> `ImageWithModal`) | **done** |
| 2 | Session replay: rrweb recording synced to step timeline | SDK + backend + frontend | **not started**, design below |
| 3 | TypeScript (`browser-use` has no maintained TS port worth tracing today; revisit with Stagehand) | JS SDK | not planned |

## Phase 2 design sketch (session replay)

1. **SDK capture.** On `BrowserSession` start, use the existing CDP session
   (`browser_session.cdp_client`) to `Page.addScriptToEvaluateOnNewDocument`
   the rrweb recorder plus a binding (`Runtime.addBinding`) that receives event
   batches. Tag each batch with the current trace id / step span id (read from
   the OTel context at flush time). Mask inputs by default (`maskAllInputs`).
2. **Transport.** Batched, gzipped NDJSON POSTs to a new endpoint, e.g.
   `POST /api/v1/traces/{trace_id}/recordings` (auth: same API key). Must fail
   open and never block the agent loop.
3. **Backend storage.** S3 under `recordings/{org}/{trace_id}/{seq}.ndjson.gz`
   (same lifecycle/retention as `logs/`), with an index row per chunk
   (trace_id, span_id, t_start, t_end, bytes). Size cap per trace, plan-gated.
4. **Frontend.** `rrweb-player` in the trace view, timeline seek driven by
   step span start times. The existing `VideoPlayer` has no timeline sync.

Cheaper interim option: Phase 1b already gives one screenshot per step, which
is a flip-book replay. Phase 2 is only worth it if customers need sub-step
detail (scrolls, typing, hovers).

## Open gaps

| # | Gap | Impact | Next step |
|---|---|---|---|
| G1 | Duplicate LLM rows when a provider instrumentation is also active. **Mitigated:** during a `browser_use.llm` span we set OTel's language-model suppression key (honoured by Respan's Anthropic instrumentation and OpenLLMetry ones) and enter Respan's OpenAI `suppress_openai_instrumentation()` (same pattern as the LiteLLM instrumentation). Verified with the real OpenAI instrumentation + a live gateway call: one span, not two. **Still open:** Respan's Google GenAI / Mistral instrumentations honour neither, so `ChatGoogle` + `GoogleGenAIInstrumentor` together would double-log. | duplicate rows / double cost for Google models only | have the remaining provider instrumentations read the OTel key (one-line check each) |
| G2 | `respan-tracing` silently drops any span without `traceloop.span.kind` / entity path / `llm.request.type`. That is why the first soak showed nothing. Every new instrumentation can hit it. | silent data loss | fixed here by setting the span kind; consider a debug warning in `is_processable_span` when an instrumentation scope's span is dropped |
| G3 | Screenshots are full-page PNG base64 (~50-300 KB each). They count toward the 20 MB per-log limit and go to S3 media storage. Step outputs carry one each; LLM prompts carry the one the model saw. | storage cost on long runs | if it becomes a problem: JPEG re-encode or downscale in the SDK, or `capture_screenshots=False` |
| G4 | Presigned media URLs only appear on the log detail view, not list views (backend behaviour). | screenshots not visible in list | frontend/backend, not SDK |
| G5 | `browser_use.beta` agent (separate `run`) is not wrapped. | beta users get LLM/action spans but no agent/step spans | wrap once the beta API settles |
| G6 | CI for Python packages only builds the wheel and smoke-imports it; it never runs package tests. These tests (including a real headless-Chrome agent run) were run locally. | regressions not caught in CI | add a pytest step to `ci.yml` for packages that have `tests/` |
| G7 | browser-use's built-in judge runs by default (`use_judge=True`) and makes one extra LLM call per run; it shows as a `browser_use.llm` span directly under the agent span. | expected, but customers may ask about the extra call | documented |
| G9 | `respan-tracing` names task spans by bare kind (only agent/tool/llm/handoff get a `.detail` suffix), so every step shows as `task` in the span list. The step number is in metadata (`browser_use.step.number`). | steps are hard to tell apart in the tree | exporter or frontend: show `traceloop.entity.name` / step number for task spans |
| G10 | **Off-scope, staging infra (root cause confirmed):** the presigned media URL for a lifted screenshot (`kwaistagingblob.blob.core.windows.net/files/media/...`) returns `AuthorizationFailure` from a public client inside its validity window. Cause: backend `iac/azure/storage.py:80-83` sets the AKS-staging storage account to `network_rule_set default_action=DENY` (VNet + private endpoint only). The backend hands browsers user-delegation SAS URLs (`clickhouse/views/logs.py` `_serve_log_media`), which can never load from outside the VNet. That affects **every** log-media image/audio on staging, not just screenshots. Prod uses S3 and is not affected. | screenshots (and all log media) do not render in the staging UI | infra decision: allow public network access with SAS-only auth (keep shared keys and anonymous access off), or proxy media through the API. Handed off, not changed here |
| G8 | Version pin `browser-use >=0.13,<0.14`. The hooks (`Agent.run/step`, `Tools.act`, `TokenCost.register_llm`) have been stable since 0.6, but private-ish seams can move. | breakage on 0.14 | widen the pin after testing each minor |

## Verification log

- Unit + e2e tests: `pytest tests` (9 tests; the e2e test drives a real headless
  Chrome with a scripted LLM and a local HTTP page).
- Staging soak (2026-10-15): real `azure/gpt-4.1-mini` through the staging
  gateway, real sites, spans exported to `staging-api.respan.ai`.
  - First soak exported nothing: agent/step/action spans lacked
    `traceloop.span.kind` and `respan-tracing` filtered them (G2). Fixed.
  - Hacker News task (trace `c1c401476a449e890637763422128b78`): 13 spans read
    back from the logs API: `tool.navigate` (browser-use's pre-step URL open),
    3 `task` steps each with `llm.azure_gpt-4.1-mini` + action
    (`tool.click`, `tool.extract`, `tool.done`), the extraction model's own LLM
    call nested inside `tool.extract`, the judge call under the agent, and the
    root `agent.browser_use`. Token counts and cost were computed server-side.
  - Step output's screenshot came back as a media-storage URL (the backend
    lifted the data URI), see G10 for the access error on staging.
