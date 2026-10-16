# Stagehand observability: coverage, parity with Laminar, open gaps

Goal: Respan traces for Stagehand v4 that match what
[Laminar's Stagehand integration](https://laminar.sh/docs/integrations/stagehand)
offers: `act` / `extract` / `observe` as spans under one run, full LLM calls,
token usage and cost per step, and a browser recording on the same timeline.

## How Laminar does it, and why this package differs

Laminar instruments Stagehand by patching `new Stagehand()` / `init()` and the v3
internals (`actHandler`, `extractHandler`, `llmClient.createChatCompletion`,
`agent()`). Stagehand v4 has none of those: it is created with
`Stagehand.create()`, and all browser and LLM work runs inside a Chrome
extension that the SDK drives over JSON-RPC. So this package wraps v4's public
seams instead:

| Seam | Span |
|---|---|
| `Stagehand.create()` ... `close()` | `stagehand.session` (agent) |
| `act` / `observe` / `extract` | `stagehand.<operation>` (task) |
| client model `generate()` (every `llm.generate` callback) | `stagehand.llm` (chat) |
| operation result `metadata.usage` (built-in models) | `stagehand.llm` (chat, usage only) |
| `Page` / `Locator` / `BrowserContext` actions | `page.goto`, `locator.fill`, ... (tool) |

## Coverage vs Laminar

| Laminar captures | Here | Notes |
|---|---|---|
| act / extract / observe nested under one run | yes | one trace per Stagehand session |
| full LLM calls (prompt, response, model) | yes, with a client model | built-in models run inside the extension; only their per-operation token totals reach the SDK (G1) |
| cost per step and per run | yes | from the chat spans; the session span sums Stagehand's token totals |
| latency | yes | every span |
| `agent()` | n/a in v4 | v4 has no `agent()`; agents are built on top (Vercel AI SDK, Mastra, Claude Agent SDK, ...) and traced by those frameworks' Respan instrumentations |
| browser session recording | no | a JPEG screenshot after each `act` instead (G2) |

## Open gaps

| # | Gap | Impact | Next step |
|---|---|---|---|
| G1 | Built-in models (`{ modelName, apiKey }`) call the provider from inside the extension. Stagehand tells the SDK only each operation's token totals, so those operations get one content-free chat span instead of one span per inference with prompts. | no prompts or completions with built-in models | use a client model for full capture; revisit if Stagehand exposes inference events with content |
| G2 | Session replay (a browser recording synced to the span timeline) is not captured. This is a known gap across Respan's browser-agent integrations. | no sub-step replay | the screenshot after each `act` is a flip-book replay meanwhile |
| G3 | The `llm.generate` callback carries no operation id or trace context. With several operations running at once on one session, each LLM call is assigned by matching the prompt against the running operations' instructions. Two operations with the same instruction running at the same time on one session can be swapped. | rare misattribution under concurrency | none in this package; needs an operation id in Stagehand's callback |
| G4 | Typed text: a credential field is detected from the page (input type, name, id, autocomplete, aria-label). A secret typed into a field that looks ordinary is exported. | possible secret exposure with unusual forms | pass secrets through `act` `variables`, or set `captureContent: false` |
| G5 | Images inside prompts (`extract({ screenshot: true })`) are exported as Stagehand sent them (PNG); act screenshots are JPEG. | larger exports on screenshot-heavy runs | `captureScreenshots: false` |
| G6 | A client model gives Stagehand no model name; `clientModelName` supplies it for pricing. Several client models in one process share that name. | cost on the wrong model if sessions mix client models | one instrumentor setting per process today |
| G7 | Stagehand v3 (`new Stagehand()` + `init()`), the Browserbase-hosted API mode and `experimentalBatch()` callbacks are not traced. | v3 users get no spans | add a v3 path if customers ask |
| G8 | A session span ends at `stagehand.close()`. If the process exits without it, the session span is never exported and its children have no root. | orphaned spans on crashes | always `close()` in a `finally` |

## Verification

- `tests/stagehand_baseline.test.mjs` (CI runs it): drives the real Stagehand
  4.2.1 SDK and its Chrome extension in headless local Chrome, against a local
  page, with a scripted client model, through Respan's real span processor and
  exporter. It checks on every exported span: customer metadata,
  `customer_identifier`, thread / session / trace group defaulting to the
  session with customer values winning, parallel sessions in separate traces,
  concurrent operations keeping their own LLM calls, content capture off,
  `variables` and credential-field text never exported, error spans, and that
  inside a Respan workflow Stagehand runs without a workflow entity path (so
  its native RPC spans cannot be exported). Each of those checks was confirmed
  to fail with the code it covers removed.
- `javascript-sdks/examples/stagehand`: 15 scenarios mirroring Stagehand's
  quickstart, cookbooks and showcase on a local browser, verified live against
  a Respan environment by reading every trace back through the logs API
  (`verify_live.py`): stored vs exported span counts, one root, one thread id
  equal to the session id, tokens and cost on every LLM span, each operation's
  LLM tokens equal to the usage Stagehand itself reports, no secrets.
