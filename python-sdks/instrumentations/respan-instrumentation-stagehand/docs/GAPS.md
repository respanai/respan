# Stagehand (Python) observability: coverage, parity with Laminar, open gaps

Goal: Respan traces for Stagehand v4 in Python that match what
[Laminar's Stagehand integration](https://laminar.sh/docs/integrations/stagehand)
offers: `act` / `extract` / `observe` as spans under one run, full LLM calls,
token usage and cost per step, and a browser recording on the same timeline.
Laminar's integration is TypeScript-only; this package mirrors Respan's own
TypeScript one (`@respan/instrumentation-stagehand`).

## What the Python package is

`stagehand` 4.x on PyPI is a full peer of the TypeScript SDK, not a client to a
hosted service: the wheel embeds the same Chrome extension
(`stagehand/_extension/service-worker.js`) and drives it over JSON-RPC on CDP.
`local_browser.launch()` runs everything on a local Chrome; no Browserbase
account or Stagehand API key is needed. Browser actions and built-in-model
inference run inside the extension; a client model's inference comes back to
the Python process as an `llm.generate` request.

Unlike the TypeScript SDK, the Python SDK opens no OpenTelemetry spans of its
own (it only injects `traceparent` into the extension's requests), so there is
nothing native to keep out of Respan exports.

## Seams

| Seam | Span |
|---|---|
| `Stagehand.create()` ... `close()` | `stagehand.session` (agent) |
| `act` / `observe` / `extract` | `stagehand.<operation>` (task) |
| client model callable (every `llm.generate` request) | `stagehand.llm` (chat) |
| operation result `metadata.usage` (built-in models) | `stagehand.llm` (chat, usage only) |
| `Page` / `Locator` / `BrowserContext` actions | `page.goto`, `locator.fill`, `context.new_page`, ... (tool) |

The `llm.generate` handler runs in Stagehand's RPC reader task, whose context
is the one `Stagehand.create()` ran in. Each call is therefore run in the
context captured when its operation started, so its span nests under the
operation and carries the attributes the caller propagated for it.

## Coverage vs Laminar

| Laminar captures (TS) | Here | Notes |
|---|---|---|
| act / extract / observe nested under one run | yes | one trace per Stagehand session |
| full LLM calls (prompt, response, model) | yes, with a client model | built-in models run inside the extension; only their per-operation token totals reach the SDK (G1) |
| cost per step and per run | yes | from the chat spans; the session span sums Stagehand's token totals |
| latency | yes | every span |
| `agent()` | n/a in v4 | v4 has no `agent()`; agents are built on top and traced by those frameworks' Respan instrumentations |
| browser session recording | no | a JPEG screenshot after each `act` instead (G2) |

## Open gaps

| # | Gap | Impact | Next step |
|---|---|---|---|
| G1 | Built-in models (`model="openai/..."`) call the provider from inside the extension. Stagehand tells the SDK only each operation's token totals, so those operations get one content-free chat span instead of one span per inference with prompts. | no prompts or completions with built-in models | use a client model for full capture; revisit if Stagehand exposes inference events with content |
| G2 | Session replay (a browser recording synced to the span timeline) is not captured. This is a known gap across Respan's browser-agent integrations. | no sub-step replay | the screenshot after each `act` is a flip-book replay meanwhile |
| G3 | The `llm.generate` request carries no operation id or trace context. With several operations running at once on one session, each LLM call is assigned by matching the prompt against the running operations' instructions. Two operations with the same instruction running at the same time on one session can be swapped. | rare misattribution under concurrency | none in this package; needs an operation id in Stagehand's request |
| G4 | Typed text: a credential field is detected from the page (input type, name, id, autocomplete, aria-label). A secret typed into a field that looks ordinary is exported. | possible secret exposure with unusual forms | pass secrets through `act` `variables`, or set `capture_content=False` |
| G5 | A client model gives Stagehand no model name; `with_model_name()` or `client_model_name` supplies it for pricing. Without either, chat spans have tokens but no model, so no cost. | no cost for unnamed client models | name the model |
| G6 | The pre-v4 Python client, the Browserbase-hosted API mode and `experimental_batch()` callbacks are not traced. | those users get no spans | add paths if customers ask |
| G7 | A session span ends at `stagehand.close()`. If the process exits without it, the session span is never exported and its children have no root. | orphaned spans on crashes | always `close()` in a `finally` |

## Verification

- `tests/test_baseline.py` (CI runs it): drives the real Stagehand 4.2.1 Python
  SDK and its Chrome extension in headless local Chrome, against a local page,
  with a scripted client model, through Respan's real span processor
  (`respan_tracing.testing.capture_exported_spans`). It checks on every
  exported span: customer metadata as flat `respan.metadata.<key>`,
  `customer_identifier`, thread / session / trace group defaulting to the
  session with customer values winning, parallel sessions in separate traces,
  concurrent operations keeping their own LLM calls, content capture off,
  `variables` and credential-field text never exported, error spans (including
  an act Stagehand reports as failed), provider instrumentation suppressed
  inside the client model, and the session nesting inside a Respan workflow.
  Each check was confirmed to fail with the code it covers removed.
  `tests/test_instrumentation.py` covers the built-in-model usage span and the
  message conversion.
- `python-sdks/examples/stagehand`: 15 scenarios mirroring Stagehand's
  quickstart, cookbooks and showcase on a local browser, with the model routed
  through the Respan gateway and Respan's OpenAI instrumentation active. They
  were run against a Respan environment and every trace read back through the
  logs API (`respan_tracing.testing.live` plus Stagehand checks): stored vs
  exported span counts, one root, one thread id equal to the session id,
  tokens and cost on every LLM span, each operation's LLM tokens equal to the
  usage Stagehand itself reports, no secrets.
