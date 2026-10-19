# browser-use: coverage and known limitations

Respan traces for [browser-use](https://github.com/browser-use/browser-use)
agents, with parity to
[Laminar's browser-use integration](https://laminar.sh/docs/integrations/browser-use#browser-use-observability)
as the target.

## How it works

browser-use ships `browser_use/observability.py`, whose `observe` decorators emit
spans only when the `lmnr` (Laminar) package is installed. Respan cannot use that
hook, so this package wraps the same seams from outside: `Agent.run`,
`Agent.step`, `Agent.rerun_history`, `Tools.act` and browser-use's chat-model
layer. Nothing in browser-use is forked; spans are standard OpenTelemetry.

## Coverage

| | Laminar | Respan |
|---|---|---|
| Agent run, steps | yes | yes (`agent.browser_use`, `task`) |
| Every LLM call (any provider), tokens, cache tokens, cost | via provider SDK instrumentation | yes, at browser-use's own model layer |
| Each browser action | debug mode only | yes (`tool.<action>`) |
| Screenshot per step | via session recording | yes, JPEG on the step span |
| Judge verdict, errors with their message | partly | yes |
| `sensitive_data` kept out of traces | n/a | yes (placeholders, incl. URL-encoded values) |
| Session replay (rrweb) | yes | **preview** (`record_session=True`): link on every span, seeking to its moment |

## Known limitations

- **Session replay** is a preview (`record_session=True` / `RESPAN_RECORD_SESSION=1`).
  The recorder (`respan_tracing.utils.session_recording`, rrweb 2.0.0-alpha.4)
  is injected over browser-use's own CDP connection into every page and new tab,
  drained every 1.5 s, before each action and at each step end, and uploaded once
  when the run ends. What it does not do yet:
  - **One tab at a time.** rrweb replays one document, so the stream follows the
    agent's focused tab (switching at that tab's fresh snapshot); background tabs
    are recorded but not in the uploaded stream.
  - **Events in flight when a page navigates away are lost** (up to the time
    since the last drain, which runs right before every action, so in practice the
    tail of the action that navigated).
  - **Not embedded in the trace view yet**: every span links to the replay page at
    its own moment (`#t=<ms>`) and the page lists every span to jump between, but
    the platform does not show the player beside the span tree.
  - **Links are reserved before the run**: if the upload then fails, the spans'
    links point at a replay that does not exist (the local `.html` is still
    written).
  - **Canvas, video, WebGL, cross-origin iframes, closed shadow roots** do not
    replay faithfully (rrweb limits; canvas recording is off). Pages that block
    script injection (`chrome://`, the PDF viewer) record nothing.
  - **Size cap:** collection stops at 20 MB raw per run (the upload limit is 25 MB
    gzipped) and the replay ends there.
  - **Masking:** every input value is masked in the page; `sensitive_data` values
    are masked in page text in the page and scrubbed from the whole payload
    (attributes, URLs, CSS) before upload. Other on-page PII (names, emails shown
    as text) is recorded as displayed.
  - Images, fonts and cross-origin stylesheets rrweb cannot read are loaded from
    the original site at playback time, so they can change or go missing later.
- **Google GenAI / Mistral double logging:** if Respan's Google GenAI or Mistral
  instrumentation is also active, a `ChatGoogle` / Mistral call can be logged
  twice. OpenAI and Anthropic instrumentations are suppressed correctly.
- **`browser_use.beta` agents** get LLM and action spans but no agent/step spans.
- **The judge** (`use_judge=True`, browser-use's default) adds one LLM call per
  run, and it uses the primary model: it does not fall back to `fallback_llm`.
  Pass `judge_llm=` if your primary model can fail.
- **Step names:** every step shows as `task`; the step number is in metadata
  (`browser_use.step.number`).
- **Screenshot links** appear on the log detail view, not in list views.
- **Version range:** `browser-use >=0.13,<0.14`. The wrapped seams have been
  stable since 0.6; the range widens after each minor is tested.

## Verification

- `tests/test_baseline.py` is the wire contract. It runs a real headless Chrome
  agent through `Respan(...)` and asserts, on every exported span: flat
  `respan.metadata.<key>`, `customer_identifier`, thread / session / trace group
  defaulting to the Agent's session id (customer values win), follow-up runs in
  the same thread, parallel agents in separate traces, content off keeps ids and
  drops content, `sensitive_data` never exported, replays in one trace. CI runs it.
  Session replay: a recorded run uploads a full snapshot plus incremental events,
  typed values and `sensitive_data` never in the payload, the reserved link lands
  on every span with its own `#t` offset (children never before their parent) and
  a marker per span is uploaded, a backend without reserve still links the root
  span, the stream follows a new tab, an upload failure leaves the run
  intact (and a local player), and content off records nothing.
- `examples/browser-use/run_all.py` runs 27 scenarios mirroring browser-use's own
  examples against a live Respan environment and checks every trace it reads
  back: exported vs stored span count, one root, a tool span per executed action,
  an LLM span per step, step totals vs browser-use's own count, a judge verdict on
  finished runs, ids on every span, no secrets. 27/27 pass on the published 0.1.2.
