# @respan/instrumentation-stagehand

Respan tracing for [Stagehand](https://github.com/browserbase/stagehand) v4
(`@browserbasehq/stagehand`) browser agents.

```bash
npm install @respan/respan @respan/instrumentation-stagehand @browserbasehq/stagehand
```

```ts
import { localBrowser, Stagehand } from "@browserbasehq/stagehand";
import { Respan } from "@respan/respan";
import { StagehandInstrumentor } from "@respan/instrumentation-stagehand";

const respan = new Respan({ instrumentations: [new StagehandInstrumentor()] }); // RESPAN_API_KEY
await respan.initialize();

const browser = await localBrowser.launch();
const stagehand = await Stagehand.create({
  browser,
  model: { modelName: "openai/gpt-5.4-mini", apiKey: process.env.OPENAI_API_KEY },
});
const [page] = await browser.context.pages();
await page.goto("https://news.ycombinator.com");
await stagehand.act("click the 'new' link");
await stagehand.close();
await browser.close();
await respan.shutdown(); // short scripts: send spans before the process exits
```

Requires Stagehand **4.2** or later (v4's `Stagehand.create` API) and Node.js 22.18+.
Stagehand v3 (`new Stagehand()` + `init()`) is not covered.

## What is traced

```
stagehand.session        agent   Stagehand.create() ... close(): settings in; operations, failures, token totals out
├─ stagehand.act         task    instruction, options (variable names only), result, cache status, screenshot after the act
│  └─ stagehand.llm      chat    every inference: prompt, completion, model, tokens
├─ stagehand.extract     task    instruction, JSON schema, extracted data
├─ stagehand.observe     task    instruction, suggested actions with their selectors
└─ page.goto, locator.fill, page.screenshot, context.newPage, ...   tool
```

- **One trace per session.** Every span of a Stagehand instance shares its trace,
  and its thread, session and trace-group identifiers default to the session id
  (the Browserbase session id when there is one). Values set with
  `respan.propagateAttributes({ thread_identifier, group_identifier, ... })` win.
  `customer_identifier` and the other propagated fields reach every span.
  Customer metadata is exported as one flat `respan.metadata.<key>` attribute
  per key, the shape Respan ingestion reads.
- **LLM calls.** Stagehand v4 runs its inference inside its Chrome extension.
  - With a **client model** (`model: { generate }`) every inference comes back
    to your process, so each one is a full `chat` span: messages (with images),
    response format or tools, completion, and tokens. Pass `clientModelName` so
    Respan knows which model it was and can price it. A client model is also how
    you route Stagehand through the Respan gateway or any OpenAI-compatible
    endpoint, since v4's built-in models take no base URL.
  - With a **built-in model** (`model: { modelName, apiKey }`), Stagehand only
    reports each operation's token totals. Each operation then gets one
    content-free `chat` span with those tokens and the model name, so usage and
    cost still show up.
- **Concurrent operations.** Operations running at once on one session (for
  example `extract` on two pages) keep their own LLM calls. Stagehand gives the
  callback no operation id, so the prompt is matched against each running
  operation's instruction; two identical instructions running at the same time
  on one session cannot be told apart.
- **Screenshots.** After each `act`, the page is captured as a JPEG and attached
  to the act span's output. Respan stores it as log media and shows it in the log
  view. That is one extra screenshot round trip per `act` (bounded to a few
  seconds); `captureScreenshots: false` skips it. `page.screenshot()` calls show
  their image too.
- **Secrets.** Values passed as `variables` to `act` / `observe` are replaced by
  their `%name%` placeholder in every span of the session, including errors.
  Text typed with `locator.fill` / `locator.type` / `page.type` is exported as
  `[redacted]` when the target field is a credential (a password input, or a
  name, id or autocomplete that says password, token, one-time code, card number,
  and so on). The page is asked for the field's attributes, since selectors from
  `observe()` say nothing about the field. When the field cannot be identified,
  the text is redacted.
- **Errors.** Failed operations and browser actions are error spans with
  `Type: message`. An `act` that Stagehand reports as unsuccessful is an error
  span too.
- **No duplicate spans.** Stagehand's own OpenTelemetry RPC spans stay out of
  Respan exports, and provider instrumentations (for example the OpenAI one)
  are suppressed inside a client model's `generate`, since the `chat` span
  already carries that call.

## Options

| Option | Default | Effect |
|---|---|---|
| `captureContent` | `true` | `false` keeps span structure, timings, models, operation names and token counts but drops instructions, prompts, completions, results, page URLs, typed text and exception messages (only the exception type is kept). `RESPAN_TRACE_CONTENT=false` or `TRACELOOP_TRACE_CONTENT=false` do the same. |
| `captureScreenshots` | `true` | `false` skips the screenshot after each `act` and replaces images in prompts with a size placeholder. |
| `clientModelName` | none | The model a client model's `generate` calls, reported on its `chat` spans. |
| `stagehandModule` | imported | The `@browserbasehq/stagehand` module to patch, when your app loads a different copy than this package would import. |

## Not yet covered

- Session replay (a browser recording synced to the timeline). See `docs/GAPS.md`.
- Stagehand v3 and the Browserbase-hosted API mode.
- `experimentalBatch()` callbacks run inside the extension and are not traced
  beyond their own operations.
