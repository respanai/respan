# Respan pi Instrumentation

Respan instrumentation plugin and pi extension for the
[pi coding agent](https://pi.dev) (`@earendil-works/pi-coding-agent`).

It traces every pi agent run — the user prompt, each LLM call with prompts,
completions, token/cache/cost usage and time-to-first-token, each tool
execution, compactions and branch summaries — as Respan-compatible OTEL spans,
for both the interactive `pi` CLI (as a pi package) and the pi SDK
(`createAgentSession`).

```ts
import { Respan } from "@respan/respan";
import { PiInstrumentor } from "@respan/instrumentation-pi";

const instrumentor = new PiInstrumentor();
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [instrumentor],
});
await respan.initialize();
```

The published Pi SDK hands lifecycle events to
extensions (`pi.on(...)`) and to SDK subscribers (`session.subscribe(...)`).
`PiInstrumentor` exposes both as adapters over one translator
(`PiSessionTracer`), so the trace shape is identical whichever way you wire it.

Supported published SDKs start at `@earendil-works/pi-coding-agent` **0.74.0**, the first release under this namespace; the current validation target is **1.1.0**. Released companions start at `@respan/respan-sdk` **1.2.0** (naming hints) and `@respan/tracing` **1.3.0** (OpenTelemetry 2 readable-span APIs and flush).

The standalone official `@earendil-works/pi-telemetry` 1.1.0 package provides a vendor-neutral contract, in-memory reference and conformance tests. It supplies no OpenTelemetry exporter, and the published agent-core/coding-agent 1.1.0 packages do not expose its `telemetryContext` plumbing. This integration therefore uses the published native subscriptions and extensions.

Content capture observes canonical `CONTEXT_KEY_ALLOW_TRACE_CONTENT` from `@traceloop/ai-semantic-conventions`, parent/readable `allow_trace_content=false`, general OpenTelemetry suppression, and the environment upper bounds `RESPAN_TRACE_CONTENT=false` and `TRACELOOP_TRACE_CONTENT=false`. An observed veto stays false through later changes, processor replacement, queueing, deactivation and reactivation. Denied spans retain safe model, usage and native identity fields while omitting payloads, free-form metadata, events and status messages. Customer getters, serializers, iterators and proxy traps are not used to copy telemetry data.

Every owned span consults the configured OpenTelemetry SDK sampler before capture. Deterministic session trace IDs require the SDK's sampler boundary rather than creating a second native probe span. The integration recognizes the installed OpenTelemetry 2.10 and 2.12 tracer identities; an unknown provider/private ABI fails closed. Initialize a supported OpenTelemetry provider before activating the instrumentor.

For extensions, a shared observer on the official `ExtensionRunner.createContext` records exact SDK getter identities and returns the native context unchanged. The last deactivation restores its method if it still owns the patch. This allows native lazy contexts while leaving customer-defined accessors passive.

Metadata is exported in the canonical JSON attribute `respan.metadata`. References such as `respan.metadata.turn_number` below describe fields inside that object, not separate dotted attributes.

## Trace shape

By default one trace per **Pi session**; each agent run adds a root to that trace. Each
run is one root agent span named `<agentName>.turn-<n>.agent` — n is the
prompt's 1-based number within the pi session, like Braintrust's "Turn n" —
and displayed as `agent.turn-<n>`. All runs of a pi session share the same
thread, session and trace-group identifier (the pi session id); with
`traceScope: "session"` they also share one trace — see
[One trace per run vs one trace per session](#one-trace-per-run-vs-one-trace-per-session).

```text
pi.turn-1.agent (agent)                         one per agent run, shown as agent.turn-1
├── pi.chat (chat)                              one per assistant message
│     prompts, completion, tool_calls, usage, TTFT, cost
├── bash.tool (tool)                            one per tool execution
├── read.tool (tool)                            skill usage detected from SKILL.md
├── pi.turn-2.steer (task)                      a user message steered into the running turn, shown as task
├── pi.chat (chat)
└── pi.compaction (task)                        when compaction happens mid-run

pi.compaction (task)                            root of its own trace when idle
pi.branch_summary (task)                        summarized /tree navigation
```

Chat and tool spans are emitted the moment they complete, so an hour-long run
streams into the dashboard while it is still running; the turn's agent span
arrives when the run ends. The turn number is read from the session history
(the user messages already on the session branch), so it survives a resume in
a new process; without a session manager the tracer counts runs itself.

### Steering

When a message is delivered into a run that is already working — pi's
`session.steer()` / `followUp()`, or typing while the agent streams — pi
injects it at the next tool boundary and the run carries on. The message gets
its own span under the turn, displayed as `task` (legacy name: `<agent>.turn-<n>.steer`), at the moment it was delivered, so it is
visible in the tree instead of only as the last input item of the next LLM
call. It is also appended to the turn span's input, and the turn span carries
`respan.metadata.steer_count`. The first LLM call that sees the steered
message is flagged with `respan.metadata.after_steer = "true"` (and
`steer_turn_numbers`), so filtering on `after_steer` lists exactly the model
calls that reacted to a steer.

Turn numbers count the session's user messages, so a steer takes the next
number: a `turn-1` run that received three steers (`turn-2`, `turn-3`,
`turn-4`) is followed by `agent.turn-5`.

## Install for the pi CLI

### With the Respan CLI

```bash
respan auth login          # or: export RESPAN_API_KEY=...
respan integrate pi        # installs the pi package and writes ~/.pi/agent/respan.json
pi
```

`respan integrate pi --local` installs into the current project instead
(`pi install -l`, config in `.pi/respan.json`). `--trace-scope session` writes
`"trace_scope": "session"` for long-lived, resumed sessions. `--disable` turns
tracing off without uninstalling; `--dry-run` shows what would happen.

### Manually

```bash
pi install npm:@respan/instrumentation-pi     # global (or `pi install -l npm:...` for the project)
export RESPAN_API_KEY=...                     # or `respan auth login`
pi
```

To load it for a single run without installing:

```bash
pi -e ./node_modules/@respan/instrumentation-pi/dist/extension.js
```

The package declares `"pi": { "extensions": ["./dist/extension.js"] }`, so pi
loads the extension automatically after `pi install`. In the interactive TUI
the footer shows `Respan: tracing` while active, `Respan: tracing off (run
\`respan integrate pi\`)`when no API key was found, or`Respan: tracing
unavailable: <reason>` if initialization failed. pi keeps running in all three
cases. After each prompt a widget below the editor shows a link to its trace on the platform (Respan cloud only).

### Configuration

Configuration is resolved once per pi process, in this order (later wins):

| Priority    | Source                    | Notes                                                                      |
| ----------- | ------------------------- | -------------------------------------------------------------------------- |
| 1 (lowest)  | defaults                  | tracing on when an API key resolves, `workflowName` `pi`, `agentName` `pi` |
| 2           | `~/.pi/agent/respan.json` | global, non-secret settings                                                |
| 3           | `<cwd>/.pi/respan.json`   | project settings, merged over global                                       |
| 4 (highest) | environment variables     | see below                                                                  |

The API key is never stored in these files. It comes from `RESPAN_API_KEY`, or
— when that is unset — from `~/.respan/credentials.json` (written by
`respan auth login`) using the profile in `~/.respan/config.json`
(`activeProfile`, default `default`). The credential's `baseUrl` is used when
no base URL is configured. Base URLs are normalized to end with `/api`.

`respan.json` keys (all optional, snake_case):

| Key             | Type                   | Meaning                                                                                                                             |
| --------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`       | boolean                | Turn tracing on/off (`respan integrate pi --disable` sets `false`)                                                                  |
| `base_url`      | string                 | Respan API base URL (self-hosted / EU)                                                                                              |
| `workflow_name` | string                 | `traceloop.workflow.name` on every turn span (`span_name` is accepted as an alias)                                                  |
| `agent_name`    | string                 | Agent name: turn spans are named `<agent_name>.turn-<n>.agent` and carry `respan.metadata.agent_name`                               |
| `trace_scope`   | `"session"` \| `"run"` | One multi-root trace per pi session (default) or one trace per agent run — see [below](#one-trace-per-run-vs-one-trace-per-session) |
| `customer_id`   | string                 | `respan.customer_params.customer_identifier` on every span                                                                          |
| `project_id`    | string                 | Recorded as `respan.metadata.project_id`                                                                                            |
| `metadata`      | object of strings      | Fields in the `respan.metadata` JSON object on every span                                                                           |

Example `~/.pi/agent/respan.json`:

```json
{
  "enabled": true,
  "workflow_name": "pi",
  "customer_id": "team-platform",
  "metadata": { "env": "dev" }
}
```

Environment variables:

| Variable                | Meaning                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| `RESPAN_API_KEY`        | API key (highest priority)                                                                        |
| `RESPAN_BASE_URL`       | API base URL                                                                                      |
| `RESPAN_PI_TRACING`     | `0`/`false`/`off`/`no` disables, `1`/`true`/`on`/`yes` enables (overrides `enabled`)              |
| `RESPAN_CUSTOMER_ID`    | Customer identifier                                                                               |
| `RESPAN_PROJECT_ID`     | Project id (→ `respan.metadata.project_id`)                                                       |
| `RESPAN_PI_TRACE_SCOPE` | `run` or `session` (overrides `trace_scope`)                                                      |
| `RESPAN_PI_DEBUG`       | Print diagnostics to stderr (and stop filtering the tracing library's `[Respan …]` console lines) |

## SDK usage

Install the package next to pi in your project:

```bash
npm install @respan/respan @respan/instrumentation-pi
```

Passing an explicit `instrumentations` list to `Respan` disables Traceloop
auto-instrumentation inside the pi process, so LLM calls are not traced twice.

### As an inline extension (`extensionFactories`)

```ts
import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
} from "@earendil-works/pi-coding-agent";
import { Respan } from "@respan/respan";
import { PiInstrumentor } from "@respan/instrumentation-pi";

const instrumentor = new PiInstrumentor({ workflowName: "mail-agent" });
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [instrumentor],
});
await respan.initialize();

const loader = new DefaultResourceLoader({
  cwd: process.cwd(), // required by pi
  agentDir: getAgentDir(), // required by pi (~/.pi/agent by default)
  extensionFactories: [instrumentor.extension],
});
await loader.reload();

const { session } = await createAgentSession({ resourceLoader: loader });
await session.prompt("Summarize the latest email in the thread");
await respan.flush();
```

Every session created through the loader gets its own tracer; nothing is shared
between sessions except the OTEL pipeline.

### By attaching to a session (`attach`)

```ts
const { session } = await createAgentSession();
const detach = instrumentor.attach(session, {
  threadIdentifier: emailChainId, // default: the pi session id
  customerIdentifier: mailboxOwnerId,
  metadata: { mailbox: "support" },
});

await session.prompt("Draft a reply");
detach(); // unsubscribes and drops the tracer (closes an interrupted run as an error)
```

`attach()` returns a detach function and works for any number of concurrent
sessions in one process (`instrumentor.activeSessionCount` reports the live
tracers). Sessions are held weakly, so one that is disposed without `detach()`
is not retained — but call `detach()` to close an interrupted run promptly.
`respan.shutdown()` closes every open run (its turn span is emitted with an
error status), detaches everything and flushes.

### Overriding correlation per prompt

Either of the adapters honors `respan.propagateAttributes()` around
`session.prompt()`:

```ts
await respan.propagateAttributes(
  {
    thread_identifier: emailChainId,
    customer_identifier: mailboxOwnerId,
    metadata: { mailbox: "support" },
  },
  () => session.prompt("Draft a reply"),
);
```

Precedence for `respan.threads.thread_identifier` /
`respan.customer_params.customer_identifier`: explicit `attach()` overrides or
`PiInstrumentor` options → `propagateAttributes()` → the pi session id.
`respan.sessions.session_identifier` is always the pi session id.

## Options

`new PiInstrumentor(options)` (also `createPiExtension(options)` for an
already-active factory):

| Option                | Default         | Meaning                                                                                                                                                                                                                                                                                               |
| --------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `traceContent`        | `true`          | Payload capture; context, parent, environment and sampling vetoes remain upper bounds                                                                                                                                                                                                                 |
| `promptCapture`       | `"full"`        | `"full"` records the whole context the model saw on every chat span; `"delta"` (opt-in) records on each chat span only the messages appended since the previous LLM call of the run (first call: from the run's user prompt onward) — see [Volume](#scale-long-running-sessions-and-resumed-sessions) |
| `captureSystemPrompt` | `true`          | Capture system instructions on each full-context chat span, including native transcript sections. Delta capture adds the system prompt on the first chat span.                                                                                                                                        |
| `captureReasoning`    | `true`          | Record assistant thinking blocks in the chat span output (`reasoning`)                                                                                                                                                                                                                                |
| `captureToolSpans`    | `true`          | Emit one tool span per tool execution                                                                                                                                                                                                                                                                 |
| `maxContentChars`     | `0` (unlimited) | Optional per-string cap for every captured prompt, completion, tool argument/output and reasoning. Nothing is truncated by default; when set, truncated strings end with ` …[truncated N chars]` and the span gets `respan.metadata.truncated = true`                                                 |
| `workflowName`        | `"pi"`          | `traceloop.workflow.name` on every turn span                                                                                                                                                                                                                                                          |
| `agentName`           | `"pi"`          | Agent name: the turn span is `<agentName>.turn-<n>.agent` (displayed as `agent.turn-<n>`) and carries `respan.metadata.agent_name`                                                                                                                                                                    |
| `traceScope`          | `"session"`     | `"run"` = one trace per agent run; `"session"` = one multi-root trace per pi session, trace id derived from the session id (see [below](#one-trace-per-run-vs-one-trace-per-session))                                                                                                                 |
| `customerIdentifier`  | —               | `respan.customer_params.customer_identifier` on every span                                                                                                                                                                                                                                            |
| `metadata`            | —               | Fields in the `respan.metadata` JSON object on every span (string/number/boolean values)                                                                                                                                                                                                              |

`PiSessionTracer` (exported for advanced use) additionally accepts
`threadIdentifier`, an `emit` sink and an `enabled` predicate.
`sessionTraceId(sessionId)` (exported) returns the trace id a session gets in
`"session"` scope.

## Captured data

| Span                           | Attributes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| all                            | `respan.entity.log_type`, `traceloop.entity.name/path`, `respan.threads.thread_identifier`, `respan.sessions.session_identifier`, `respan.trace.trace_group_identifier` (all three = the pi session id), `respan.customer_params.customer_identifier`, `respan.metadata` (JSON), `telemetry.sdk.name/version`, OTel ERROR status and `error.message` on failures                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `<agent>.turn-<n>.agent`       | `traceloop.workflow.name`, input `[{role: "user", content: prompt}, ...steered messages]`, output = final assistant text, `respan.metadata.steer_count` when the run was steered, `respan.metadata.agent_name`, `respan.metadata.turn_number`, `respan.metadata.{pi_version, thinking_level, session_file, cwd, turn_count, tool_call_count, stop_reason, continuation}`. A structural span: no `gen_ai.request.model` (the model is on the chat spans)                                                                                                                                                                                                                                                                                                                                                                                         |
| `pi.chat`                      | `gen_ai.system` (provider), `llm.request.type = chat`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.prompt.N.role/content/tool_calls`, `gen_ai.completion.0.role/content/tool_calls`, `traceloop.entity.input/output` (output carries `reasoning`), `llm.request.functions` (tool definitions, capped at `maxContentChars`), `gen_ai.usage.input_tokens` / `gen_ai.usage.prompt_tokens` (= input + cacheRead + cacheWrite), `gen_ai.usage.output_tokens` / `gen_ai.usage.completion_tokens`, `llm.usage.total_tokens`, `gen_ai.usage.cache_read.input_tokens` / `llm.usage.cache_read_input_tokens`, `gen_ai.usage.cache_creation.input_tokens`, `respan.metadata.{reasoning_tokens, estimated_cost_usd, time_to_first_token_ms, stop_reason, response_id, turn_index, thinking_level, api, prompt_capture, prompt_message_offset}` |
| `<tool>.tool`                  | `traceloop.entity.input` = `{name, arguments}`, `traceloop.entity.output` = the complete native result JSON (`content`, `details`, `structuredContent`, and any native usage/outcome fields), `gen_ai.tool.call.id`, `respan.metadata.skill_name` when a `SKILL.md` is read or the `skill` tool is used                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `<agent>.turn-<n>.steer`       | input `[{role: "user", content: steered message}]`, `respan.metadata.turn_number` (the steer's own number), `respan.metadata.steered_into_turn`, `respan.metadata.steer_index`, `respan.metadata.delivered_after` (`tool_results` \| `assistant_reply` \| `run_start`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `pi.compaction`                | input `{reason, willRetry, tokensBefore}`, output `{summary, tokensBefore, tokensAfter, firstKeptEntryId}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `pi.branch_summary`            | input `{targetId, oldLeafId, label}`, output `{summary, label, id, fromId, ...}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `<agent>.turn-<n>.agent` (git) | `respan.metadata.git_repository` (remote URL, credentials stripped), `respan.metadata.git_branch`, `respan.metadata.git_commit` of the working directory, cached per directory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

Errors: an assistant message with `stopReason` `error`/`aborted` marks its chat
span (and the run) with OTel ERROR status and `error.message`; a tool result
with `isError` marks the tool span; a session shut down mid-run closes the run
with `Session shut down before the agent run completed` and any in-flight LLM
call / tool with `Interrupted before completion`. In delta mode
`respan.metadata.prompt_message_offset` is the index of the first captured
message in the full context, so readers know earlier messages were elided.
Images are never captured (rendered as `[image:<mimeType>]`).

## Scale, long-running sessions, and resumed sessions

This package was designed for fleets of unattended pi agents (thousands of
wakes per day, runs from a few turns to well over an hour, sessions that are
resumed after days of silence).

- **One trace per wake.** A run starts at the user prompt and ends at
  `agent_end`. No span is ever held open across idle time: a week-long pause
  between two prompts of the same session costs nothing, and resuming the
  session (same pi session file) simply adds new traces to the same thread —
  or new roots to the same trace with `traceScope: "session"` (see the next
  section).
- **Correlation across pauses.** Every span carries
  `respan.threads.thread_identifier = respan.sessions.session_identifier = <pi session id>`,
  which pi persists in the session file, so all wakes of one conversation
  (say, one email chain) line up in one thread. Override the thread per
  session with `attach(session, { threadIdentifier })` or per prompt with
  `respan.propagateAttributes()`.
- **Nothing is written to disk.** Spans go straight to the in-memory OTLP batch
  exporter of `@respan/tracing` (exported every few seconds, retried by the
  exporter). There is no journal, no log file; diagnostics go to stderr only
  when `RESPAN_PI_DEBUG` is set.
- **Bounded memory.** Per-run state (the pending LLM call's converted input
  delta, in-flight tool calls, counters) is discarded when the run closes; the
  tracer never retains message history beyond what the current LLM call needs.
  Extension tracers are held weakly and dropped on `session_shutdown`;
  attached sessions are held weakly too, and `attach()` returns a detach
  function.
- **Nothing is dropped by default.** Every chat span records the whole
  context the model saw (`promptCapture: "full"`), the system prompt is
  recorded with each full-context chat (`captureSystemPrompt: true`), and no
  string is truncated (`maxContentChars: 0`). Size is handled at ingest by the
  Respan backend, which tiers large span bodies to object storage. Note that
  full capture is quadratic in turn count: a run with N LLM calls re-sends the
  growing conversation N times (the tool catalog in `llm.request.functions` is
  also re-sent on every chat span).
- **Volume controls (opt-in).** For very high-volume deployments — many
  wakes per day with hour-long runs — set `promptCapture: "delta"` so each
  chat span records only the messages appended since the previous LLM call of
  the run (volume becomes linear; the full conversation is still
  reconstructable from the trace, and `respan.metadata.prompt_message_offset`
  says where each delta starts), and/or `maxContentChars` to cap individual
  strings, `captureReasoning: false`, or `captureToolSpans: false`.
- **Streaming.** Chat and tool spans are emitted as soon as they complete, so
  long runs are visible while they run; the turn span closes the run.

## One trace per run vs one trace per session

`traceScope` decides what one Respan _trace_ is for a pi session:

|                                                                           | `"session"` (default)                                                                                                                                | `"run"`                                                                            |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Trace                                                                     | one per pi session, shared by every run                                                                                                              | one per agent run (user prompt → `agent_end`)                                      |
| Trace id                                                                  | derived from the pi session id — the dash-stripped UUID, or a SHA-256 prefix for non-UUID ids (`sessionTraceId(id)`); an active OTEL span is ignored | random, or the active OTEL span's trace when there is one (the run nests under it) |
| Root spans                                                                | one turn span per run; all of them are roots of the same trace (Session > turn-1, turn-2, …)                                                         | one turn span (`pi.turn-<n>.agent`) per trace                                      |
| Compaction / branch summary outside a run                                 | another root of the session trace                                                                                                                    | root of its own trace                                                              |
| `respan.threads.thread_identifier` / `respan.sessions.session_identifier` | pi session id                                                                                                                                        | pi session id                                                                      |

In session scope the Respan backend groups the roots under a synthetic
"Trace Root" node ordered by start time, so the run before a pause appears
before the run after it; the trace's duration spans from the first wake to the
last, and the trace list shows the latest root's name/input/output with cost
and tokens summed over all runs. Nothing changes about _when_ spans are
emitted: every run still emits its own spans as it goes, and no span is held
open across a pause. Turn span ids are random, so runs of one
session emitted from different processes never collide inside the shared
trace.

`"session"` is the default: a pi session — one email chain that wakes up every
few days, from whichever process handles it — is one trace you can open as a
whole, the same shape as Braintrust's session trace. Pick `"run"` when each
prompt should be a self-contained trace with its own cost and latency (short
jobs, or when a run must nest under an outer OTEL trace); the Threads page
(thread id = session id) still shows the whole conversation either way.

Set it with the `traceScope` option, `"trace_scope": "session"` in
`respan.json` (`respan integrate pi --trace-scope session`), or
`RESPAN_PI_TRACE_SCOPE=session`.

## Constant ownership

pi's raw event and message field names stay inside this package. Shared
attributes come from their canonical sources:

- Traceloop keys (`traceloop.*`, `gen_ai.prompt.*`, `llm.*`) from
  `@traceloop/ai-semantic-conventions`.
- GenAI usage keys from `@opentelemetry/semantic-conventions/incubating`
  (`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.cache_creation.input_tokens`).
- Respan-owned keys (`respan.entity.*`, `respan.threads.*`,
  `respan.sessions.*`, `respan.metadata` (JSON), ...) from `@respan/respan-sdk`.
- Tool definitions use `llm.request.functions`; this-turn tool calls use
  `gen_ai.completion.0.tool_calls`; tool spans carry the call in
  `traceloop.entity.input`. Deprecated aliases (`respan.span.tools`,
  `tool_calls`, `model`, `prompt_tokens`, `traceloop.span.kind` on
  auto-emitted spans, ...) are not emitted.

## Fail-open

Tracing must never break the agent. Every handler is wrapped so it cannot
throw into pi; if the API key is missing, the network is down or
initialization fails, pi keeps running and the TUI footer shows why. Spans are
only emitted while the instrumentor is active (after `Respan.initialize()`);
state is tracked regardless so a late initialization does not corrupt runs.

Exports never sit on pi's event path either: pi awaits extension handlers
before it finishes a run, so the flush started after each run (and after
compactions / branch summaries) runs in the background — the batch exporter
ships spans on its own schedule anyway. The final flush and shutdown on
`session_shutdown` are awaited, because pi exits right after them on quit, but
for at most 3 seconds (`SHUTDOWN_FLUSH_TIMEOUT_MS`). A Respan outage therefore
delays neither a response nor `/quit`; at worst the last spans of a quit
during an outage are lost — see [Delivery guarantees](#delivery-guarantees)
for the Collector setup that removes that window. In the pi CLI the tracing library's `[Respan …]`
console lines are filtered out of stdout/stderr (pi's TUI and `--mode json`
own stdout; set `RESPAN_PI_DEBUG` to see them) — pi's own output and other
extensions' output are untouched.

## Delivery guarantees

What the SDK guarantees on its own, and how to get at-least-once delivery
across outages and restarts **without** anything piling up on the machine.

**In-process (this package + `@respan/tracing`), the OpenTelemetry standard
behaviour:**

- Spans are batched in memory by the OTel `BatchSpanProcessor` (default queue
  2048 spans, export every 5 s; tune with `OTEL_BSP_MAX_QUEUE_SIZE`,
  `OTEL_BSP_SCHEDULE_DELAY`, `OTEL_BSP_EXPORT_TIMEOUT`).
- The OTLP exporter retries 429/502/503/504 with exponential backoff (5
  attempts, 1 s → 5 s). Short outages are absorbed.
- `respan.flush()` / `respan.shutdown()` drain the queue; the pi extension
  entry does this on `session_shutdown`.
- **Nothing is written to disk.** A queue overflow, an outage longer than the
  retry window, or a process crash loses the spans that were still queued.
  This is the same trade-off Braintrust's, Langfuse's, Arize's and LangSmith's
  SDKs make; none of them persist locally.

**For long-running production agents where losing spans is not acceptable,
run a local OpenTelemetry Collector with a persistent queue.** This is the
OpenTelemetry-recommended pattern: the SDK exports to the Collector on
localhost (always reachable, sub-millisecond), and the Collector owns
durability — a write-ahead log on disk that is bounded by configuration,
retried indefinitely, and replayed after a restart.

Respan ships this as a packaged collector — see [`collector/`](../../../collector/)
in the monorepo (Docker Compose sidecar, or `respan collector start` from the
Respan CLI, which runs Docker or the pinned `otelcol-contrib` binary). Point the
SDK at it and nothing else changes:

```bash
export RESPAN_BASE_URL=http://127.0.0.1:4318   # the SDK posts to <base>/api/v2/traces
```

or, for the pi extension, `respan integrate pi --with-collector`.

What the packaged configuration does (all overridable by environment variables,
see the collector README):

- receives OTLP/HTTP on `127.0.0.1:4318` at `/api/v2/traces`, the path the SDK
  posts to;
- persists batches in a write-ahead log (`file_storage`) capped at 512 MiB
  (`RESPAN_COLLECTOR_MAX_SIZE_BYTES`) and 10 000 batches
  (`RESPAN_COLLECTOR_QUEUE_SIZE`); entries are deleted as soon as Respan
  acknowledges them, so the directory is nearly empty in normal operation;
- retries forever with backoff (`max_elapsed_time: 0`), exporter timeout 30 s,
  uncompressed JSON (Respan's ingest does not inflate gzip);
- exposes health on `:13133` and Prometheus metrics on `:8888`.

When both caps are exhausted the collector drops new batches and counts them in
`otelcol_exporter_enqueue_failed_spans`, which is the metric to alert on; watch
`otelcol_exporter_queue_size` and `otelcol_exporter_send_failed_spans` too.
Verified end to end against otelcol-contrib 0.160.0: spans sent during a 503
outage were persisted, survived `kill -9` of the collector, and were delivered
after restart.

**Re-delivery and duplicates.** Retries re-send whole batches, so a batch
that timed out after the server stored it arrives twice. Respan's trace view
shows one span per span id, but until ingest de-duplicates by span id,
duplicated batches can double-count cost and token aggregates for those runs.
Keep `timeout` generous (30 s) so this stays rare.

## Troubleshooting

- **No traces.** Run `RESPAN_PI_DEBUG=1 pi` — the extension logs (to stderr)
  which configuration sources it used, whether it initialized, and any flush
  errors. Check the footer status: `tracing off` means no API key resolved
  (`export RESPAN_API_KEY=...` or `respan auth login`); `tracing unavailable`
  includes the initialization error.
- **`pi: command not found`.** Install pi first:
  `npm install -g @earendil-works/pi-coding-agent` (see https://pi.dev), then
  `respan integrate pi`.
- **The extension does not load.** `pi install npm:@respan/instrumentation-pi`
  installs the package under `~/.pi/agent/npm`; run `pi` once and check the
  Extensions list in the startup banner. The current Pi 1.1.0 SDK requires Node >= 22.19; these examples and validation use Node 24.
- **Duplicate LLM spans.** Only one tracer should observe a session: either the
  installed pi package (CLI), `instrumentor.extension` in `extensionFactories`,
  or `instrumentor.attach(session)` — not two of them. Always pass an explicit
  `instrumentations` list to `Respan` inside the pi process.
- **Self-hosted / EU.** Set `base_url` in `respan.json` or `RESPAN_BASE_URL`;
  URLs are normalized to end with `/api`.
- **Volume is higher than expected.** Full context capture is quadratic in
  turn count. Switch to `promptCapture: "delta"`, set `maxContentChars`, or
  set `captureToolSpans: false` / `captureReasoning: false`.
