# @respan/instrumentation-eve

Respan instrumentation for [Eve](https://eve.dev), Vercel's agent framework.
It maps Eve turns and sessions into the canonical Respan span contract and
translates Eve's nested AI SDK 7 model and tool spans. The optional
`withEveLineage` helper groups delegated sessions under their root session and
records their immediate parent lineage.

The AI SDK translator is included in this package. It does not depend on
`@respan/instrumentation-vercel`, and it does not register another
`@ai-sdk/otel` integration. Eve owns that registration.

## Install

```bash
npm install @respan/respan @respan/instrumentation-eve eve@0.26.1 ai@7.0.26
```

Eve 0.26 requires Node.js 24 or newer and AI SDK 7. The initial supported
ranges are `eve >=0.26.0 <0.27.0` and `ai >=7.0.26 <8.0.0`.

## Configure Eve

Create `agent/instrumentation.ts`. Initialize Respan at module scope and await
it before exporting Eve's instrumentation definition:

```ts
import { Respan } from "@respan/respan";
import {
  EveInstrumentor,
  withEveLineage,
} from "@respan/instrumentation-eve";
import { defineInstrumentation } from "eve/instrumentation";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new EveInstrumentor()],
});

await respan.initialize();

export default withEveLineage(
  defineInstrumentation({
    recordInputs: false,
    recordOutputs: false,
  }),
);
```

Do not start Respan from Eve's synchronous `setup` callback. Awaiting
initialization at module scope ensures the translator is installed before Eve
creates its first turn span.

Eve detects `agent/instrumentation.ts` and enables its vendored AI SDK
OpenTelemetry integration. No separate `@vercel/otel` or
`@ai-sdk/otel` registration is needed.

Do not activate `VercelAIInstrumentor` alongside `EveInstrumentor`; this
package already handles Eve-translated AI SDK spans.

This configuration disables model input and output recording because conversations
can contain sensitive data. Set either option to `true` only after deciding
that exporting that content matches your privacy and retention requirements.

All user-authored non-`eve.*` values returned from `step.started` as
`runtimeContext` are exported under `respan.metadata.eve.runtime_context`, even
when `recordInputs` and `recordOutputs` are `false`; do not put secrets in that
runtime context.

## Delegated session lineage

Eve exposes parent lineage to `events["step.started"]`, but does not put that
lineage on its raw OpenTelemetry spans. `withEveLineage` composes that event
hook, mirrors the root lineage onto the active `ai.eve.turn`, and returns it as
runtime context for the nested AI SDK spans.

If you already have a `step.started` hook, keep it inside
`defineInstrumentation`; the helper calls it once and preserves its runtime
context:

```ts
export default withEveLineage(
  defineInstrumentation({
    events: {
      "step.started"(input) {
        return {
          runtimeContext: {
            "app.tenant_id":
              input.session.auth.current?.attributes.tenantId ?? "",
          },
        };
      },
    },
    recordInputs: false,
    recordOutputs: false,
  }),
);
```

Eve reserves authored `eve.*` runtime-context keys and discards them. The
helper therefore uses the private `__respan_eve` namespace; it owns and
overwrites that one key while preserving every other authored context key. The
private bridge attributes are consumed and stripped before export.

With the helper enabled, the current Eve session remains the Respan session and
thread identifier, while the root session becomes the trace-group identifier.
For delegated sessions, `respan.metadata.eve.parent` contains the immediate
parent session id, tool call id, and turn id/sequence. These are workflow
lineage values, not OpenTelemetry span ids, so they are not written to
`respan.entity.log_parent_id`.

Eve 0.26 starts the delegated child session in a fresh OpenTelemetry trace.
When the helper's exact parent-session and parent-turn lineage matches one
caller, semantic export moves the complete child agent subtree into the caller
trace. The later caller-side `invoke_agent` usage event contains no unique
content and duplicates the child model usage, so it is suppressed only after
that exact correlation succeeds. If lineage is missing or ambiguous, the
translator leaves the original traces and usage event intact rather than
assigning them to the wrong caller.

Without `withEveLineage`, each Eve session falls back to its own session id as
the trace-group identifier.

## What is captured

- Each `ai.eve.turn` span as an agent execution
- Eve's instrumentation `functionId` as the workflow name on every nested span
- Current Eve session IDs as Respan session and thread identifiers
- Root-session trace grouping and immediate parent metadata when
  `withEveLineage` is enabled
- Eve version, environment, turn, step, and channel data in JSON metadata
- Delegated child agent, task, and model spans nested under the caller trace
- AI SDK model input/output, provider, model, streaming state, and token usage
- Tool execution input/output
- Embedding spans, model details, and token usage; input and vectors only when
  AI SDK telemetry emits those supplemental attributes

In semantic span-name mode, Eve's AI SDK `invoke_agent <model>` transport
wrappers are omitted. Their task, model, and tool children are reparented to
the real Eve turn, so the visible tree does not contain a second agent for
every model step. Legacy mode preserves Eve's original emitted tree.

Eve's `$eve.*` Workflow run tags are not OpenTelemetry attributes and are
therefore outside this span processor's input.

## License

Apache-2.0
