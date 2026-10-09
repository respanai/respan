# @respan/instrumentation-eve

Translate [Eve](https://eve.dev) agent, model, tool, session, and memory spans into
Respan's canonical span contract. Supports Eve `>=0.26.0 <0.76.0` and AI SDK
`>=7.0.26 <8.0.0`, including Eve 0.75.1 with AI SDK 7.0.136. Requires Node.js 24.

## Current Eve

Eve owns its OpenTelemetry provider and durable span IDs. Add one settings file
and one Respan destination under `agent/instrumentation/`:

```bash
npm install @respan/instrumentation-eve eve@0.75.1 ai@7.0.136 \
  @opentelemetry/exporter-trace-otlp-http
```

```ts
// agent/instrumentation/otel.ts
import { otel } from "eve/instrumentation/otel";

export default otel({
  functionId: "support-agent",
  tracePolicy: () => ({
    emit: true,
    recordInputs: false,
    recordOutputs: false,
  }),
});
```

```ts
// agent/instrumentation/respan.ts
import { EveSpanProcessor, withEveLineage } from "@respan/instrumentation-eve";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { otelIntegration } from "eve/instrumentation/otel";

const translator = new EveSpanProcessor();
const exporter = new OTLPTraceExporter({
  url: "https://api.respan.ai/api/v2/traces",
  headers: { Authorization: "Bearer " + process.env.RESPAN_API_KEY },
});

export default withEveLineage(
  otelIntegration({
    spanProcessors: [translator],
    traceExporter: translator.wrapExporter(exporter),
  }),
);
```

Keep `wrapExporter` around the destination exporter. It applies the immutable
content policy again when a queued batch actually exports, then applies Respan's
semantic or legacy display rules. Eve manages batching, flush, and shutdown.

Add authored JSON metadata with `otelIntegration({ runtimeContext(input) { ... } })`.
`withEveLineage` calls that resolver once and preserves its native failure and
invalid Promise return behavior. It adds root and parent session lineage without
changing native local dispatch span IDs or tree structure.

## Eve 0.26

The older SDK discovers `agent/instrumentation.ts`. Initialize Respan before
exporting the native instrumentation definition:

```ts
import { RespanTelemetry } from "@respan/tracing";
import { EveInstrumentor, withEveLineage } from "@respan/instrumentation-eve";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { defineInstrumentation } from "eve/instrumentation";

const instrumentor = new EveInstrumentor();
const exporter = new OTLPTraceExporter({
  url: "https://api.respan.ai/api/v2/traces",
  headers: { Authorization: "Bearer " + process.env.RESPAN_API_KEY },
});
const respan = new RespanTelemetry({
  apiKey: process.env.RESPAN_API_KEY,
  disabledInstrumentations: [
    "openAI",
    "anthropic",
    "azureOpenAI",
    "cohere",
    "bedrock",
    "googleVertexAI",
    "googleAIPlatform",
    "pinecone",
    "together",
    "langChain",
    "llamaIndex",
    "chromaDB",
    "qdrant",
  ],
  exporter: instrumentor.wrapExporter(exporter),
});
await respan.initialize();
instrumentor.activate();

export default withEveLineage(
  defineInstrumentation({ recordInputs: false, recordOutputs: false }),
);
```

This generation's `events["step.started"]` hook remains supported. The helper
preserves authored callbacks and uses exact parent session/turn lineage to
connect delegated child traces. Missing or ambiguous lineage keeps the original
native traces. Do not activate `VercelAIInstrumentor` in the same process.

## Captured fields

- Agent/task/tool types, real model and provider identifiers, and native tool-call IDs
- Complete emitted system instructions, prompt history, current output/tool calls, and tool schemas
- False, zero, empty-string, null, vector, and structured tool results
- Provider-reported input/output/cache usage, with modern and legacy token keys; total usage only when actually supplied
- Session/thread identifiers, conversation grouping, root/parent lineage, and Eve lifecycle metadata
- Current Eve memory records when the SDK emits them

The package adds no content length or vector dimension limit. Eve 0.75.1 itself
limits several native telemetry attributes to 32 KiB. A native result can therefore
remain complete while Eve's emitted span contains a truncated value. The
translator cannot restore content absent from its source span.

Input/output capture follows Eve's `tracePolicy` (or legacy `recordInputs` and
`recordOutputs`), constructor limits on `EveSpanProcessor`, the imported
Traceloop context flag, observed ancestor vetoes, and `RESPAN_TRACE_CONTENT` /
`TRACELOOP_TRACE_CONTENT`. A false decision remains false for that span and its
descendants. General OTel suppression, sampled-out parents, and the historical
JavaScript LM suppression context are honored. Unknown local parents without an
observable policy deny content; current Eve's own durable parent IDs retain the
framework's policy.

Content denial also removes content-bearing runtime metadata, exception events,
and status descriptions from the actual SDK `ReadableSpan` before export. The
translator does not call caller getters, `toJSON`, unknown Proxy traps, or custom
iterators to collect telemetry.

`RESPAN_SPAN_NAME_STYLE=legacy` preserves native names and structural wrappers;
semantic mode uses `agent.<name>`, `tool.<name>`, and `llm.<model>` names. Neither
mode changes content policy or native execution results.

Runtime dependency floors are `@respan/respan-sdk >=1.2.0` for the required
internal span constants and `@respan/tracing >=1.6.0` for the public batch display
API used by the destination wrapper.

## License

Apache-2.0
