# @respan/instrumentation-mastra

Connect Mastra's native observability events to the active OpenTelemetry provider and Respan tracing pipeline. Agent, model, tool, workflow, and RAG spans retain their native parent relationships. The adapter leaves Mastra results, streams, callbacks, and errors under Mastra's control.

Requires Node.js 22.13 or later, `@mastra/core >=1.36.0`, and `@mastra/observability >=1.13.0`. Tested with those exact minimum releases and with core 1.75.0 / observability 1.18.4.

```ts
import { Agent } from "@mastra/core/agent";
import { Mastra } from "@mastra/core/mastra";
import { createMockModel } from "@mastra/core/test-utils/llm-mock";
import { Observability, SamplingStrategyType } from "@mastra/observability";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { RespanTelemetry } from "@respan/tracing";
import { MastraInstrumentor } from "@respan/instrumentation-mastra";

const instrumentor = new MastraInstrumentor();
const telemetry = new RespanTelemetry({
  apiKey: "local-example",
  exporter: new InMemorySpanExporter(),
  disableBatch: true,
});
await telemetry.initialize();

const agent = new Agent({
  id: "assistant",
  name: "Assistant",
  instructions: "Be helpful.",
  model: createMockModel({ mockText: "Hello from Mastra" }),
});
const mastra = new Mastra({
  agents: { agent },
  observability: new Observability({
    configs: {
      default: {
        serviceName: "mastra-app",
        sampling: { type: SamplingStrategyType.ALWAYS },
        exporters: [instrumentor],
      },
    },
  }),
});

try {
  const result = await telemetry.withWorkflow(
    { name: "mastra_assistant" },
    () => mastra.getAgent("agent").generate("Hello"),
  );
  console.log(result.text);
} finally {
  await mastra.shutdown();
  await telemetry.shutdown();
}
```

This sample uses Mastra's released deterministic model and keeps spans in memory. To export to Respan, supply your Respan API key to `RespanTelemetry` and omit the in-memory exporter. The same exporter can be passed in `Observability.configs.*.exporters` when using the `Respan` facade.

`new MastraInstrumentor({ traceContent: false })` omits inputs, outputs, tool definitions, metadata, and error messages. The adapter also honors `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, the upstream content context, tracing suppression, and observed ancestor content vetoes. A denied decision cannot be re-enabled later. A RECORD_ONLY or DROP sampling decision prevents payload conversion. Actual recording spans remain guarded against payload changes made by late processors or while queued for export. Provider sampling runs before payload conversion.

`excludeSpanTypes` adds native span types to the default `model_chunk` exclusion. Descendants of excluded spans attach to the closest included parent. `activate()` enables future native spans; `deactivate()` and `shutdown()` close inflight spans without copying their pending content. `flush()` leaves active spans open; flush the tracing runtime to export completed spans. An ended-only event replay cannot create a span without an observed start event.

Complete native input and output envelopes are JSON encoded, including multimodal parts, structured results, tool history, scalar values, and embedding vectors when Mastra includes them. Model tool schemas come from native `attributes.tools`; older native events exposing only `availableTools` retain names only. Token totals and HTTP status are emitted only when actually supplied. Mastra can redact, filter, or truncate data before exporter events: its default serializer bounds arrays to 50 items and depth to 8. Configure Mastra's `serializationOptions` when larger native payloads are needed. The adapter cannot restore data already removed upstream. Ambient OpenTelemetry context supplies external parent relationships. Mastra exporter events do not carry remote OpenTelemetry trace flags, so an external parent ID without ambient context is retained in metadata rather than assigned an invented sampling flag.

[`RespanMastraExporter`](#respaninstrumentation-mastra) remains an alias of `MastraInstrumentor`.
