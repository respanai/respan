# Respan instrumentation for Superagent

Trace the native Superagent `safety-agent` TypeScript SDK through Respan's shared OpenTelemetry pipeline. Verified with stable 0.1.7 and minimum 0.1.6.

## Install

```bash
npm install @respan/respan @respan/instrumentation-superagent safety-agent
```

## Usage

```typescript
import { Respan } from "@respan/respan";
import { SuperagentInstrumentor } from "@respan/instrumentation-superagent";
import * as safetyAgentModule from "safety-agent";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new SuperagentInstrumentor({ safetyAgentModule })],
});
await respan.initialize();
try {
  const client = safetyAgentModule.createClient({
    apiKey: process.env.SUPERAGENT_API_KEY,
  });
  const result = await client.guard({
    input: "Check this message",
    model: "openai/gpt-4o-mini",
    chunkSize: 0,
  });
  console.log(result.classification);
} finally {
  await respan.shutdown();
}
```

## Captured behavior

`guard()` emits a guardrail operation; `redact()` and `scan()` emit tool operations. The native provider's public request and response transformations produce one chat span per native provider transformation. Chunked guard calls therefore produce multiple model spans, and 0.1.7 fallback calls preserve separate attempts. Scan executes remotely through Daytona and exposes only a scan result, so it emits no invented model child.

Model spans preserve native messages, multimodal parts, response schemas, complete request metadata and completions, observed provider/model identity, and native usage including explicit zeros. Missing totals and HTTP statuses are not inferred. Failed requests retain their native error and have no fabricated output. The plugin preserves returned promise/result/error identity, native chunks, and provider callbacks.

## Content controls and lifecycle

`SuperagentInstrumentor` accepts `traceContent`, `recordInputs`, `recordOutputs`, `metadata`, an optional `methods` subset, and the native `safetyAgentModule`. Constructor restrictions, `RESPAN_TRACE_CONTENT=false`, canonical Traceloop context, suppression, observed ancestor vetoes, and late span vetoes are ceilings; later opt-in cannot restore denied content. Sampling runs before request/response copying. Dropped and unsampled recording spans add no payload work. Actual span attributes, events, and status are guarded against late content writes.

Await `activate()` for standalone use and call `deactivate()` to restore owned patches. Multiple owners, pending activation cancellation, in-flight draining, and foreign patches are handled without replacing unrelated wrappers. Respan's released public transformer API observes ancestor policy; a standalone OTel provider still applies sampling and owned-span content guards.

Operation parents carry common fields and canonical JSON metadata; model fields stay on actual model children. Respan's shared exporter applies semantic naming and removes internal naming hints. Auto-emitted operations do not set `traceloop.span.kind`.

The standalone controlled examples cover guard/redact/workflow/scan, chunking, fallback, privacy, errors, and large native payloads. The explicit `fallbackModel` option is absent from minimum 0.1.6; its native regression is skipped only in that profile.

The SDK may retry a timed-out HTTP request with a reused transformed body. Such transport retries do not expose another provider transformation, so this adapter cannot distinguish them as separate model spans. Explicit `fallbackModel` calls do expose distinct transformations and are covered.
