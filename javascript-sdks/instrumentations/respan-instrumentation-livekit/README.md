# @respan/instrumentation-livekit

Respan instrumentation for the LiveKit Agents TypeScript SDK.

The instrumentor patches LiveKit's `telemetry.tracer` span creation methods and
adds canonical Respan attributes before LiveKit spans end. It also points
LiveKit's dynamic tracer at the active Respan OpenTelemetry provider during
`activate()`.

```ts
import { Respan } from "@respan/respan";
import { LiveKitInstrumentor } from "@respan/instrumentation-livekit";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new LiveKitInstrumentor()],
});

await respan.initialize();
```

Captured LiveKit spans include agent sessions, agent turns, LLM nodes, function
tools, user turns, and TTS spans. LiveKit raw `lk.*` attributes are preserved,
while Respan canonical fields such as `respan.entity.log_type`,
`traceloop.entity.input`, `gen_ai.prompt.*`, and
`gen_ai.completion.0.content` are added for ingestion.
