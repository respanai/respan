# @respan/instrumentation-braintrust

Respan instrumentation for Braintrust TypeScript tracing.

```ts
import { Respan } from "@respan/respan";
import { BraintrustInstrumentor } from "@respan/instrumentation-braintrust";

const respan = new Respan({
  instrumentations: [new BraintrustInstrumentor()],
});

await respan.initialize();
```

The plugin installs a Braintrust background logger bridge and translates
Braintrust project log and span rows into Respan OTEL spans. It preserves
Braintrust metadata, metrics, scores, tags, merge updates, parent span
relationships, and error rows while emitting Respan canonical GenAI, LLM, and
entity attributes.
