# @respan/instrumentation-beeai

Trace BeeAI agents, chat models, streams, structured generation, tools, and
embeddings with Respan. The instrumentor preserves BeeAI `Run` objects and their
observers, middleware, cancellation, results, and errors.

## Install

```bash
npm install @respan/respan @respan/instrumentation-beeai beeai-framework@0.1.31
```

Install the provider adapter your application uses, for example
`@ai-sdk/openai@^3.0.26` for BeeAI 0.1.31. BeeAI 0.1.31 also needs
`uuid@^11.1.1` for `RequirementAgent`; its released manifest omits that dependency.

## Usage

```typescript
import * as beeaiFramework from "beeai-framework";
import { BeeAIInstrumentor } from "@respan/instrumentation-beeai";
import { Respan } from "@respan/respan";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new BeeAIInstrumentor({ sdkModule: beeaiFramework })],
});
await respan.initialize();

// Run your BeeAI models, agents, or tools here.
await respan.shutdown();
```

Pass the application-resolved `sdkModule` in ESM, linked, or bundled installs so
Respan instruments the same BeeAI classes your application uses.

## Supported versions

`beeai-framework >=0.1.9 <0.2.0`. Tests exercise released 0.1.9 and 0.1.31 with
their compatible provider adapters. The default integration is native: released
OpenInference BeeAI instrumentation skips 0.1.14 and newer, and its older event
serializer can inspect application values before privacy and sampling gates.
OpenInference packages are not required.

Chat traces retain full history, current-turn tool calls, available tool schemas,
provider usage including zero counts, and native message and tool-call IDs.
Embedding traces retain complete vectors. Usage is emitted only when the SDK
returns a provider usage field. OpenTelemetry export limits still apply; configure
`OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT` when exporting long indexed message histories.

## Privacy

Set `traceContent: false` on the instrumentor, `RESPAN_TRACE_CONTENT=false`, or
`TRACELOOP_TRACE_CONTENT=false` to disable payload capture. Context and observed
ancestor vetoes remain effective when a child enables content capture. General
OpenTelemetry suppression skips instrumentation; the compatibility context key
`suppress_language_model_instrumentation` skips chat and embedding spans.

Telemetry does not invoke application getters, `toJSON`, string methods, iterators,
or schema factories. It observes tool and structured schemas when BeeAI itself
converts them. Caller proxies remain opaque. BeeAI's native watched message arrays
are recognized with a trap-free Node.js shape probe and retain their content;
arrays backed by application proxies remain opaque.

`instrumentationClass` and `delegateFactory` remain available for explicit custom
delegate compatibility. Custom delegates own their mapping, privacy, and
serialization behavior; the native guarantees above apply to the default path.
