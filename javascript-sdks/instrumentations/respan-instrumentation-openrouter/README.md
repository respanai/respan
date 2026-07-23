# @respan/instrumentation-openrouter

Respan tracing instrumentation for the OpenRouter TypeScript SDK.

## Usage

```ts
import { Respan } from "@respan/tracing";
import { OpenRouterInstrumentor } from "@respan/instrumentation-openrouter";

await Respan.init({
  apiKey: process.env.RESPAN_API_KEY,
  appName: "openrouter-typescript-example",
  instrumentModules: {
    openrouter: new OpenRouterInstrumentor(),
  },
});
```

The instrumentation records OpenRouter chat, streamed chat, tool-calling chat, and embeddings calls using canonical GenAI/OpenTelemetry attributes and Respan-owned span attributes.
