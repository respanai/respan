# Respan Together AI TypeScript Instrumentation

Native Respan instrumentation for the [`together-ai`](https://www.npmjs.com/package/together-ai) TypeScript SDK.

```ts
import { Respan } from "@respan/respan";
import { TogetherAIInstrumentor } from "@respan/instrumentation-together-ai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new TogetherAIInstrumentor()],
});

await respan.initialize();
```

The instrumentor patches the Together AI SDK resource methods for chat completions, text completions, embeddings, image generation, rerank, text-to-speech, transcription, and translation. It emits Respan-compatible spans with canonical GenAI and Traceloop semantic-convention attributes.
