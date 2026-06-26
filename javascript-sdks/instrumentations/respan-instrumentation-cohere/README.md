# @respan/instrumentation-cohere

Respan instrumentation for the Cohere TypeScript SDK. The package instruments `cohere-ai` v1 and v2 client calls for chat, streaming chat, generation, streaming generation, embeddings, and rerank.

```ts
import * as Cohere from "cohere-ai";
import { CohereClientV2 } from "cohere-ai";
import { CohereInstrumentor } from "@respan/instrumentation-cohere";
import { Respan } from "@respan/respan";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [new CohereInstrumentor({ sdkModule: Cohere })],
});
await respan.initialize();

const cohere = new CohereClientV2({
  token: process.env.COHERE_API_KEY,
});

try {
  const result = await respan.withWorkflow({ name: "cohere_chat" }, async () => {
    return cohere.chat({
      model: "command-a-03-2025",
      messages: [{ role: "user", content: "Say hello from Cohere." }],
    });
  });
  console.log(result.message?.content);
} finally {
  await respan.shutdown();
}
```

Pass the imported `cohere-ai` module through `sdkModule` in ESM applications so the instrumentor patches the same module instance used by the application.
