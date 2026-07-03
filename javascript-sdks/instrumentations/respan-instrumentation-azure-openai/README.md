# Azure OpenAI + Respan

TypeScript instrumentation for Azure OpenAI clients.

The primary supported surface is the current `openai` package's `AzureOpenAI`
client. The instrumentor also patches the older `@azure/openai` `OpenAIClient`
methods when that legacy module is available.

## Install

```bash
npm install @respan/respan @respan/instrumentation-azure-openai openai
```

## Usage

```ts
import { AzureOpenAI } from "openai";
import * as OpenAI from "openai";
import { Respan } from "@respan/respan";
import { AzureOpenAIInstrumentor } from "@respan/instrumentation-azure-openai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [
    new AzureOpenAIInstrumentor({ openAIModule: OpenAI }),
  ],
});
await respan.initialize();

const client = new AzureOpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY,
  endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  apiVersion: process.env.OPENAI_API_VERSION,
  deployment: "gpt-4o-mini",
});

await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Say hello." }],
});

await respan.shutdown();
```

## Captured Operations

- `AzureOpenAI.chat.completions.create()`
- `AzureOpenAI.completions.create()`
- `AzureOpenAI.embeddings.create()`
- streaming chat and text completion calls
- legacy `@azure/openai` `OpenAIClient` chat, completion, embedding, and streaming methods

The instrumentation emits canonical GenAI/LLM attributes from upstream
semantic-convention packages and uses `@respan/respan-sdk` only for
Respan-owned log type and method attributes.
