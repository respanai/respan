# @respan/instrumentation-vertexai

Respan instrumentation plugin for the Google Vertex AI TypeScript SDK.

The package patches `@google-cloud/vertexai` generation and chat methods to emit
canonical Respan chat spans for:

- `GenerativeModel.generateContent()`
- `GenerativeModel.generateContentStream()`
- `ChatSession.sendMessage()`
- `ChatSession.sendMessageStream()`

## Installation

```bash
npm install @respan/respan @respan/instrumentation-vertexai @google-cloud/vertexai
```

## Usage

```typescript
import { VertexAI } from "@google-cloud/vertexai";
import { Respan } from "@respan/respan";
import { VertexAIInstrumentor } from "@respan/instrumentation-vertexai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new VertexAIInstrumentor()],
});
await respan.initialize();

const vertexAI = new VertexAI({
  project: process.env.GOOGLE_CLOUD_PROJECT,
  location: process.env.GOOGLE_CLOUD_LOCATION ?? "us-central1",
});
const model = vertexAI.getGenerativeModel({ model: "gemini-2.0-flash" });

await respan.withWorkflow({ name: "vertexai_generate_content_example" }, async () => {
  const result = await model.generateContent("Say hello in one sentence.");
  console.log(result.response.candidates?.[0]?.content?.parts?.[0]?.text);
});

await respan.flush();
```
