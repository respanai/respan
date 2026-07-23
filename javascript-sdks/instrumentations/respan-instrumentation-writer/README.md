# Respan Writer Instrumentation

Respan instrumentation plugin for the official `writer-sdk` TypeScript SDK.

```ts
import Writer from "writer-sdk";
import { Respan } from "@respan/respan";
import { WriterInstrumentor } from "@respan/instrumentation-writer";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new WriterInstrumentor()],
});
await respan.initialize();

const client = new Writer({ apiKey: process.env.WRITER_API_KEY });
const completion = await client.chat.chat({
  model: "palmyra-x5",
  messages: [{ role: "user", content: "Write a one-line product update." }],
});
```

The instrumentor emits canonical Respan chat/text spans for:

- `client.chat.chat(...)`, including non-streaming and `stream: true`
- `client.chat.parse(...)`
- `client.chat.stream(...)`
- `client.completions.create(...)`, including streaming completions

For runtimes that resolve multiple `writer-sdk` module instances, pass the same module instance your app imports:

```ts
const writerModule = await import("writer-sdk");
const respan = new Respan({
  instrumentations: [new WriterInstrumentor({ sdkModule: writerModule })],
});
```
