# Respan Writer Instrumentation

Trace the official `writer-sdk` TypeScript SDK with canonical Respan chat and text spans. Requires `@respan/tracing` 1.3.2 or later. Supports Writer SDK 2.0.0 through 3.x; `chat.parse` and the event streaming helper are available only in SDK versions that provide them (2.3.2 and 3.0.0 are tested).

```ts
import Writer from "writer-sdk";
import { Respan } from "@respan/respan";
import { WriterInstrumentor } from "@respan/instrumentation-writer";

const instrumentor = new WriterInstrumentor();
const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [instrumentor],
});
await respan.initialize();
const client = new Writer({ apiKey: process.env.WRITER_API_KEY });
const completion = await client.chat.chat({
  model: "palmyra-x5",
  messages: [{ role: "user", content: "Write a one-line product update." }],
});
await respan.shutdown();
```

The instrumentor observes `chat.chat`, `chat.parse`, `chat.stream`, and `completions.create`, including native SSE streams, early return, cancellation, `tee()`, `asResponse()`, and `withResponse()`. It preserves native promises, parsed results, responses, stream controllers, iterators, chunks and errors. Raw-response-only calls emit request metadata and HTTP status without parsing or consuming the response body.

Prompt history, structured content, tool definitions, schemas, response tool calls, parsed output and provider usage retain their native values. Historical tool messages do not create execution spans. Wrap application tool execution with `respan.withTool` to trace that work explicitly. Response IDs and response models come from actual SDK responses; missing totals and HTTP statuses are omitted.

`traceContent: false` withholds both input and output. `recordInputs: false` and `recordOutputs: false` control each direction. The canonical Traceloop `CONTEXT_KEY_ALLOW_TRACE_CONTENT`, observed ancestor `allow_trace_content: false`, and `RESPAN_TRACE_CONTENT=false` / `TRACELOOP_TRACE_CONTENT=false` are upper bounds. A denial remains effective through completion and later readable-span processing. Unsampled calls, OTel general suppression and the historical local language-model suppression context do not emit Writer spans. Content conversion reads data properties only and never calls customer getters, proxies, `toJSON`, `toString` or custom iterators.

When an application resolves a separate SDK copy, pass its module:

```ts
import { WriterInstrumentor } from "@respan/instrumentation-writer";

const writerModule = await import("writer-sdk");
const instrumentor = new WriterInstrumentor({ sdkModule: writerModule });
await instrumentor.activate();
// Use writerModule.default for application clients.
instrumentor.deactivate();
```

Overlapping instrumentor instances share owned patches. Deactivation preserves foreign overrides and lets existing calls finish. Existing mapping and stream-state helper exports remain available.
