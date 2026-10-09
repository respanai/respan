# OpenRouter instrumentation

Trace the official `@openrouter/sdk` from Respan. Supported SDK versions start at **0.13.7**; the current validation target is **1.4.25**. Use an OpenTelemetry 2 provider. The released Respan tracing companion starts at **1.3.0** for OpenTelemetry 2.

```bash
npm install @respan/respan @respan/instrumentation-openrouter @openrouter/sdk
```

```ts
import { OpenRouter } from "@openrouter/sdk";
import { Respan } from "@respan/respan";
import { OpenRouterInstrumentor } from "@respan/instrumentation-openrouter";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new OpenRouterInstrumentor()],
});
await respan.initialize();

try {
  const client = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
  await respan.withWorkflow({ name: "answer" }, async () => {
    return client.chat.send({
      chatRequest: {
        model: process.env.OPENROUTER_CHAT_MODEL!,
        messages: [{ role: "user", content: "Explain this result." }],
      },
    });
  });
} finally {
  await respan.shutdown();
}
```

Initialize tracing before activating the instrumentor. When using an existing OTel provider, call `await instrumentor.activate()` and `await instrumentor.deactivate()` directly. Activation is shared across instances; the final deactivation restores the package's patches. Existing requests retain their original capture decision while finishing.

The instrumentor observes the SDK's native request and response boundaries for:

- `client.chat.send`, including streaming, all choices, historical and current tool calls, full tool schemas, provider usage, and requested/resolved models.
- `client.embeddings.generate`, including complete vectors and provider token counts.
- `client.responses.send` and `client.beta.responses.send`.
- The SDK's lazy `client.callModel`, including each model turn, resolved request fields, tool results in continuation history, and native stream/error outcomes.
- The matching standalone SDK functions and their native `APIPromise.$inspect()` HTTP metadata.

The SDK's `callModel` remains present in 1.4.25. The separate `@openrouter/agent` 0.11.0 package uses its own SDK dependency and is outside this integration's supported surface. Model spans record requested tool calls and continuation results; this package does not emit individual local tool-execution spans.

Promises, `ModelResult` objects, HTTP responses, `EventStream` objects, readers, and iterator results retain their native identity. Telemetry observes the SDK's existing reads; it does not consume another body, start a lazy request, or change cancellation behavior. Caller cancellation is recorded as a cancellation outcome without fabricating a provider error. Failed native requests use OTel ERROR status and preserve the thrown SDK error.

Payload capture follows actual sampling, general tracing suppression, `traceContent: false`, the canonical Traceloop content context, and the upper bounds `RESPAN_TRACE_CONTENT=false` and `TRACELOOP_TRACE_CONTENT=false`. An observed parent or readable-span `allow_trace_content=false` remains a veto even after a later `true`. Queued spans apply that veto to late attributes, metadata, events, and status messages. Native scalar models, IDs, and usage remain available when content is denied. Telemetry does not invoke customer getters, serializers, or iterators to copy request data.

Payloads are not capped by this translator. OTel provider span limits still apply to indexed attributes; configure `attributeCountLimit` and `attributeValueLengthLimit` for your application's histories. The full messages, tool schemas, vectors, and Responses output are also retained in their canonical structured attributes.

See the paired TypeScript examples for controlled transport fixtures, native errors and cancellation, standalone APIPromise inspection, full histories/vectors, and optional live-provider requests. Controlled examples use the real vendor client and native parsers; only HTTP transport is supplied by the fixture.
