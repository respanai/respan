# @respan/instrumentation-cohere

Respan instrumentation for the Cohere TypeScript SDK on Node.js 18 or later.

Tested with `cohere-ai` 8.1.0 and the supported minimum 7.7.5. Version 7.7.5 exposes the v1 client; v2 features require a SDK version that provides `CohereClientV2` or `client.v2`.

| Native API                      | Captured data                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| v1/v2 `chat`, `chatStream`      | Full messages, multimodal content, tool definitions and returned tool-call IDs/arguments         |
| v1/v2 `embed`                   | Input texts/images/inputs and complete float, int8, uint8, binary and ubinary vectors            |
| v1/v2 `rerank`                  | Query, documents and complete returned ranking results; search units remain in response metadata |
| v1 `generate`, `generateStream` | Prompt and all returned generations                                                              |
| v2 `parse`                      | Document request and all returned pages in a tool span                                           |

Spans use the active OpenTelemetry tracer and sampler. Native promise, `withRawResponse()`, stream, controller, iterator and chunk objects are preserved. Instrumentation observes consumption without reading the stream ahead of the application. Deactivation restores owned methods and lets calls already in progress finish.

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
  const result = await respan.withWorkflow(
    { name: "cohere_chat" },
    async () => {
      return cohere.chat({
        model: "command-a-03-2025",
        messages: [{ role: "user", content: "Say hello from Cohere." }],
      });
    },
  );
  console.log(result.message?.content);
} finally {
  await respan.shutdown();
}
```

Pass the imported `cohere-ai` module through `sdkModule` in ESM applications so the instrumentor patches the same module instance used by the application.

Content tracing is enabled by default. Disable both lanes with `traceContent: false`, or set `recordInputs: false` / `recordOutputs: false` independently on `CohereInstrumentor`. `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, canonical Traceloop context denial, and an ancestor's observed `allow_trace_content=false` are ceilings that a later `true` cannot override. OTel suppression and non-sampled spans do not inspect payloads. Denied payloads, events, metadata and status messages also stay removed after late processor updates.

Usage is copied only from actual provider fields. The adapter does not infer total tokens, HTTP success codes, default models or historical tool execution. Document parsing is represented as service tool work without LLM token attributes. Client administration, dataset jobs, model listing and audio endpoints are outside this adapter's scope.

The native test suite uses the real SDK with local HTTP/SSE fixtures. To test another installed version, set `COHERE_NATIVE_MODULE` to that package directory before `npm test`. The paired examples live in `respan-example-projects/typescript/tracing/cohere`; their default collector is local, and hosted export requires an explicit opt-in.
