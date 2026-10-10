# Azure OpenAI instrumentation

`AzureOpenAIInstrumentor` observes the real Azure SDK through the active
OpenTelemetry tracer provider. It does not configure a provider or exporter.

```sh
npm install @respan/respan @respan/instrumentation-azure-openai openai
```

```ts
import * as OpenAI from "openai";
import { Respan } from "@respan/respan";
import { AzureOpenAIInstrumentor } from "@respan/instrumentation-azure-openai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new AzureOpenAIInstrumentor({ openAIModule: OpenAI })],
});
await respan.initialize();
const client = new OpenAI.AzureOpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY,
  endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  apiVersion: process.env.OPENAI_API_VERSION,
  deployment: "your-deployment",
});
const request = client.chat.completions.create({
  model: "your-deployment",
  messages: [{ role: "user", content: "Say hello." }],
});
const { data, response } = await request.withResponse();
console.log(data.choices[0]?.message.content, response.status);
await respan.shutdown();
```

## Compatibility

The modern client is `AzureOpenAI` from `openai >=4.47.2 <8`. Native SDK
releases **4.47.2** and **7.32.0** are exercised by the tests with controlled
HTTP/SSE transport responses. Version 7.32.0 was the current official release
checked on October 10, 2026. Use Node.js 22 or later with the current SDK;
older SDKs have their own runtime requirements.

`@azure/openai` **2.0.0** is a companion for Azure-specific types, not an
`OpenAIClient` implementation. Legacy `@azure/openai` 1.x clients are also
supported, with native **1.0.0-beta.1** and **1.0.0-beta.12** tested. See the
[Microsoft migration guide](https://github.com/Azure/azure-sdk-for-js/blob/main/sdk/openai/openai/MIGRATION.md)
and [OpenAI JavaScript SDK documentation](https://developers.openai.com/api/reference/typescript).

| Native operation                                                                                   | Captured data                                                              |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `chat.completions.create()`                                                                        | Complete messages, multimodal content, tools, choices and provider usage   |
| `completions.create()`                                                                             | Text prompts, returned choices and provider usage                          |
| `embeddings.create()`                                                                              | Inputs, complete returned vectors and provider usage                       |
| `responses.create()`                                                                               | Instructions, input items, output items, function calls and provider usage |
| Chat and Responses `parse()` / `stream()` helpers                                                  | Native delegated requests traced once                                      |
| Legacy `getChatCompletions`, `getCompletions`, `getEmbeddings`                                     | Native camelCase response fields normalized                                |
| Legacy `listChatCompletions` / `listCompletions` and `streamChatCompletions` / `streamCompletions` | Streaming methods present in the installed legacy release                  |

Responses and helper APIs are traced only when the installed SDK provides them.
Deployment availability and API version support still come from your Azure
resource. Images/audio generation, Realtime/WebSockets, Assistants, batch job
execution, and Responses retrieval/background polling are outside this adapter's
coverage. Standard `OpenAI` clients are excluded.

## Native behavior and lifecycle

The original `APIPromise` and stream objects retain their identity, helpers,
controllers and delivered chunks. `.asResponse()` alone ends the span without
reading the response body. Consumed streams finish on exhaustion, iterator return,
errors or controller abort. Consume or abort streams before flushing; streams
that are never consumed or cancelled remain in flight.

HTTP status comes from the actual response or legacy Azure pipeline. Usage
comes only from provider fields; missing totals are not estimated. The adapter
captures function-tool definitions and model calls; use `respan.withTool()`
around your tool implementation to record execution.

Activation coalesces concurrent requests and supports multiple SDK modules and
plugin instances. Last-owner deactivation restores only its own wrappers.
In-flight requests retain their spans. `isActive()` reports lifecycle state.
Avoid enabling another Azure/OpenAI adapter on the same client.

## Content controls

`traceContent: false`, `recordInputs: false`, and `recordOutputs: false` restrict
capture. `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, the
canonical Traceloop trace-content context, and observed ancestor/span vetoes
are ceilings: later opt-in cannot restore denied content. OpenTelemetry
suppression is respected. Real sampling runs before payload copying; dropped
and unsampled recording spans do not inspect input/output for telemetry.

Private spans retain a precise set of model, operation, usage and HTTP fields.
Their actual attributes, events and status remain guarded against late payload
writes. Telemetry snapshots read data properties without invoking customer
getters or `toJSON`. Input/output capture has no translator truncation limit.

## Development

```sh
npm test
npm pack --dry-run
```

Paired [TypeScript examples](https://github.com/respanai/respan-example-projects/tree/main/typescript/tracing/azure-openai)
run real SDK transports against fixtures and a local OTLP collector by default.
Live Azure calls and hosted Respan export are explicit opt-ins.
