# @respan/instrumentation-aws-bedrock

Capture calls made through the AWS SDK's `BedrockRuntimeClient.send` in the active
OpenTelemetry provider, including the Respan runtime.

```bash
npm install @respan/respan @respan/instrumentation-aws-bedrock @aws-sdk/client-bedrock-runtime
```

```typescript
import {
  BedrockRuntimeClient,
  ConverseCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { Respan } from "@respan/respan";
import { AWSBedrockInstrumentor } from "@respan/instrumentation-aws-bedrock";

const instrumentor = new AWSBedrockInstrumentor();
const respan = new Respan({ instrumentations: [instrumentor] });
await respan.initialize();
const client = new BedrockRuntimeClient({ region: "us-east-1" });
try {
  await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [
        {
          role: "user",
          content: [{ text: "Write a sentence about observability." }],
        },
      ],
    }),
  );
} finally {
  client.destroy();
  await respan.shutdown();
}
```

The supported operations are `Converse`, `ConverseStream`, `InvokeModel`, and
`InvokeModelWithResponseStream`. Promise and callback overloads use the native
client; responses, promises, streams, chunks, and errors keep their original
identity. Streaming spans end when consumption finishes, fails, or the iterator
returns early. An abandoned stream remains open until it is closed or consumed.
Deactivation restores the final owner's patch and lets admitted calls finish.

Spans contain complete input messages, tool definitions, raw structured output,
stream events, fragmented tool arguments, and provider usage. Invoke responses
support Anthropic messages, Amazon Nova/Titan text, generic completion fields,
and embedding vectors. Usage totals are recorded only when returned by the
provider. Request metadata, structured output configuration, multimodal content,
reasoning, citations, and new content blocks are retained in canonical JSON.
Native spans record the provider's HTTP response status. The shared export
processor in `@respan/tracing` 1.6.1 strips `http.*` attributes, so that status
does not reach OTLP.
The adapter records model calls; historical tool context creates no tool
execution spans. Other AWS operations, including token counting, guardrails,
async inference, and bidirectional inference, pass through without a span.

```typescript
const privateInstrumentor = new AWSBedrockInstrumentor({ traceContent: false });
// Capture only output, or only input:
const outputOnly = new AWSBedrockInstrumentor({ recordInputs: false });
const inputOnly = new AWSBedrockInstrumentor({ recordOutputs: false });
```

Sampling runs before payload capture. DROP and RECORD_ONLY spans do not capture
content. Content ceilings also honor `RESPAN_TRACE_CONTENT=false`,
`TRACELOOP_TRACE_CONTENT=false`, the canonical Traceloop content context, tracing
suppression, and observed ancestor `allow_trace_content=false`. A later true
value cannot restore denied capture. Late processor attributes, events, and error
messages on the actual span remain guarded. The native child sampler decides
even when the parent is unsampled. Exporter suppression does not revoke spans
already admitted.

`activate()` coalesces concurrent calls; `deactivate()` cancels pending activation;
`isActive()` reports the current state. `sdkModule` and `clientClass` can select an
explicit native SDK copy. Multiple instrumentors share one patch per prototype,
and any owner's content opt-out applies to the shared call.

Compatibility is tested with AWS SDK `3.1149.0` and `3.704.0`, the first published
version satisfying the declared `>=3.700.0` peer range. The exact `3.700.0` Bedrock
package was never published. The tests use real released AWS clients, Smithy HTTP
serialization/deserialization, and encoded AWS eventstream frames without AWS
credentials or network provider calls. See the paired `typescript/tracing/aws-bedrock`
examples for fixture transport, opt-in live AWS calls, and opt-in Respan export.
