# @respan/instrumentation-anthropic

Capture native Anthropic SDK calls in the active OpenTelemetry provider,
including the Respan tracing runtime.

```bash
npm install @respan/respan @respan/instrumentation-anthropic @anthropic-ai/sdk
```

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { Respan } from "@respan/respan";
import { AnthropicInstrumentor } from "@respan/instrumentation-anthropic";

const respan = new Respan({ instrumentations: [new AnthropicInstrumentor()] });
await respan.initialize();
const client = new Anthropic();
try {
  const message = await client.messages.create({
    model: "claude-sonnet-5-5",
    max_tokens: 128,
    messages: [
      { role: "user", content: "Write one sentence about observability." },
    ],
  });
  console.log(message.content);
} finally {
  await respan.shutdown();
}
```

The adapter supports stable and beta message creation and their native
`parse()` and `stream()` helpers, token counting, batch submission and JSONL
results, legacy completions, and the beta Messages `toolRunner()` when the
installed SDK exposes them. Managed Agents, session runners, file operations,
and administration APIs are outside this package's coverage.

Message spans preserve full multimodal blocks, thinking/signatures, citations,
structured output configuration, tool schemas, tool calls, cache usage, and
false/zero/empty/null values. Streaming merges native content-block fragments,
including tool input JSON and thinking deltas. Totals are recorded only when the
provider returns a total field. Historical tool context stays in the prompt;
it creates no tool execution spans.

The native tool runner creates one agent parent, individual native model spans,
and tool spans only for actual `tool.run()` callbacks. Tool IDs come from the
SDK's execution context. Runner tool definitions stay in canonical metadata;
agent and tool spans have no model-call attributes. Tool replacement and
repeated parameter changes preserve customer functions and avoid duplicate
execution spans. Content-denied runners pass original tool payloads to the SDK
without inspecting or copying callbacks; individual callback spans are omitted
when observing them would require denied tool traversal. Token counting and batches produce task spans; batch result
rows retain their native success/error state without fabricated model spans.

Native APIPromise, raw response helpers, messages, streams, controllers, iterator
objects, chunks, callback results and errors keep their identities. Telemetry
observes `APIPromise.parseResponse` lazily, so `asResponse()` leaves the response
body unread. A raw-only response ends at the actual response headers, without reading or
capturing its body. Stream spans finish on consumption, error, or early iterator
return. Abandoned streams remain open until consumed or closed. Deactivation
lets admitted work finish and restores the final owner's patch after pending
runner work drains. New calls outside an admitted runner pass through.

```typescript
const privateInstrumentor = new AnthropicInstrumentor({ traceContent: false });
const outputOnly = new AnthropicInstrumentor({ recordInputs: false });
const inputOnly = new AnthropicInstrumentor({ recordOutputs: false });
```

The real OTel sampler runs before telemetry copies payloads and receives safe
provider/model identity. DROP and RECORD_ONLY capture no content. Constructor
options, `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, canonical
Traceloop content context, tracing suppression, and observed ancestor content
vetoes apply as immutable ceilings. Later true values cannot restore capture.
Late attributes, events and error status on the actual span remain guarded;
telemetry skips customer getters, proxies and `toJSON` methods. Native SDK
serialization can still execute callbacks required by the SDK itself.
Exporter suppression does not revoke already admitted content.

`activate()` coalesces concurrent calls; `deactivate()` cancels pending
activation; `isActive()` is a method. Multiple owners share one patch per native
resource prototype and combine content opt-outs. `sdkModule` or `clientClass`
can select an explicit native SDK copy. Default discovery supports host and
package copies of the ESM and CommonJS SDK.

Compatibility is tested with current SDK `0.133.0` and the declared minimum
`0.30.0`. New helpers absent in the minimum SDK are explicitly skipped. Tests
use genuine released SDK methods with controlled HTTP, SSE and JSONL responses,
without provider credentials or network calls. See the paired
`typescript/tracing/anthropic` examples for standalone feature coverage.
