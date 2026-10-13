# @respan/instrumentation-llama-index

Trace native LlamaIndex TypeScript chat, completion, streaming, agent, retrieval, synthesis, document parsing, and OpenAI embedding calls through the active Respan runtime.

```typescript
import { Respan } from "@respan/respan";
import { LlamaIndexInstrumentor } from "@respan/instrumentation-llama-index";
import { OpenAI } from "@llamaindex/openai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new LlamaIndexInstrumentor()],
});
await respan.initialize();
// Initialize tracing before constructing clients whose decorators bind methods.
const llm = new OpenAI({
  model: "gpt-4o-mini",
  apiKey: process.env.OPENAI_API_KEY,
});
try {
  await respan.withWorkflow({ name: "answer" }, () =>
    llm.chat({ messages: [{ role: "user", content: "What is tracing?" }] }),
  );
} finally {
  await respan.shutdown();
}
```

Install `llamaindex >=0.12.0`, or use the modular `@llamaindex/core >=0.6.22` and a provider such as `@llamaindex/openai >=0.4.22`. The meta package is optional for modular applications. Verified releases are `llamaindex` 0.12.0 and 0.12.1, core 0.6.22 and 0.6.23, and OpenAI 0.4.22 and 0.4.23. The current meta package embeds core 0.6.22 while current modular providers use 0.6.23; the instrumentor observes both module copies.

`activate()` and `deactivate()` can also be called directly after configuring an OTel provider. `isActive()` reports lifecycle state. Repeated activation is coalesced, multiple owners share patches, foreign wrappers survive deactivation, and observed calls finish after deactivation. Pass `llamaIndexModule` as a module or module array when resolving explicit SDK copies.

```typescript
const instrumentor = new LlamaIndexInstrumentor({
  traceContent: true,
  recordInputs: true,
  recordOutputs: false,
});
```

Constructor flags, `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, the canonical Traceloop content context, tracing suppression, and an observed ancestor `allow_trace_content=false` constrain capture. A later `true` does not restore an observed veto. Sampling runs before payload copying; DROP and RECORD_ONLY spans do not inspect payloads. Inputs and outputs have independent lanes. The actual recording span also rejects late private attributes, events, status text, and property replacement.

Chat spans preserve complete message history, multimodal content, tool schemas, emitted tool calls, and native usage including zero values. Canonical input, tools, output, and usage precede optional indexed attributes under the default OTel attribute budget. Tool execution spans retain the native call ID and `{name, arguments}`. Workflow, task, and agent spans use common fields. OpenAI embedding spans preserve complete native vectors; LlamaIndex discards the provider's usage before returning vectors, so the adapter does not reconstruct that count. Provider names, token totals, HTTP status, and request models are never guessed.

Generic native callback support remains available across providers and callback manager instances. A provider that emits a start but no terminal callback on rejection cannot supply a standalone terminal result through callbacks; native observed parent failures close their pending callback children as failures. For direct OpenAI calls, method observation also covers rejected promises, iterator errors, and early returns. Native OpenAI Responses calls are captured through `OpenAIResponses.chat`.

The adapter uses native observation because the released upstream adapters replace stream/promise objects and do not provide these privacy guarantees, complete vectors, or native tool execution mapping. It emits real spans into the configured tracer provider; the Respan runtime owns normalization, semantic names, and export.
