# @respan/instrumentation-langchain

Attach a Respan callback to released LangChain runnables, models, tools, retrievers, and LangGraph invocations. The callback creates real OpenTelemetry spans through the active Respan tracing runtime. It does not patch LangChain globally.

```ts
import { Respan } from "@respan/respan";
import { LangChainInstrumentor } from "@respan/instrumentation-langchain";
import { ChatOpenAI } from "@langchain/openai";

const langchain = new LangChainInstrumentor();
const respan = new Respan({ instrumentations: [langchain] });
await respan.initialize();
try {
  const model = new ChatOpenAI({ model: "gpt-4o-mini" });
  await model.invoke(
    "Say hello.",
    langchain.addCallback({ runName: "greeting" }),
  );
} finally {
  await respan.shutdown();
}
```

`RespanCallbackHandler`, `getCallbackHandler`, and `addRespanCallback` also work when a tracer provider is already configured. Reuse one handler for a workflow so native callback run IDs resolve to actual OTel parent spans. `LangchainInstrumentor` remains an alias of `LangChainInstrumentor`.

The plugin accepts new roots after `activate()` and stops accepting new roots after `deactivate()`. Runs already started still finish, including their child callbacks. Multiple instrumentors may share a handler; deactivating one owner leaves other active owners enabled.

## Capture options

```ts
const langchain = new LangChainInstrumentor({
  callbackHandlerOptions: {
    includeContent: true,
    includeMetadata: true,
    recordInputs: true,
    recordOutputs: false,
    maxCachedRuns: 4096,
  },
});
```

The real tracer sampler runs before payload capture. DROP and RECORD_ONLY runs do not traverse inputs, outputs, metadata, tool definitions, or stream tokens. Constructor options, `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, the canonical Traceloop content context, and an observed ancestor's `allow_trace_content=false` are immutable ceilings for each run. A later `true` cannot restore capture on that run. Input and output lanes can be disabled independently; metadata requires both lanes. Capture reads data properties without executing getters or `toJSON` methods. Privacy also guards the actual readable span against late processor attributes, events, and status messages.

LLM spans preserve complete canonical messages, multimodal content, tool definitions, final output, and provider usage before optional indexed attributes consume OTel's default attribute budget. Tool execution spans contain `{name, arguments}`, the real result, and `gen_ai.tool.call.id` when the native callback exposes it. Agent action notifications remain events on their existing run, so they do not invent tool executions.

## Compatibility

Native transport suites cover `@langchain/core` 0.2.0 with `@langchain/openai` 0.0.30, and core 1.2.17 with OpenAI 1.6.2. Companion examples cover `langchain` 1.5.16 and `@langchain/langgraph` 1.4.21, including agents, tools, structured output, streaming, retries, fallbacks, errors, and graphs. The production floors are `@respan/tracing` 1.6.1 and `@respan/respan-sdk` 1.3.4.

The core 0.2.0 tool callback does not expose a tool invocation ID, and OpenAI 0.0.30 omits zero usage counts before its callback. The adapter leaves unavailable fields absent. Returning early from a native model stream may skip both terminal callbacks; this callback-only adapter cannot close that run without a terminal callback. Deactivation leaves that admitted span open for draining; if the native terminal callback never arrives, the span remains unfinished and is not exported. Consume the stream, or abort it and await the resulting native error. LangChain embedding methods do not provide this callback lifecycle; this package does not invent embedding spans. Langflow examples demonstrate explicitly attached TypeScript component callbacks, not an official native Langflow TypeScript SDK.

For explicitly grouped component callbacks, `getCallbackHandler()` enables `groupLangflowRootRuns`; reuse it only within one component invocation:

```ts
import {
  addRespanCallback,
  getCallbackHandler,
} from "@respan/instrumentation-langchain";

const handler = getCallbackHandler();
const config = addRespanCallback(
  {
    tags: ["langflow"],
    metadata: { framework: "langflow", langflow_component: "RoutingComponent" },
  },
  handler,
);
```
