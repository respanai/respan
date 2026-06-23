# Respan Cursor SDK Instrumentation

Native Respan instrumentation for the Cursor TypeScript SDK (`@cursor/sdk`).

The instrumentor patches the Cursor SDK module you provide and emits canonical Respan spans for:

- `Agent.prompt()`
- `Agent.create()` and `Agent.resume()`
- `SDKAgent.send()`
- `Run.stream()` and `Run.wait()`
- Cursor SDK message types: assistant, user, thinking, status, task, request, and tool calls
- per-run custom tools passed through `local.customTools`

## Usage

```ts
import * as CursorSDK from "@cursor/sdk";
import { Respan } from "@respan/respan";
import { CursorSDKInstrumentor } from "@respan/instrumentation-cursor";

const cursorSdkModule = { ...CursorSDK };

const respan = new Respan({
  instrumentations: [
    new CursorSDKInstrumentor({
      sdkModule: cursorSdkModule,
      agentName: "cursor-agent",
    }),
  ],
});

await respan.initialize();

const agent = await cursorSdkModule.Agent.create({
  name: "Cursor docs agent",
  model: { id: "gpt-4o" },
});

const run = await agent.send("Summarize this repository.");
for await (const event of run.stream()) {
  console.log(event.type);
}

await respan.flush();
```

## Constant ownership

This package keeps Cursor SDK raw field names local and emits the public Respan span contract:

- Traceloop keys come from `@traceloop/ai-semantic-conventions`.
- GenAI keys come from `@opentelemetry/semantic-conventions/incubating`.
- Respan-owned keys come from `@respan/respan-sdk`.
- Tool definitions use `llm.request.functions`.
- This-turn tool calls use `gen_ai.completion.0.tool_calls`.
- Deprecated aliases such as `respan.span.tools`, `respan.span.tool_calls`, `tools`, and `tool_calls` are not emitted.
