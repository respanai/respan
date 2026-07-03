# Respan Codex SDK Instrumentation

Respan instrumentation plugin for the OpenAI Codex TypeScript SDK.

```ts
import { Respan } from "@respan/respan";
import { CodexSDKInstrumentor } from "@respan/instrumentation-codex-sdk";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new CodexSDKInstrumentor()],
});

await respan.initialize();
```

The plugin patches `Thread.run()` and `Thread.runStreamed()` from
`@openai/codex-sdk`, converts Codex turn events into Respan-compatible OTEL
spans, and injects them into the active Respan tracing pipeline.

Captured turn features include:

- buffered and streamed turns
- final agent messages
- reasoning summaries
- command execution
- MCP tool calls
- web search events
- file changes
- todo lists
- structured output usage
- failed turns and stream errors

SDK-specific Codex event fields stay inside this package. Shared GenAI and
Traceloop attributes are imported from `@traceloop/ai-semantic-conventions`,
and Respan-owned attributes are imported from `@respan/respan-sdk`.
