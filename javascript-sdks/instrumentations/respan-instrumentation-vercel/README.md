# @respan/instrumentation-vercel

Respan instrumentation for the Vercel AI SDK. It translates AI SDK 4 through 7 telemetry into the canonical Respan span contract, including model calls, embeddings, tool executions, agents, and workflows.

This integration is explicit-only: add `VercelAIInstrumentor` to your Respan configuration.

## Install

For AI SDK 7, install the OpenTelemetry adapter alongside the AI SDK:

```bash
npm install @respan/respan @respan/instrumentation-vercel ai @ai-sdk/otel
```

For AI SDK 4 through 6, `@ai-sdk/otel` is not required:

```bash
npm install @respan/respan @respan/instrumentation-vercel ai
```

Supported peer ranges are `ai >=4 <8` and `@ai-sdk/otel >=1 <2`.

## Activate the instrumentation

Initialize Respan before making AI SDK calls:

```ts
import { Respan } from "@respan/respan";
import { VercelAIInstrumentor } from "@respan/instrumentation-vercel";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new VercelAIInstrumentor()],
});

await respan.initialize();
```

The translator is provider-independent. The same activation works with Vercel AI SDK providers such as OpenAI, Anthropic, Google Gemini, and other providers that emit AI SDK telemetry.

### AI SDK 7

During activation, the instrumentor detects AI SDK 7 and registers the application-resolved `@ai-sdk/otel` `OpenTelemetry` integration. Registration is shared across multiple instrumentor instances and is removed after the final owner deactivates.

The registered adapter supplies the OpenTelemetry integration, but telemetry still must be enabled on each AI SDK operation:

```ts
import { generateText } from "ai";

const result = await generateText({
  model,
  prompt: "Write a one-line release note.",
  telemetry: {
    isEnabled: true,
  },
});
```

If your application registers the AI SDK telemetry integration itself, prevent duplicate ownership:

```ts
const respan = new Respan({
  instrumentations: [
    new VercelAIInstrumentor({ autoRegisterAISDKTelemetry: false }),
  ],
});
```

### AI SDK 4 through 6

These versions use their native experimental telemetry path. Enable telemetry on each AI SDK operation:

```ts
import { generateText } from "ai";

const result = await generateText({
  model,
  prompt: "Write a one-line release note.",
  experimental_telemetry: {
    isEnabled: true,
  },
});
```

The instrumentor translates those emitted spans; it does not turn on `experimental_telemetry` for individual calls.

## What is captured

- Chat and text input/output, model/provider, streaming state, and token usage
- Embedding inputs and vectors
- Tool execution input/output without duplicate tool-call aliases
- Agent, task, and workflow structure
- Customer, session, thread, trace-group, and JSON metadata fields

## License

Apache-2.0
