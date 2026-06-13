# @respan/instrumentation-google-adk

Respan instrumentation plugin for [Google ADK TypeScript](https://github.com/google/adk-js).

The Google ADK JavaScript SDK already emits OpenTelemetry spans from its
`gcp.vertex.agent` tracer. This package installs a Respan-local translator hook
that normalizes those ADK runner, agent, LLM, and tool spans into canonical
Respan fields before export.

## Install

```bash
npm install @respan/respan @respan/instrumentation-google-adk @google/adk
```

## Quickstart

```typescript
import { Respan } from "@respan/respan";
import { GoogleADKInstrumentor } from "@respan/instrumentation-google-adk";

const respan = new Respan({
  instrumentations: [new GoogleADKInstrumentor()],
});
await respan.initialize();

// Import ADK after Respan initializes so ADK's module-level tracer is created
// from the active OpenTelemetry provider.
const { InMemoryRunner, LlmAgent } = await import("@google/adk");

const agent = new LlmAgent({
  name: "weather_agent",
  model: "gemini-2.5-flash",
  instruction: "Answer weather questions concisely.",
});

const runner = new InMemoryRunner({
  appName: "google-adk-demo",
  agent,
});

for await (const event of runner.runEphemeral({
  userId: "demo-user",
  newMessage: {
    role: "user",
    parts: [{ text: "What is the weather in Tokyo?" }],
  },
})) {
  console.log(event.content?.parts?.map((part) => part.text).join(""));
}

await respan.flush();
```

## Captured Spans

- ADK runner invocations as `workflow` spans
- Agent invocations as `agent` spans
- Model calls as `chat` spans with prompt, completion, tool definition, and token fields
- Tool executions as `tool` spans with normalized input and output

ADK-specific `gcp.vertex.agent.*` attributes are translator-local raw inputs and
are stripped before export.
