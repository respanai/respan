# @respan/instrumentation-google-adk

Translate the native OpenTelemetry spans emitted by [Google ADK TypeScript](https://github.com/google/adk-js) into Respan workflow, agent, chat, tool, and task records.

## Install

```bash
npm install @respan/tracing @respan/instrumentation-google-adk @google/adk
```

Requires Google ADK **1.2.x or 2.x** and `@respan/tracing` **1.3.2 or later**. The adapter uses the public span transformer registry introduced in tracing 1.3.2. Initialize the tracing runtime before activating it.

## Quickstart

Set `RESPAN_API_KEY` and `GOOGLE_GENAI_API_KEY`, then run:

```typescript
import { RespanTelemetry } from "@respan/tracing";
import { GoogleADKInstrumentor } from "@respan/instrumentation-google-adk";

const telemetry = new RespanTelemetry({ appName: "google-adk-demo" });
await telemetry.initialize();
const instrumentor = new GoogleADKInstrumentor();
instrumentor.activate();

try {
  // ADK creates a module-level tracer. Import it after tracing initializes.
  const { InMemoryRunner, LlmAgent } = await import("@google/adk");
  const runner = new InMemoryRunner({
    appName: "google-adk-demo",
    agent: new LlmAgent({
      name: "weather_agent",
      model: "gemini-2.5-flash",
      instruction: "Answer weather questions concisely.",
    }),
  });
  for await (const event of runner.runEphemeral({
    userId: "demo-user",
    newMessage: {
      role: "user",
      parts: [{ text: "What is the weather in Tokyo?" }],
    },
  })) {
    console.log(event.content?.parts?.map((part) => part.text ?? "").join(""));
  }
  await telemetry.flush();
} finally {
  instrumentor.deactivate();
  await telemetry.shutdown();
}
```

For `new Respan({ instrumentations: [new GoogleADKInstrumentor()] })`, the facade activates the adapter after `await respan.initialize()`.

## Captured operations

- Runner invocations become `workflow` spans.
- Agent invocations become `agent` spans.
- Model calls become `chat` spans with native model, messages, tool schemas, streaming completions, and reported usage.
- Tool executions become `tool` spans. `gen_ai.tool.call.id` connects each execution with the model call and tool response. ADK's merged-tool bookkeeping span becomes a `task` so it does not count as another execution.
- ADK 2.x `Workflow` invocations become `workflow` spans. `execute_node` and retry attempt spans become `task` spans with node paths and reported execution metadata.

Graph workflow APIs are absent in ADK 1.2.0. Tests for those APIs skip at that minimum. Native tracing fields beginning with `gcp.vertex.agent.*` and graph translator inputs are removed after normalization. Other instrumentation scopes are not classified from their operation names.

## Content and lifecycle controls

```typescript
const instrumentor = new GoogleADKInstrumentor({ traceContent: false });
```

The adapter also honors `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, ADK's `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false`, the Traceloop content context, observed span vetoes, and OpenTelemetry suppression. A recorded content veto remains in effect for that invocation. Sampling stays with the active OpenTelemetry provider.

Repeated activation is idempotent; multiple owners share one translation. Deactivation stops admission of new spans while spans already in flight finish translating. Native SDK methods, returned events, and iterator behavior are unchanged.

ADK itself determines which fields exist. For example, it omits inline request bytes from native telemetry and can represent a provider failure as an error event while leaving span status unset. This adapter does not reconstruct withheld data or invent usage, status, or model responses.

See the [controlled examples](https://github.com/respanai/respan-example-projects/tree/main/typescript/tracing/google-adk) for Gemini transport fixtures that exercise the real SDK without a provider key.
