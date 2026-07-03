# Respan Arize TypeScript Instrumentation

This package routes spans created by `@arizeai/phoenix-otel` and its
OpenInference helpers into Respan.

`@arizeai/phoenix-otel` owns helper APIs such as `traceAgent`, `traceChain`,
`traceTool`, `withSpan`, `observe`, context setters, and attribute builders.
`ArizeInstrumentor` does not register a Phoenix exporter. It installs a Respan
translation hook on the active OpenTelemetry provider so OpenInference spans
are promoted to Respan's canonical span contract before export.

## Installation

```bash
npm install @respan/respan @respan/instrumentation-arize @arizeai/phoenix-otel
```

## Usage

```typescript
import { traceAgent, traceChain, traceTool } from "@arizeai/phoenix-otel";
import { ArizeInstrumentor } from "@respan/instrumentation-arize";
import { Respan } from "@respan/respan";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [new ArizeInstrumentor()],
});
await respan.initialize();

const lookupDocs = traceTool(async (query: string) => [`Doc for ${query}`], {
  name: "lookup-docs",
});

const summarize = traceChain(async (docs: string[]) => docs.join("\n"), {
  name: "summarize-docs",
});

const agent = traceAgent(
  async (question: string) => summarize(await lookupDocs(question)),
  { name: "support-agent" },
);

await respan.withWorkflow({ name: "arize-support.workflow" }, async () => {
  await agent("How do I trace TypeScript helpers?");
});

await respan.shutdown();
```

Use Arize helper and semantic-convention constants from the Arize packages.
Use Respan SDK constants only for Respan-owned attributes.
