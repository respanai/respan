# @respan/instrumentation-flue

Respan instrumentation plugin for the Flue TypeScript runtime.

The instrumentor subscribes to Flue's `observe()` runtime event stream and
translates stable Flue activity events into canonical Respan spans:

- workflow run lifecycle
- direct agent lifecycle
- prompt, skill, task, shell, and compaction operations
- delegated tasks
- model turns
- tool executions
- structured application logs

## Installation

```bash
npm install @respan/instrumentation-flue @flue/runtime
```

## Usage

```ts
import { Respan } from "@respan/respan";
import { FlueInstrumentor } from "@respan/instrumentation-flue";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  instrumentations: [
    new FlueInstrumentor({
      workflowName: "Flue App.workflow",
    }),
  ],
});

await respan.initialize();
```

Register the instrumentor once during application startup before Flue handles
workflow or agent activity.
