# @respan/instrumentation-claude-agent-sdk

Respan instrumentation plugin for the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).

This package patches `query()` and, when available, `prewarm()` on a mutable
Claude Agent SDK module, merges in
Claude hook callbacks for tool lifecycle tracking, and emits OTEL spans that
match the Respan tracing pipeline.

## Install

```bash
yarn add @anthropic-ai/claude-agent-sdk @respan/instrumentation-claude-agent-sdk
```

## Quickstart

```ts
import "dotenv/config";
import * as _ClaudeAgentSDK from "@anthropic-ai/claude-agent-sdk";
import { Respan } from "@respan/respan";
import { ClaudeAgentSDKInstrumentor } from "@respan/instrumentation-claude-agent-sdk";

// ESM namespace objects are read-only. Patch a mutable copy instead.
const ClaudeAgentSDK = { ..._ClaudeAgentSDK };

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [
    new ClaudeAgentSDKInstrumentor({
      sdkModule: ClaudeAgentSDK,
      agentName: "claude-agent-sdk",
    }),
  ],
});

await respan.initialize();

const result = await ClaudeAgentSDK.query({
  prompt: "Write a haiku about tracing.",
  options: {
    maxTurns: 1,
  },
});

for await (const event of result) {
  if (event.type === "result") {
    console.log(event.result);
  }
}

await respan.flush();
```

## Notes

- `sdkModule` should be the same mutable module object your app calls.
- Existing Claude hook callbacks are preserved and merged with the
  instrumentation hooks.
- Tool executions are emitted as OTEL tool spans and linked to the enclosing
  agent span.

## Prewarmed sessions

With Claude Agent SDK 0.3.282 or later, call `prewarm()` through the same mutable
SDK module passed to the instrumentor. Hooks are installed before the process
starts, and tracing begins when `spare.claim()` starts a session:

```ts
const spare = await ClaudeAgentSDK.prewarm({
  options: { permissionMode: "default", tools: [] },
});
try {
  const query = spare.claim({
    prompt: "Write a haiku about tracing.",
    options: { cwd: process.cwd(), model: "claude-sonnet-4-6" },
  });
  for await (const message of query) {
    if (message.type === "result" && message.subtype === "success") {
      console.log(message.result);
    }
  }
} finally {
  spare.close();
}
```

Claimed sessions retain the synchronous Query API, existing hooks, and tool
success/failure spans. Closing an unused spare emits no agent span. The
instrumentor also accepts a mutable copy of `@anthropic-ai/claude-agent-sdk/core`.
