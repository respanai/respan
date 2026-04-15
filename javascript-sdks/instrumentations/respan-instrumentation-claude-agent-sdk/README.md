# @respan/instrumentation-claude-agent-sdk

Respan instrumentation plugin for the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).

This package patches `query()` on a mutable Claude Agent SDK module, merges in
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
