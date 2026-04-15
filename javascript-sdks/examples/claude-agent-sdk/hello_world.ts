/** Claude Agent SDK — Simple query with Respan tracing. */

import "dotenv/config";
import * as _ClaudeAgentSDK from "@anthropic-ai/claude-agent-sdk";
import { ClaudeAgentSDKInstrumentor } from "@respan/instrumentation-claude-agent-sdk";
import { Respan } from "@respan/respan";

// Create a mutable copy of the ESM module so the instrumentor can patch query()
const ClaudeAgentSDK = { ..._ClaudeAgentSDK };

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [
    new ClaudeAgentSDKInstrumentor({ sdkModule: ClaudeAgentSDK }),
  ],
});
await respan.initialize();

const result = await ClaudeAgentSDK.query({
  prompt: "Write a haiku about recursion in programming.",
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
