/** Claude Agent SDK — Simple query with Respan tracing via OpenInference. */

import "dotenv/config";
import * as _ClaudeAgentSDK from "@anthropic-ai/claude-agent-sdk";
import { ClaudeAgentSDKInstrumentation } from "@arizeai/openinference-instrumentation-claude-agent-sdk";
import { Respan } from "@respan/respan";
import { OpenInferenceInstrumentor } from "@respan/instrumentation-openinference";

// Create a mutable copy of the ESM module so the OI instrumentor can monkey-patch it
const ClaudeAgentSDK = { ..._ClaudeAgentSDK };

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [
    new OpenInferenceInstrumentor(ClaudeAgentSDKInstrumentation, ClaudeAgentSDK),
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
