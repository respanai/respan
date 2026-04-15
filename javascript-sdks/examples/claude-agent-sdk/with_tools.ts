/** Claude Agent SDK — Query that triggers tool usage for multiple spans. */

import "dotenv/config";
import * as _ClaudeAgentSDK from "@anthropic-ai/claude-agent-sdk";
import { ClaudeAgentSDKInstrumentor } from "@respan/instrumentation-claude-agent-sdk";
import { Respan } from "@respan/respan";

const ClaudeAgentSDK = { ..._ClaudeAgentSDK };

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [
    new ClaudeAgentSDKInstrumentor({ sdkModule: ClaudeAgentSDK }),
  ],
});
await respan.initialize();

// Ask something that will trigger Claude Code to use tools (read files, run commands)
const result = await ClaudeAgentSDK.query({
  prompt: "List the files in the current directory using the bash tool, then tell me how many there are.",
  options: {
    maxTurns: 5,
  },
});

for await (const event of result) {
  if (event.type === "result") {
    console.log("Result:", event.result);
  }
}

await respan.flush();
