/** Hello World — Minimal agent with Respan tracing (mirrors Python example). */

import "dotenv/config";
import { Agent, run } from "@openai/agents";
import { Respan } from "@respan/respan";
import { OpenAIAgentsInstrumentor } from "@respan/instrumentation-openai-agents";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [new OpenAIAgentsInstrumentor()],
});
await respan.initialize();

const agent = new Agent({
  name: "Assistant",
  instructions: "You only respond in haikus.",
});

const result = await run(agent, "Tell me about recursion in programming.");
console.log(result.finalOutput);
