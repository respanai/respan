/** Attributes — Attach customer info and metadata to traces. */

import "dotenv/config";
import OpenAI from "openai";
import { Respan } from "@respan/respan";
import { OpenAIInstrumentor } from "@respan/instrumentation-openai";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [new OpenAIInstrumentor()],
});
await respan.initialize();

const client = new OpenAI({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
});

async function handleRequest(userId: string, question: string) {
  await respan.propagateAttributes(
    {
      customer_identifier: userId,
      thread_identifier: "conv_001",
      metadata: { plan: "pro" },
    },
    async () => {
      return respan.withWorkflow({ name: "handle_request" }, async () => {
        const response = await client.chat.completions.create({
          model: "gpt-4.1-nano",
          messages: [{ role: "user", content: question }],
        });
        console.log(`[${userId}] ${response.choices[0].message.content}`);
        return response;
      });
    }
  );
}

await handleRequest("user_alice", "What is an API gateway?");
await handleRequest("user_bob", "Explain rate limiting.");
