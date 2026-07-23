/** Anthropic SDK — Chat completion with Respan tracing via OpenInference. */

import "dotenv/config";
import Anthropic from "@anthropic-ai/sdk";
import { AnthropicInstrumentation } from "@arizeai/openinference-instrumentation-anthropic";
import { Respan } from "@respan/respan";
import { OpenInferenceInstrumentor } from "@respan/instrumentation-openinference";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL,
  instrumentations: [
    new OpenInferenceInstrumentor(AnthropicInstrumentation, Anthropic),
  ],
});
await respan.initialize();

const client = new Anthropic();

const message = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 100,
  messages: [
    { role: "user", content: "Write a haiku about recursion in programming." },
  ],
});

const text = message.content[0];
if (text.type === "text") {
  console.log(text.text);
}
