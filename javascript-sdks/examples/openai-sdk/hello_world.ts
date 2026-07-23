/** Hello World — Simplest possible: one OpenAI call, auto-traced. */

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
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const response = await client.chat.completions.create({
  model: "gpt-4.1-nano",
  messages: [{ role: "user", content: "Say hello in three languages." }],
});
console.log(response.choices[0].message.content);
