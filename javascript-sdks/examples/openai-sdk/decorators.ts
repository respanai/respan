/** Decorators — Use withWorkflow and withTask to structure traces. */

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

async function generateOutline(topic: string): Promise<string> {
  return respan.withTask({ name: "generate_outline" }, async () => {
    const response = await client.chat.completions.create({
      model: "gpt-4.1-nano",
      messages: [
        { role: "system", content: "Generate a 3-point outline. Be concise." },
        { role: "user", content: topic },
      ],
    });
    return response.choices[0].message.content!;
  });
}

async function writeDraft(outline: string): Promise<string> {
  return respan.withTask({ name: "write_draft" }, async () => {
    const response = await client.chat.completions.create({
      model: "gpt-4.1-nano",
      messages: [
        { role: "system", content: "Write a short paragraph from this outline." },
        { role: "user", content: outline },
      ],
    });
    return response.choices[0].message.content!;
  });
}

async function run(topic: string) {
  return respan.withWorkflow({ name: "content_pipeline" }, async () => {
    const outline = await generateOutline(topic);
    console.log(`Outline:\n${outline}\n`);

    const draft = await writeDraft(outline);
    console.log(`Draft:\n${draft}`);
    return draft;
  });
}

await run("Benefits of open-source software");
