# @respan/instrumentation-strands-agents

Respan instrumentation plugin for the
[Strands Agents TypeScript SDK](https://strandsagents.com/).

This package consumes Strands Agents' native OpenTelemetry spans and normalizes
them into the Respan span contract before the existing Respan OTLP exporter
sends them. It does not require OpenInference at runtime.

## Install

```bash
npm install @respan/instrumentation-strands-agents @strands-agents/sdk
```

## Quickstart

```typescript
import { Agent, tool } from "@strands-agents/sdk";
import { OpenAIModel } from "@strands-agents/sdk/models/openai";
import { Respan } from "@respan/respan";
import { StrandsAgentsInstrumentor } from "@respan/instrumentation-strands-agents";
import { z } from "zod";

const respanApiKey = process.env.RESPAN_API_KEY!;
const respanBaseURL = process.env.RESPAN_BASE_URL ?? "https://api.respan.ai/api";

const respan = new Respan({
  apiKey: respanApiKey,
  baseURL: respanBaseURL,
  instrumentations: [new StrandsAgentsInstrumentor()],
});
await respan.initialize();

const model = new OpenAIModel({
  api: "chat",
  modelId: "gpt-4.1-nano",
  apiKey: respanApiKey,
  clientConfig: { baseURL: respanBaseURL },
});

const getWeather = tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string() }),
  callback: ({ city }) => `The weather in ${city} is sunny and 72F.`,
});

const agent = new Agent({
  name: "WeatherAgent",
  model,
  tools: [getWeather],
  systemPrompt: "You are a concise weather assistant.",
  printer: false,
});

await agent.invoke("What is the weather in Seattle?");
await respan.flush();
```

## Notes

- Initialize `Respan` before constructing or running Strands agents so the
  Strands tracer uses the active Respan OpenTelemetry provider.
- Tool definitions are enabled by default through Strands'
  `gen_ai_tool_definitions` semantic-convention opt-in and are exported as
  canonical `llm.request.functions` attributes when Strands emits them.
- The instrumentor translates Strands agent, model, tool, loop, graph, swarm,
  and node spans without adding off-contract `respan.span.*` aliases.
