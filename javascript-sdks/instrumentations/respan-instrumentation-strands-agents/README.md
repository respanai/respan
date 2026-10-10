# @respan/instrumentation-strands-agents

Translate the [Strands Agents TypeScript SDK](https://strandsagents.com/) native
OpenTelemetry spans into Respan agent, chat, tool, task, and workflow records.
Supported SDK versions: `>=1.0.0 <2.0.0`; tested with 1.0.0 and 1.20.0.
Use Node.js 22 or newer with the current SDK.

```bash
npm install @respan/respan @respan/instrumentation-strands-agents @strands-agents/sdk openai zod \
  @opentelemetry/exporter-trace-otlp-http@^0.219.0 @opentelemetry/exporter-metrics-otlp-http@^0.219.0
```

Strands 1.20 declares the 0.219 exporter peers above. npm keeps Respan’s
newer exporter dependency separate so both packages have compatible versions.

```typescript
import { Agent, tool } from "@strands-agents/sdk";
import { OpenAIModel } from "@strands-agents/sdk/models/openai";
import { Respan } from "@respan/respan";
import { StrandsAgentsInstrumentor } from "@respan/instrumentation-strands-agents";
import { z } from "zod";

const respan = new Respan({
  apiKey: process.env.RESPAN_API_KEY!,
  instrumentations: [new StrandsAgentsInstrumentor()],
});
await respan.initialize();
try {
  const agent = new Agent({
    name: "WeatherAgent",
    model: new OpenAIModel({
      api: "chat",
      modelId: "gpt-4.1-nano",
      apiKey: process.env.OPENAI_API_KEY!,
    }),
    tools: [
      tool({
        name: "get_weather",
        description: "Get the weather for a city.",
        inputSchema: z.object({ city: z.string() }),
        callback: ({ city }) => ({ city, forecast: "sunny" }),
      }),
    ],
    printer: false,
  });
  await agent.invoke("What is the weather in Seattle?");
} finally {
  await respan.shutdown();
}
```

Initialize Respan before constructing Strands agents. The plugin consumes native
spans without wrapping SDK calls, promises, iterators, callbacks, or tools. It
translates an export copy; other span processors retain the original SDK span.
The exported `enrichStrandsAgentsSpan(span)` helper retains its explicit in-place
attribute translation API.

Tool execution IDs, full system instructions, tool results, message history,
structured output, multimodal content, and observed usage are preserved in both
stable and experimental Strands conventions. Model and tool definitions are
captured only when Strands emits them. Agent tool names and definitions are
merged into canonical metadata; model request fields remain on chat spans. The SDK currently identifies its tracing
service in the provider field, so the plugin preserves that value without
inferring a model provider or missing usage totals.

Set `traceContent: false` on `StrandsAgentsInstrumentor`,
`RESPAN_TRACE_CONTENT=false`, or `TRACELOOP_TRACE_CONTENT=false` to omit content.
The canonical Traceloop context veto and OpenTelemetry suppression also apply.
A parent veto stays in effect for its observed descendants even if a child
scope enables content. Sampling and content checks happen before payload copies;
exported events and error messages follow the same policy. These controls apply
to Respan export copies; Strands itself owns its native spans and local traces.

Tool-definition collection uses Strands' `gen_ai_tool_definitions` opt-in by
default. Set `includeToolDefinitions: false` to avoid adding that opt-in.
Repeated activation is idempotent, multiple owners share one translation, and
spans already started when the final owner deactivates finish with their
original translation policy.
