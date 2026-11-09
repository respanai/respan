# @respan/ai-sdk-provider

The [Respan gateway](https://www.respan.ai/docs/ai-gateway/quickstart) as a provider for the [Vercel AI SDK](https://ai-sdk.dev). One Respan API key calls OpenAI, Anthropic, Google and every other model in the catalog.

Each model is called in its own native format, so provider options such as `providerOptions.anthropic` keep working:

| Model | Format |
|---|---|
| Claude (`anthropic/...`, `claude-...`) | Anthropic Messages |
| Gemini (`google/...`, `gemini/...`, `gemini-...`) | Gemini generateContent |
| OpenAI (`openai/...`, `gpt-...`, `o4-...`) | OpenAI Responses |
| Everything else, and route names | OpenAI Chat Completions |

## Install

```bash
npm install ai @respan/ai-sdk-provider
```

Requires `ai` 7. Set `RESPAN_API_KEY`, or pass `apiKey`.

## Use

```ts
import { generateText } from "ai";
import { respan } from "@respan/ai-sdk-provider";

const { text } = await generateText({
  model: respan("anthropic/claude-sonnet-4.5"),
  prompt: "Hello!",
});
```

To keep string model IDs, make Respan the default provider once at startup:

```ts
import { respan } from "@respan/ai-sdk-provider";

globalThis.AI_SDK_DEFAULT_PROVIDER = respan;

const { text } = await generateText({ model: "anthropic/claude-sonnet-4.5", prompt: "Hello!" });
```

## Migrate from Vercel AI Gateway

Set the default provider as above and keep your model IDs. `openai/`, `anthropic/` and `google/` name the model's maker, as they do in Vercel AI Gateway: the gateway routes the model across every provider that serves it, with fallback. Any other prefix pins a Respan provider, such as `azure/gpt-5.5`.

## Model IDs

| ID | Sent to the gateway as |
|---|---|
| `anthropic/claude-sonnet-4.5` | `claude-sonnet-4.5` |
| `google/gemini-3.5-flash` | `gemini-3.5-flash` |
| `openai/gpt-5.5` | `gpt-5.5` |
| `azure/gpt-5.5` | `azure/gpt-5.5` (pinned to Azure) |
| `smart` | `smart` (a route) |

Version dots and dashes are interchangeable: `claude-sonnet-4.5` and `claude-sonnet-4-5` are the same model.

## Options

```ts
import { createRespan } from "@respan/ai-sdk-provider";

const respan = createRespan({
  apiKey: process.env.RESPAN_API_KEY,   // default: RESPAN_API_KEY
  baseURL: "https://gateway.respan.ai", // default: RESPAN_GATEWAY_URL, then this
  headers: {},                          // sent with every request
});
```

`respan.embeddingModel(id)` and `respan.imageModel(id)` work the same way.
