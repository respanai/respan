# Gateway

Set up and use the Respan **gateway** — routing an app's LLM calls through the Respan
proxy for logging, caching, key management, fallbacks, and model switching.

This file covers **both** the setup steps and the feature reference. Gateway is a
separate setup from tracing — do one at a time, never both in the same pass. Gateway
setup does **not** install instrumentation packages; it repoints the LLM client's base
URL at the Respan proxy and applies the framework's specific wiring.

Docs: `https://respan.ai/docs/documentation/features/gateway/gateway-quickstart.md`
Full docs index: `https://www.respan.ai/docs/llms.txt`

---

## Setup

Use this section when the user asks to set up the Respan gateway.

### Hard Rules

- **Interactive mode:** Ask the user questions when you need input. Do not assume.
- **Only add Respan wiring.** Do not refactor or modify unrelated code.
- **The gateway serves only the completion / text-generation operation**, exposed per provider:
  - OpenAI-compatible → `https://api.respan.ai/api/chat/completions`
  - Anthropic Messages → `https://api.respan.ai/api/anthropic/`
  - Google Gemini → `https://api.respan.ai/api/google/gemini/...`

  **Use the exact base path from the per-SDK gateway doc you fetch in Step 4** — it is authoritative. Do **NOT** repoint clients used for any other operation — embeddings, moderation, images, audio/speech, assistants, batch, fine-tuning — they are not supported and will fail (e.g. 404). Leave those untouched.
- **Use chat completions, not the Responses API.** The gateway serves chat completions (`/api/chat/completions`), not the OpenAI Responses API (`/api/responses`). SDKs that *default* to the Responses API are still fully supported — you just switch them to chat-completions mode, per their per-SDK doc:
  - **OpenAI Agents SDK** — Python: `set_default_openai_client(AsyncOpenAI(base_url=..., api_key=os.environ["RESPAN_API_KEY"]))` + `set_default_openai_api("chat_completions")` (TS: `setDefaultOpenAIClient(...)` + `setOpenAIAPI("chat_completions")`).
  - **Vercel AI SDK / `@ai-sdk/openai`** — use `provider.chat(model)`, not the default `provider(model)` (which targets the Responses API).
- **Scope the wiring — do not blanket-rewrite.** Wire only the client(s) that make the completion call you are setting up. Do **NOT** repoint every client instance in the repo (especially in example/monorepo projects with many independent scripts).
- **Do NOT install instrumentation packages.** Gateway routing does not use them. (That is tracing setup — see `tracing.md`.)
- **Do not guess APIs or doc slugs.** Use only the patterns from the live integration doc you fetch, and the exact slug from the map below.
- **Pin exact versions** if you install a framework SDK that is genuinely missing.
- **If the gateway is already wired up, do not duplicate work.** Check for an existing `base_url`/`baseURL` pointing at `api.respan.ai` first.
- **Read the code before proposing changes.** Find where the client is actually instantiated.

The API key is stored in `.env` as `RESPAN_API_KEY`. Gateway base URL: `https://api.respan.ai/api` (the proxy completions path is `https://api.respan.ai/api/chat/completions`).

### 1. Detect

Identify:
- **Language & package manager** — `package.json` (JS/TS), `pyproject.toml` / `requirements.txt` (Python), `Gemfile` (Ruby). Detect the package manager from lock files.
- **LLM library / framework in use** — match the project's dependencies against the **Detection → slug map** below.

### 2. Map to slug

Match the detected package(s) against the map. **Use the slug from the map verbatim — never auto-derive a slug from a display name** (the kebab-casing is irregular).

Apply the **priority rule:** a high-level framework wins over the raw SDK beneath it.
For example, a project with both `crewai` and `openai` maps to `crew-ai` (not `open-ai-sdk`),
because their gateway wiring differs. **If two frameworks match, ask the user which to configure.**

#### Detection → slug map

URL pattern: `https://respan.ai/docs/integrations/gateway/<slug>.md`

| Detected package(s) | Slug |
|---|---|
| `@openai/agents` / `openai-agents` | `open-ai-agents` |
| `claude-agent-sdk` | `claude-agent-sdk` |
| `ai` (Vercel AI SDK) | `vercel-ai-sdk` |
| `pydantic-ai` | `pydantic-ai` |
| `crewai` | `crew-ai` |
| `haystack-ai` | `haystack` |
| `langchain` | `lang-chain` |
| `llama-index` | `llama-index` |
| `autogen` / `autogen-agentchat` | `auto-gen` |
| `dspy` / `dspy-ai` | `ds-py` |
| `google-adk` | `google-adk` |
| `openai` | `open-ai-sdk` |
| `anthropic` / `@anthropic-ai/sdk` | `anthropic-sdk` |
| `google-genai` / `@google/genai` | `google-gen-ai` |
| `litellm` | `lite-llm` |
| `ruby_llm` (gem) | `ruby-llm` |
| `google-cloud-aiplatform` | `vertex-ai` |

> Namespace note: OpenAI, Anthropic, and Vertex each have both an SDK-level gateway page (`open-ai-sdk`, `anthropic-sdk`, `vertex-ai`) and a `model-providers/*` page. The SDK-level slugs above are the correct targets for this map.

### 3. Confirm

Show the user what was detected and which doc will be used. Wait for confirmation.

> Detected:
> - Language: [TypeScript/Python/Ruby]
> - Framework / SDK: [e.g. OpenAI Agents]
> - Gateway doc: https://respan.ai/docs/integrations/gateway/open-ai-agents.md
>
> Set up the gateway for this? (yes/no)

### 4. Fetch the live doc

Fetch `https://respan.ai/docs/integrations/gateway/<slug>.md` for the chosen slug.
Use the slug from the map verbatim.

If a pinned URL fails to resolve, that signals the **map is stale** and the table should be
updated — it does **not** mean you should guess an alternative slug.

### 5. Read the project's client code

Find where the LLM client / `base_url` is instantiated, and identify **which client(s) make
completion / text-generation calls** (chat completions, Anthropic `messages.create`, Gemini
`generate_content`) — that is the only path you repoint. This is the step that determines the
correct wiring — do not skip it.

If the project has multiple clients or many scripts (e.g. an examples repo), repoint **only**
the completion ones. Leave clients that call embeddings, moderation, images, audio,
assistants, batch, or fine-tuning exactly as they are — the gateway does not serve them.

### 6. Apply the framework-specific wiring

Apply the exact pattern from the fetched doc:

- Ensure `RESPAN_API_KEY` is set (it is in `.env`).
- Repoint `base_url` / `baseURL` to the gateway path from the fetched doc (e.g. `https://api.respan.ai/api` for OpenAI-compatible, `…/api/anthropic/` for Anthropic) — **only on the completion client(s)** identified in step 5, not on every client in the project.
- Apply the framework's exact wiring — e.g. OpenAI Agents requires `set_default_openai_client(AsyncOpenAI(base_url=...))`, **not** env vars.
- Install the framework SDK only if it is genuinely missing.
- **Do NOT install instrumentation packages.**

### 7. Verify (final test — always run this)

As the final step, **run the user's program once with a small request** through the
configured gateway and confirm it works:

```bash
respan logs list --limit 5
```

Confirm:
- The app runs without errors
- The request appears in the gateway logs (via the CLI above, or the Respan platform at https://platform.respan.ai)

If no log entry appears, the request is not actually routing through the gateway — debug
the `base_url`/wiring and re-run before declaring setup done.

---

## Reference

### Wiring Patterns

Point the LLM client's `base_url` at the Respan gateway. The Respan API key authenticates both the proxy and tracing.

#### Python (OpenAI SDK)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

#### TypeScript (OpenAI SDK)

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.RESPAN_API_KEY,
  baseURL: process.env.RESPAN_BASE_URL || "https://api.respan.ai/api",
});

const response = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hello!" }],
});
```

#### Python (Anthropic SDK)

```python
import os
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)
```

#### Environment variables

Set once, works everywhere:

```bash
export RESPAN_API_KEY="your-key"
export OPENAI_API_KEY="$RESPAN_API_KEY"
export OPENAI_BASE_URL="https://api.respan.ai/api"
```

### Framework-Specific Gateway Setup

#### CrewAI

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

from crewai import Agent, Task, Crew
# CrewAI uses the env vars automatically
```

#### Haystack

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

from haystack.components.generators import OpenAIGenerator
# OpenAIGenerator picks up env vars automatically
```

#### OpenAI Agents SDK

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
```

#### Pydantic AI

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

from pydantic_ai import Agent
agent = Agent(model="openai:gpt-4o-mini")
```

### Model Switching

The gateway supports 250+ models across providers. Use any model name:

```python
# OpenAI models
response = client.chat.completions.create(model="gpt-4o", ...)

# Anthropic models (via OpenAI SDK!)
response = client.chat.completions.create(model="claude-sonnet-4-20250514", ...)

# Google models
response = client.chat.completions.create(model="gemini-2.0-flash", ...)

# Open-source models
response = client.chat.completions.create(model="llama-3.1-70b", ...)
```

No code changes needed — just change the model string. View all models at `https://platform.respan.ai/platform/models`.

### Advanced Features

#### Fallback Models

Automatic failover if the primary model fails:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_body={
        "fallback_models": ["claude-sonnet-4-20250514", "gemini-2.0-flash"],
    },
)
```

#### Retries

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_body={
        "retries": 3,
    },
)
```

#### Caching

Enable response caching to save cost on repeated calls:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_body={
        "cache_enabled": True,
        "cache_ttl": 3600,  # seconds
    },
)
```

#### Streaming

Streaming works normally through the gateway:

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

#### Disable Logging

For sensitive data, disable logging on specific requests:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_body={
        "disable_log": True,
    },
)
```

### Key Benefits

- **Unified key management** — one Respan API key for all providers
- **Automatic logging** — every request logged as a span
- **Cost tracking** — per-request cost tracking across providers
- **Caching** — cache repeated LLM calls to reduce cost and latency
- **Fallbacks** — automatic failover between providers
- **Model switching** — swap models without code changes
- **~50-150ms added latency** — minimal overhead
