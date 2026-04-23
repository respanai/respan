# Gateway Reference

Route LLM calls through the Respan proxy for logging, caching, key management, fallbacks, and model switching.

Docs: `https://respan.ai/docs/documentation/features/gateway/gateway-quickstart.md`

---

## Setup

Point the LLM client's `base_url` at the Respan gateway. The Respan API key authenticates both the proxy and tracing.

### Python (OpenAI SDK)

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

### TypeScript (OpenAI SDK)

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

### Python (Anthropic SDK)

```python
import os
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)
```

### Environment variables

Set once, works everywhere:

```bash
export RESPAN_API_KEY="your-key"
export OPENAI_API_KEY="$RESPAN_API_KEY"
export OPENAI_BASE_URL="https://api.respan.ai/api"
```

---

## Gateway + Tracing (Both)

Combine SDK tracing and gateway routing:

### Python

```python
import os
from openai import OpenAI
from respan import Respan

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

respan = Respan(api_key=respan_api_key, base_url=respan_base_url)

client = OpenAI(
    api_key=respan_api_key,
    base_url=respan_base_url,
)
```

### TypeScript

```typescript
import OpenAI from "openai";
import { Respan } from "@respan/respan";

const respanApiKey = process.env.RESPAN_API_KEY!;
const respanBaseUrl = process.env.RESPAN_BASE_URL || "https://api.respan.ai/api";

const respan = new Respan({ apiKey: respanApiKey, baseUrl: respanBaseUrl });
await respan.initialize();

const client = new OpenAI({
  apiKey: respanApiKey,
  baseURL: respanBaseUrl,
});
```

---

## Framework-Specific Gateway Setup

### CrewAI

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

from crewai import Agent, Task, Crew
# CrewAI uses the env vars automatically
```

### Haystack

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

from haystack.components.generators import OpenAIGenerator
# OpenAIGenerator picks up env vars automatically
```

### OpenAI Agents SDK

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
```

### Pydantic AI

```python
import os
os.environ["OPENAI_API_KEY"] = os.environ["RESPAN_API_KEY"]
os.environ["OPENAI_BASE_URL"] = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

from pydantic_ai import Agent
agent = Agent(model="openai:gpt-4o-mini")
```

---

## Model Switching

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

---

## Advanced Features

### Fallback Models

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

### Retries

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_body={
        "retries": 3,
    },
)
```

### Caching

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

### Streaming

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

### Disable Logging

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

---

## Key Benefits

- **Unified key management** — one Respan API key for all providers
- **Automatic logging** — every request logged as a span
- **Cost tracking** — per-request cost tracking across providers
- **Caching** — cache repeated LLM calls to reduce cost and latency
- **Fallbacks** — automatic failover between providers
- **Model switching** — swap models without code changes
- **~50-150ms added latency** — minimal overhead
