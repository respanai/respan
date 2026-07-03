# respan-instrumentation-openai

Respan instrumentation plugin for direct OpenAI SDK usage. A native, Traceloop-free instrumentation: it patches the OpenAI SDK directly and emits Respan's own LLM span conventions (`llm.request.type`, `gen_ai.*`, `traceloop.entity.*`, `respan.entity.log_type`), so chat, responses, completions, and embeddings calls are traced as real LLM calls with token/cost roll-up. Covers sync, async, and streaming.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-openai
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai`. |

All vendor-specific variables (e.g. `OPENAI_API_KEY`) are derived from these in your application code.

## Quickstart

### 3. Run Script

```python
import os
from openai import OpenAI
from respan import Respan
from respan_instrumentation_openai import OpenAIInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[OpenAIInstrumentor()],
)

client = OpenAI(
    api_key=respan_api_key,
    base_url=f"{respan_base_url}/api/openai",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

See the [examples/openai-sdk/](../../examples/openai-sdk/) directory for runnable examples including streaming, tool use, Responses API, and gateway routing.
