# Respan Helicone Exporter

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-helicone/)**

Helicone integration for exporting manual logs to Respan.

## Installation

```bash
pip install respan-exporter-helicone
```

## Quick Start

Add two lines to your existing Helicone code:

```python
from respan_exporter_helicone import HeliconeInstrumentor
HeliconeInstrumentor().instrument(api_key="your-respan-api-key")
```

That's it. All Helicone logs are now also sent to Respan.

## Full Example

```python
import json
import openai

from respan_exporter_helicone import HeliconeInstrumentor
from helicone_helpers import HeliconeManualLogger

# Add Respan instrumentation (your only addition)
HeliconeInstrumentor().instrument(api_key="your-respan-api-key")

# Initialize Helicone logger
logger = HeliconeManualLogger(api_key="your-helicone-api-key")
client = openai.OpenAI(api_key="your-openai-api-key")

# Make LLM calls - traces are sent to both Helicone and Respan
request = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
}

def chat_operation(result_recorder):
    response = client.chat.completions.create(**request)
    result_recorder.append_results(json.loads(response.to_json()))
    return response

logger.log_request(
    provider="openai",
    request=request,
    operation=chat_operation
)
```

## Configuration

| Env Var | Required | Default |
|---------|----------|---------|
| `RESPAN_API_KEY` | Yes (or pass `api_key=`) | — |
| `RESPAN_ENDPOINT` | No | `https://api.respan.ai/api/v1/traces/ingest` |

## What gets captured

- Model name & provider
- Input messages / prompt
- Output / completion
- Token usage (prompt, completion, total)
- Latency
- Helicone metadata (user ID, session ID, custom headers)

## License

MIT
