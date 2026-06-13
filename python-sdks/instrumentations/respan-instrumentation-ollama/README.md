# respan-instrumentation-ollama

Respan instrumentation plugin for [Ollama](https://ollama.com/). It instruments the official Ollama Python client and emits chat, completion, and embedding spans into the Respan tracing pipeline.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-ollama ollama
```

`ollama` is the official Ollama Python client. A running Ollama server is required for real model calls.

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |
| `OLLAMA_HOST` | No | Ollama server URL. Defaults to the Ollama client default. |
| `OLLAMA_MODEL` | No | Model used by your application. |

## Quickstart

```python
import os

from dotenv import load_dotenv
from ollama import Client
from respan import Respan, workflow
from respan_instrumentation_ollama import OllamaInstrumentor

load_dotenv()

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[OllamaInstrumentor()],
)
client = Client(host=os.getenv("OLLAMA_HOST"))


@workflow(name="ollama_quickstart")
def run() -> str:
    response = client.chat(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        messages=[{"role": "user", "content": "Reply with one concise sentence."}],
    )
    return response["message"]["content"]


print(run())
respan.flush()
respan.shutdown()
```

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/ollama) for runnable scripts.
