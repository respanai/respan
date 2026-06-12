# respan-instrumentation-cohere

Respan instrumentation plugin for the Cohere Python SDK. It activates `opentelemetry-instrumentation-cohere` and normalizes Cohere spans to the Respan span contract before export.

## Install

```bash
pip install respan-ai respan-instrumentation-cohere cohere python-dotenv
```

## Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Respan API key used to export traces. |
| `RESPAN_BASE_URL` | No | Respan API base URL. Defaults to `https://api.respan.ai/api`. |
| `CO_API_KEY` | Yes | Cohere API key used by the Cohere SDK. |

## Quickstart

```python
import os

import cohere
from dotenv import load_dotenv
from respan import Respan, workflow
from respan_instrumentation_cohere import CohereInstrumentor

load_dotenv()

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[CohereInstrumentor()],
)
client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])


@workflow(name="cohere_chat_quickstart")
def run_chat() -> str:
    response = client.chat(
        model=os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025"),
        messages=[
            {
                "role": "user",
                "content": "Say hello in three languages.",
            }
        ],
    )
    return response.message.content[0].text


print(run_chat())
respan.flush()
```

## Notes

The upstream Cohere OpenTelemetry instrumentor emits Cohere SDK spans. This package adds a Respan span processor that:

- sets `gen_ai.system` to `cohere`
- adds `respan.entity.log_type`
- publishes both modern and legacy token usage attributes
- converts indexed tool definitions and tool calls into JSON string attributes
- strips off-contract shortcut aliases before export
