# respan-instrumentation-cohere

Respan instrumentation plugin for the Cohere Python SDK. It activates `opentelemetry-instrumentation-cohere` and normalizes Cohere spans to the Respan span contract before export.

## Install

```bash
pip install respan-ai respan-instrumentation-cohere cohere python-dotenv
```

## Supported SDK versions

Supports Cohere `>=5.0.0,<8`: V1 clients in 5.0 and V2 clients where the SDK provides them. Compatibility is checked with Cohere 5.0.0 / OpenTelemetry Cohere 0.60.0 and Cohere 7.2.0 / OpenTelemetry Cohere 0.62.4. The wrapper updates the upstream dependency check to this range; dependency validation remains enabled.

Chat, streaming chat, embeddings, and rerank work with synchronous and asynchronous clients. Embedding spans preserve text, image, or structured inputs and returned vectors. Streaming spans end on completion, transport failure, cancellation, or explicit early close. Content capture follows `TRACELOOP_TRACE_CONTENT` and the upstream context override.

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

## Examples

The companion [Cohere examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/cohere) cover chat, sync/async streaming, tools, embeddings, rerank, and controlled failures with released SDK HTTP transports. Use the paired examples update and its local adapter installation instructions before this instrumentation change is released.
