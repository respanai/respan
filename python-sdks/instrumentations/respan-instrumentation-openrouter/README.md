# Respan OpenRouter instrumentation

Trace OpenRouter calls made through the OpenAI-compatible Python client with Respan.

OpenRouter's Python usage is OpenAI-compatible, so this package delegates SDK
patching to `respan-instrumentation-openai` and adds a package-local processor
that normalizes those spans as OpenRouter spans before export.

## Install

```bash
pip install respan-ai respan-instrumentation-openrouter openai
```

## Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Respan API key for trace export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for direct OpenRouter calls. |
| `OPENROUTER_BASE_URL` | No | Defaults to `https://openrouter.ai/api/v1`. |
| `OPENROUTER_MODEL` | No | Defaults to `openai/gpt-4o-mini`. |

The examples can also run through a Respan gateway configuration by providing
`OPENROUTER_USE_RESPAN_GATEWAY=true`, `RESPAN_GATEWAY_API_KEY`,
`RESPAN_GATEWAY_BASE_URL`, and `RESPAN_MODEL`. Without direct OpenRouter or
explicit gateway configuration, the example set uses a local OpenAI-compatible
mock so instrumentation and Respan export can still be verified.

## Usage

```python
import os

from openai import OpenAI
from respan import Respan, workflow
from respan_instrumentation_openrouter import OpenRouterInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[OpenRouterInstrumentor()],
)

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)


@workflow(name="openrouter_chat_example")
def run_chat() -> str:
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": "Reply with one concise sentence about observability.",
            }
        ],
    )
    return response.choices[0].message.content or ""


try:
    print(run_chat())
finally:
    respan.flush()
    respan.shutdown()
```

`OpenRouterInstrumentor(normalize_all_openai_spans=True)` is the default because
the package is intended for OpenRouter-specific OpenAI-compatible clients. Set
`normalize_all_openai_spans=False` if the same process also emits regular OpenAI
spans and only spans with OpenRouter URL markers should be rewritten.
