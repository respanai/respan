# Respan LiteLLM Instrumentation

Respan instrumentation plugin for [LiteLLM](https://docs.litellm.ai/). It
registers a LiteLLM `CustomLogger` callback and emits canonical Respan chat
spans into `respan-tracing`.

## Installation

```bash
pip install respan-ai respan-instrumentation-litellm litellm
```

## Usage

```python
import os

import litellm
from respan import Respan, workflow
from respan_instrumentation_litellm import LiteLLMInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    app_name="litellm-example",
    instrumentations=[LiteLLMInstrumentor()],
)


@workflow(name="litellm_quickstart.workflow")
def run_completion() -> str:
    response = litellm.completion(
        api_key=os.environ["RESPAN_API_KEY"],
        api_base=os.getenv("RESPAN_GATEWAY_BASE_URL", "https://api.respan.ai/api"),
        model=os.getenv("RESPAN_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    return response.choices[0].message.content


print(run_completion())
respan.flush()
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_content` | `bool` | `True` | Capture request messages and assistant response content on chat spans. |

Use `Respan(..., customer_identifier=..., thread_identifier=..., metadata=...)`
or `respan.propagate_attributes(...)` to attach Respan attributes to LiteLLM
spans.
