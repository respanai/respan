# Respan Mistral AI instrumentation

Trace the official `mistralai` Python SDK with Respan.

This package wraps `openinference-instrumentation-mistralai` and registers
Respan's OpenInference translator so Mistral AI spans are emitted with the
canonical `traceloop.*`, `gen_ai.*`, `llm.*`, and `respan.*` fields expected by
the Respan OTLP pipeline.

## Install

```bash
pip install respan-ai respan-instrumentation-mistralai mistralai
```

## Usage

```python
import os

from mistralai.client import Mistral
from respan import Respan
from respan_instrumentation_mistralai import MistralAIInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[MistralAIInstrumentor()],
)

with Mistral(api_key=os.environ["MISTRAL_API_KEY"]) as client:
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": "Reply with one concise sentence about tracing.",
            }
        ],
    )
    print(response.choices[0].message.content)

respan.flush()
respan.shutdown()
```

Any keyword arguments passed to `MistralAIInstrumentor(...)` are forwarded to the
underlying OpenInference instrumentor.
