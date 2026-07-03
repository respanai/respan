# Respan Replicate Instrumentation

Trace the official `replicate` Python SDK with Respan.

The instrumentation patches the Replicate client lifecycle and emits canonical
Respan spans for `run`, `async_run`, `stream`, `async_stream`, prediction
creation, prediction waiting, and prediction lookup/cancel operations.

## Install

```bash
pip install respan-ai respan-instrumentation-replicate replicate
```

## Usage

```python
import os

import replicate
from respan import Respan, workflow
from respan_instrumentation_replicate import ReplicateInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[ReplicateInstrumentor()],
)


@workflow(name="replicate_quickstart.workflow")
def run_prediction() -> str:
    output = replicate.run(
        "meta/meta-llama-3-8b-instruct",
        input={"prompt": "Reply with one concise sentence about tracing."},
    )
    return "".join(str(chunk) for chunk in output) if not isinstance(output, str) else output


print(run_prediction())
respan.flush()
respan.shutdown()
```

Use `Respan(..., customer_identifier=..., thread_identifier=..., metadata=...)`
or `respan.propagate_attributes(...)` to attach Respan attributes to Replicate
spans.
