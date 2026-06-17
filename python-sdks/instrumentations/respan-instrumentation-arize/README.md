# Respan Instrumentation for Arize

Respan instrumentation plugin for the `arize` Python SDK.

## Installation

```bash
pip install respan-ai respan-instrumentation-arize arize
```

## Usage

```python
from arize import ArizeClient
from respan import Respan
from respan_instrumentation_arize import ArizeInstrumentor

respan = Respan(instrumentations=[ArizeInstrumentor()])
client = ArizeClient(api_key="...")

client.datasets.list(space="my-space")
respan.flush()
```

The instrumentor monkey-patches Arize SDK public client and subclient methods
and emits each SDK operation into the shared Respan OpenTelemetry pipeline as a
task span.
