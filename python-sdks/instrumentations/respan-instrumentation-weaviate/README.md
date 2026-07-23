# respan-instrumentation-weaviate

Respan instrumentation for the Weaviate Python client v4. It emits canonical
task spans for synchronous and asynchronous collection lifecycle, data,
reference, vector and keyword query, aggregate, config, batch, and tenant
operations.

## Install

```bash
pip install respan-ai respan-instrumentation-weaviate "weaviate-client>=4.22.0,<5"
```

## Quickstart

```python
import weaviate
from weaviate.classes.config import Configure
from respan import Respan, workflow
from respan_instrumentation_weaviate import WeaviateInstrumentor

respan = Respan(instrumentations=[WeaviateInstrumentor()])


@workflow(name="weaviate_vector_query_workflow")
def run_query():
    client = weaviate.connect_to_local()
    collection = client.collections.create(
        "RespanDocs",
        vector_config=Configure.Vectors.self_provided(),
    )
    collection.data.insert(
        {"text": "Respan traces vector-store operations."},
        vector=[0.1, 0.2, 0.3],
    )
    result = collection.query.near_vector(
        near_vector=[0.1, 0.2, 0.3],
        limit=1,
    )
    client.close()
    return result


print(run_query())
respan.shutdown()
```

Pass `capture_content=False` to omit operation arguments and results while
retaining names, status, and database attributes. Activation/deactivation and
the `instrument()` / `uninstrument()` aliases are idempotent.

The adapter targets `weaviate-client>=4.22.0,<5` and its v4 collection API.
Generative search calls are left to an LLM-specific integration because they require chat
semantics rather than vector-store task semantics.

See `respan-example-projects/python/tracing/weaviate` for runnable sync, async,
batch, aggregate, query, mutation, and deterministic error examples.
