# respan-instrumentation-qdrant

Respan instrumentation for synchronous and asynchronous
[`qdrant-client`](https://github.com/qdrant/qdrant-client) operations. It emits
canonical Respan task spans for collection management, point writes, payload
updates, retrieval, filtering, vector queries, facets, snapshots, and cluster
administration.

## Install

```bash
pip install respan-ai respan-instrumentation-qdrant qdrant-client
```

## Quickstart

```python
from qdrant_client import QdrantClient, models
from respan import Respan, workflow
from respan_instrumentation_qdrant import QdrantInstrumentor

respan = Respan(instrumentations=[QdrantInstrumentor()])


@workflow(name="qdrant_local_query_workflow")
def run_query():
    client = QdrantClient(":memory:")
    client.create_collection(
        "docs",
        vectors_config=models.VectorParams(
            size=3,
            distance=models.Distance.COSINE,
        ),
    )
    client.upsert(
        "docs",
        points=[models.PointStruct(id=1, vector=[0.1, 0.2, 0.3])],
    )
    return client.query_points(
        "docs",
        query=[0.1, 0.2, 0.3],
        limit=1,
    )


print(run_query())
respan.shutdown()
```

`QdrantInstrumentor(capture_content=False)` keeps operation names, status, and
database attributes while omitting request and response bodies. Activation and
deactivation are idempotent, and `instrument()` / `uninstrument()` aliases are
also available.

See `respan-example-projects/python/tracing/qdrant` for collection, point,
query, async, and deterministic error examples.
