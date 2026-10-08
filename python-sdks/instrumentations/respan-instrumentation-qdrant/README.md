# Respan Qdrant instrumentation

Trace real `QdrantClient` and `AsyncQdrantClient` operations with Respan. The adapter supports released `qdrant-client` 1.9.0 through 1.19.1, including the current universal query, prefetch, fusion, sparse and multivector models. Methods absent from an older SDK are left alone; its legacy search and recommendation APIs remain instrumented.

## Install

Install this checkout or its built wheel before running the companion examples; the published 0.1.0 package does not contain these repairs.

```bash
pip install ./python-sdks/instrumentations/respan-instrumentation-qdrant
```

The package requires Python 3.11–3.13, `respan-sdk>=2.6.26`, and `respan-tracing>=2.17.0,<3`.

## Use

Initialize Respan tracing, then activate the instrumentor before making SDK calls. The local Qdrant engine works without a server or Qdrant credentials.

```python
from qdrant_client import QdrantClient, models
from respan_tracing import RespanTracer

from respan_instrumentation_qdrant import QdrantInstrumentor

RespanTracer(app_name="qdrant-example")
client = QdrantClient(":memory:")
instrumentor = QdrantInstrumentor()
instrumentor.activate()
try:
    client.create_collection(
        "documents",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT),
    )
    client.upsert(
        "documents",
        [models.PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"active": False})],
    )
    records = client.retrieve("documents", [1], with_vectors=True)
    assert records[0].payload["active"] is False
finally:
    instrumentor.deactivate()
    client.close()
```

The same instrumentor covers initialized clients, native async methods, collection configuration, aliases, payload indexes, queries, mutations and uploads. Client factories and `close()` do not create operation spans. SDK return values, native exceptions and upload iterators retain their behavior. In particular, instrumentation does not consume a generator before the uploader reads it.

## Captured data

Each operation produces a task span with canonical `traceloop.entity.input` and `traceloop.entity.output`, database attributes and the original SDK operation name. The Respan exporter applies its semantic or legacy naming style. Full known native models, protobuf messages, primitive NumPy arrays, vectors, batches and payloads are captured without row, dimension or character preview limits. False, zero and empty values are retained. Credential fields, URL credentials and authorization tokens are redacted.

Opaque objects and iterators are represented as `null`; the adapter does not invoke user getters, serializers, string conversion or iteration to obtain telemetry. This also means a generator upload's point contents are not copied into the input span. No embedding model, token usage, error result or successful HTTP status is invented. Native HTTP errors can provide their actual response status; local errors have no HTTP status.

## Content controls

Set `capture_content=False` on the instrumentor to keep operation structure while removing input, output, error descriptions and events. Canonical `respan_enable_content_tracing=False`, legacy `trace_content=False` / `override_enable_content_tracing=False`, and `RESPAN_TRACE_CONTENT` / `TRACELOOP_TRACE_CONTENT` false values also disable content. General and language-model OTel suppression bypass instrumentation; unsampled calls do not serialize content.

```python
from opentelemetry import context
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


def retrieve_private(client):
    token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    try:
        return client.retrieve("documents", [1])
    finally:
        context.detach(token)
```

A veto observed on a local ancestor or during a pending operation remains effective through export. Unknown local parents fail closed; remote parents are allowed unless their supplied context disables content. Compatible owners share hooks, conflicting configurations raise, and the last owner restores only hooks it owns. Telemetry faults do not replace native results or exceptions.

## Validation and examples

Native tests use the embedded Qdrant engine, real HTTP clients and SDK models on current and minimum versions. Companion examples live at `python/tracing/qdrant` in `respan-example-projects`; they run locally by default and export only when explicitly enabled. The native adapter remains necessary because the released upstream delegate consumes upload generators before the SDK and does not cover the current async query boundaries correctly.
