# respan-instrumentation-pinecone

Respan instrumentation for Pinecone data, control-plane, and inference operations,
including the synchronous, asynchronous, and gRPC clients.

```bash
pip install respan-ai respan-instrumentation-pinecone pinecone
```

```python
from pinecone import Pinecone
from respan import Respan
from respan_instrumentation_pinecone import PineconeInstrumentor

respan = Respan(instrumentations=[PineconeInstrumentor()])
index = Pinecone().index(name="documents")
print(index.query(vector=[0.1, 0.2], top_k=3))
respan.flush()
```

## Supported SDKs and operations

The adapter supports Pinecone 5.1 through 10.x. The legacy sync vector client
remains instrumented. Pinecone 10 adds document upsert, batch upsert, search,
fetch, update, delete, and listing for both sync and async clients. Listing
preserves the SDK paginator, including `pages()`, `to_list()`, and pagination
tokens. Each fetched page produces a span; creating a paginator does not issue
a request or emit a completed request span.

Document responses preserve fields from Pinecone's msgspec structs and document
models in canonical entity output. Existing content capture limits and redaction
still apply. Set `PineconeInstrumentor(capture_content=False)` to omit inputs and
outputs. Error spans keep the service status while the original exception is
raised to the caller.

Compatibility tests use the released SDK against a local HTTP fixture. A live
Pinecone service requires an existing index and provider credentials.

Async vector, namespace, and bulk-import iterators remain lazy, record one span
per consumed iterator, and finalize on early close or errors. Control-plane index
listing retains its native paginator. Existing gRPC patch targets remain; this
compatibility suite validates HTTP clients and does not exercise a gRPC service.

See the [Python Pinecone examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/pinecone)
for complete deterministic scenarios.
