# respan-instrumentation-lancedb

Trace embedded LanceDB connection, table, merge and query execution as Respan
`task` spans with native database operation and collection attributes. Tested
against LanceDB 0.20.0 and 0.40.0 with released tracing dependencies.

```bash
pip install respan-instrumentation-lancedb lancedb pandas
```

```python
from tempfile import TemporaryDirectory

import lancedb
from opentelemetry.sdk.trace import TracerProvider

from respan_instrumentation_lancedb import LanceDBInstrumentor

provider = TracerProvider()
instrumentor = LanceDBInstrumentor()
instrumentor.activate(tracer_provider=provider)
try:
    with TemporaryDirectory() as directory:
        db = lancedb.connect(directory)
        table = db.create_table("documents", [{"vector": [0.1, 0.2], "text": "hello"}])
        print(table.search([0.1, 0.2]).limit(1).to_list())
finally:
    instrumentor.deactivate()
    provider.shutdown()
```

Supply your provider/exporter or load the `lancedb` Respan instrumentation plugin.
Activation is idempotent and shared across instrumentor instances with matching
configuration. `capture_content=False`, Respan/Traceloop context flags, both
suppression flags, and `RESPAN_TRACE_CONTENT`/`TRACELOOP_TRACE_CONTENT` environment
vetoes prevent payload capture. A veto cannot subsequently widen, including
observed active/finished ancestors. Unknown local ancestors fail closed.

Supported execution includes connection create/open/drop/list; table
add/update/delete, indexes and optimize; merge execution; and native vector,
scan, full-text, hybrid and take query `to_list`, `to_arrow`, `to_pandas`,
`explain_plan` and `to_batches`. Search/merge builder construction and modifiers
retain their native synchronous/async behavior and do not emit execution spans.
Async merge's synchronous public `execute` dispatch remains native; its awaited
`AsyncTable._do_merge` boundary records the actual result. Internal SDK nesting
emits one operation span. Existing deprecated convenience methods retain native
warnings; current LanceDB recommends `create_index(config=...)` and `list_tables`.

Full known builtin JSON, numpy vectors, native Arrow tables/batches/schema,
native result fields and ordinary numpy/primitive-Arrow-backed pandas results are retained.
Arrow schema/field metadata is retained in owned capture-gated
`respan.metadata.lancedb.request` / `respan.metadata.lancedb.result` alongside
complete canonical rows.
There are no item/vector/text limits. Credential values are redacted, including
quoted authorization strings, JSON arguments and URL credentials; schema
property names survive with sensitive defaults redacted. Unknown classes,
customer Arrow extension types and custom dataframe blocks are represented by
structural type information; instrumentation does not call their getters or
conversion hooks. It does not consume write iterators, evaluate a reranker, call
an embedding service, infer token usage/models, or manufacture HTTP statuses or
error-as-output values. Recorded errors come from the native operation.

Sync `to_batches` returns the exact native C `pyarrow.RecordBatchReader` with its
original identity, Arrow interoperability, iteration, `read_all`,
`read_next_batch`, close and context-manager behavior. Its creation span records
structural reader output and finishes when the native method returns. Lazy rows
and errors after that return are **not observed**: telemetry never drains or
replaces that reader. Use `to_arrow`/`to_list` to record complete sync results.
Async `to_batches` retains the native `AsyncRecordBatchReader` type and identity,
with an owned private iterator tap that records only batches the caller consumes.
Its span finishes at exhaustion, deactivation or garbage collection without
closing caller-owned resources. The native reader has no public close/aclose
API. Partial consumption records consumed batches when finalizing; unread readers
have no invented output. Native batch-size hints and default limits are preserved
(0.20.0 scan defaults differ from 0.40.0). Builder parameters created before
activation are available only where the SDK exposes safe Python storage; opaque
Rust query settings are not retrieved by extra SDK calls. The SDK's own span
attribute-count limit can bound convenience metadata; full canonical I/O is
written last.

The compatible examples in `python/tracing/lancedb` run the real local engine
with local OTel recording by default. `RESPAN_EXAMPLE_EXPORT=1` explicitly exports
to Respan; embedded operations need no provider key or production service calls.
Remote/cloud-specific SDK classes and embedding provider integrations are outside
this package's observed boundary.

The minimum test profile uses Arrow 17.0.0 with LanceDB 0.20.0. Arrow 14.0.1,
which that SDK declares compatible, fails bare native vector creation because
its `RecordBatch.set_column` API is absent; this is a vendor dependency limit,
not a tracing replacement or alias workaround.
