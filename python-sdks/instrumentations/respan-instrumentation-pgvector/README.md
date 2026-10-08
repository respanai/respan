# pgvector instrumentation

Trace actual pgvector registration and psycopg 3 sync/async execute/fetch APIs as
Respan TASK spans. Native connections, cursors, row objects, generators,
transactions, callbacks and resource cleanup stay under the driver. The adapter
never fetches extra rows or replaces a cursor factory.

```bash
python -m pip install 'psycopg[binary]>=3.1.12,<4' pgvector respan-instrumentation-pgvector
```

The optional `psycopg2` extra traces pgvector's psycopg 2 registration call; it
does not wrap psycopg 2 query/fetch APIs. For review before the paired change is
published, install this checkout with `python -m pip install .` from the package
directory or install its validated wheel. The older published plugin retains
preview limits and does not contain these repairs.

Configure an initialized native connection and an OTel provider/exporter in the
application, then pass that provider explicitly:

```python
import pgvector.psycopg as vector_adapter
import psycopg
from opentelemetry.sdk.trace import TracerProvider

from respan_instrumentation_pgvector import PGVectorInstrumentor


def fetch_vectors(connection: psycopg.Connection, provider: TracerProvider):
    instrumentor = PGVectorInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    try:
        vector_adapter.register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute("SELECT embedding FROM items ORDER BY id")
            return cursor.fetchall()
    finally:
        instrumentor.deactivate()
```

Without an explicit provider, calls resolve the application's current global
provider. A vector extension must be installed in the actual PostgreSQL server.
The generic upstream Psycopg instrumentor changes native cursor factories and
traces SQL execution; it does not supply canonical pgvector registration/fetch
results. This package retains its native API boundary to preserve concrete
cursor types and capture the actual values consumed by callers.

Canonical JSON input preserves actual arguments/kwargs, known native pgvector
values, JSON/JSONB wrappers, SQL composition storage and compiled SQL where the
driver exposes it. Fetch output retains complete rows, vectors, histories,
schemas, false, zero, empty and null values. No adapter size/item cap applies.
Vector and half-vector values are full arrays; sparse vectors retain dimensions,
all indices and values; bit values retain length and full bits. Native bytes
remain full base64 with textual credentials redacted in the captured copy.
Current pgvector 0.5 can return bytes for binary bit queries; the actual byte
result is captured without inventing a Bit return object. Pgvector 0.3 uses its
native `pgvector.utils` models and NumPy storage instead.

Execute calls that return a cursor capture its actual cached/libpq metadata;
they do not drain rows. Actual fetch APIs emit the values they return. Native
`None`, including registration and executemany returns, remains JSON null.
Native cursor protocol/column metadata is retained at
`respan.metadata.pgvector.result`. Unknown objects, custom row factories and
parameter iterators are opaque to telemetry; their conversion or consumption
hooks are never called solely for capture. Known finite parameter lists retain
all items; a caller generator is consumed only by psycopg and is not materialized
by the adapter. Native iteration, `nextset`, context managers, commit/rollback,
and cursor closure keep the original behavior.

Spans carry upstream database system/operation/namespace/query/server fields
from actual native data. Raised native errors carry their actual type and SQLSTATE
where available. No HTTP status, embedding generation, model or token usage is
invented, and an error result is never fabricated. Actual bare ERROR statuses
set by application processors survive without invented error diagnostics.

Content capture defaults on. Use `capture_content=False`, canonical Respan or
Traceloop context/attribute vetoes, or `RESPAN_TRACE_CONTENT=false` /
`TRACELOOP_TRACE_CONTENT=false` for bodyless spans. Sampling and both general/LLM
suppression precede extraction. Initial and observed ancestor vetoes cannot
widen later; unknown local carriers fail closed, true remote context propagates.
Credential values in native JSON, quoted text, URLs, sensitive schema defaults,
and tuples with actual credential column names are redacted. Known metadata
uses static builtin/native storage and actual libpq descriptions, avoiding
customer getters and adapter lookups. Late vetoes scrub retained live and
ReadableSpan content/events/status descriptions, including immutable attribute
stores. Approved structural run markers remain available.

Activation/deactivation and `instrument`/`uninstrument` are idempotent; matching
owners share patches and conflicting active configuration raises `ValueError`.
Rollback/restoration uses descriptor presence and wrapper identity, retaining
foreign wrappers/processors. Telemetry startup/serialization/attribute/context/
end faults preserve native outcomes. Deactivating during an async query ends its
owned span once with content cleared; it leaves the native query/cursor open to
complete normally.

Native local-engine examples are in `python/tracing/pgvector` in the paired
examples repository. They use real bundled PostgreSQL or an explicitly selected
isolated binary installation, with a temporary data directory and cleanup.
The validated engine is PostgreSQL 16.2 with pgvector extension 0.8.7. The pgserver
0.1.4 wheel bundles extension 0.6.2; half/sparse scenarios require extension 0.7+
and explicitly skip when those native types are absent. PostgreSQL and pgvector
retain their own dimension/index bounds. Synthetic export and existing-server
access have separate explicit opt-ins; HTTP export success does not establish
stored-trace semantic acceptance.
