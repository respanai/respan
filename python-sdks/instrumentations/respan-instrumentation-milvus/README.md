# respan-instrumentation-milvus

Trace native PyMilvus client operations as Respan task spans. Capture complete requests, typed results, collection schemas, hybrid search requests, metadata and vectors. Database work does not invent embedding models, token usage or HTTP status codes. Native `None` values remain JSON `null`; errors have no fabricated result.

```bash
pip install respan-instrumentation-milvus pymilvus milvus-lite
```

The plugin supports PyMilvus `>=2.4.1,<4` on Python 3.11–3.13. It requires the canonical constants in `respan-sdk>=2.6.26`, upstream Milvus instrumentation `>=0.54.0`, and the upstream OTel 1.38/AI semantic convention 0.5.1 floors. The newest upstream 0.62.4 metadata currently requires unpublished AI semantic conventions 0.5.2; the healthy current profile uses released upstream 0.60.0 with AI 0.5.1. This plugin does not bypass that upstream dependency constraint.

## Local native recording

```python
import tempfile
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pymilvus import DataType, MilvusClient

from respan_instrumentation_milvus import MilvusInstrumentor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
owner = MilvusInstrumentor(tracer_provider=provider)
owner.activate()
try:
    with tempfile.TemporaryDirectory() as directory:
        client = MilvusClient(uri=str(Path(directory) / "native.db"))
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)
        client.create_collection("native_docs", schema=schema)
        client.insert("native_docs", [{"id": 1, "vector": [0.0, 1.0, 2.0, 3.0]}])
        result = client.query("native_docs", filter="id >= 0", output_fields=["vector"])
        print(len(result))
        client.close()
finally:
    owner.deactivate()
    provider.shutdown()
```

The local `.db` example uses current PyMilvus 3.0.2 and Milvus Lite 3.2.1. PyMilvus 2.4.1 requires a server URI and old import dependencies (`setuptools<81`, `marshmallow<4`). A server running current Lite needs a compatible current SDK protobuf environment; run it separately from the minimum client environment. No client or server modules are replaced by the instrumentation.

To export, add the released exporter to the provider before activation:

```python
import os

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from respan_tracing.exporters.respan import RespanSpanExporter

provider.add_span_processor(
    SimpleSpanProcessor(
        RespanSpanExporter(
            endpoint="https://api.respan.ai/api/v2/traces",
            api_key=os.environ["RESPAN_API_KEY"],
        )
    )
)
```

The plugin is registered as `milvus` under `respan.instrumentations`.

## Coverage and native behavior

The package preserves the released upstream synchronous delegate for collection creation, insert, upsert, delete, search, get, query and hybrid search. Complete capture-gated canonical payloads replace unsafe or lossy upstream summaries on those native spans. Upstream methods absent from the selected SDK are capability-gated rather than fabricated.

Additional boundaries cover exposed client collection/database/index/partition lifecycle, collection properties/fields/functions, aliases, snapshots, analyzer, flush/load/release, query/search iterators and optimization. Async client methods are installed when exposed. SDK 2.4.1 has no async client, client iterators, hybrid search, flush or optimization API. Remote-only or unsupported Lite operations keep their native capability errors.

The result object remains the SDK's exact object. Current lazy `HybridExtraList` rows are decoded on an owned native shadow, leaving the returned object's materialization flags untouched. Native extra result metadata is capture-gated under `respan.metadata.milvus.result`. Telemetry uses builtins and known native storage descriptors; unknown getters, mapping/iterator conversion hooks and metaclass hooks are not called solely for observation.

Iterators and optimization tasks retain their native handles and API methods. Spans defer until real consumption/result or resource termination. Fully exhausted native iterators succeed; partial/abandoned handles remain unset unless a real native error was observed. Collected output contains actual native batches. Creation does not drain an iterator. SDK-owned CPU optimization workers retain their operation parent and initial suppression through package-owned start/run hooks; global thread or ambient context implementations are not patched. Client closure, cancellation, deactivation and garbage collection clear owned observation state without adding vendor calls.

## Content and lifecycle controls

`MilvusInstrumentor(capture_content=False, tracer_provider=provider)` disables bodies. Canonical `respan_enable_content_tracing=False`, supported legacy flags and `RESPAN_TRACE_CONTENT`/`TRACELOOP_TRACE_CONTENT` off values (`false`, `0`, `no`, `off`) also bound capture. Initial, supplied, ambient and observed ancestor choices cannot widen. Unknown local ancestors fail closed; valid remote parents remain usable. Native sampling and general/LLM suppression precede extraction.

Late vetoes clear bodies, diagnostic messages/descriptions and events, including the actual SDK ReadableSpan end snapshot. Credential values are redacted without truncating vectors, history or schema property names. Native span attribute count limits remain configured by OTel; canonical bodies are written last.

Owners can share identical configuration. Conflicts fail clearly, partial installs roll back, and deactivation restores only owned descriptors while retaining foreign replacements. Telemetry start, serialization, context, attribute, delegate metric and end failures preserve native outcomes. Bare foreign error status survives cleanup without invented error type/message/output.

See the [paired native examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/milvus).
