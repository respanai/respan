# Respan Weaviate instrumentation

Trace Weaviate Python v4 collection operations as Respan task spans. Sync and async collection, data, query, aggregate, configuration, tenant and batch manager calls keep their native results and errors.

```bash
pip install respan-ai respan-instrumentation-weaviate weaviate-client
```

Use the Respan facade to enable the plugin:

```python
from respan import Respan

respan = Respan(api_key="your-respan-key", instrumentations=["weaviate"])
# Use your connected WeaviateClient or WeaviateAsyncClient normally.
respan.shutdown()
```

For local recording with your own OpenTelemetry provider:

```python
from opentelemetry.sdk.trace import TracerProvider

from respan_instrumentation_weaviate import WeaviateInstrumentor

provider = TracerProvider()
instrumentor = WeaviateInstrumentor(capture_content=True)
instrumentor.activate(tracer_provider=provider)
try:
    # Use your connected Weaviate client here.
    pass
finally:
    instrumentor.deactivate()
    provider.shutdown()
```

Content uses canonical JSON input and output attributes. Complete supported native model fields, lists, IDs, vectors and false/zero/empty values are retained; there is no instrumentation payload limit. Credential values are redacted. Unknown customer objects are represented by their type without calling conversion, iteration or formatting hooks. A collection handle result records its name and type rather than connection internals. Inputs retain supplied positional and keyword arguments; SDK defaults are not invented.

Capture can be disabled with `capture_content=False`, `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, or the canonical Respan/native Traceloop context veto. Observed ancestor vetoes are irreversible, including finished parents. Unknown local parents fail closed. Sampling and both general and language-model suppression precede content inspection. Privacy removes owned input, output, diagnostics and events before export. Database system, operation and collection metadata use standard OpenTelemetry attributes; no model, embedding usage, HTTP status or error result is fabricated.

This package retains its native adapter because the released upstream Weaviate instrumentor does not provide these canonical results or equivalent async manager coverage and can emit overlapping synchronous query spans. Supported native clients are `weaviate-client>=4.23.0,<5`; async manager methods are awaited at their native boundary. The SDK's own retry, callbacks, query pagination, batching and resource cleanup remain in charge. Native collection iterators keep their concrete type and identity; constructing an iterator emits no span, and each executed query page is traced. The adapter does not wrap client/channel resources or drain iterators.

`data.ingest` records the complete returned native batch result. A caller iterable is consumed solely by the SDK, so an opaque generator input is represented structurally. Direct batch `add_object`/`add_reference` spans describe queue calls and their actual return values; `flush` describes the SDK wait boundary. They do not claim server completion, batch-worker HTTP/gRPC status, or errors that the SDK does not return from that call. Nested calls within a traced `ingest` operation produce one outer span. No thread or ambient-context patch is installed.

Multiple matching activations share ownership; conflicting configurations raise `ValueError`. Deactivation restores only owned descriptors/processors and retains foreign wrappers. Telemetry failures fail closed while preserving native outcomes and the prior ambient context. The SDK and OpenTelemetry may still impose their own limits or raise native validation/service errors.

The paired [examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/weaviate) use initialized native clients and real local HTTP/gRPC protocol handlers by default. They exercise native decoding and callbacks, not a Weaviate storage engine or production relevance/consistency guarantees. Real-service access and Respan export are separate opt-ins.
