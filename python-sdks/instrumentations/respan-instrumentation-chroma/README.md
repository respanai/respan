# respan-instrumentation-chroma

Trace native Chroma Python collection and client operations as Respan task spans. Requests and responses retain full records, metadata, query results and vectors, including numpy arrays. Database queries and writes do not invent embedding model names or token usage. A native `None` return is captured as JSON `null`.

```bash
pip install respan-instrumentation-chroma chromadb
```

This plugin supports Chroma `>=0.5.0,<2.0.0` on Python 3.11–3.13. Chroma 0.5 requires a compatible numpy release (`numpy>=1.26.4,<2`); install that constraint when using the minimum Chroma release. The plugin requires `respan-sdk>=2.6.26`, which exposes the canonical span constants, and released upstream Chroma instrumentation `>=0.54.0` with its OTel 1.38 and AI semantic convention 0.5.1 dependency floor.

## Local recording

```python
import tempfile

import chromadb
from chromadb.config import Settings
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from respan_instrumentation_chroma import ChromaInstrumentor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
owner = ChromaInstrumentor(tracer_provider=provider)
owner.activate()
try:
    with tempfile.TemporaryDirectory() as directory:
        client = chromadb.PersistentClient(
            path=directory, settings=Settings(anonymized_telemetry=False)
        )
        collection = client.create_collection("native_docs", embedding_function=None)
        collection.add(
            ids=["doc-1"], documents=["Native Chroma"], embeddings=[[0.0, 1.0]]
        )
        result = collection.query(query_embeddings=[[0.0, 1.0]], n_results=1)
        print(result["ids"])
        if hasattr(client, "close"):
            client.close()
        else:
            client._system.stop()
finally:
    owner.deactivate()
    provider.shutdown()
```

To export these spans, add the released Respan exporter to your provider before activating the plugin:

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

The plugin is also registered under the `chroma` entry point in `respan.instrumentations` for Respan discovery.

## Native API coverage

The plugin retains the released upstream synchronous delegate for `Collection.add/get/peek/query/modify/update/upsert/delete` and the upstream `SegmentAPI._query` boundary where the native SDK uses it. It replaces lossy upstream summaries and unsafe exception coercion with complete capture-gated canonical input and output on those spans. This can produce a nested SegmentAPI query span on older Chroma engines.

Additional spans cover client collection creation, lookup, listing/counting/deletion and reset; collection count, indexing status, fork/fork count, search and attached-function operations; and released async client/collection operations. A method is installed only when the selected Chroma release actually exposes it. Chroma 0.5 has no async client API. New cloud-only operations such as fork/search retain their native local-engine capability errors; instrumentation does not enable unsupported server features. Client construction, embedding function calls and handle cleanup keep their native behavior.

Each database span uses `respan.entity.log_type=task`, standard database fields and JSON `traceloop.entity.input/output`. The input contains the bound native arguments and collection model when available. Return values, exceptions, callback invocation counts and resource lifetimes remain those of Chroma. Native embedding callbacks may compute vectors during a database operation; that database span still represents database work.

## Content controls

Use `ChromaInstrumentor(capture_content=False, tracer_provider=provider)` to disable bodies. The canonical `respan_enable_content_tracing=False`, supported legacy context flags, `RESPAN_TRACE_CONTENT` and `TRACELOOP_TRACE_CONTENT` off values (`false`, `0`, `no`, `off`) also disable content. The initial choice cannot widen later. Supplied context and ambient context both bound capture, and observed local ancestors retain vetoes through span end. Unknown local ancestors fail closed; valid remote parents remain usable.

Native OTel sampling and general/LLM suppression are checked before content extraction. Later vetoes remove captured bodies, error messages/descriptions and events before export, including the SDK ReadableSpan end snapshot. Credential values are redacted while schema property names and complete native data shapes remain available. Unknown object conversion/getter/iterator hooks are not invoked solely for tracing. OTel's configured span attribute count remains its native limit; complete canonical bodies are written last.

Multiple owners can share identical provider/context/capture/callback configuration. Conflicting configurations fail clearly. Deactivation restores only instrumentation-owned wrappers and context hooks, and preserves foreign replacements. Telemetry start, context, serialization, attribute and end failures do not replace native results or errors.

Runnable local persistent-engine and native HTTP examples live in [python/tracing/chroma](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/chroma).
