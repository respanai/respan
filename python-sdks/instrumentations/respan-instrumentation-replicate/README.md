# Respan Replicate instrumentation

Trace the released Replicate Python SDK through its native clients, prediction objects, HTTP responses, and SSE parser. Checked with Replicate 1.0.7 and the declared minimum 1.0.0. Pre-release SDK versions are outside this compatibility claim.

```python
import replicate
from respan_instrumentation_replicate import ReplicateInstrumentor

instrumentor = ReplicateInstrumentor()
instrumentor.activate()
try:
    output = replicate.run("owner/model", input={"prompt": "Hello"})
finally:
    instrumentor.deactivate()
```

Configure a tracing provider/exporter through `respan-tracing`, or pass an existing `tracer_provider` to the instrumentor. The `replicate` entry point in `respan.instrumentations` supports plugin discovery. Activation is idempotent, compatible owners share wrappers, conflicting owners fail explicitly, and deactivation restores only wrappers owned by this integration.

## Covered APIs

- Module functions and `Client.run`, `async_run`, `stream`, and `async_stream`.
- `Client.predictions` create/get/list/cancel and their async variants.
- Model and deployment prediction creation, including async creation.
- Native `Prediction` wait/reload/cancel/stream and their async variants.

Nested SDK requests contribute their actual HTTP status and prediction metrics to the owning call. Native provider failures remain the original exceptions. HTTP status comes from an observed response or provider error; usage comes from numeric prediction metrics. Missing totals and HTTP status are omitted.

Prediction, page, file, scalar, and container results are returned unchanged. Native `FileOutput` values remain usable for caller-owned reads. The SDK's own `use_file_output` defaults differ between 1.0.0 and 1.0.7; pass the option explicitly if your application needs identical behavior across these versions.

## Streams and payloads

Streams remain lazy. The integration delegates next/send/throw/close or their async equivalents without draining the source, and retains the identity of each yielded native event. Exhaustion and explicit close end the span. Closing after partial consumption captures only observed events; closing without reading and garbage collection do not synthesize output. Abandoned proxies end a bodyless span without advancing the native iterator. Native SDK finalization owns its resource cleanup.

The stream return value is a protocol proxy around the native generator. Its concrete type and object identity differ from the uninstrumented generator. Keep explicit `close()`/`aclose()` calls for deterministic resource cleanup, especially for asynchronous streams.

JSON input, output, histories, tool schemas, token metrics, and vectors have no integration-imposed length cap. A provider embedding response with `object="embedding"` and an `embedding` list maps the complete vector to canonical output; the native response envelope and prediction snapshot are preserved in capture-gated `respan.metadata.replicate.*` fields. False, zero, and empty values are retained. OTel attribute-count and downstream storage/display limits still apply.

Serialization reads built-in values and known native SDK storage only. It does not call arbitrary user `str`, `repr`, model-dump methods, or getters for telemetry. Credential fields, authorization strings, quoted JSON secrets, and URL credentials are redacted while schema field names remain intact.

## Content controls

Use `ReplicateInstrumentor(capture_content=False)` to retain structural telemetry while excluding bodies and diagnostics. The integration also honors canonical and legacy content-context flags, `RESPAN_TRACE_CONTENT`, `TRACELOOP_TRACE_CONTENT`, general OTel suppression, and language-model suppression. Sampling and suppression run before content extraction.

Content permission is bounded by the call's initial context and cannot be widened later. Observed local ancestors, explicit context detach, stream resumes, span completion, and deactivation can irreversibly veto content. Unknown local parents fail closed; remote parents remain usable. A veto clears owned input/output, indexed content, metadata, diagnostics, events, status descriptions, and retained buffers. Native error type and sourced structural fields may remain. Pending calls are scrubbed before deactivation ends their spans.

## Validation

The companion [Replicate examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/replicate) exercise actual released SDK HTTP/SSE parsing against controlled local transports, with optional Respan trace export. Tests cover current and minimum SDK versions, complete 5,001-component vectors and 251-event streams, async prediction lifecycle, privacy, native errors/resources, shared owners, abandonment, and observer faults.
