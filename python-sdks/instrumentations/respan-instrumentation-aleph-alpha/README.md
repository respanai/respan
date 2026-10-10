# Aleph Alpha instrumentation

Trace the released `aleph-alpha-client` Python SDK with native OpenTelemetry sampling. The supported SDK range starts at 11.5.0; controlled native HTTP/SSE validation also covers 11.5.1. The [official SDK repository](https://github.com/Aleph-Alpha/aleph-alpha-client) was archived in September 2026.

```python
from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry.sdk.trace import TracerProvider
from respan_instrumentation_aleph_alpha import AlephAlphaInstrumentor

provider = TracerProvider()
# Add your local or Respan exporter to this provider.
instrumentor = AlephAlphaInstrumentor(tracer_provider=provider)
instrumentor.activate()
client = Client(token="your-provider-token", host="https://api.aleph-alpha.com/")
try:
    response = client.complete(
        CompletionRequest(Prompt.from_text("Explain tracing."), maximum_tokens=32),
        model="your-model",
    )
finally:
    client.session.close()
    instrumentor.deactivate()
    provider.shutdown()
```

The plugin entry point is `aleph-alpha`. A Respan facade can load `AlephAlphaInstrumentor()` using its `instrumentations` option. An explicit `tracer_provider` supports local recording; without one, the instrumentor resolves the global provider at each call, including a provider installed after activation.

Supported native methods are sync and async `chat`, `complete`, `embed`, `semantic_embed`, `batch_semantic_embed`, `instructable_embed`, `evaluate`, and `explain`; sync `embeddings`; and async `chat_with_streaming` and `complete_with_streaming`. SDK translation, reranking, tokenization, tokenizer downloads, steering-concept creation, and administrative endpoints are outside this package's declared scope. No managed deployment or agent API is instrumented. The existing native adapter remains necessary: the released upstream instrumentor covers only sync completion and raises `IndexError` for an otherwise successful empty completion response.

Typed results and exceptions pass through unchanged. Streaming returns an `AsyncGenerator` protocol proxy that forwards `__anext__`, `asend`, `athrow`, and `aclose` to the native generator. Its concrete Python type differs from an async generator. The SDK's HTTP transport, parsers, retries and client context managers remain native. Exhausted streams and eager successful calls have OK status; partial, unconsumed or abandoned streams have UNSET status. Explicit close forwards native cleanup. Garbage collection ends telemetry without pulling data or starting network activity; callers should still close their native streams and clients explicitly.

Canonical input/output retain complete messages, multimodal prompt parts, response envelopes, and embedding vectors without an instrumentation truncation limit. Embedding metadata retains the native non-vector envelope, including model version and actual usage. Tool definitions and native tool calls map to GenAI fields. Capture-gated `respan.metadata.aleph_alpha.request` records native request configuration and JSON bodies already produced by the SDK, including actual structured-output schemas, steering settings, false, zero and empty values. Convenience indexed histories can be bounded by OpenTelemetry's native attribute limit; complete canonical bodies and tools are written after those indexes.

Usage comes only from native provider fields. No missing total, HTTP success code, output, role, or model is synthesized; `model_version` remains native metadata rather than replacing the requested model. HTTP error status is observed from the SDK's actual status-check path. Unknown object conversion, string, property and iterator hooks are never invoked solely for telemetry.

Set `capture_content=False`, any supported context flag (`respan_enable_content_tracing`, `trace_content`, `override_enable_content_tracing`) to `False`, or `RESPAN_TRACE_CONTENT` / `TRACELOOP_TRACE_CONTENT` to `false`, `0`, `no`, or `off` to disable content capture. Ambient and supplied context restrictions are combined. Capture restrictions latch for the call, including active or finished observed local ancestors and restrictions visible before context detach. Unknown local parent identities fail closed; remote parents remain valid. Both general and language-model suppression are honored before capture. Credential values in known JSON, URLs and text are redacted; schema property names remain intact while sensitive defaults/examples are redacted.

Activation is shared and idempotent for matching settings. Conflicting provider, context or capture settings raise `RuntimeError`. Deactivation restores only wrappers/processors still owned by this instrumentor and clears pending captured bodies. Telemetry failures do not change native SDK results or cleanup.

Run the native fixture tests with `pytest tests`. Paired examples under `python/tracing/aleph-alpha` default to local recording and controlled HTTP/SSE. Export requires explicit `RESPAN_EXAMPLE_EXPORT=1`; real provider calls use a separate opt-in. Installed Respan SDK 2.6.26 is the supported shared-constant floor; 2.6.1 does not provide the required `span_attributes` module. Package versions are managed by the repository release workflow.
