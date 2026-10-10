# Respan IBM watsonx.ai instrumentation

Observe the official `ibm-watsonx-ai` Python SDK with native OpenTelemetry spans. Tested with IBM SDK 1.8.0 and the declared minimum 1.6.3; the checkout package version remains 0.1.0 (published 0.1.1).

```bash
pip install respan-tracing respan-instrumentation-watsonx ibm-watsonx-ai
```

Activate after configuring your application's tracer provider. The caller creates and closes IBM clients normally.

```python
from ibm_watsonx_ai.foundation_models import ModelInference
from opentelemetry.sdk.trace import TracerProvider
from respan_instrumentation_watsonx import WatsonxInstrumentor


def generate_with_tracing(model: ModelInference, provider: TracerProvider):
    instrumentor = WatsonxInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    try:
        return model.generate(prompt="Explain native instrumentation.")
    finally:
        instrumentor.deactivate()
```

`ModelInference` supports `generate`, `generate_text`, `generate_text_stream`, `chat`, `chat_stream`, `agenerate`, `agenerate_stream`, `achat`, and `achat_stream`. `Embeddings` supports `generate`, `embed_documents`, `embed_query`, `agenerate`, `aembed_documents`, and `aembed_query`. Nested convenience calls produce one inference span. Native batched requests retain all public inputs and the aggregate native result; observed HTTP request bodies are retained under `native_requests`. Native worker threads may perform HTTP calls outside the calling context: unobserved transport status and aggregate usage are omitted.

Canonical entity input retains the complete sanitized public request and observed native HTTP request, including full history, historical tool IDs, schemas, source settings, false values, zero values, and empty fields. Entity output retains the native response, including reasoning, provider feedback, tools, custom fields, and streamed frames. Embedding output is the full vector list; nonvector response fields remain in `respan.metadata.watsonx.result`. Usage, response model, and HTTP status are captured only when observed in the native SDK response/transport. Independent batch usage is retained inside the raw responses without inventing aggregate totals.

Native SSE is consumed by the IBM parser. Instrumentation never eagerly reads or replaces response bytes. Stream results use a protocol proxy around the native generator: concrete generator identity/type changes, while iteration, `send`/`throw`, `asend`/`athrow`, `close`/`aclose`, and native generator attributes delegate to it. Full consumption, partial close, pre-first close, errors, and instrumentation deactivation finalize once. Close async streams explicitly with `await stream.aclose()`; garbage collection finalizes telemetry, while the Python async generator lifecycle owns native asynchronous cleanup. Deactivation ends pending telemetry without closing customer resources.

Sampling and both OpenTelemetry/LLM suppression precede telemetry content extraction. `capture_content=False`, `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, canonical Respan content context, and Traceloop content context/attributes all veto capture irreversibly, including observed active/finished ancestors and late stream vetoes. Unknown local carriers are conservative; remote parents retain trace ancestry. Credentials are redacted from structured data, text, URLs, schemas, and fragmented streams. Opaque customer serialization hooks are not invoked for telemetry. Diagnostics, events, and retained content are cleared on veto. Privacy preserves structural provider/model/type/status fields.

Telemetry failures preserve native results, exceptions, contexts, and resource cleanup. Shared activation is idempotent, incompatible provider/content configuration raises `ValueError`, and deactivation removes only owned wrappers/processors/hooks. Direct dependencies are verified against released companions; `respan-sdk>=2.6.26` is required for the imported canonical attribute module.

The OpenTelemetry SDK's default 128-attribute limit can bound indexed convenience prompt/completion fields. Full canonical input/output, tools, and native metadata are written last and are not arbitrarily truncated by this integration. Controlled local examples and explicit export/provider modes are in `python/tracing/watsonx` in the examples repository. Stored Respan projections have a separate validation gate; source preservation does not imply all backend convenience fields are populated.
