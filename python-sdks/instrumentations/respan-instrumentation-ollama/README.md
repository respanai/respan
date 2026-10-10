# respan-instrumentation-ollama

Observe the official Ollama Python SDK with OpenTelemetry sampling, connected client spans, and Respan's canonical attributes. `Client` and `AsyncClient` support `chat`, `generate`, `embed`, and legacy `embeddings`. The adapter observes the SDK's native HTTPX response and NDJSON decoding after the SDK has serialized the request.

## Install and use

```bash
pip install 'respan-instrumentation-ollama[instruments]' 'opentelemetry-semantic-conventions-ai>=0.5.1'
```

A running Ollama server and an installed model are needed for real inference. The paired examples use controlled native HTTPX transports by default and need neither provider access nor Respan credentials.

```python
import os
import ollama
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from respan_tracing.exporters.respan import RespanSpanExporter
from respan_instrumentation_ollama import OllamaInstrumentor

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(
        RespanSpanExporter(
            api_key=os.environ["RESPAN_API_KEY"],
            endpoint="https://api.respan.ai/api/v2/traces",
        )
    )
)
instrumentor = OllamaInstrumentor(tracer_provider=provider)
instrumentor.activate()
client = ollama.Client()
try:
    with provider.get_tracer("example").start_as_current_span("ollama_workflow"):
        response = client.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": "Reply with one concise sentence."},
            ],
        )
        print(response.message.content)
finally:
    # Works on both the minimum SDK and versions with Client.close().
    client._client.close()
    instrumentor.deactivate()
    provider.shutdown()
```

Pass the application's tracer provider, or omit `tracer_provider` to use `RespanTracer().tracer_provider`. Activate each instance once and deactivate it when done. Multiple owners may share matching configuration; conflicting content/provider configuration raises `ValueError`. Deactivation restores only this adapter's patches, hooks, and processors and ends pending spans without closing caller-owned stream resources.

## Captured source data

Full sanitized native requests and chat/generation responses remain in `traceloop.entity.input` and `traceloop.entity.output`, including complete histories, options, structured formats, images, context, thinking, logprobs, tool schemas, and tool calls. Embedding output follows the canonical vector(s) shape, with complete vectors and sourced model/usage attributes; the native envelope excluding the remapped vector field stays in capture-gated `respan.metadata.ollama.result`. Stream output contains the native frames actually yielded. The HTTP decoder observation retains provider fields that the SDK's typed models omit, such as `prompt_eval_cached_count`, without reading ahead or changing native response objects.

Indexed prompt/completion attributes provide convenient projections. The OpenTelemetry SDK's attribute-count limit can restrict those indexed fields for long histories; canonical bodies, tool definitions, model descriptors, and metadata are written last. This adapter does not truncate history, strings, schemas, frames, or vectors. Native SDK serialization and typed-field validation still apply.

Request and reported response models remain separate. Usage, cache counts, HTTP status, and errors come from actual source fields, including zero values. Missing totals and status codes are omitted. `generate` uses the completion request type and text log type. Current response tool calls stay separate from historical request tool calls.

Native unary responses and yielded chunks retain identity. Streams use a protocol proxy over the SDK's `Iterator`/`AsyncIterator` contract so close/aclose before the first read ends the span. Concrete generator identity changes; `send`, `throw`, `asend`, `athrow`, and native generator attributes delegate to the original generator. Close, errors, EOF, and partial output finalize once. Consume or explicitly close streams to complete their spans.

## Content controls

Set `capture_content=False` on the instrumentor, `RESPAN_TRACE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, or the released Respan/Traceloop context content flag. Initial and observed late vetoes are irreversible for the call and its observed ancestor chain, including finished carriers. Unknown local carriers conservatively omit content; genuine remote parents remain connected. General and LLM instrumentation suppression bypass capture, and unsampled calls do not extract payloads. OpenTelemetry's temporary exporter suppression does not veto unrelated pending sibling calls.

Credential-like values, authorization strings, URL credentials, and sensitive schema defaults/examples are redacted. Schema property identifiers remain intact. Fragmented stream credentials are also redacted. Serialization inspects builtin containers and exact installed native models and never calls arbitrary customer mapping, iterator, string, or model serialization hooks. Telemetry startup, mapping, processor, and cleanup failures preserve native outcomes and ambient context.

Validated profiles use Ollama 0.6.3 and 0.6.0, released `respan-tracing` 2.20.1 and 2.17.0, `respan-sdk` 2.7.6 and 2.6.26, and AI semantic conventions 0.5.1. The minimum OpenTelemetry set is 1.38.0 with instrumentation/semantic conventions 0.59b0. Native capabilities differ by SDK version: 0.6.0 requires a generation response field and lacks the later client close convenience method. Model-management and cloud `web_search`/`web_fetch` helpers are outside this inference adapter's scope.

See the [runnable examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/ollama) for local validation, explicit Respan export, and optional real Ollama inference.
