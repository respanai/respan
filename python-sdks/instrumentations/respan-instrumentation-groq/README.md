# Respan Groq Instrumentation

Trace the official Groq Python SDK with Respan.

## Installation

```bash
pip install respan-ai respan-instrumentation-groq
```

## Usage

```python
from groq import Groq
from respan import Respan
from respan_instrumentation_groq import GroqInstrumentor

respan = Respan(instrumentations=[GroqInstrumentor()])
client = Groq()

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)
print(response.choices[0].message.content)

respan.flush()
respan.shutdown()
```

The package delegates SDK patching to `openinference-instrumentation-groq`
and uses Respan's OpenInference translator to normalize emitted spans into
the Respan tracing pipeline.

## Supported APIs

Validated with Groq **1.7.0** and the minimum **0.9.0**, using
`openinference-instrumentation-groq >=0.1.31` and the released Respan
OpenInference bridge (`>=1.2.5`).

- Sync and async chat completions, including streamed text, fragmented tool
  calls, history, structured output controls, multimodal inputs and returned
  reasoning content. Tool-only replies keep empty text; historical calls stay
  in input context.
- Embeddings: input, returned float/base64 vectors and provider-reported usage.
- Audio transcription and translation: request metadata and returned text.
- Speech generation on SDK versions exposing that resource: returned audio,
  including lazy binary streaming. Audio is base64 encoded with a 64 KiB capture
  limit and an explicit `truncated` flag. Upload metadata records a filename;
  instrumentation does not read or seek caller-owned upload files.
- Native raw and streaming HTTP response objects remain usable. JSON response
  bodies are captured when `parse()` is called; closing without parsing records
  the request without inventing output. Streaming spans finish at exhaustion,
  explicit close, or failure, with partial output on early close.

`TraceConfig` supplied to `GroqInstrumentor(config=...)` controls OpenInference
chat privacy; native inference paths honor input/output and embedding privacy
settings too. Standard OpenTelemetry suppression is respected. Automatic
spans do not represent execution of a tool: decorate actual tool functions
separately.

Files, batch administration and model administration are not instrumented.
The compatibility checks use released SDKs with deterministic HTTP fixtures;
this does not establish live model availability or account access.

See the [companion Python examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/groq)
for the complete synthetic suite and direct Groq setup.

Content policy also honors `TRACELOOP_TRACE_CONTENT=false` and Respan's
`ENABLE_CONTENT_TRACING_KEY=False` context, captured when the SDK call starts.
The package uses the active Respan tracer provider for both delegated and native
spans. The released OpenInference bridge does not support selecting a separate
provider through the instrumentor's `tracer_provider` keyword.
