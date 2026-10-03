# respan-instrumentation-google-genai

Respan instrumentation plugin for the official
[Google Gen AI SDK](https://googleapis.github.io/python-genai/).

The package patches `google.genai.models.Models` and `AsyncModels` generation
and embedding methods and emits canonical Respan spans through the active OTEL pipeline.
It captures sync calls, async calls, streaming calls, prompt and completion
content, token usage, tool definitions, and model function calls.

## Installation

```bash
pip install respan-ai respan-instrumentation-google-genai google-genai
```

## Usage

```python
from google import genai
from respan import Respan
from respan_instrumentation_google_genai import GoogleGenAIInstrumentor

respan = Respan(instrumentations=[GoogleGenAIInstrumentor()])
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in three languages.",
)
print(response.text)
respan.flush()
```

## Embeddings and supported SDK versions

Sync `client.models.embed_content()` and async
`client.aio.models.embed_content()` emit embedding spans. The adapter preserves
the requested model, text or structured multimodal input, and all returned
vectors. Vertex embedding `statistics.token_count` values are summed only when
every returned embedding has a valid token count. Responses without token
statistics leave usage unset; billable character counts are never treated as
tokens. Provider exceptions are re-raised and recorded as failed spans.

```python
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["Tracing", "Observability"],
    config={"output_dimensionality": 768},
)
print(result.embeddings[0].values)
```

Compatibility tests exercise released `google-genai` 1.0.0 and 2.28.0 using
real SDK models, serializers, responses, and deterministic HTTP transports.
The current SDK's Gemini Embedding 2 multimodal input and built-in Exa search
tool definitions are covered. Other Google SDK subsystems such as files,
batches, live sessions, and image/video generation are not instrumented by
this package.

Content capture honors `TRACELOOP_TRACE_CONTENT=false` and the standard
`override_enable_content_tracing` context override. Model and reported usage
remain available when content capture is disabled. Multiple active instrumentor
instances share patches until the final instance deactivates.

Google `model` messages are translated to the canonical `assistant` role;
function responses use `tool`. Prior function calls, including automatic
function-calling history, remain in prompt fields. Completion tool calls include
only calls present in the response candidates.
