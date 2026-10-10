# respan-instrumentation-together

Respan instrumentation for the official [Together Python SDK v2](https://github.com/togethercomputer/together-py). It observes the native resource methods and emits real OpenTelemetry spans through the active provider.

## Installation

```bash
pip install 'respan-instrumentation-together[instruments]'
```

The Together dependency is optional. Importing and activating the plugin without Together installed is a no-op. The supported native boundary starts at `together==2.0.0`; the native compatibility suites also run against `2.39.0`. The direct Respan SDK floor is `2.6.26`, which supplies the shared span-attribute constants used here.

## Usage

```python
from together import Together
from respan import Respan
from respan_instrumentation_together import TogetherInstrumentor

respan = Respan(instrumentations=[TogetherInstrumentor()])
with Together() as client:
    response = client.chat.completions.create(
        model="your-together-model",
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(response.choices[0].message.content)
respan.flush()
```

Supply the normal Together and Respan credentials for live calls and export. The companion examples default to controlled native HTTP responses and local recording.

## Native coverage

The adapter covers synchronous and asynchronous chat completions, text completions, embeddings, reranking, and image generation. Chat and text streaming are observed as the caller consumes the native stream. Tool definitions, history, emitted tool-call IDs and arguments, reasoning and other native response fields remain available in the sanitized full payload. Empty choices and provider feedback are retained even when no message projection exists. Usage and HTTP error status are recorded only when the native SDK supplies them.

The public `together.Stream` and `together.AsyncStream` objects retain their identity and concrete type. The adapter installs an owned observation tap on their private iterator and close method; original chunks, errors, callbacks, retries, context managers and HTTP resources remain SDK-owned. It never drains an unread stream or invokes the provider again. Closing an unread stream produces no invented result. Deactivation finishes pending telemetry without closing caller-owned streams. Foreign replacements are retained during restoration.

Embedding output contains the complete vector(s). Other native embedding response fields are retained as sanitized JSON in capture-gated `respan.metadata.together.result`, excluding the vectors already mapped to canonical output. Full native request/response JSON, histories, schemas, byte values and vectors have no adapter truncation limit. The provider's OpenTelemetry attribute-count limit can bound indexed convenience projections; canonical full payload and tool-schema attributes are written last.

## Privacy and lifecycle

`TogetherInstrumentor(capture_content=False)` disables content capture. Canonical Respan and Traceloop content flags, `RESPAN_TRACE_CONTENT` and `TRACELOOP_TRACE_CONTENT`, and both general and LLM suppression are honored. Ambient and supplied context decisions are combined. Initial denial, observed ancestor denial and later explicit denial cannot widen again. Unknown local ancestors fail closed. Denial removes owned input/output, embedding envelope metadata, error messages, status descriptions and events before span end. Credential values are redacted while JSON and tool-schema property names remain valid.

Sampling and suppression precede payload inspection. Telemetry faults preserve native values, error identities, resource cleanup and ambient context. Shared activation is reference-counted; conflicting configurations fail explicitly. Only owned descriptors and processors are restored on rollback or deactivation.

## Validation

The native tests use released Together clients and typed models with HTTPX transport responses and actual SSE bytes, including native retries, errors and close behavior. They cover current and exact minimum dependencies. The companion examples separately opt into Respan export and live Together calls. Stored-trace projection is a separate acceptance gate from successful local capture or HTTP export.
