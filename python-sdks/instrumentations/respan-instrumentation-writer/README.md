# respan-instrumentation-writer

Respan instrumentation for the official [Writer Python SDK](https://github.com/writer/writer-python). It observes existing native resource methods and records real OpenTelemetry spans through the active provider.

## Install

```bash
pip install respan-instrumentation-writer
```

`writerai==4.0.1` is an alias distribution with an unconstrained dependency on `writer-sdk`. This package also requires `writer-sdk>=2.3.0,<4`: 2.3.0 is the first stable release with all existing resource families, including direct tools. Current compatibility is tested against the released SDK 3.0.0. The Respan SDK floor is 2.6.26, which supplies the canonical span-attribute module used here.

## Usage

```python
from respan import Respan
from respan_instrumentation_writer import WriterInstrumentor
from writerai import Writer

respan = Respan(instrumentations=[WriterInstrumentor()])
with Writer() as client:
    result = client.chat.chat(
        model="your-writer-model",
        messages=[{"role": "user", "content": "Say hello."}],
    )
    print(result.choices[0].message.content)
respan.flush()
```

Normal Writer and Respan credentials are required for live provider calls and export. Companion examples default to controlled native HTTPX responses and local recording.

## Coverage and native behavior

All sixteen existing synchronous/asynchronous resource methods are covered: chat, text completions, Knowledge Graph questions, application content generation, vision, translation, web search and PDF parsing. Native structured `chat.parse()` delegates through chat and produces one call span; the SDK owns schema generation and parsed values. Chat, completion, graph and application streams are observed as the caller consumes them. PDF parsing remains available in the current SDK but is natively deprecated; its warnings and behavior are forwarded.

The public `writerai.Stream` and `writerai.AsyncStream` retain their identity and concrete type. Owned private iterator and close taps preserve native chunks, error identities, retries, callbacks, context-manager values and HTTP resources. No unread stream is drained and no provider call is repeated. Unread close produces no invented output. Deactivation ends pending telemetry without closing caller-owned streams; garbage collection also ends owned telemetry.

Native request JSON is observed after the SDK serializes it, so caller generators are consumed only by the SDK and full histories remain intact. Supplied application/file IDs and query controls are preserved. Full known native response JSON, provider feedback, reasoning, tool schemas, arguments, IDs, false/zero/empty values and extra vectors are captured without adapter truncation. Writer exposes no embedding resource in this supported boundary; vectors returned as actual native response data remain in the full payload. The provider's OpenTelemetry attribute-count limit can bound indexed convenience projections; full canonical input/output and schemas are written last.

Only actual supplied/returned model and usage fields are mapped. Unknown custom objects become structural type descriptions without executing their serialization, iterator, numeric, equality or formatting hooks. Credentials are redacted with valid, idempotent JSON and retained schema property names. The upstream Writer instrumentor currently declares SDK versions below 3 and only covers chat/completions; this package retains its existing broader native boundary.

## Privacy and lifecycle

`WriterInstrumentor(capture_content=False)` disables content. Ambient and supplied Respan/Traceloop decisions, both content environment flags and general/LLM suppression are honored. Initial or observed ancestor denial cannot widen again. Active/finished ancestors and unknown local carriers are handled conservatively. Owned input/output, metadata, error messages, status descriptions, events and retained buffers are removed before detach/end when denied.

Sampling and suppression precede payload inspection. Telemetry startup, serialization, attribute, end and context failures preserve native outcomes and ambient context. Shared activation is reference-counted, conflicting settings fail explicitly, and rollback/deactivation restore only owned descriptors/processors while retaining foreign changes.

## Validation

Native regression tests exercise released SDK clients and typed models with actual HTTPX JSON/SSE bytes, SDK retries, callbacks, streaming errors, close/deactivation/GC and privacy/lifecycle faults. Current and exact minimum profiles are tested separately. Package byte provenance and stored-trace semantic acceptance are independent gates; successful local recording or HTTP export does not by itself prove the stored projection.
