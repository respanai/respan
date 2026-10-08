# Respan instrumentation for Arize

Trace public operations of the Arize Python SDK through Respan's OpenTelemetry
pipeline. This adapter covers the control-plane SDK, rather than treating
historical span data or integration configuration as new LLM or agent executions.

Install the facade and adapter:

```bash
pip install respan-ai respan-instrumentation-arize
```

```python
from arize import ArizeClient
from respan import Respan
from respan_instrumentation_arize import ArizeInstrumentor

respan = Respan(instrumentations=[ArizeInstrumentor()])
client = ArizeClient(api_key="YOUR_ARIZE_API_KEY")
try:
    page = client.datasets.list()
finally:
    respan.flush()
    respan.shutdown()
```

Validated with released Arize 8.35.0 and 8.57.0. Later clients for traces, audit
logs, integrations, webhooks, and newer dataset/evaluator/annotation operations
are instrumented when the installed SDK exposes them. Arize labels several
released REST endpoints as alpha or beta; the SDK's own validations, return
models, native helper tracers, exceptions, and pagination behavior are retained.

Every operation starts a real task span before native execution. Compatible
instrumentor instances share their wrappers. Deactivation preserves foreign
replacements and stops owned observations without canceling native work.
ML streaming uploads keep their original concurrent Future, and the span ends
on its actual completion, failure, or cancellation. Synchronous and coroutine
wrappers keep scoped context; opaque iterators are never consumed.

Content capture honors `capture_content=False`,
`TRACELOOP_TRACE_CONTENT=false`, and the Respan context policy.
A start opt-out cannot be re-enabled for that call, and an observed later veto
clears input/output. Future completion observes its completion context and the
current environment policy. Suppression and sampling apply before serialization.
Arize's own experiment worker tracing keeps its native provider/configuration;
the adapter's privacy policy applies to the adapter's operation spans.

Known Arize models, dataframe rows, Arrow tables, SDK embedding tuples, full dense
and sparse vectors, and tool-call data are serialized safely with secrets
redacted. Ordinary payloads are bounded; vectors and known tool payloads remain
complete when content is enabled. The adapter reports only actual native HTTP
status/error information, emits no invented error output, and does not infer
LLM usage from administrative calls or historical trace records.

See the paired `python/tracing/arize` examples for controlled
released-SDK REST/protobuf fixtures, optional Respan export, and explicit skips
for APIs absent from the minimum SDK. Live Arize accounts, Flight servers,
remote integrations/webhook delivery, and every CRUD variant need separate
validation.
