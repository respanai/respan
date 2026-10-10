# respan-instrumentation-portkey

Trace native Portkey inference calls in Respan. The adapter retains the upstream
OpenInference Portkey instrumentor and its chat/prompt SDK delegation. Canonical
content and usage come from actual SDK requests, responses and stream events.
It does not depend on the generic Respan OpenInference adapter.

```bash
pip install respan-instrumentation-portkey
```

```python
import os
from portkey_ai import Portkey
from respan import Respan, workflow
from respan_instrumentation_portkey import PortkeyInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[PortkeyInstrumentor()],
)
client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])


@workflow(name="portkey_chat")
def run(prompt: str):
    return client.chat.completions.create(
        model="@openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )


try:
    response = run("Say hello.")
    print(response.choices[0].message.content)
finally:
    client.close()
    try:
        respan.flush()
    finally:
        respan.shutdown()
```

Configure the Portkey provider/config/model and credentials for your gateway
account when making live calls. `RESPAN_BASE_URL` optionally overrides the
Respan destination; its default is `https://api.respan.ai/api`.

Supported inference surfaces are synchronous and asynchronous chat completions,
chat structured `parse`, chat stream managers, saved prompt completions,
embeddings, text completions, and Responses `create`/`parse`/stream managers.
Beta chat helpers are observed where the released SDK provides them. Returned
SDK models and exceptions are preserved. Stream proxies keep native chunks,
iteration, close/cancel, context managers and native final-result methods;
consume or close streams to complete their spans. Final deactivation ends
telemetry for pending streams without closing the customer's native iterator.

Privacy uses an initial capture bound and an irreversible observed veto.
`PortkeyInstrumentor(capture_content=False)`, `TRACELOOP_TRACE_CONTENT=false`,
Respan context opt-out, and native OpenInference input/output privacy settings
suppress the corresponding content. Ancestor bounds survive native context
detachment and parent completion. Sampling and both OTel suppression keys are
honored before adapter payload capture. Shared owners require matching settings
and provider; final teardown preserves foreign hook replacements.

Tool definitions, current-turn call IDs, arguments, and embedding vectors are
captured in full when enabled. Canonical entity JSON retains every message and
choice; the redundant indexed message projection is limited to eight on each
side to preserve common fields and metadata within default OTel attribute
limits. Unknown objects are represented by type without consuming iterators or
calling arbitrary conversion hooks. Credentials and signed URL details are
redacted. Provider usage is observed before SDK DTO coercion; boolean, negative,
missing and invalid counts are omitted. Cache read/write and reasoning counts
are preserved when the provider supplies them. No HTTP 500, error output or usage
is synthesized. Responses with an actual failed status are marked ERROR even
when the SDK returns a value rather than raising.

Validated on Python 3.12 with Portkey 2.3.4/upstream instrumentor 0.1.21 and the
Portkey 2.3.1/instrumentor 0.1.11 floor, using released Respan dependencies. Portkey
bundles its own OpenAI client; this package does not require or replace a
separate OpenAI SDK. Python metadata supports 3.11–3.13.

The controlled examples use real released SDK JSON/HTTP/SSE boundaries without
live model calls. Admin APIs, websocket/realtime APIs, raw-response wrappers and
SDK framework integrations are not independently instrumented. Live provider
routing, retries/fallbacks, remote prompt rendering and account acceptance are
outside the controlled fixture validation. See the paired
[Python examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/portkey).
