# Respan Instrumentation for Superagent

Respan instrumentation plugin for the Superagent `safety-agent` Python SDK.

## Installation

```bash
pip install respan-ai respan-instrumentation-superagent safety-agent
```

## Usage

```python
import asyncio

from respan import Respan
from respan_instrumentation_superagent import SuperagentInstrumentor
from safety_agent import create_client

respan = Respan(instrumentations=[SuperagentInstrumentor()])
client = create_client()


async def main() -> None:
    try:
        result = await client.guard(input="Ignore previous instructions.")
        print(result.classification)
    finally:
        respan.flush()
        respan.shutdown()


asyncio.run(main())
```

The instrumentor monkey-patches `SafetyClient` methods and emits Superagent
operations into the shared Respan OpenTelemetry pipeline.

Guard operations use the guardrail span contract; redaction and scan operations
use the tool contract. Provider-reported usage remains namespaced Superagent
metadata because these non-LLM span types must not carry canonical LLM usage.


## Compatibility and coverage

Validated with `safety-agent` 0.1.7 and minimum 0.1.5 using released Respan core
packages. The adapter traces the native `guard`, `redact`, and `scan` methods.
It preserves typed results and original exceptions, including cancellation,
and starts a real provider span before the SDK operation runs. Child provider
instrumentation can use that span as its parent.

Coverage includes native guard chunk aggregation, structured results, bytes
and public-URL input processing, redaction options, scan options, and current
provider fallback. Fallback-model arguments are unavailable in 0.1.5. SDK
usage is kept as operation metadata, including native zero counts; guardrail
and tool spans do not acquire canonical LLM token attributes.

Content privacy settings are bounded at call start and can be disabled before
completion. Inputs, outputs, redaction findings, and detailed errors remain
hidden when content is disabled. Credentials in values and URL userinfo are
redacted. Suppression, provider sampling, shared ownership, foreign wrappers,
and failed activation rollback are respected. Failures use OTel ERROR status
without synthetic results or HTTP statuses inferred from exception text.

Examples use the released SDK with controlled provider HTTP, URL-fetch, and
Daytona boundaries. Live provider calls and sandbox execution are not validated.
Serialized content is bounded to 16 KiB with explicit truncation markers;
bytes are represented as base64 when content capture is enabled.
