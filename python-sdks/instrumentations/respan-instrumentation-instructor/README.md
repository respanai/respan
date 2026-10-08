# respan-instrumentation-instructor

Native Respan tracing for Instructor's public clients and `patch(create=...)`
callables. Tested with Instructor **1.17.0** and the declared minimum **1.3.7**.
The adapter supports the older flat modules and the current v2 module layout.

```bash
pip install respan-tracing respan-instrumentation-instructor
```

```python
import os

import instructor
from openai import OpenAI
from pydantic import BaseModel
from respan_instrumentation_instructor import InstructorInstrumentor
from respan_tracing import RespanTelemetry


class User(BaseModel):
    name: str
    age: int


telemetry = RespanTelemetry(
    api_key=os.environ["RESPAN_API_KEY"],
    is_auto_instrument=False,
)
instrumentor = InstructorInstrumentor()
instrumentor.activate()
try:
    # OpenAI uses its own OPENAI_API_KEY; tracing configuration is separate.
    with OpenAI() as provider:
        client = instructor.from_openai(provider)
        result = client.create(
            response_model=User,
            messages=[{"role": "user", "content": "Ada is 36 years old."}],
            model="gpt-4o-mini",
        )
        print(result.model_dump())
finally:
    instrumentor.deactivate()
    telemetry.flush()
    telemetry.tracer.tracer_provider.shutdown()
```

`RESPAN_BASE_URL` selects the tracing API base. Configure provider endpoints and
credentials on the native provider client. Installing the `respan-ai` facade is
optional; its current aggregate dependencies may require newer provider SDKs
than an old Instructor environment allows.

The integration covers sync/async `create`, `create_with_completion`,
`create_partial`, `create_iterable`, and low-level `patch`. Current Responses
clients, `from_provider`, TypedDict schemas, completion hooks, and retry token
budgets use the native SDK behavior. Instructor 1.3.7 predates those newer APIs.
Native return types, exceptions, cancellation, retry counts, and iterator
advance/send/throw/close remain unchanged. Some current stream paths return a
native list response; the adapter retains that return type.

Chat spans contain canonical messages, schemas, current response tool calls,
source call IDs, response IDs, and actual reported usage. Completed and failed
retry totals use the native observed aggregate. Missing, invalid, or SDK-only
default counts are omitted. Known schema/tool payloads remain complete; normal
payloads use explicit 16 KiB/50-item bounds and credential redaction. Private data
is not copied into schema/output attributes when content capture is disabled.

Content opt-out: `InstructorInstrumentor(trace_content=False)`,
`TRACELOOP_TRACE_CONTENT=false`, or Respan's content context flag. The initial
policy bounds the whole call; a later veto removes earlier captured content.
Sampling and OTel/Traceloop suppression are honored. Compatible owners share
hooks; conflicting provider/privacy settings are rejected. Final release
finishes pending telemetry without advancing or closing native streams, and
preserves foreign wrappers.

[Eleven runnable examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/instructor)
exercise released SDKs with controlled HTTP responses by default. Fixtures cover
current and minimum APIs, streams, schemas, retry/error usage, user hooks,
privacy, and cancellation; live providers are optional and separately configured.
The controlled suite does not establish live provider availability, billing,
cache services, or every optional provider transport.
