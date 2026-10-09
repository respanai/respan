# respan-instrumentation-anthropic

Respan instrumentation for the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

```bash
pip install 'respan-instrumentation-anthropic[instruments]'
```

```python
import os
from anthropic import Anthropic
from respan import Respan
from respan_instrumentation_anthropic import AnthropicInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[AnthropicInstrumentor()],
)
with Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]) as client:
    message = client.messages.create(
        model=os.environ["ANTHROPIC_MODEL"],
        max_tokens=64,
        messages=[{"role": "user", "content": "Write one line about tracing."}],
    )
respan.flush()
```

The native adapter observes stable and beta `messages.create()`, available
`messages.parse()`, raw `create(stream=True)`, and helper `messages.stream()`
for sync and async clients. It preserves native Message, Stream, parsed models,
and stream-manager objects. Streams are observed as the caller consumes them;
closing early does not drain the response to obtain a final message.

Available `beta.sessions.events.stream()` APIs are observed as agent spans.
Final native input/output events and source usage are retained; preview deltas
are not appended twice. Instrumentation does not create sessions, deploy agents,
or execute remote tools. Transmitted tool definitions, arguments, results, IDs,
thinking/signature blocks, schemas, and complete histories remain in their native
content. Model tool requests and historical tool results do not imply a locally
observed tool execution span.
Remaining native request options, including beta, thinking, structured-output,
and context-management configuration, are captured under
`respan.metadata.anthropic.request`. Native max tokens, temperature, and top-p
also use their GenAI attributes. Credential header values are redacted.

Content capture defaults to enabled. Disable it with
`AnthropicInstrumentor(capture_content=False)`, `TRACELOOP_TRACE_CONTENT=false`,
`RESPAN_TRACE_CONTENT=false`, or an OTel context using the canonical Respan
`ENABLE_CONTENT_TRACING_KEY=False` (legacy `trace_content=False` also works).
General and LLM suppression skip tracing.
Capture policy narrows throughout the call, including observed active/finished
ancestors and stream consumption. Unknown local parent carriers fail closed.
Temporary veto contexts latch before detach. The observer retains at most 4096
bodyless finished ancestry records; older unknown local ancestry fails closed.
Credential values are redacted while schema property names remain intact.
Sampling is applied before content serialization. Telemetry faults do not change
native SDK outcomes or resource cleanup.

Multiple instrumentors with the same configuration share patches until the final
owner deactivates. Conflicting configurations remain inactive. Deactivation
restores only this adapter's wrappers and retains foreign wrappers/processors.
The SDK is optional; install the `instruments` extra when needed. Support starts
at Anthropic 0.39.0. Newer optional parse, beta helper, and managed-session APIs
are discovered when the installed SDK exposes them.

Controlled native examples live in `python/tracing/anthropic` in
[respan-example-projects](https://github.com/respanai/respan-example-projects).
