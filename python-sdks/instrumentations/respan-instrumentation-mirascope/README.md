# Respan instrumentation for Mirascope

Traces the released Mirascope 2.5 Model API: `call`, `context_call`, their async
variants, and all four corresponding stream methods. The native `Toolkit`,
`ContextToolkit`, `AsyncToolkit` and `AsyncContextToolkit` execution methods emit
common-only tool spans. Mirascope 2.5's XAI provider works through the same native
Model path; provider-specific APIs that bypass Model or Toolkit are outside this
adapter's scope.

```python
from respan_instrumentation_mirascope import MirascopeInstrumentor

instrumentor = MirascopeInstrumentor(capture_content=True)
instrumentor.activate()
# Use native mirascope.llm.Model and Toolkit APIs.
instrumentor.deactivate()
```

Calls return their original native response objects. Stream sources delegate
iteration, send/throw/close and async equivalents only when the native source
supports them. Each advance attaches and detaches its span independently.
Exhaustion, native source close, error and response abandonment end the span;
closing an outer Mirascope text generator follows Mirascope's own behavior and
may leave its inner source open. Consume it or close the native `_chunk_iterator`
when abandoning a response. Unadvanced or failed-before-first-content streams
do not invent assistant output.

`capture_content=False`, `TRACELOOP_TRACE_CONTENT=false` (also `0`, `off`, `no`),
and Respan's content context omit messages, schemas, arguments, results and
error diagnostics. The initial bound cannot be widened, and active/finished
ancestor and final policy vetoes lower it before native context detach. Safe
model/provider identifiers, actual usage, HTTP status and error types remain.
General and language-model suppression skip instrumentation; non-recording
spans skip content inspection.

Canonical chat attributes retain all messages and current/history tool calls
with source IDs and single-encoded JSON arguments. Known schemas and dense or
sparse tool vectors are complete; indexed prompt/completion attributes project
up to eight messages each to protect OpenTelemetry's default attribute budget.
Generic unknown values are described without invoking serialization hooks;
generic text and lists remain bounded. Credentials are redacted, including
quoted values and unfinished streamed arguments. Sensitive JSON-schema property
names remain structural, while credential defaults/examples are redacted.

Usage is taken from retained provider source data. OpenAI Completions, Responses
and consumed streaming events are observed before DTO boolean-to-integer
coercion. Actual zero counters remain zero; missing or invalid counters remain
absent. Mirascope's default-zero Usage fields without a raw provider source
cannot establish provider usage and are omitted. A total computed from invalid
native counts is omitted. Errors retain their original exceptions and actual
HTTP status when available; they do not create model output or inferred status.

Compatible owners share hooks; incompatible provider or capture settings raise
`ValueError`. Activation failure rolls back owned changes, and deactivation
restores only hooks still owned by this adapter. Retained foreign wrappers are
inert after deactivation. An optional `tracer_provider` selects an OTel SDK
provider explicitly. If activation precedes SDK provider initialization, the
privacy processor cannot observe that provider's initial span policy: native
structural spans and actual usage remain, but content stays hidden. Activate
with the initialized SDK provider to capture permitted content. Recording
parents whose initial policy was unobserved also keep descendant content hidden.

The verified dependency floor is Mirascope 2.5,
respan-sdk 2.7.6, respan-tracing 2.17, OTel 1.38 and AI semantic conventions 0.5.1.
Mirascope's OpenAI extra requires OpenAI >=2.15,<3.

Do not combine this adapter with `mirascope.ops.instrument_llm()` unless two
independent telemetry pipelines are intentional. Paired examples provide
controlled real-SDK fixtures and an explicit live-provider opt-in. Local and
wire validation do not establish backend field projection fidelity.
