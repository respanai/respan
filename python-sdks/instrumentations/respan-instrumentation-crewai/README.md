# respan-instrumentation-crewai

First-party Respan instrumentation for CrewAI's official lifecycle events.
It emits crew workflows, tasks, agents, tool calls, native LLM calls, and
user-defined Flow workflows/methods through the active Respan tracer provider.
Internal CrewAI execution flows remain transparent so agent trees stay connected.

## Installation and usage

```bash
pip install respan-ai respan-instrumentation-crewai
```

```python
from respan import Respan
from respan_instrumentation_crewai import CrewAIInstrumentor

respan = Respan(instrumentations=[CrewAIInstrumentor()])
# Run your Crew, Agent, Flow, or native LLM calls here.
from crewai.events.event_bus import crewai_event_bus

crewai_event_bus.flush()  # Wait for CrewAI's background lifecycle handlers.
respan.shutdown()
```

The supported CrewAI range is `>=1.10.1,<2`, tested against **1.10.1** and
**1.15.23** with released Respan tracing 2.20.1. The current matrix uses OpenAI
2.54.0 and OpenTelemetry 1.45.0. CrewAI 1.10.1 pins OpenTelemetry 1.34.x, so its
isolated matrix uses AI semantic conventions 0.4.13 and `respan-tracing` directly.
It cannot be coinstalled with the current `respan-ai` facade's Together adapter,
which requires AI semantic conventions >=0.5.1 and OpenTelemetry >=1.38.
No dependency constraint needs to be bypassed for either supported matrix.

## Coverage

- Native sync/async Chat Completions and Responses, streaming calls, and current
  `LLM.stream_events()` sessions retain their SDK results and caller context.
- Chat spans retain request/response content, response IDs when emitted by the
  SDK, current tool calls, historical tool messages, and actual reported token,
  cache, and reasoning usage, including zero counts. Agent/task/tool usage is
  not inferred from child model calls.
- Native executor tool IDs are retained when the SDK exposes an unambiguous
  matching current tool call. Ambiguous or unavailable IDs are omitted.
- User-defined Flow methods retain parentage and failure status. Paused/failed
  lifecycle events close their scopes when the installed SDK exposes them.
- Current `ToolFailure` results mark the finished tool span as an error while
  preserving the real returned output. Failed model calls have error metadata
  and OTel ERROR status, without a fabricated completion or HTTP status.

Embedding providers and vectors are outside CrewAI's lifecycle event surface
covered by this adapter. Instrument the underlying embedding provider separately.

## Privacy and lifecycle

Set `TRACELOOP_TRACE_CONTENT=false`, or use Respan's `ENABLE_CONTENT_TRACING_KEY`
context override, to omit model/function/agent/flow content. Decisions are captured
when CrewAI emits each event, before its background handler runs. Disabling content
at the start of a request remains effective through awaits and stream completion;
model identity and reported usage remain available. Error messages remain error
metadata. OpenTelemetry instrumentation suppression is also respected.

Multiple instrumentor instances share one listener on the same tracer provider.
The last deactivation removes owned handlers and restores hooks. Foreign wrappers
are preserved, and retained adapter wrappers become inactive. Event and usage
buffers are bounded and cleaned up on final deactivation.

See the [companion examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/crewai)
for a complete deterministic runner using real SDK APIs and local HTTP fixtures.
