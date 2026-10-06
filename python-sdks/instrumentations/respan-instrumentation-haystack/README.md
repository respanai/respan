# respan-instrumentation-haystack

Respan's Haystack adapter uses OpenInference and the released Respan tracing
runtime to capture pipelines, components, model calls, embeddings and agents.
It is tested with Haystack 2.18.0 and 3.3.0, OpenInference Haystack 0.1.44 and
Respan's OpenInference bridge 1.2.5.

```bash
pip install respan-instrumentation-haystack
```

The minimum Haystack version is 2.18, matching the upstream OpenInference
instrumentor. Haystack 3 uses `Pipeline.run_async` and `Pipeline.stream`; the
adapter also supports Haystack 2's separate `AsyncPipeline`.

## Quickstart with Haystack 3

The native mock generator runs locally. The configured Respan key exports only
the resulting fixture traces.

```python
import os

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url="https://api.respan.ai/api",
    instrumentations=[HaystackInstrumentor()],
)
try:
    agent = Agent(chat_generator=MockChatGenerator(responses="Hello from Haystack"))
    print(agent.run(messages=[ChatMessage.from_user("Hello")])["last_message"].text)
finally:
    respan.shutdown()
```

## Coverage

- Sync pipelines, async pipelines, async generators and the Haystack 3 stream
  handle retain their results, exceptions and parent relationships.
- Components registered after activation are traced. Async components that
  delegate to their own sync method emit one component span.
- Haystack 3 agents include individual sync and async tool executions. Current
  model tool calls, historical calls, tool results and execution IDs correlate.
- Embeddings retain every returned vector dimension, inputs and reported usage.
  Token usage is recorded on model/embedding calls, without agent rollups.
- Parallel retrieval workers retain the calling trace and propagated metadata.
- Shared instrumentor instances keep tracing active until the final owner
  deactivates. The first owner's configuration applies while shared ownership
  is active; patches and processors are restored after final deactivation.

Use OpenInference's `TraceConfig` to control content capture:

```python
from openinference.instrumentation import TraceConfig

instrumentor = HaystackInstrumentor(
    config=TraceConfig(hide_inputs=True, hide_outputs=True)
)
```

This hides model, agent, tool and embedding payloads while keeping operation
metadata and reported usage. The adapter also honors Respan's
`TRACELOOP_TRACE_CONTENT=false` switch and scoped content-tracing opt-out. The native Haystack content-tracing switch applies
to Haystack's own tracer; OpenInference capture is configured separately.

The tested runtime uses released `respan-ai==4.2.3`, `respan-tracing==2.20.1`,
`respan-sdk==2.7.6`, `respan-instrumentation-openinference==1.2.5`, OpenTelemetry
1.45.0 and AI semantic conventions 0.5.1. Only this adapter needs an editable
install when validating changes before publication.

See the [companion Python examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/haystack)
for the full deterministic runner, retrieval, routing, converters, evaluators,
agents, tools, streams, embeddings, controlled errors and privacy checks.
Gateway calls and managed-prompt creation are separate explicit runner options.

OpenInference Haystack 0.1.44 retains its own OpenTelemetry context across an
async-generator yield. Closing that iterator from a different asyncio task can
log an upstream context-detach warning; the same behavior occurs without this
adapter. Respan's pipeline context is restored before each yield and its
iterator cleanup succeeds across tasks.
