# respan-instrumentation-agentscope

Trace [AgentScope](https://docs.agentscope.io/) through the active Respan
OpenTelemetry provider. Tested with AgentScope **2.0.3 and 2.0.9**, using
released `respan-tracing>=2.20.0` dependencies.

```bash
pip install respan-ai respan-instrumentation-agentscope
```

```python
import os
from respan import Respan
from respan_instrumentation_agentscope import AgentScopeInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[AgentScopeInstrumentor()],
)
# Build and run AgentScope agents, models and tools as usual.
respan.shutdown()
```

## Captured operations

- `Agent.reply()` and `reply_stream()`: agent input, actual message/event output,
  child model/tool calls, and errors.
- Chat-model `__call__()` and `generate_structured_output()`: messages, current
  turn tool calls, tool definitions, model/provider, and SDK-reported usage.
  Zero counts are retained; missing counts are not replaced with zero.
- Embedding model `_call_api()` batches: input, **complete embedding vectors**,
  and available provider usage. Each actual/cache batch has one span. Cached
  results have vectors but no fabricated usage; SDK aggregate usage is not used.
- `Toolkit.call_tool()`: actual tool result, lifecycle, error state, and canonical
  tool-call ID linking the invocation to the model's call.
- `TeamPipeline.reply_stream()` and `GoalPipeline.reply_stream()` on SDK versions
  that provide them: workflow spans around their agent trees.

Streams retain Python's async-generator API. Spans finish on exhaustion,
explicit `aclose()`, or error. The adapter attaches context only while advancing
or closing the SDK generator. Early close records only observed output/usage;
provider errors remain exceptions and do not become model responses or invented
HTTP status codes. Nested cleanup that the underlying SDK does not propagate is
subject to that SDK's generator lifecycle.

## Configuration and ownership

```python
AgentScopeInstrumentor(
    models=[planner_model, reviewer_model],  # custom chat model classes
    embedding_models=[custom_embedding_model],
    capture_content=False,
)
```

Use `model=...` for one custom chat model or `models=...` for several; these
options are mutually exclusive. Custom model instances patch their classes,
matching Python's special-method lookup. `agent=...` and `toolkit=...` scope
custom object hooks. `instrument_models`, `instrument_embeddings`, and
`instrument_tools` can disable those groups independently.

`capture_content=False`, `TRACELOOP_TRACE_CONTENT=false`, and the Respan content
context disable payload capture while retaining usage and operation metadata.
Content policy is captured when the call/stream is created. Standard OTel
suppression and provider sampling are respected. Shared owners retain hooks
until final deactivation; conflicting content settings are rejected, and later
foreign wrappers survive teardown.

Standalone realtime audio, TTS connections, classifier APIs, remote A2A
transport internals, storage and admin operations are not automatically traced
by this adapter. A custom agent passed with `agent=...` can still have its
ordinary `reply`/`reply_stream` boundary traced. No live provider service is
required for the deterministic validation examples.

[Feature-compatible Python examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/agentscope)
cover models, tools, multi-agent work, streams, structured output, embeddings,
privacy, controlled errors, and TeamPipeline delegation.
