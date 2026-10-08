# respan-instrumentation-smolagents

Trace smolagents agents, planning/action steps, model calls, local tools, and managed agent invocations through Respan. Validated against released smolagents 1.24.0 and 1.26.0.

```bash
pip install respan-ai respan-instrumentation-smolagents
```

```python
from respan import Respan
from respan_instrumentation_smolagents import SmolagentsInstrumentor
from smolagents import CodeAgent, InferenceClientModel

respan = Respan(instrumentations=[SmolagentsInstrumentor()])
agent = CodeAgent(tools=[], model=InferenceClientModel(model_id="your-model"))
try:
    print(agent.run("Explain recursion in one sentence."))
finally:
    respan.shutdown()
```

Set `RESPAN_API_KEY` for trace export. The model provider needs its own credentials. `RESPAN_BASE_URL` defaults to `https://api.respan.ai/api`.

The native adapter preserves returned `ChatMessage` and `RunResult` objects, exceptions, streamed chunks, and generator `send`, `throw`, and `close`. Streams attach their span context during each advance and detach it before returning to the caller. Model subclasses defined before or during activation are instrumented. Multiple plugin instances share hooks; final deactivation restores only hooks still owned by this adapter.

`TRACELOOP_TRACE_CONTENT=false` or Respan's `ENABLE_CONTENT_TRACING_KEY=False` disables message, schema, tool input/output, and agent input/output capture. A call that starts with capture disabled cannot enable it later. Disabling capture before completion vetoes its buffered content. Model/provider, actual SDK usage fields, and error classes remain available. Sampling and OTel suppression do not inspect content. Credential fields and credential assignments are redacted; binary and unsupported Python values are described by type without invoking `repr`.

Model token usage comes from `ChatMessage.token_usage` and streamed usage deltas. Cache and reasoning counts are mapped only when the response's raw usage exposes them. Agent totals are not copied onto parent spans. Tool execution IDs are correlated with the current step's model calls when the name and arguments identify one source call. Identical parallel calls retain their model IDs; their tool execution spans omit an ambiguous ID because the SDK does not pass it to tool dispatch. Code-agent tools do not acquire an invented model tool-call ID when the SDK exposes no such ID.

The adapter does not execute remote provider calls or remote sandboxes itself. Local fixtures validate tracing at the released SDK's model boundary; they do not establish live provider availability, Exa search access, or remote executor support. Configure content through Respan or the environment; the optional `tracer_provider` constructor argument supports a custom OTel provider. Existing `config` privacy options are accepted without requiring OpenInference; enabling any input/output hiding option conservatively disables content for the whole call. Other forwarded legacy options remain accepted.

See the [paired examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/smolagents) for ten runnable fixture scenarios and optional live model usage.
