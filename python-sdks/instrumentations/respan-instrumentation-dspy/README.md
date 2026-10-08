# respan-instrumentation-dspy

Respan tracing for DSPy 3.x using native callbacks and actual sampled OpenTelemetry spans. Tested with released DSPy 3.0.0 and 3.4.0.

```bash
pip install respan-instrumentation-dspy respan-ai
```

```python
import os
import dspy
from respan import Respan
from respan_instrumentation_dspy import DSPyInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"], instrumentations=[DSPyInstrumentor()]
)
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"], cache=False)
)
try:
    print(dspy.Predict("question -> answer")(question="What is DSPy?").answer)
finally:
    respan.shutdown()
```

The callback observes modules, models, tools, adapters and evaluation. On SDKs that expose them, native compile and interpreter callbacks are also recognized. Small owned hooks capture actual model responses independently of DSPy history, observe sync/async Embedder results, and correlate ReActV2 tool executions with unambiguous native call IDs. Original responses, exceptions, NumPy arrays and DSPy streaming iterators retain their native behavior.

DSPy 3.4 native `lm15.Request`/`Response`, custom engines, `streamify`, and async ReActV2 are covered by controlled released-SDK examples. Usage comes from observed response fields; cache hits omit historical token usage. Callable Embedder returns complete vectors but exposes no provider usage, so the adapter does not manufacture token counts. Failed calls have error spans and no invented output or HTTP status. Remote interpreters, external tools and live provider streaming have not been validated by these fixtures.

`DSPyInstrumentor(target=None, include_content=True, tracer_provider=None)` supports global or target registration. Shared compatible registrations emit one subtree and remain active until their final owner deactivates. Foreign callbacks and wrappers are retained. Set `include_content=False`, `TRACELOOP_TRACE_CONTENT=false`, or the Respan content context to disable payload capture. The initial policy bounds the call; a later opt-out removes captured payload before export. OTel suppression and sampling are respected. Credential values are redacted, unknown application objects are described without calling their string or dump hooks, and complete tool arguments/results and vectors are preserved.

See the [paired fixture examples](https://github.com/Keywords-AI/respan-example-projects/tree/main/python/tracing/dspy). They default to deterministic local model fixtures and can export traces without making provider requests. Live Gateway mode is an explicit option.
