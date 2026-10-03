# respan-instrumentation-guardrails

Translate [Guardrails AI](https://github.com/guardrails-ai/guardrails) native OpenTelemetry spans into the Respan contract. Supports released `guardrails-ai>=0.9.3,<0.12`, including 0.11.0 and its `guardrails-ai-types` results.

```bash
pip install respan-ai respan-instrumentation-guardrails
```

Initialize Respan before running a guard:

```python
import os
from guardrails import Guard
from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_guardrails import GuardrailsInstrumentor


class Reply(BaseModel):
    answer: str


respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[GuardrailsInstrumentor()],
    is_auto_instrument=False,
)
try:
    guard = Guard.for_pydantic(Reply)
    guard.configure(allow_metrics_collection=False)
    result = guard.parse('{"answer":"A local fixture reply"}', num_reasks=0)
    assert result.validation_passed
finally:
    respan.shutdown()
```

The adapter normalizes synchronous and asynchronous Guard execution, local Pydantic parsing, registered validators, repair/reask iterations, LLM calls, and native streaming spans. It preserves Guardrails' returned values, exceptions, and stream topology. Validation outcomes remain available as `validation_passed` and validator result content; a validation rejection with `on_fail="noop"` is not an execution exception.

LLM spans capture actual messages, request tools, current response tool calls, model parameters, and source-reported usage. Local validation remains `guardrail`, and observed LLM call counts do not depend on token usage being available. Historical tool calls remain in prompt fields. Native streaming telemetry does not always provide a final model response or token usage; the adapter preserves that absence rather than inventing values. Guardrails may emit linked spans as stream results are consumed.

Set `TRACELOOP_TRACE_CONTENT=false` to omit native guard/validator input, output, message and tool-definition payloads. Respan's `override_enable_content_tracing` context override is supported. Duplicate OpenInference payload attributes are removed after translation. Multiple active plugin instances share one processor per provider; deactivating the last owner removes it from its original provider and leaves unrelated processors intact.

Examples load `RESPAN_API_KEY` and optional `RESPAN_BASE_URL` (default `https://api.respan.ai/api`) and export traces to Respan. See the [paired Python examples](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/guardrails) for nine runnable local scenarios, including async execution, streaming, reasks, content privacy, and controlled errors.
