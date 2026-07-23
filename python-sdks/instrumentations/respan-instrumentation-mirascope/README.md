# Respan instrumentation for Mirascope

This package traces Mirascope 2.x model calls, context-aware calls, sync and
async streams, and toolkit execution. Model calls become canonical Respan chat
spans; toolkit executions become canonical tool spans. Streaming spans finish
when the underlying stream is consumed and preserve iterator errors.

```python
from respan_instrumentation_mirascope import MirascopeInstrumentor

instrumentor = MirascopeInstrumentor(capture_content=True)
instrumentor.activate()
```

`capture_content=False` omits messages, tool definitions, arguments, and model
outputs while retaining models, providers, usage, status, and errors. Activation
is reference-counted and safe to repeat.

Do not combine this adapter with `mirascope.ops.instrument_llm()` unless you
intentionally want two independent telemetry pipelines for each operation.
