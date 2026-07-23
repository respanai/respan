# Respan instrumentation for Ragas

This package traces Ragas 0.4 evaluation and experiment surfaces as canonical
Respan task spans. It covers `evaluate`, `aevaluate`, single-turn and multi-turn
metrics, experiment runs, and each experiment row. Sync and async failures keep
their original exception while OpenTelemetry records error status and exception
details.

```python
from respan_instrumentation_ragas import RagasInstrumentor

instrumentor = RagasInstrumentor(capture_content=True)
instrumentor.activate()
```

`capture_content=False` keeps operation names and lifecycle/error metadata but
omits evaluation inputs and outputs. Activation and deactivation are idempotent,
and patches are removed only after the final active instrumentor is deactivated.
