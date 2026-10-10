# Respan instrumentation for Ragas

Native task spans for Ragas 0.4.3 evaluations, metrics, and experiments. This
package preserves native results, exceptions, callbacks, and deferred execution.

```python
import ragas
from opentelemetry.sdk.trace import TracerProvider
from respan_instrumentation_ragas import RagasInstrumentor

provider = TracerProvider()  # Add your application's span processors/exporters.
instrumentor = RagasInstrumentor(tracer_provider=provider)
instrumentor.activate()
try:
    result = ragas.metrics.collections.ExactMatch().score(
        reference="Paris", response="Paris"
    )
    print(result.value)
finally:
    instrumentor.deactivate()
    provider.shutdown()
```

## Supported APIs

| Native API | Captured task lifecycle |
| --- | --- |
| `ragas.evaluate`, `ragas.aevaluate`, their evaluation-module aliases | One evaluation span with connected metric children and the native result. |
| `return_executor=True` | The original `Executor` is returned immediately. The evaluation span ends on `results`, `aresults`, `cancel`, collection, or deactivation. Unconsumed/cancelled evaluations have no invented output. |
| Legacy single-turn and multi-turn metrics | `single_turn_score`, `single_turn_ascore`, `multi_turn_score`, and `multi_turn_ascore`, including native extension subclasses. |
| Collection and decorated metrics | `score`, `ascore`, `batch_score`, and `abatch_score`; numeric, discrete, and ranking decorators created after activation are included. |
| Experiments | Native decorated experiment `__call__` and `arun`, with connected row spans. |

A metric decorator's raw `metric(...)` call retains its SDK behavior: it calls the
application function directly without validated scoring or a metric span. Use
`score` or `ascore` for traced scoring. Result lists, datasets, metric traces,
NumPy arrays, false values, zeros, and empty results remain complete. Nested calls
on the same metric object are deduplicated; separate metric objects remain
separate operations. Native errors record `error.type` and OpenTelemetry error
status without fabricated HTTP status or error output. If Ragas catches a callback
error and returns `MetricResult(value=None, reason=...)`, that original result is
preserved and captured as a successful SDK call.

## Content and lifecycle

Sampling and general/LLM instrumentation suppression are checked before reading
content. `capture_content=False`, canonical `ENABLE_CONTENT_TRACING_KEY=False`,
legacy `trace_content=False`/`override_enable_content_tracing=False`, and
`RESPAN_TRACE_CONTENT=false`/`TRACELOOP_TRACE_CONTENT=false` omit bodies. Content
bounds from observed active/finished ancestors cannot widen; unknown local parents
fail closed. A later veto clears owned I/O, error descriptions, SDK events, and
retained buffers, including the readable span snapshot before export.

Credential fields, authentication text, and URL credentials are redacted. JSON
schema field names are preserved while sensitive defaults/examples are redacted.
Serialization reads known native storage without invoking user `model_dump`,
string, representation, iterator, equality, or hash hooks for telemetry. Unknown
objects are represented by their type name. No model, token usage, HTTP status, or
synthetic result is inferred for task spans.

Compatible owners share patches until the final owner deactivates; conflicting
configuration raises `ValueError`. Deactivation restores only owned patches and
leaves foreign replacements in place. Observer faults cannot replace native
results or exceptions. Application OTel limits can still evict attributes; the
package does not change provider configuration.

## Compatibility and examples

Ragas 0.4.3 is the current supported release. `langchain-community>=0.3,<0.4`
is retained because released Ragas 0.4.3 unconditionally imports `ChatVertexAI`
from a namespace removed in Community 0.4.2; the bare SDK fails to import with
that newest companion. This is an upstream compatibility limitation.

The paired examples in `respan-example-projects/python/tracing/ragas` use actual
released Ragas APIs and deterministic native metrics/application callbacks. They
run locally without credentials and explicitly opt into Respan trace export.
The latest/minimum companion profiles and a clean wheel built from the sdist are
validated independently; stored Respan projection fidelity is a separate gate.
