# respan-instrumentation-braintrust

Observe released Braintrust native spans through Respan. Tested with Braintrust 0.44.0 and the declared minimum 0.5.0.

```python
import os
import braintrust
from respan import Respan
from respan_instrumentation_braintrust import BraintrustInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"], instrumentations=[BraintrustInstrumentor()]
)
logger = braintrust.init_logger(project="My project")
try:
    with logger.start_span(name="lookup", type="tool") as span:
        result = {"answer": "controlled result"}
        span.log(input={"question": "example"}, output=result)
finally:
    logger.flush()
    respan.shutdown()
```

The adapter creates sampled OpenTelemetry spans at native span construction and observes incremental records when Braintrust resolves its lazy exports. Braintrust's sink receives the same native records and return values. Its native customizers remain in order and run once. Compatible owners share hooks; final teardown restores owned methods while preserving foreign wrappers and outstanding context cleanup. The legacy class alias and context-manager API remain available.

`BraintrustInstrumentor(tracer_provider=None, include_content=True)` honors OTel suppression/sampling, `TRACELOOP_TRACE_CONTENT=false`, and Respan's content context. Initial privacy bounds descendants; an observed later opt-out cannot be undone before deferred export. Unknown application objects are described without calling string/dump hooks. Credentials are redacted. Usage is mapped only from native provider fields, including explicit zero; absent/invalid counters remain absent.

Manual spans, scores/tags, sync/async traced functions, native generators, current/history tool calls and complete tool results are covered. Scoped released OpenAI wrapper observations preserve actual response objects, provider usage and complete embedding vectors; Braintrust's own embedding record currently retains only vector length. Streaming remains a native iterator. Generator functions decorated before activation retain native caller context; functions decorated during activation also retain generator/async-generator introspection and detached OTel advances.

A sink-local Braintrust masking function runs after lazy record resolution. The adapter conservatively disables Respan content capture when that native mask is active, avoiding observation of pre-mask payload and duplicate mask invocation. Adapter `set_masking_function` masks Respan payload; changing its policy while multiple owners are active requires matching settings. Its initial mask remains a bound if removed during a span. Queue limits forward to the native sink.

Native Braintrust logging remains enabled unless the application configures a local sink. The [paired examples](https://github.com/Keywords-AI/respan-example-projects/tree/main/python/tracing/braintrust) use a controlled local sink and HTTP fixtures by default, so they export only to Respan and require no Braintrust/model-provider account. They validate SDK compatibility, not live provider, remote evaluation/function, attachment-upload, dataset or platform eligibility. OTel compatibility mode keeps Braintrust's native context/traceparent APIs intact; cross-framework parenting in that optional mode has not been independently validated.
