# respan-instrumentation-openinference

Respan's generic lifecycle wrapper and canonical span translator for
[OpenInference](https://github.com/Arize-ai/openinference) instrumentors. It keeps
native OpenInference instrumentation and decorator APIs; translation happens
before export on real sampled OTel spans.

```python
from openinference.instrumentation.openai import OpenAIInstrumentor
from respan import Respan
from respan_instrumentation_openinference import OpenInferenceInstrumentor

respan = Respan(instrumentations=[OpenInferenceInstrumentor(OpenAIInstrumentor)])
```

Install the individual OpenInference delegates you use. Standard instrumentors
receive options through their native `instrument()` method; processor delegates
receive constructor options. Activations share one delegate per class and one
translator. The first active configuration wins, the last owner tears down only
its owned delegate, and externally installed delegates/processors remain owned
by their original caller. Processor ordering is restored without removing
processors added by other owners.

The translator maps OpenInference kinds, input/output, model/provider, actual
nonnegative provider usage, modern typed message content, current-turn tools,
tool execution IDs, and embeddings into the Respan span contract. It preserves
complete tool schemas, arguments/results, dense vectors and numeric-index sparse
payloads. Ordinary text is bounded and common credential shapes are redacted;
unknown objects are not stringified. Historical calls remain in input messages.
Consumed raw OI fields and legacy aliases are removed before export.

`TRACELOOP_TRACE_CONTENT=false` and Respan's content-disabled context are
snapshotted at span start and checked again at completion. One owned native
OITracer scope observer captures callback vetoes before SDK context detachment. A private start or
ancestor cannot be reenabled; privacy also removes duplicate native GenAI
payloads and private exception messages. Upstream OI masking, sampling and
instrumentation suppression retain their native behavior. This processor gate
controls exported attributes; it does not prevent upstream instrumentors from
creating their own in-process data.

Validated current releases: OI instrumentation 0.1.70, semantic conventions
0.1.41, OpenAI delegate 0.1.63, OpenAI SDK 2.54.0. Minimum test matrix: core,
semconv and OpenAI delegate 0.1.32, OpenAI 1.78.0, Respan tracing 2.16.1/SDK 2.7.6,
OTel 1.38.0 and OTel semantic conventions 0.59b0. Current native retriever,
reranker, guardrail and evaluator decorators and optional GenAI projection are
covered; those APIs are skipped on older versions where absent. Each other
OpenInference delegate has its own SDK requirements and needs separate testing.

The wrapper requires an OTel SDK provider with a mutable processor chain. Keep
it active until in-flight SDK spans finish, then deactivate; deactivation removes
its translator. Telemetry serialization failures are isolated from native
application results and exceptions. Controlled examples exercise the native
OpenAI delegate and OITracer decorators; they do not establish live-provider or
backend stored-payload acceptance.
