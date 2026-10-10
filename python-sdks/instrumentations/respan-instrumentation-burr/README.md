# Respan instrumentation for Apache Burr

This package attaches a Burr lifecycle adapter when
`ApplicationBuilder.build()` or `abuild()` is called. It supports Apache Burr
0.42 and maps Burr's own execution model:

- application execution methods (`run`, `arun`, `iterate`, streaming methods)
  become workflow spans
- state-machine actions become task spans with their declared reads, writes,
  tags, inputs, sequence ID, result, and state transition
- Burr custom `ActionSpan` instances become nested task spans
- Burr stream lifecycle callbacks become span events
- attributes logged through Burr's tracing API are retained in Respan metadata

Burr application IDs are mapped to Respan trace-group identifiers and Burr
partition keys are mapped to thread identifiers.

Install `respan-ai` alongside this package when using the unified Respan entry
point below. Activate Respan before building the application:

```python
from burr.core import ApplicationBuilder, Result, State, action, default, expr
from respan import Respan
from respan_instrumentation_burr import BurrInstrumentor

respan = Respan(
    api_key="...",
    instrumentations=[BurrInstrumentor()],
)


@action(reads=["count"], writes=["count"])
def increment(state: State) -> State:
    return state.update(count=state["count"] + 1)


result = Result("count").with_name("result")

app = (
    ApplicationBuilder()
    .with_identifiers(app_id="counter-app", partition_key="user-42")
    .with_actions(increment, result)
    .with_transitions(("increment", "increment", expr("count < 2")))
    .with_transitions(("increment", "result", default))
    .with_entrypoint("increment")
    .with_state(count=0)
    .build()
)

try:
    _, _, state = app.run(halt_after=["result"])
    print(state["count"])
finally:
    respan.shutdown()
```

Set `capture_content=False` to retain Burr operation identity and OTel status
while omitting state, action inputs/results, stream items, logged attributes,
and exception messages, stack traces, and events. The adapter also honors
`TRACELOOP_TRACE_CONTENT=false`, `RESPAN_TRACE_CONTENT=false`, Respan's content
context, and OTel suppression. Sampling is decided before content is read.
A capture veto remains in force for descendants and cannot be widened later in
the same execution. Already-recording local parents that predate the adapter's
privacy observation use a private bound.

Activate multiple owners with the same content setting to share one adapter.
A conflicting setting raises `RuntimeError`; the active setting stays in force.
Deactivation restores only wrappers this package still owns. Built applications
retain a disabled adapter, and native build results, exceptions, and streaming
containers keep their original behavior.

The adapter records structured Burr state and logged attributes, including
nested histories, schemas, arguments, and vectors. It does not infer model calls,
token usage, or HTTP status from application actions. Custom span output is not
invented when Burr provides no result. The stream metadata retains the first 32
items and its total item count; item events retain each received item. Fully
consume streams or call their native `get()` method to deliver Burr's terminal
lifecycle callbacks. Unknown Python objects are represented by their type;
arbitrary iterators are never consumed for telemetry.
