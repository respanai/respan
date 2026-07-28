# Respan instrumentation for Apache Burr

This package attaches a Burr lifecycle adapter when
`ApplicationBuilder.build()` is called. It maps Burr's own execution model:

- application execution methods (`run`, `arun`, `iterate`, streaming methods)
  become workflow spans
- state-machine actions become task spans with their declared reads, writes,
  tags, inputs, sequence ID, result, and state transition
- Burr custom `ActionSpan` instances become nested task spans
- Burr stream lifecycle callbacks become span events
- attributes logged through Burr's tracing API are retained in Respan metadata

Burr application IDs are mapped to Respan trace-group identifiers and Burr
partition keys are mapped to thread identifiers.

Activate Respan before building the application:

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

Set `capture_content=False` to retain Burr operation identity and status while
omitting state, action inputs/results, stream items, and logged attribute
values.
