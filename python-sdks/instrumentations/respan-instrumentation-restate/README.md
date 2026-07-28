# Respan instrumentation for Restate

This package instruments the Restate Python SDK through its
`invocation_context_managers` extension point. It adds one span around each
handler invocation attempt registered on a:

- `restate.Service`
- `restate.VirtualObject`
- `restate.Workflow`

The span records Restate-specific service, handler, invocation, replay,
object/workflow key, scope, limit-key, and idempotency-key context. Request
content is deserialized with the handler's configured Restate serde. Restate
does not expose the serialized handler result to invocation context managers,
so the adapter records completion status without inventing a response body.

Activate Respan before registering handlers:

```python
import restate
from respan import Respan
from respan_instrumentation_restate import RestateInstrumentor

respan = Respan(
    api_key="...",
    instrumentations=[RestateInstrumentor()],
)

greeter = restate.Service("Greeter")

@greeter.handler()
async def greet(ctx: restate.Context, name: str) -> str:
    return f"Hello, {name}!"
```

Restate invocation IDs are mapped to Respan trace-group identifiers. Object
and workflow keys are mapped to thread identifiers so repeated invocations can
be correlated. Set `capture_content=False` to omit the deserialized request.
