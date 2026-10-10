# Respan instrumentation for Restate

This package uses Restate's `invocation_context_managers` extension point to
observe one attempt of a registered Service, VirtualObject, or Workflow handler.
It does not add durable operations, retry handlers, or wrap durable futures.
Restate keeps responsibility for replay, journaling, cancellation, serialization,
and extension-data cleanup.

Activate the instrumentor before registering handlers:

```python
import restate
from respan import Respan
from respan_instrumentation_restate import RestateInstrumentor

respan = Respan(api_key="...", instrumentations=[RestateInstrumentor()])
greeter = restate.Service("Greeter")


@greeter.handler()
async def greet(ctx: restate.Context, name: str) -> str:
    return f"Hello, {name}!"
```

The adapter records service and handler identity, attempt replay state, invocation
ID, object/workflow key, scope, limit key, and idempotency key. Invocation IDs map
to trace groups; object and workflow keys map to thread identifiers. Standard JSON
inputs retain the complete JSON structure. Secret fields and credential-bearing
URLs are redacted.

Request capture reads native JSON buffers without invoking serde callbacks again.
Opaque custom serde and journal-codec input is omitted. The invocation hook does
not expose the handler result, so it emits no response body or invented usage.
A successful attempt has an OTel OK status; an exception has an ERROR status.
Only status codes supplied by the native error are captured.

Set `capture_content=False`, `TRACELOOP_TRACE_CONTENT=false`, or the Respan
content context to disable request, metadata, and diagnostic content. Ambient and
supplied parent contexts both apply. A privacy veto remains in effect for the
observed ancestor chain, including completed ancestors, and clears content before
the attempt span ends. An unobserved local recording parent fails closed.
Sampling and OTel suppression are checked before request inspection. Telemetry
failures preserve the native handler result or exception and native cleanup.

The supported Restate range starts at 1.0.1 because Restate 1.0.0 was yanked for a
stability bug. `respan-sdk>=2.7.0` supplies the required span-attribute module;
released 2.6.1 and 2.6.2 do not contain it. The package has been exercised with
Restate 1.0.1 and 1.0.5, and with the declared Respan and OTel dependency floors.
