# respan-instrumentation-temporal

Respan instrumentation for the Temporal Python SDK. It adapts Temporal's official replay-aware OpenTelemetry interceptor instead of patching workflow execution directly, preserving Temporal context propagation and replay semantics while emitting canonical Respan workflow and task fields.

The instrumentor injects its interceptor into `Client.connect()` automatically. The same interceptor is exposed as `instrumentor.interceptor` for Temporal test environments and custom client factories that accept an explicit `interceptors` list.

## Install

```bash
pip install respan-ai respan-instrumentation-temporal temporalio
```

## Usage

```python
from temporalio.client import Client
from respan import Respan
from respan_instrumentation_temporal import TemporalInstrumentor

instrumentor = TemporalInstrumentor()
respan = Respan(instrumentations=[instrumentor])

# The interceptor is appended automatically and is inherited by workers
# created from this client.
client = await Client.connect("localhost:7233")
```

For `WorkflowEnvironment`:

```python
from temporalio.testing import WorkflowEnvironment

environment = await WorkflowEnvironment.start_time_skipping(
    interceptors=[instrumentor.interceptor]
)
```

If an explicit Temporal `TracingInterceptor` is already present, Respan does not append a second tracing interceptor, avoiding duplicate spans.

## Content capture

```python
TemporalInstrumentor(capture_content=False)
```

Content capture includes workflow/activity arguments and Temporal identifiers when the relevant interceptor input exposes them. With capture disabled, spans retain stable operation and workflow/activity type information but omit arguments and IDs. Temporal headers are never captured. Errors record OpenTelemetry error status plus backend-visible `status_code` and `error.message` fields. `activate()` and `deactivate()` are idempotent.
