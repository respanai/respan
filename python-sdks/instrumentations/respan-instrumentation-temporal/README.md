# respan-instrumentation-temporal

Respan instrumentation for Temporal Python 1.30–1.x. It adapts Temporal's official OpenTelemetry interceptor, preserving its context propagation, sandbox execution, completed-span extern hook and history replay behavior.

## Install

```bash
pip install respan-ai respan-instrumentation-temporal
```

The package requires `respan-tracing>=2.17` and `respan-sdk>=2.6.26`; earlier SDK wheels do not publish the constants this adapter imports.

## Usage

```python
from temporalio.client import Client
from respan import Respan
from respan_instrumentation_temporal import TemporalInstrumentor

instrumentor = TemporalInstrumentor()
respan = Respan(instrumentations=[instrumentor])

# The interceptor is appended automatically and inherited by workers.
client = await Client.connect("localhost:7233")

# Stop workers before flushing and shutting down telemetry.
respan.flush()
respan.shutdown()
```

For Temporal's ephemeral test server or custom client factories, pass the same interceptor explicitly:

```python
from temporalio.testing import WorkflowEnvironment

environment = await WorkflowEnvironment.start_time_skipping(
    interceptors=[instrumentor.interceptor]
)
```

An existing native `TracingInterceptor` is respected, avoiding duplicate tracing. Multiple active instances must use matching options; the final owner restores only its own `Client.connect` wrapper. Existing clients retain their interceptor after deactivation.

## Captured data

Workflow/activity starts record the arguments and identifiers exposed by the native input. Activity execution and workflow completion record their actual results; client queries record their actual responses. Standard JSON containers, dataclasses, supported Pydantic models, protobuf messages and binary values are preserved. Temporal headers are excluded. Capture uses complete payloads by default; `max_attribute_chars` explicitly opts into a JSON preview size limit.

Spans use canonical workflow/task input and output fields. The adapter records sanitized OpenTelemetry error status and `error.message`; it does not invent HTTP status codes, usage or successful result placeholders. Native values, exceptions and cancellation outcomes pass through unchanged when telemetry fails.

```python
TemporalInstrumentor(capture_content=False)
```

Content capture also honors `TRACELOOP_TRACE_CONTENT=false` and Respan's `ENABLE_CONTENT_TRACING_KEY` context veto. Ambient and supplied contexts are combined, denied ancestor decisions persist after completion, and later vetoes scrub owned content before the span ends. Unobserved recording parents fail closed. Sampling and instrumentation suppression are checked before payload access. Exception events are omitted to keep raw tracebacks from bypassing content controls.

Temporal's native workflow spans are instantaneous and replay-aware. A cancellation while waiting can finish the workflow without emitting a native `CompleteWorkflow` span. Worker-side query/signal spans describe native operation boundaries; query results are available on the client span. This adapter retains those boundaries.

## Validation

```bash
pytest tests -m 'not integration'
pytest tests -m integration
```

Integration tests run bare and instrumented SDK workflows on the actual local time-skipping server, checking complete payloads, failures, signals, queries, cancellation and history replay. Temporal downloads its test-server binary on first use; `TEMPORAL_TEST_SERVER_PATH` can select an existing compatible binary.
