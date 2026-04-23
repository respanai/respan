# Tracing Reference

Advanced tracing configuration: decorators, context propagation, processors, and span attributes.

Docs: `https://respan.ai/docs/documentation/features/tracing/traces/tracing-sdk.md`

---

## Decorators

Wrap functions to create structured span hierarchies. All decorators share the same signature.

### Types

| Decorator | Purpose |
|-----------|---------|
| `@workflow` / `withWorkflow` | Root span — top-level pipeline |
| `@task` / `withTask` | Step within a workflow |
| `@agent` / `withAgent` | Agent loop span |
| `@tool` / `withTool` | Tool/function call span |

### Python

```python
from respan import Respan, workflow, task, agent, tool

Respan()

@workflow(name="write_article")
def write_article(topic: str):
    outline = generate_outline(topic)
    return write_draft(outline)

@task(name="generate_outline")
def generate_outline(topic: str):
    return client.chat.completions.create(...)

@task(name="write_draft")
def write_draft(outline: str):
    return client.chat.completions.create(...)
```

### TypeScript

```typescript
import { Respan, withWorkflow, withTask } from "@respan/respan";

const respan = new Respan();
await respan.initialize();

const result = await withWorkflow({ name: "write_article" }, async () => {
  const outline = await withTask({ name: "generate_outline" }, async () => {
    return await client.chat.completions.create({...});
  });
  return await withTask({ name: "write_draft" }, async () => {
    return await client.chat.completions.create({...});
  });
});
```

### Decorator Parameters

```python
@workflow(
    name="my_workflow",               # Display name (default: function name)
    version=1,                        # Version number
    processors="my_processor",        # Route to specific processor(s)
    sample_rate=0.5,                  # 0.0-1.0, fraction of spans exported
    export_filter={                   # Conditional export
        "metadata.env": {"operator": "", "value": "production"}
    },
)
```

All parameters are optional. Without any, the function name is used as the span name.

### Async Support

Decorators work with `async def`, generators, and async generators automatically.

---

## Context Propagation

Attach attributes to **all spans** within a scope, including auto-instrumented LLM calls.

### propagate_attributes

```python
from respan import Respan, propagate_attributes

Respan()

with propagate_attributes(
    customer_identifier="user_123",
    thread_identifier="conv_abc",
    environment="production",
    metadata={"plan": "pro", "team": "search"},
):
    # All spans here (including auto-instrumented OpenAI calls) get these attributes
    result = run_pipeline()
```

#### Available attributes

| Attribute | Description |
|-----------|-------------|
| `customer_identifier` | User/customer ID — enables per-user analytics |
| `customer_email` | Customer email |
| `customer_name` | Customer display name |
| `thread_identifier` | Conversation thread ID — groups related calls |
| `custom_identifier` | Indexed custom identifier for fast queries |
| `group_identifier` | Group related traces together |
| `environment` | Environment name (e.g. `"production"`, `"staging"`) |
| `metadata` | Dict of custom key-value pairs (merged in nested contexts) |
| `prompt` | Dict with `prompt_id` and `variables` for prompt logging |

#### TypeScript

```typescript
const result = await respan.propagateAttributes(
  {
    customerIdentifier: "user_123",
    threadIdentifier: "conv_abc",
    metadata: { plan: "pro" },
  },
  async () => {
    return await runPipeline();
  }
);
```

#### Nesting behavior

- Nested `propagate_attributes` calls merge with the outer context
- `metadata` dicts are merged (not replaced)
- For duplicate keys, inner values override outer
- Async-safe via `contextvars`

### respan_span_attributes

Attaches attributes to the **current active span only** (not auto-instrumented child spans):

```python
from respan import respan_span_attributes

with respan_span_attributes({
    "customer_identifier": "user-123",
    "metadata": {"priority": "high"},
}):
    pass
```

Use `propagate_attributes` when you need attributes on all nested spans (most common). Use `respan_span_attributes` when you only want to tag the current span.

---

## Imperative Span Creation

For runtime-determined span names, use the client API instead of decorators:

```python
from respan import get_client

client = get_client()

with client.start_span(name="dynamic_step", kind="task") as span:
    result = do_work()
    # span is automatically closed and exported
```

---

## Updating the Current Span

Add attributes or change status on the active span at runtime:

```python
from respan import get_client

client = get_client()

# Inside a decorated function or start_span context:
client.update_current_span(
    respan_params={
        "customer_identifier": "user_123",
        "metadata": {"step": "final"},
    },
    name="renamed_span",
)

# Add events
client.add_event("checkpoint_reached", attributes={"items": 42})

# Record exceptions (span continues)
try:
    risky_operation()
except Exception as e:
    client.record_exception(e)
```

---

## Custom Processors

Route spans to multiple destinations with filtering:

```python
from respan_tracing import RespanTelemetry

telemetry = RespanTelemetry(api_key="...", base_url="...")

# Add a second exporter for debug logging
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

telemetry.add_processor(
    exporter=ConsoleSpanExporter(),
    name="debug",
    filter_fn=lambda span: span.attributes.get("environment") == "debug",
)
```

Route specific spans to a processor using the `processors` parameter on decorators:

```python
@task(name="sensitive_step", processors="debug")
def sensitive_step():
    ...
```

---

## Flush

Always call `flush()` before process exit to ensure all spans are exported:

```python
respan = Respan()
# ... your code ...
respan.flush()
```

In serverless/Lambda: call `flush()` at the end of every handler invocation.

---

## Trace Hierarchy Example

```
workflow: handle_request
  +-- task: classify_intent
  |     +-- llm: openai.chat (auto-captured)
  +-- agent: support_agent
  |     +-- tool: lookup_order
  |     +-- llm: openai.chat (auto-captured)
  |     +-- tool: process_refund
  +-- task: generate_response
        +-- llm: openai.chat (auto-captured)
```

LLM calls within decorated functions are automatically nested as child spans.
