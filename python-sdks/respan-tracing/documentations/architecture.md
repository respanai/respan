# respan-tracing — Architecture (decorator → span → batch → export)

This doc explains what actually happens when application code calls `@workflow(...)` + `client.update_current_span(attributes=...)`. The short version: emission is microseconds and asynchronous; export is batched on a background thread; the dogfood/audit endpoints CAN return slow / 5xx without backpressuring the calling request.

Read this before reasoning about the cost of instrumentation.

## End-to-end path of one decorated call

```
@workflow(name="X", processors="dogfood")
def my_view(request, ...):
    client = get_client()
    if client:
        client.update_current_span(attributes={"domain.org_id": org_id})
    return do_work()
```

1. **`@workflow(...)` wraps the function** (`src/respan_tracing/decorators/base.py::_create_entity_method_decorator`). On call:
   - `_setup_span(...)` opens an OTel span via `setup_span` (`utils/span_setup.py`). Pure in-process — context push, span allocation, attribute initialization. No I/O.
   - The wrapper invokes the user function inside a `try/finally` and `_cleanup_span(...)` ends the span on exit (`span.end()`).

2. **`client.update_current_span(attributes={...})`** (`src/respan_tracing/core/client.py::RespanClient.update_current_span`):
   - Resolves the active span via the OTel context.
   - Loops `span.set_attribute(key, value)` per attribute — each call is a dict write on the span object. No I/O.

3. **`span.end()`** triggers each registered `SpanProcessor`'s `on_end(span)`:
   - In our SDK every processor is wrapped in `FilteringSpanProcessor` (`src/respan_tracing/processors/base.py::FilteringSpanProcessor`).
   - `on_end` evaluates the processor's `filter_fn` (the `processors="X"` discriminator routes here). If the filter rejects, the span is dropped at this layer — no enqueue, no export.
   - If the filter accepts, the wrapped `BatchSpanProcessor` (default — `is_batching_enabled=True` at `processors/base.py:282–285`) appends to its bounded in-memory queue.

4. **A background thread inside `BatchSpanProcessor`** flushes the queue on a timer (or when the queue hits its size threshold) and POSTs one OTLP JSON batch to the exporter's endpoint (`src/respan_tracing/exporters/respan.py::RespanSpanExporter.export`).

The application thread that called `@workflow` returns as soon as `span.end()` returns — typically microseconds. **No HTTP round-trip happens on the calling thread.**

## Cost model (what callers should assume)

| Operation | Cost | I/O? |
|---|---|---|
| `@workflow` entry | Span allocation + context push | No |
| `client.update_current_span(attributes=...)` | N dict writes (`span.set_attribute`) | No |
| `@workflow` exit (`span.end()`) | Queue enqueue + filter eval | No |
| Background batch flush | One OTLP JSON POST per batch | Yes — async |

Implications:
- **Adding attributes is free.** A dozen `update_current_span` calls in one function is a dozen dict writes — not a dozen network calls.
- **A slow / 5xx audit endpoint cannot block the request thread.** The batch processor's queue is bounded; if it fills, spans are dropped (with logged warnings) but the request still returns.
- **`processors="X"` filtering is a per-processor decision, not a global one.** Routing a span to the audit processor does not also send it to dogfood — `FilteringSpanProcessor.on_end` checks the filter that was wired in at `add_processor` time.

## When `is_batching_enabled=False` is the right choice

`add_processor(..., is_batching_enabled=False)` swaps `BatchSpanProcessor` for `SimpleSpanProcessor`. `on_end` then calls `exporter.export([span])` synchronously on the caller's thread. **Use this only when:**

- The exporter is in-process (no network) — e.g., a `DebugFileExporter` writing to local disk in `DEBUG=True`, or an in-memory `InMemorySpanExporter` for tests.
- An in-process exporter that ITSELF triggers a downstream sync action (e.g., `DatasetWorkflowSpanExporter` invoking `log_dataset_request()` directly).

For any HTTP-bound exporter (`RespanSpanExporter`), `is_batching_enabled=True` is mandatory — otherwise every `@workflow` exit becomes a synchronous network round-trip on the request thread.

## Trace boundaries inside batch handlers (`is_new_trace_root`)

OTel propagates `trace_id` implicitly through the active `Context`. Every span created while another span is active attaches as that span's child and inherits the parent's `trace_id`. This is the correct default for request-scoped work, but it is the **wrong** default at any execution boundary where the inner work is an independent unit.

The case that drove the flag — and the canonical case it is meant for — is a **batch consumer dispatching per-message work**:

```python
@workflow(name="my_consumer_handle_batch")          # 1 span per BATCH
async def _handle_batch(consumer, messages):
    for message in messages:
        # Each per-message Celery task creates its own @workflow.
        # By default it ATTACHES as a child of _handle_batch and shares
        # its trace_id, even though the messages are independent.
        await asyncio.to_thread(per_message_task.run, **message.payload)
```

If `per_message_task` is decorated `@workflow(name="run_X")`, every message in one batch ends up with the same `trace_id`. Downstream readers that do `count(distinct trace_id)` (per-trace UIs, "how many runs did this experiment produce") collapse N messages into ~1 trace per batch.

`is_new_trace_root=True` declares the decorated function a fresh trace root: the wrapper attaches an empty `Context()` before creating the span, so OTel sees no active parent and allocates a new `trace_id`. On exit the root context token is detached LIFO-last to restore the caller's original trace.

```python
@workflow(name="run_X", is_new_trace_root=True)    # FRESH trace per call
def per_message_task(...):
    ...
```

**When to set the flag:**

- The decorated function is invoked from inside another active `@workflow` span (a batch handler, a request handler, an outer workflow).
- The work being decorated is conceptually one independent unit — one message, one row, one job — not a sub-step of the caller.
- Downstream readers count or aggregate by `trace_id`.

**When NOT to set it:**

- The decorated function is a request handler, a top-level Celery task, or a CLI entry point — there is no active OTel context to detach from. The flag is harmless here but signals the wrong intent.
- The work IS conceptually a sub-step (a child task within a workflow) and should be queryable as part of the caller's trace.

### Why a flag, not a separate API

The alternative was `client.start_root_workflow(...)` as a sibling primitive. Rejected because every existing `@workflow`-decorated function would then have two entry points to maintain, and callers would have to know which one to use. A flag on the existing decorator keeps the surface flat: one decorator, one mental model, one call site to update when a function moves from "child of caller" to "fresh trace root."

The cost is one extra optional kwarg on the public `workflow()` factory, one extra context token threaded through `setup_span`/`cleanup_span`, and zero runtime overhead when `is_new_trace_root=False` (the default). When `True`, the cost is one `context_api.attach(Context())` and one matching `detach` — a few hundred nanoseconds per call.

### Why expose it on `@workflow` and `client.start_span` only

The flag is conceptually meaningful only for **entry-point** spans (`workflow`/`agent` kinds). `task` and `tool` spans are by definition sub-steps within a containing workflow; making one a fresh root would orphan it from its logical parent. The public `task()` / `agent()` / `tool()` decorator factories therefore do NOT expose the kwarg, even though the lower-level `create_entity_method` accepts it so the imperative `client.start_span(..., is_new_trace_root=True)` can work for kinds the caller picks at runtime.

### Span-link, not trace inheritance, when you need to correlate

If a fresh-root span needs to be findable from the caller's trace (e.g. "follow the request → kicked-off background job"), use **span links** (`links=[SpanLink(...)]` on the decorator), not trace inheritance. Links record a typed cross-trace pointer in the OTel data model. Trace inheritance is the wrong tool for cross-trace correlation: it tells readers "these two spans are part of the SAME trace," which is false when the two pieces of work have independent lifecycles.

## Failure modes

- **Exporter network failure:** `BatchSpanProcessor` retries internally; eventually drops with a logged warning. The application thread doesn't see the failure.
- **Span queue overflow:** Same — drops with a warning. The application thread doesn't see the failure.
- **Exception inside the decorated function:** The `@workflow` wrapper sets `Status(StatusCode.ERROR)` on the span and re-raises. The span still ends and is exported (carrying the error status as an attribute).
- **`get_client()` returns `None`:** Telemetry is uninitialized. Calls become no-ops at the application level (the codebase guards with `if client:` before `update_current_span`).

## Common confusions worth dispelling

- **"`update_current_span` sends data."** No — it mutates the in-memory span object. The span is shipped (batched) when it ends.
- **"More attributes = slower."** Within reason, no. Each attribute is one dict write. The serialization cost is paid once per batch on the export thread.
- **"My processor's `processors=` argument routes to that processor only."** Correct, but the way it's enforced is via per-processor `filter_fn` registered when `add_processor(..., name=X)` was called — a span without `processor=X` is filtered out at that processor's `on_end`. The span itself isn't tagged for routing; the routing is a downstream filter.
- **"A 10× span emission means a 10× HTTP cost."** No — those 10 spans are batched into one POST (or dropped if the queue is full). The 10× cost is span allocation + downstream storage + dashboard noise, not request-path latency.
- **"Filtering with `export_filter` is free."** Almost — the span is still CREATED (allocation, attribute init, context push). `export_filter` only suppresses the EXPORT. For very-high-volume paths (>10k/s), even the creation cost matters — see "SDK gap: no conditional span creation" in the backend dogfood architecture doc.

## Source map

| File | Role |
|---|---|
| `src/respan_tracing/main.py::RespanTelemetry` | Public entry point — `__init__`, `add_processor`, `flush` |
| `src/respan_tracing/core/tracer.py::RespanTracer` | Singleton tracer; owns the OTel `TracerProvider` and registered processors |
| `src/respan_tracing/core/client.py::RespanClient` | Thin wrapper that resolves the active span and exposes `update_current_span` |
| `src/respan_tracing/decorators/base.py` | `@workflow` / `@task` decorator factories — `_setup_span` / `_cleanup_span` lifecycle |
| `src/respan_tracing/processors/base.py::FilteringSpanProcessor` | Per-processor `filter_fn` + wraps either `BatchSpanProcessor` or `SimpleSpanProcessor` |
| `src/respan_tracing/exporters/respan.py::RespanSpanExporter` | OTLP JSON over HTTPS to `<endpoint>/v2/traces` |
| `src/respan_tracing/utils/span_setup.py` | Span creation / cleanup primitives shared by the decorator. The `is_new_trace_root` boundary control lives here: `setup_span` attaches an empty `Context()` when flagged; `cleanup_span` detaches it LIFO-last to restore the caller's original context. |
