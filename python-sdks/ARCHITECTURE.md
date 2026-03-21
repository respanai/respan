# Python SDK Architecture

## Package Roles

```
respan-tracing          The engine. OTEL TracerProvider, processors, exporter, decorators,
                        auto-instrumentation, client API. Standalone — works without `respan`.

respan                  The user-facing entry point. Thin wrapper that initializes respan-tracing,
                        activates instrumentation plugins, provides propagate_attributes() and
                        log_batch_results(). Re-exports decorators for convenience.

respan-instrumentation-openai
                        Plugin: activates opentelemetry-instrumentation-openai + sync prompt patch.
                        Spans flow through the existing OTEL pipeline — no custom export logic.

respan-instrumentation-openai-agents
                        Plugin: registers a TracingProcessor with the OpenAI Agents SDK.
                        Converts SDK Trace/Span objects to ReadableSpan and injects them
                        into the OTEL pipeline via span_factory.

respan-sdk              Shared types, constants, OTLP attribute maps. No runtime behavior.
```

## Data Flow

```
                    ┌──────────────────────────────────────────────────┐
                    │              Span Sources                       │
                    ├──────────────────────────────────────────────────┤
                    │                                                  │
                    │  @workflow / @task / @agent / @tool              │
                    │      └─ tracer.start_as_current_span()          │
                    │         └─ OTEL creates ReadableSpan            │
                    │                                                  │
                    │  Auto-instrumentation (openai, anthropic, ...)  │
                    │      └─ OTEL instrumentors create ReadableSpan  │
                    │                                                  │
                    │  Instrumentation plugins                        │
                    │      └─ Convert SDK events → ReadableSpan       │
                    │      └─ inject_span() → processor.on_end()      │
                    │                                                  │
                    │  log_batch_results()                            │
                    │      └─ build_readable_span() per result        │
                    │      └─ inject_span()                           │
                    │                                                  │
                    └──────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────────────────┐
                    │         TracerProvider (singleton)               │
                    │                                                  │
                    │  on_start(span):                                │
                    │    RespanSpanProcessor                           │
                    │      ├─ inject entity_path from context         │
                    │      ├─ inject workflow_name from context       │
                    │      ├─ inject trace_group_id from context      │
                    │      ├─ inject respan params from context       │
                    │      └─ bridge _PROPAGATED_ATTRIBUTES onto span │
                    │           (customer_id, thread_id, metadata)    │
                    │                                                  │
                    │  on_end(span):                                  │
                    │    RespanSpanProcessor                           │
                    │      ├─ is_processable_span() filter            │
                    │      │    accept if has: traceloop.span.kind    │
                    │      │                   traceloop.entity.path  │
                    │      │                   llm.request.type       │
                    │      │                   gen_ai.system          │
                    │      │    reject otherwise (HTTP noise, etc.)   │
                    │      └─ export_filter evaluation (if present)   │
                    │                                                  │
                    │  BatchSpanProcessor                             │
                    │      └─ batches spans, calls exporter.export()  │
                    │                                                  │
                    └──────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────────────────┐
                    │          RespanSpanExporter                      │
                    │                                                  │
                    │  1. Enrich: inject llm.request.type="chat"      │
                    │     for GenAI spans missing it (EnrichedSpan)   │
                    │                                                  │
                    │  2. Root promotion: clear parent for standalone  │
                    │     spans (is_root_span_candidate)              │
                    │                                                  │
                    │  3. Serialize: ReadableSpan → OTLP JSON         │
                    │                                                  │
                    │  4. POST → /v2/traces                           │
                    │     (with _SUPPRESS_INSTRUMENTATION to prevent  │
                    │      recursive traces from the HTTP call)       │
                    └──────────────────────────────────────────────────┘
```

## What Each Package Actually Does

### respan-tracing (the engine)

Everything below exists on main and should not be duplicated:

| Module | Purpose |
|--------|---------|
| `main.py` | `RespanTelemetry` — top-level init, logging config |
| `core/tracer.py` | `RespanTracer` — singleton, sets up TracerProvider + processors + exporter |
| `core/client.py` | `RespanClient` — programmatic access: get trace_id, update span, add events |
| `decorators/` | `@workflow`, `@task`, `@agent`, `@tool` — create spans with traceloop.* attributes |
| `contexts/span.py` | `respan_span_attributes()` — set customer/thread/metadata on active span |
| `processors/base.py` | `RespanSpanProcessor` — enrichment + filtering on_start/on_end |
| | `BufferingSpanProcessor` — ContextVar-based span buffering |
| | `FilteringSpanProcessor` — route spans to specific exporters |
| | `SpanBuffer` — manual span collection for deferred export |
| `exporters/respan.py` | `RespanSpanExporter` — OTLP JSON serialization + POST to /v2/traces |
| | `EnrichedSpan` — proxy that injects missing attributes before export |
| `utils/instrumentation.py` | Auto-instrumentation: 47 OTEL instrumentors (OpenAI, Anthropic, etc.) |
| `utils/preprocessing/` | `is_processable_span()`, `is_root_span_candidate()` — filtering logic |
| `utils/context.py` | `get_entity_path()` — read decorator hierarchy from OTEL context |

### respan-tracing additions (this branch)

| Addition | Why it's needed |
|----------|----------------|
| `auto_instrument` param on `RespanTracer`/`RespanTelemetry` | When plugins handle tracing, we need to disable auto-instrumentation to avoid duplicate spans |
| `EnrichedSpan` + `_get_enrichment_attrs()` in exporter | GenAI spans from `opentelemetry-instrumentation-openai` lack `llm.request.type`, backend needs it |
| `is_processable_span()` accepting `gen_ai.system` | Without this, Responses API spans are silently filtered out |
| `_PROPAGATED_ATTRIBUTES` bridge in `RespanSpanProcessor.on_start()` | Auto-instrumented spans need customer/thread context from `propagate_attributes()` |
| Quieter logging (error → debug for missing instrumentations) | Users without all SDKs installed shouldn't see error-level noise |

### span_factory.py — why it exists

`span_factory.py` provides `build_readable_span()` and `inject_span()`. These are needed by:

1. **`respan-instrumentation-openai-agents`** — converts OpenAI Agents SDK Trace/Span objects into `ReadableSpan` and injects them into the OTEL pipeline. The SDK has its own span format; this bridge converts them.

2. **`respan.log_batch_results()`** — constructs `ReadableSpan` objects for batch API results that arrived asynchronously (no live OTEL context).

Without span_factory, these use cases would need to either:
- Use the OTEL tracer API (`tracer.start_as_current_span()`) which requires an active context and doesn't support explicit trace/span IDs
- Bypass the OTEL pipeline entirely (losing all processor enrichment and filtering)

`span_factory` also contains `_PROPAGATED_ATTRIBUTES` and `propagate_attributes()`:
- **`_PROPAGATED_ATTRIBUTES`** — ContextVar storing customer_identifier, thread_identifier, metadata, etc.
- **`propagate_attributes()`** — context manager to set these for a scope
- **`read_propagated_attributes()`** — reads the ContextVar and maps values to OTEL span attribute keys (`respan.customer_params.*`, `respan.threads.*`, etc.)

**Overlap with `respan_span_attributes()`**: The existing `respan_span_attributes()` in `contexts/span.py` sets attributes directly on the *current active span*. `propagate_attributes()` stores attributes in a ContextVar that gets bridged onto *all* spans created within the scope (including auto-instrumented ones that don't have a parent decorator span). They serve different use cases:

| | `respan_span_attributes()` | `propagate_attributes()` |
|---|---|---|
| Target | Current active span only | All spans in scope |
| Requires decorator | Yes (@workflow/@task) | No |
| Works with auto-instrumented spans | Only if inside a decorator | Yes (bridged in on_start) |
| Works with plugins | No | Yes |

### respan (the wrapper)

| What | Why |
|------|-----|
| `Respan.__init__()` | Initializes `RespanTelemetry` with `auto_instrument=False` when plugins provided |
| `Respan.propagate_attributes()` | Static method wrapping `span_factory.propagate_attributes()` |
| `Respan.log_batch_results()` | Constructs ReadableSpans for OpenAI Batch API results |
| `Respan.flush()` / `.shutdown()` | Delegates to telemetry + deactivates plugins |
| Re-exports | `workflow`, `task`, `agent`, `tool`, `get_client`, `propagate_attributes` |

### respan-instrumentation-openai (plugin)

Self-contained. Does two things:
1. Calls `opentelemetry-instrumentation-openai`'s `instrument()` — this patches the OpenAI SDK to emit OTEL spans automatically
2. Applies sync prompt-capture patch (same code that exists in `respan-tracing/utils/instrumentation.py` for auto-instrumentation)

**Why a separate package?** When using the `Respan()` entry point with explicit plugins, auto-instrumentation is disabled. This plugin enables *only* OpenAI instrumentation without pulling in all 47 instrumentors.

### respan-instrumentation-openai-agents (plugin)

Registers a `TracingProcessor` with the OpenAI Agents SDK. On each trace/span end:
1. `_otel_emitter.py` dispatches by span data type (Agent, Response, Function, Handoff, etc.)
2. Each emitter builds a `ReadableSpan` via `span_factory.build_readable_span()` with proper `traceloop.*` and `gen_ai.*` attributes
3. Calls `inject_span()` to push it into the OTEL pipeline

## Attribute Propagation Summary

```
User code:
    with propagate_attributes(customer_identifier="user_123"):
        client.chat.completions.create(...)

    ↓ ContextVar stores {"customer_identifier": "user_123"}

RespanSpanProcessor.on_start():
    ↓ reads ContextVar via read_propagated_attributes()
    ↓ maps to: span.set_attribute("respan.customer_params.customer_identifier", "user_123")

RespanSpanExporter.export():
    ↓ serializes span attributes to OTLP JSON
    ↓ POST to /v2/traces

Backend:
    ↓ reads respan.customer_params.customer_identifier
    ↓ associates the log with the customer
```
