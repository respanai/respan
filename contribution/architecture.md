# Repository Architecture

This repository is an SDK monorepo with one active architecture shared across Python and JavaScript.

The active system has five layers:

1. Contract layer
2. Tracing runtime layer
3. Product facade layer
4. Instrumentation adapter layer
5. Operator CLI layer

Anything under `python-sdks/legacy/` or `javascript-sdks/legacy/` is outside this architecture.

## Architectural Goal

The core design goal is:

- one tracing model
- one export pipeline
- multiple language front doors
- multiple instrumentation adapters

That means package boundaries matter more than directory boundaries.

## Layer Model

### 1. Contract Layer

This layer defines shared types, attribute keys, and wire-shape constants. It should not own runtime initialization.

Python contract package:

- `python-sdks/respan-sdk`

JavaScript contract package:

- `javascript-sdks/respan-sdk`

Responsibilities:

- public parameter types such as `RespanLogParams`, `RespanParams`, filter types, and usage/message models
- canonical attribute keys such as `RespanSpanAttributes`
- OTLP field names and promotion rules used by exporters and serializers
- low-level utility code that is safe to reuse from runtime packages

Important Python entrypoints:

- [__init__.py](../python-sdks/respan-sdk/src/respan_sdk/__init__.py)
- [span_attributes.py](../python-sdks/respan-sdk/src/respan_sdk/constants/span_attributes.py)
- [otlp_constants.py](../python-sdks/respan-sdk/src/respan_sdk/constants/otlp_constants.py)

Boundary rule:

- `respan-sdk` defines contracts and constants
- `respan-sdk` does not initialize tracing, patch vendor SDKs, or own exporter lifecycle

### 2. Tracing Runtime Layer

This layer owns OpenTelemetry setup, decorators, processor routing, context propagation, span mutation, and export.

Python runtime package:

- `python-sdks/respan-tracing`

JavaScript runtime package:

- `javascript-sdks/respan-tracing`

This is the real engine of the system.

## Python Runtime

Core objects:

- [RespanTelemetry](../python-sdks/respan-tracing/src/respan_tracing/main.py)
  - high-level runtime facade
  - configures logging
  - creates the singleton tracer
  - exposes decorators as instance methods
  - exposes `add_processor()`, `flush()`, and `get_client()`
- [RespanTracer](../python-sdks/respan-tracing/src/respan_tracing/core/tracer.py)
  - singleton OTEL owner
  - creates the `TracerProvider`
  - installs default Respan exporter when `api_key` exists
  - manages additional processors via `add_processor()`
  - controls auto-instrumentation via `_setup_instrumentations()`
- [RespanClient](../python-sdks/respan-tracing/src/respan_tracing/core/client.py)
  - imperative API over current OTEL context
  - reads current trace/span ids
  - updates span attributes via `update_current_span()`
  - records events and exceptions
- [create_entity_method()](../python-sdks/respan-tracing/src/respan_tracing/decorators/base.py)
  - decorator factory used by `workflow`, `task`, `agent`, and `tool`
  - creates spans around sync, async, generator, and async-generator functions
  - serializes input/output when content tracing is enabled
- [RespanSpanExporter](../python-sdks/respan-tracing/src/respan_tracing/exporters/respan.py)
  - transforms `ReadableSpan` objects into Respan OTLP payloads
  - enriches spans before export
  - performs exporter-only synthetic span generation when needed

Supporting subsystems:

- `contexts/`
  - span links and context helpers
- `processors/`
  - filtering, buffering, and span collection behavior
- `utils/span_setup.py`
  - common span setup/cleanup logic used by decorators and clients
  - hosts the `is_new_trace_root` boundary control (see "Trace boundaries" below)
- [instruments.py](../python-sdks/respan-tracing/src/respan_tracing/instruments.py)
  - enum of built-in auto-instrumentable libraries

Design rule:

- all Python instrumentation packages should eventually terminate into this runtime layer

#### Trace boundaries inside batch handlers (`is_new_trace_root`)

OTel propagates `trace_id` implicitly through the active `Context`. Every span created while another span is active attaches as that span's child and inherits the parent's `trace_id`. This is the right default for request-scoped work, but it is the wrong default at any execution boundary where the inner work is an independent unit.

The canonical case is a batch consumer dispatching per-message work:

```python
@workflow(name="my_consumer_handle_batch")          # 1 span per BATCH
async def _handle_batch(consumer, messages):
    for message in messages:
        # Each per-message Celery task creates its own @workflow.
        # By default it ATTACHES as a child of _handle_batch and shares
        # its trace_id, even though the messages are independent.
        await asyncio.to_thread(per_message_task.run, **message.payload)
```

If `per_message_task` is decorated `@workflow(name="run_X")`, every message in one batch ends up with the same `trace_id`. Downstream readers that do `count(distinct trace_id)` — per-trace UIs, "how many runs did this experiment produce" aggregations — collapse N messages into ~1 trace per batch.

`is_new_trace_root=True` declares the decorated function a fresh trace root: the wrapper attaches an empty `Context()` before creating the span, so OTel sees no active parent and allocates a new `trace_id`. On exit the root context token is detached LIFO-last to restore the caller's original trace.

```python
@workflow(name="run_X", is_new_trace_root=True)    # FRESH trace per call
def per_message_task(...):
    ...
```

When to set the flag:

- the decorated function is invoked from inside another active `@workflow` span (a batch handler, a request handler, an outer workflow)
- the work being decorated is conceptually one independent unit — one message, one row, one job — not a sub-step of the caller
- downstream readers count or aggregate by `trace_id`

When NOT to set it:

- the decorated function is a request handler, a top-level Celery task, or a CLI entry point — there is no active OTel context to detach from. Harmless but signals the wrong intent.
- the work IS conceptually a sub-step (a child task within a workflow) and should be queryable as part of the caller's trace

Design choices:

- **Flag, not a separate API.** The alternative was `client.start_root_workflow(...)` as a sibling primitive. Rejected because every existing `@workflow`-decorated function would then have two entry points to maintain, and callers would have to know which one to use. A flag on the existing decorator keeps the surface flat: one decorator, one mental model, one call site to update when a function moves from "child of caller" to "fresh trace root."
- **Cost.** One extra optional kwarg, one extra context token threaded through `setup_span`/`cleanup_span`, zero runtime overhead when `is_new_trace_root=False` (the default). When `True`, the cost is one `context_api.attach(Context())` and one matching `detach` — a few hundred nanoseconds per call.
- **Surfaced on `@workflow` and `client.start_span` only.** The flag is conceptually meaningful only for entry-point spans. `task` and `tool` kinds are by definition sub-steps within a containing workflow; making one a fresh root would orphan it from its logical parent. The lower-level `create_entity_method` accepts the kwarg so the imperative `client.start_span(..., is_new_trace_root=True)` works for kinds the caller picks at runtime.
- **Use span links, not trace inheritance, for cross-trace correlation.** If a fresh-root span needs to be findable from the caller's trace (e.g. "follow the request → kicked-off background job"), use `links=[SpanLink(...)]` on the decorator. Trace inheritance is the wrong tool for cross-trace correlation: it tells readers "these two spans are part of the SAME trace," which is false when the two pieces of work have independent lifecycles.

Production driver: a 55-run experiment showed up as 3 traces in the UI because every per-row `run_automation_workflow` span inherited the `workflow_execution_writer_handle_batch` (Pulsar consumer) trace.

## JavaScript Runtime

Core objects:

- [RespanTelemetry](../javascript-sdks/respan-tracing/src/main.ts)
  - high-level runtime facade
  - owns explicit async initialization through `initialize()`
  - exposes `withWorkflow`, `withTask`, `withAgent`, `withTool`
  - exposes `addProcessor()`, `getClient()`, and `getSpanBufferManager()`
- [instrumentation/manager.ts](../javascript-sdks/respan-tracing/src/instrumentation/manager.ts)
  - discovers and configures OTEL/Traceloop instrumentations
  - maintains loaded instrumentation instances
  - applies trace-content config to instrumentors
- [MultiProcessorManager](../javascript-sdks/respan-tracing/src/processor/manager.ts)
  - routes spans to named processors
  - supports processor-name routing plus custom filters
  - preserves a default route for backward compatibility
- `utils/tracing.ts`
  - bootstraps provider/export pipeline
  - adds processors to the runtime
  - exposes lower-level span injection helpers

Design rule:

- JS runtime owns initialization order
- JS product packages should avoid directly configuring OTEL internals outside this layer

### 3. Product Facade Layer

This layer is what end users import first. It should be small and opinionated.

Python facade package:

- `python-sdks/respan`

JavaScript facade package:

- `javascript-sdks/respan`

These packages are thin orchestration layers over the tracing runtime.

## Python Facade

Key exports:

- [Respan](../python-sdks/respan/src/respan/_core.py)
- [OTELInstrumentor](../python-sdks/respan/src/respan/_otel_instrumentor.py)
- decorator and client re-exports from `respan_tracing`

`Respan` owns three things:

- creating `RespanTelemetry`
- activating explicitly supplied instrumentation plugins
- exposing convenience helpers such as `propagate_attributes()` and `log_batch_results()`

Important methods:

- `__init__()`
  - wires API key, defaults, and auto-instrumentation policy
- `_activate()`
  - activates a plugin implementing the Respan instrumentation protocol
- `propagate_attributes()`
  - pushes Respan attributes into contextvars so child spans inherit them
- `log_batch_results()`
  - converts delayed OpenAI batch results into trace-linked chat spans

`OTELInstrumentor` is a compatibility wrapper:

- adapts `.instrument()` / `.uninstrument()` style instrumentors into `.activate()` / `.deactivate()`

## JavaScript Facade

Key exports:

- [Respan](../javascript-sdks/respan/src/_core.ts)
- [OTELInstrumentor](../javascript-sdks/respan/src/_otel_instrumentor.ts)
- [OpenInferenceInstrumentor](../javascript-sdks/respan/src/_openinference_instrumentor.ts)

`Respan` in JS owns:

- constructing `RespanTelemetry`
- explicit `initialize()` sequencing
- activating pending plugins after the runtime exists
- forwarding convenience methods like `addProcessor()`, `propagateAttributes()`, and `logBatchResults()`

Important methods:

- `initialize()`
  - must run before plugin activation
- `addProcessor()`
  - forwards routing config into the tracing runtime
- `propagateAttributes()`
  - executes a closure inside an OTEL attribute propagation scope
- `logBatchResults()`
  - injects synthetic chat spans for delayed OpenAI batch responses

### 4. Instrumentation Adapter Layer

This layer translates vendor SDK activity into spans that the runtime can understand.

Python active instrumentations live in:

- `python-sdks/instrumentations/`

JavaScript active instrumentations live in:

- `javascript-sdks/instrumentations/`

There are two valid adapter styles:

1. wrap an upstream OTEL/OpenInference/Traceloop instrumentor
2. native patching that emits spans directly in the Respan model

Representative examples:

- [Python Anthropic instrumentation](../python-sdks/instrumentations/respan-instrumentation-anthropic/src/respan_instrumentation_anthropic/_instrumentation.py)
  - monkey-patches Anthropic clients
  - normalizes messages, tools, tool calls, and token usage
  - emits GenAI semantic-convention attributes
- [JavaScript OpenAI instrumentation](../javascript-sdks/instrumentations/respan-instrumentation-openai/src/index.ts)
  - wraps `@traceloop/instrumentation-openai`
  - points it at the global tracer provider
  - manually patches the OpenAI module

Instrumentation package contract:

- package exposes one clear instrumentor object
- object implements `activate()` and `deactivate()`
- object must emit or route into the active tracing runtime
- object must not depend on `legacy/`

Architectural rule:

- instrumentations are adapters, not foundations
- core packages must not depend back on any concrete instrumentation package

### 5. Operator CLI Layer

This layer exists only in JavaScript:

- `javascript-sdks/respan-cli`

The CLI is not part of the tracing runtime. It is an operator and integration tool.

Core responsibilities:

- authenticate against Respan
- store credentials and config under `~/.respan`
- provide CRUD and summary commands for datasets, logs, traces, prompts, experiments, evaluators, and users
- integrate external tools such as Codex CLI, Claude Code, Gemini CLI, and Opencode

Representative modules:

- [auth.ts](../javascript-sdks/respan-cli/src/lib/auth.ts)
  - resolves auth from flags, env, or stored credentials
  - refreshes JWT tokens when needed
- [config.ts](../javascript-sdks/respan-cli/src/lib/config.ts)
  - persists credentials and defaults in `~/.respan`
- [src/commands/](../javascript-sdks/respan-cli/src/commands)
  - command surface grouped by product area

Boundary rule:

- the CLI may consume SDK contracts or APIs
- the core tracing runtime must not depend on the CLI

## End-To-End Data Flow

The normal flow is:

1. user enters through `respan` or directly through `respan-tracing`
2. runtime initializes OTEL provider, processors, propagation, and exporter
3. decorators or instrumentation adapters create spans
4. `RespanClient` or equivalent helpers mutate the active span when needed
5. exporter transforms spans into Respan OTLP payloads
6. backend receives traces, logs, and derived metrics

Two span creation paths coexist by design:

- decorator path
  - user wraps functions with `workflow` / `task` / `agent` / `tool`
- instrumentation path
  - vendor SDK calls are patched and emitted as spans automatically

Those paths must merge into the same runtime and the same export semantics.

## Dependency Direction

The intended dependency direction is:

- facade -> tracing runtime
- facade -> contract layer
- instrumentation -> tracing runtime
- instrumentation -> contract layer when needed
- CLI -> product API / contracts

Avoid:

- tracing runtime -> concrete instrumentation package
- contract layer -> runtime initialization code
- active packages -> `legacy/`

## Source Of Truth Docs

This file explains runtime architecture and package responsibilities.

Related docs:

- [writing-instrumentations.md](writing-instrumentations.md)
- [cicd.md](cicd.md)
- [publish.md](publish.md)
