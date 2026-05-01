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
- [instruments.py](../python-sdks/respan-tracing/src/respan_tracing/instruments.py)
  - enum of built-in auto-instrumentable libraries

Design rule:

- all Python instrumentation packages should eventually terminate into this runtime layer

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
