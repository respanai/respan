# AGENT.md

This file provides repository guidance for coding agents working in this repo.

The source of truth for contributor expectations lives under [`contribution/`](contribution/).
If this file conflicts with those docs, follow the contributor docs.

## Start Here

Read these before making structural changes:

- [`contribution/architecture.md`](contribution/architecture.md)
- [`contribution/writing-instrumentations.md`](contribution/writing-instrumentations.md)
- [`contribution/cicd.md`](contribution/cicd.md)
- [`contribution/publish.md`](contribution/publish.md)

## Repository Shape

This monorepo has one active architecture across Python and JavaScript.

Active package areas:

- `python-sdks/respan-sdk`
- `python-sdks/respan-tracing`
- `python-sdks/respan`
- `python-sdks/instrumentations/*`
- `javascript-sdks/respan-sdk`
- `javascript-sdks/respan-tracing`
- `javascript-sdks/respan`
- `javascript-sdks/respan-cli`
- `javascript-sdks/instrumentations/*`

Anything under `python-sdks/legacy/` or `javascript-sdks/legacy/` is outside the active architecture and outside the main CI/CD and release flow.

## Architecture Rules

The active system has five layers:

1. Contract layer
2. Tracing runtime layer
3. Product facade layer
4. Instrumentation adapter layer
5. Operator CLI layer

Important boundaries:

- `respan-sdk` defines shared contracts, constants, and low-level helpers.
- `respan-sdk` must not own tracing setup or vendor patching.
- `respan-tracing` owns OTEL runtime setup, processors, decorators, export, and preprocessing.
- `respan` is a thin product facade over the runtime.
- Instrumentation packages are adapters into the runtime, not new foundations.
- Core packages must not depend back on concrete instrumentation packages.

## Instrumentation Conventions

For new active instrumentations:

- place them under `python-sdks/instrumentations/respan-instrumentation-<name>` or `javascript-sdks/instrumentations/respan-instrumentation-<name>`
- expose one clear instrumentor entrypoint
- implement `activate()` / `deactivate()` for Respan-facing plugins
- keep translation helpers private
- add focused tests for mapping logic

Preferred module split:

- Python lifecycle entrypoint: `_instrumentation.py`
- JavaScript lifecycle entrypoint: `src/index.ts`
- `_processor.*` for real live-span processors
- `_otel_emitter.*` for synthetic OTEL span builders/injectors
- `_translator.*` for pure schema translation helpers
- `_constants.*` for integration-local raw attrs only

Thin wrappers may keep everything in `_instrumentation.py` or `src/index.ts`, but once a package grows beyond a thin wrapper, split helpers by role. Do not use `instrumentor.py` for new Python packages, and do not name a real `SpanProcessor` module `translator`.

There are only two accepted patterns:

1. thin wrapper around an existing OTEL/OpenInference/Traceloop instrumentor
2. native instrumentation only when custom translation or patching is actually required

When resolving semantic-convention keys:

1. default to the Traceloop / `opentelemetry.semconv_ai` constants already used by the active runtime or instrumentation
2. use OpenInference semconv only for integrations that already speak OpenInference
3. use `respan-sdk` only for Respan-owned keys

Do not duplicate upstream semconv constants inside `respan-sdk`.

In this repo, OpenInference semconv is mainly for OpenInference-specific adapters or integrations that already emit it. It is not the default constant source for every instrumentation.

Constant ownership:

- integration-specific raw attrs stay in that instrumentation's `_constants.py`
- common translated GenAI attrs should come from shared constants, preferably `opentelemetry.semconv_ai`
- shared Respan-owned attrs should come from `respan-sdk`
- if a common translated attr is used across instrumentations and is not available from the supported upstream semconv baseline, define it once in `respan-sdk` rather than keeping it local to one instrumentation

Normalization pipeline:

1. instrumentation reads vendor or upstream OTEL attrs
2. instrumentation writes canonical internal attrs
3. `respan-tracing` derives final export attrs
4. redundant raw or helper attrs are stripped after promotion is complete

Canonical internal attrs include:

- `traceloop.*`
- `llm.*`
- `respan.*`
- `RESPAN_SPAN_TOOLS` for normalized tool definitions
- `RESPAN_SPAN_TOOL_CALLS` for normalized tool invocations

Ownership split:

- instrumentation packages own vendor-specific translation and vendor-specific stripping
- `respan-tracing` owns shared enrichment, final promotion, and shared helper stripping before OTLP serialization

Do not treat a bare `tool_calls` attr as the source of truth, and do not strip canonical helper attrs before `respan-tracing` has promoted them.

## Release And CI Rules

Release-managed packages are validated through:

- `.github/release-packages.json`
- `.release-intents/*.json`

If a PR changes a release-managed package:

1. update the code
2. add one release intent file under `.release-intents/`
3. include every changed release-managed package in that intent
4. do not hand-edit final release versions just to satisfy CI

Current release automation covers:

- Python: `respan-ai`, `respan-sdk`, `respan-tracing`, and all packages under `python-sdks/instrumentations/`
- JavaScript: `@respan/respan`, `@respan/cli`, `@respan/respan-sdk`, `@respan/tracing`, and all packages under `javascript-sdks/instrumentations/`

For new active instrumentations that should publish:

- add the package to `.github/release-packages.json`
- add a release intent covering that package

Useful validation commands:

```bash
python3 scripts/release_inventory.py --validate
python3 scripts/release_intents.py validate
python3 scripts/release_intents.py plan --ecosystem all --changed-from <base> --changed-to <head>
```

## Editing Rules

- Prefer existing shared helpers and constants over new local utilities.
- Use stdlib or OTEL utilities directly when a wrapper adds no value.
- Keep package boundaries consistent with `contribution/architecture.md`.
- Do not add new active packages under `legacy/`.
- Do not add duplicate contributor docs under package subtrees.

## Practical Review Checklist

Before merging a package-level change, check:

- the package sits in the right layer and directory
- dependencies follow the layer boundaries
- constants come from canonical sources
- tests cover the mapping or translation being introduced
- release metadata is complete for any release-managed package change
