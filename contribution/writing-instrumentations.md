# Writing Instrumentations

This guide is for adding a new package under:

- `python-sdks/instrumentations/`
- `javascript-sdks/instrumentations/`

It does not apply to legacy packages under `*/legacy/`.

## Choose The Right Shape

There are only two acceptable patterns:

1. Thin wrapper around an existing OTEL/OpenInference/Traceloop instrumentor
2. Native instrumentation package that translates vendor events into the Respan tracing model

Prefer the thin wrapper when a mature upstream instrumentor already exists. Only build a native integration when you actually need custom event translation or patching.

## Package Placement

New active instrumentations must live here:

- Python: `python-sdks/instrumentations/respan-instrumentation-<name>`
- JavaScript: `javascript-sdks/instrumentations/respan-instrumentation-<name>`

Do not add new instrumentation work under `legacy/`.

## Package Naming

### Python

- Distribution name: `respan-instrumentation-<name>`
- Import package: `respan_instrumentation_<name>`

### JavaScript

- Package name: `@respan/instrumentation-<name>`
- Directory name: `respan-instrumentation-<name>`

Use the same `<name>` across both languages where possible.

## Minimum Package Structure

### Python

```text
python-sdks/instrumentations/respan-instrumentation-<name>/
├── pyproject.toml
├── README.md
├── src/
│   └── respan_instrumentation_<name>/
│       ├── __init__.py
│       ├── _instrumentation.py
│       └── ...
└── tests/
```

### JavaScript

```text
javascript-sdks/instrumentations/respan-instrumentation-<name>/
├── package.json
├── tsconfig.json
├── README.md
├── src/
│   └── index.ts
└── tests/
```

## Required Integration Steps

For every new active instrumentation:

1. Add the package directory
2. Add tests
3. Add the package to the release-managed inventory if it should publish
4. Add it to the JS root workspace if it is a JS package
5. Add examples only if they add real coverage or onboarding value

### Inventory

Release-managed instrumentations must be added to:

- [.github/release-packages.json](../.github/release-packages.json)

If the package is not in release inventory, CI/CD will not treat it as part of the active release surface.

### JavaScript Workspace

Release-managed JS instrumentations must be listed in:

- [package.json](../javascript-sdks/package.json)

Legacy packages must stay out of that workspace list.

## Python Guidance

Python instrumentation packages should generally depend on:

- `respan-tracing`
- `respan-sdk` only when they need Respan-owned shared constants or types
- the vendor SDK or upstream OTEL instrumentor they wrap

Good defaults:

- keep package metadata in `pyproject.toml`
- expose one clear instrumentor entrypoint from `__init__.py`
- keep translation/serialization helpers private
- put unit tests under `tests/`

If the package is intended to be loaded as a plugin, make the plugin entrypoint explicit in `pyproject.toml`.

## Constant Resolution Order

See [`span-contract.md`](./span-contract.md#source-rules) for the
canonical source rules — this section is a how-to overlay, that doc is
the source of truth.

When an instrumentation needs semantic-convention keys or attribute constants, resolve them in this order:

1. Traceloop / GenAI semantic-convention packages already used by that instrumentation
2. OpenInference semantic-convention packages already used by that instrumentation
3. `respan-sdk` constants, but only for Respan-owned keys that do not exist upstream
4. **SDK-specific keys** (e.g. LangChain callback field names, Vercel AI SDK `ai.*`, n8n event keys): keep as local constants inside the instrumentation package that owns the SDK. Do NOT promote them into `respan-sdk` — they are translator-internal, not part of the public span contract.

Rules:

- Do not re-declare a Traceloop GenAI or OpenInference semantic-convention key inside `respan-sdk`.
- Do not create local ad hoc string constants in an instrumentation if an upstream constant already exists.
- Add a new `respan-sdk` constant only when the key is Respan-specific and cannot be sourced from Traceloop or OpenInference.
- Prefer importing upstream constants directly in the instrumentation package that uses them.
- SDK-specific input keys (whatever the instrumented library names its fields) stay co-located with the translator that reads them.

Examples of keys that belong upstream rather than in `respan-sdk`:

- GenAI semantic-convention attributes
- OpenInference semantic-convention attributes
- other vendor-neutral tracing keys already published by an upstream semconv package

Examples of keys that may belong in `respan-sdk`:

- Respan-specific metadata keys
- Respan-specific log type identifiers
- Respan-owned plugin or registry keys shared across packages

Examples of keys that stay inside the instrumentation package:

- LangChain callback handler field names
- Vercel AI SDK `ai.*` raw attribute keys
- n8n / Langflow / vendor-specific event payload keys

## Semantic Span Names

Both SDKs export semantic span names **by default** so trace trees look the
same across languages and frameworks. Users opt out with
`RESPAN_SPAN_NAME_STYLE=legacy` (or `spanNameStyle: "legacy"` in JS), which
preserves original instrumentation span names and the emitted span tree
exactly, apart from stripping internal `respan.internal.*` hint attributes.

The naming layer is an export/display rule only. Do not change log type, span
kind, input/output, usage, cost, or metadata mapping just to make the visible
span name look right.

Canonical semantic span names use lowercase operation prefixes:

```text
<operation>
<operation>.<detail>
```

Use a suffix only when it is part of the shared display contract:

- `llm.<model>` for LLM chat/text/response/generation calls. The suffix must be
  the resolved model name, such as `llm.gpt-4o` or `llm.claude-3-5-sonnet`.
  If the model cannot be resolved, emit bare `llm` — never an SDK operation
  name such as `doGenerate`, `chat`, `completion`, or `generation`.
- `agent.<name>` for agent execution spans.
- `tool.<name>` for tool/function-tool execution spans.
- `handoff.<from>_to_<to>` for agent handoffs.

All other semantic operation prefixes are emitted without a suffix:
`workflow`, `task`, `embedding`, `transcribe`, `speech`, `guardrail`, and
`span` as the last-resort fallback.

Suffixes must be stable, human-readable identifiers (model, agent, tool names,
handoff routes). Never put prompt text, user input, URLs, request IDs, customer
IDs, secrets, timestamps, or other high-cardinality values in `span.name`. The
transformers sanitize whitespace, arrows, and unsupported punctuation in both
SDKs, but instrumentations should still provide concise suffixes up front.

### Where naming happens

- **JavaScript**: `@respan/tracing` renames at export time
  (`processor/spanName.ts`), driven by span kind, log type, entity name, and
  model attributes.
- **Python**: `respan-tracing` renames at OTLP export time
  (`exporters/respan.py`), from the same attributes.

Both exporters resolve the LLM model in the same order: GenAI request model
(`gen_ai.request.model`), then OpenInference aliases (`llm.model_name`), then
generic `model`. Instrumentations must set the model attribute for LLM spans
so the exporter can build `llm.<model>`.

### Internal naming hints

When the desired prefix/detail cannot be derived from normal span kind, entity
name, log type, or model attributes, instrumentations may set internal hints.
Import the keys from the SDK — never hard-code the strings:

```ts
import { RespanSpanAttributes } from "@respan/respan-sdk";

attrs[RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_KIND] = "tool";
attrs[RespanSpanAttributes.RESPAN_INTERNAL_SPAN_NAME_DETAIL] = toolName;
```

```py
from respan_sdk.constants.span_attributes import (
    RESPAN_INTERNAL_SPAN_NAME_DETAIL,
    RESPAN_INTERNAL_SPAN_NAME_KIND,
)
```

These hints (and `RESPAN_INTERNAL_DROP_SPAN` /
`RESPAN_INTERNAL_EXPORT_PARENT`) are exporter-only: both exporters strip them
before export in both styles. They must never appear in exported
customer-visible attributes.

### Structural wrapper spans

Some SDKs emit wrapper spans that carry no user-visible work of their own
(e.g. Vercel AI SDK `ai.generateText` wrapping the detailed
`ai.generateText.doGenerate`). The owning instrumentation may mark these with
`RESPAN_INTERNAL_DROP_SPAN` **at span start**, and must stamp each child with
`RESPAN_INTERNAL_EXPORT_PARENT` (the wrapper's own parent span id, or `""`
when the wrapper is a root span) so the exporter can drop the wrapper and
reparent children per-span, immune to export-batch boundaries. The semantic
style drops marked wrappers; the legacy style exports them untouched.

Do not use drop marking to hide errors, tools, handoffs, or spans with unique
user-visible input/output.

### Rules

- Vendor-specific span names and attribute keys (`ai.*`, `openai.*`, etc.)
  stay inside the instrumentation package that owns the translator. The core
  `@respan/tracing` / `respan-tracing` naming layer keys off shared attributes
  only — never off vendor span names.
- Do not use brand, package, provider, or product prefixes such as
  `respan.*`, `openai.*`, `anthropic.*`, or `vercel.*` as semantic span names.
- Add focused tests for both semantic and legacy behavior when an
  instrumentation sets span-name hints or drop markers, and Python exporter
  tests when changing Python semantic-name inference.

## JavaScript Guidance

JS instrumentation packages should generally:

- compile with `tsc`
- expose one clear entrypoint from `src/index.ts`
- keep package metadata and `repository.directory` accurate
- avoid coupling themselves to `legacy/` packages

If a JS package has a real test suite, wire it through the package `test` script so CI picks it up automatically.

## Testing Expectations

Minimum expectation for a new instrumentation:

- it builds
- it can be packaged
- it has at least one focused unit or smoke test for its core mapping logic

Current CI behavior already gives you:

- build validation
- package smoke validation
- affected-package execution

If your integration has subtle event translation, add direct unit tests for those mappings. Do not rely only on end-to-end examples.

## Release Expectations

If you touch a release-managed instrumentation package in a PR, you must add one release intent file under:

- `.release-intents/`

Use one of:

- `none`
- `new`
- `patch`
- `minor`
- `major`

See [publish.md](publish.md) for the release workflow.

## Anti-Patterns

Do not do these:

- add new active packages under `legacy/`
- bypass `.github/release-packages.json`
- keep duplicate contributor docs in package subtrees
- introduce circular dependencies between core packages and instrumentations
- rely on manual post-merge version editing
- duplicate upstream semantic-convention constants inside `respan-sdk`
