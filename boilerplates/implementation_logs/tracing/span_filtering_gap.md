# Span Filtering Gap: Standalone Auto-Instrumentation

**Status:** Known gap, duck-taped
**Date:** 2026-03-02
**Files:** `python-sdks/respan-tracing/src/respan_tracing/utils/preprocessing/span_processing.py`

## Problem

`is_processable_span()` was designed with only two modes:
1. User-decorated spans (`TRACELOOP_SPAN_KIND`) — always process
2. Child spans inside decorator context (`TRACELOOP_ENTITY_PATH`) — always process
3. Everything else — filter as noise

This dropped standalone auto-instrumented LLM spans (e.g. `openai.chat.completions.create()` without a `@workflow` wrapper) because they have neither attribute. The `gen_ai.prompt.*` attributes were set correctly but the span was silently discarded before reaching the exporter.

## Current Fix (Duck-tape)

Added a third check: if a span has `llm.request.type` (set by all OpenLLMetry LLM instrumentors), allow it through and root-promote it.

This works for LLM instrumentors but **will not cover** future non-LLM instrumentors (vector DB, retrieval, tool-use, etc.) that also lack Traceloop decorator context.

## Proper Fix

Replace attribute-based detection with an **instrumentation scope allowlist**.

Every OTel instrumentor tags its spans with a scope name (e.g. `opentelemetry.instrumentation.openai`). The `ReadableSpan` already carries this as `span.instrumentation_scope.name`. We should:

1. Maintain an allowlist of recognized scope name prefixes (e.g. `opentelemetry.instrumentation.`)
2. In `is_processable_span()`, check `span.instrumentation_scope.name` against the allowlist
3. Any span from a recognized instrumentor passes through — no per-provider attribute checks needed
4. Unknown scopes without decorator context still get filtered as noise

This would handle any instrumentor (current and future) without needing attribute-specific duck-tape.

## Also Affected

`is_root_span_candidate()` has the same gap — standalone non-LLM spans would need root promotion too. The scope allowlist fix would apply to both functions.
