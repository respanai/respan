# respan-instrumentation-pytest

An opt-in pytest plugin that records one workflow for each native session and one task across each test's setup, call and teardown. Native application spans remain nested under their test. The plugin supports current pytest 9.1.1 and the declared 7.4.0 floor, async tests with pytest-asyncio, and independent xdist worker trees.

```bash
pip install respan-instrumentation-pytest
pytest --respan-tracing
```

The `pytest11` entry point is explicit. Enable with `RESPAN_PYTEST_ENABLED=true` or `respan_tracing = true` in pytest.ini. Optional `RESPAN_PYTEST_WORKFLOW_NAME` / `respan_workflow_name` names the session. The plugin uses an existing released OTel provider when available; otherwise released RespanTelemetry configures export from `RESPAN_API_KEY` and optional `RESPAN_BASE_URL`.

The actual finalized reports supply pass, skip, xfail, XPASS, strict XPASS failures, phase durations and exit codes. Native collection errors and interruption keep their original outcomes. Failed spans use OTel error status and native exception types; permitted failure messages are redacted. HTTP status, model usage and failure-as-output are never invented. Stored backend status and payload projection are separate acceptance checks.

## Content controls

`--no-respan-capture-content`, `RESPAN_PYTEST_CAPTURE_CONTENT=false`, `TRACELOOP_TRACE_CONTENT=false`, `RESPAN_TRACE_CONTENT=false`, or the canonical content-disabled OTel context omit input/output and failure messages. Initial bounds cannot widen. Ambient and supplied suppression, observed finished ancestors, unknown local parents, and context exit before span end are checked. Parameter IDs are stripped from structural names/paths; outcome and phase summaries remain visible. Fixture return values are never inspected.

Permitted builtin JSON parameters retain full arrays, false/zero, and strings. Credential values are redacted; schema field names remain with sensitive defaults redacted. Unknown custom conversion hooks and iterators are not called. An explicit `max_attribute_chars` requests a JSON truncation summary; the default keeps complete known JSON.

Observer faults preserve native pytest exits and clean up owned contexts. Activation/deactivation is idempotent, retains foreign plugins/processors and does not reset the shared Respan tracer singleton. The companion examples run locally by default and cover the same behaviors using the released dependencies.
