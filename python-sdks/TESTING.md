# Python SDK Testing Conventions

Every exporter package MUST have two test files: unit tests and an integration test.

---

## Test Structure

```
respan-exporter-<name>/
├── tests/
│   ├── test_instrument.py                  # Unit tests (always run)
│   └── test_real_gateway_integration.py    # Integration test (opt-in, live API)
└── pyproject.toml                          # pytest config under [tool.pytest.ini_options]
```

### pytest config goes in `pyproject.toml`, not `pytest.ini`

```toml
[tool.pytest.ini_options]
markers = [
    "integration: live network tests (requires API keys)",
]
```

---

## Unit Tests (`test_instrument.py`)

Unit tests verify instrumentation wiring without making network calls.

**Required test cases:**

| Test | What it verifies |
|------|-----------------|
| `test_instrument_global` | After `instrument_<name>()`, global instrumentation is active |
| `test_instrument_disabled` | When `RespanTelemetry(is_enabled=False)`, instrumentation is skipped |
| `test_instrument_specific_agent` | When an `agent` arg is passed, only that agent is instrumented |

**Required fixture:** Reset singleton state between tests.

```python
import pytest
from respan_tracing.core.tracer import RespanTracer

@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()
```

---

## Integration Tests (`test_real_gateway_integration.py`)

Integration tests verify the full pipeline: real LLM call → instrumentation → spans captured.

### Gate: Opt-in via env vars

Integration tests are **always skipped by default**. They require explicit opt-in:

```python
if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
    self.skipTest("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run.")
```

**Required env vars** (skip gracefully if missing):

| Env var | Purpose |
|---------|---------|
| `IS_REAL_GATEWAY_TESTING_ENABLED=1` | Master switch |
| `RESPAN_API_KEY` | Telemetry export (from Respan dashboard) |
| `OPENAI_API_KEY` | LLM call (if using OpenAI models) |

### Use `InMemorySpanExporter` — NEVER patch HTTP clients

The OTLP exporter's HTTP client can change between OTel versions (urllib → requests → httpx).
Patching a specific HTTP client makes tests fragile.

**Instead, use `InMemorySpanExporter` from `respan_tracing.testing`:**

```python
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from respan_tracing.testing import InMemorySpanExporter

span_exporter = InMemorySpanExporter()

telemetry = RespanTelemetry(
    app_name="integration-test",
    api_key=respan_api_key,
    base_url=gateway_base_url,
    is_enabled=True,
    is_batching_enabled=False,  # Synchronous export for deterministic testing
)

# Add directly to tracer provider — bypasses RespanSpanProcessor filtering
# which drops spans without Traceloop attributes (native OTel spans use gen_ai.*)
telemetry.tracer.tracer_provider.add_span_processor(
    SimpleSpanProcessor(span_exporter)
)
```

**Why `add_span_processor` directly instead of `telemetry.add_processor()`?**

`add_processor()` wraps the exporter in `FilteringSpanProcessor` → `RespanSpanProcessor`,
which filters out spans missing `TRACELOOP_SPAN_KIND` or `TRACELOOP_ENTITY_PATH`.
Pydantic AI, OpenAI Agents, and other frameworks emit standard `gen_ai.*` OTel spans
that don't have Traceloop attributes — they'd be silently dropped.

### Assert on spans, not HTTP status codes

```python
# Run the instrumented code
result = await agent.run('Reply with exactly "gateway_ok".')

# Flush to ensure all spans are exported synchronously
telemetry.flush()

# Assert spans were captured
spans = span_exporter.get_finished_spans()
assert len(spans) > 0, "Instrumentation did not produce any spans."

# Assert expected attributes exist
span_attrs = {k for s in spans for k in (s.attributes or {}).keys()}
assert any("gen_ai" in attr for attr in span_attrs), (
    f"No gen_ai attributes found. Span names: {[s.name for s in spans]}"
)
```

### Cleanup: Always reset singleton

```python
RespanTracer.reset_instance()  # Before test
# ... test body ...
RespanTracer.reset_instance()  # After test (in finally or tearDown)
```

---

## Running Tests

```bash
cd python-sdks/respan-exporter-<name>

# Install package in editable mode
pip install -e .

# Unit tests (always work, no env vars needed)
pytest tests/test_instrument.py -v

# Integration test (requires env vars)
IS_REAL_GATEWAY_TESTING_ENABLED=1 \
RESPAN_API_KEY="your-key" \
OPENAI_API_KEY="your-key" \
pytest tests/test_real_gateway_integration.py -v

# All tests (integration auto-skips without env vars)
pytest tests/ -v
```

### Using backend `.env` for keys

If you have the backend repo checked out with a `.env`:

```bash
set -a && source /path/to/backend/.env && set +a
IS_REAL_GATEWAY_TESTING_ENABLED=1 \
RESPAN_API_KEY="$KEYWORDSAI_API_KEY" \
pytest tests/test_real_gateway_integration.py -v
```

Note: Backend uses `KEYWORDSAI_API_KEY`, not `RESPAN_API_KEY`. Map it explicitly.

---

## Test Utilities (`respan_tracing.testing`)

Centralized test utilities live in `respan-tracing` so all exporter packages can import them
(every exporter already depends on `respan-tracing`).

### Available utilities

| Class | Import | Purpose |
|-------|--------|---------|
| `InMemorySpanExporter` | `from respan_tracing.testing import InMemorySpanExporter` | Captures spans in memory for assertions |

### Adding new test utilities

Add to `python-sdks/respan-tracing/src/respan_tracing/testing/` and re-export
from `__init__.py`. All exporters inherit the utility via their `respan-tracing` dependency.

---

## Checklist Before Submitting an Exporter PR

```
├── tests/test_instrument.py exists?
│   ├── test_instrument_global ✓
│   ├── test_instrument_disabled ✓
│   └── test_instrument_specific_agent (if applicable) ✓
├── tests/test_real_gateway_integration.py exists?
│   ├── Uses InMemorySpanExporter (NOT urllib/requests patching) ✓
│   ├── Gated by IS_REAL_GATEWAY_TESTING_ENABLED ✓
│   ├── Skips gracefully when keys missing ✓
│   └── Resets RespanTracer singleton ✓
├── pytest config in pyproject.toml (not pytest.ini) ✓
├── Unit tests pass: pytest tests/test_instrument.py ✓
└── Integration test passes: IS_REAL_GATEWAY_TESTING_ENABLED=1 pytest tests/ ✓
```
