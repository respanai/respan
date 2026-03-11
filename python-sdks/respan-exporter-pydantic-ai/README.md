# Respan Exporter for Pydantic AI

**[respan.ai](https://respan.ai)** · **[Documentation](https://docs.respan.ai)** · **[PyPI](https://pypi.org/project/respan-exporter-pydantic-ai/)**

Instrument [Pydantic AI](https://ai.pydantic.dev/) agents with Respan: traces, spans, and metrics are sent to Respan via OpenTelemetry and standard semantic conventions. Requires [respan-tracing](https://pypi.org/project/respan-tracing/) (installed automatically).

---

## Install

```bash
pip install respan-exporter-pydantic-ai
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Respan API key (used when `api_key` is not passed to `RespanTelemetry`) |
| `RESPAN_BASE_URL` | No | Respan API base URL (default: `https://api.respan.ai/api`) |

## Quickstart

```python
import os
from pydantic_ai import Agent
from respan_tracing import RespanTelemetry
from respan_exporter_pydantic_ai.instrument import instrument_pydantic_ai

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

# Use Respan as the LLM gateway (no separate OpenAI key needed)
os.environ["OPENAI_BASE_URL"] = respan_base_url
os.environ["OPENAI_API_KEY"] = respan_api_key

# 1. Initialize Respan telemetry
telemetry = RespanTelemetry(app_name="my-app", api_key=respan_api_key)

# 2. Instrument Pydantic AI (global: all agents)
instrument_pydantic_ai()

# 3. Use your agent
agent = Agent("openai:gpt-4o")
result = agent.run_sync("What is the capital of France?")
print(result.output)
```

To instrument a single agent instead of globally:

```python
agent = Agent("openai:gpt-4o")
instrument_pydantic_ai(agent=agent)
```

## API Reference

### `RespanTelemetry`

Initialize once before calling `instrument_pydantic_ai()`:

| Argument | Description |
|----------|-------------|
| `app_name` | Application name shown in Respan. |
| `api_key` | Optional if `RESPAN_API_KEY` is set. |
| `base_url` | Optional; overrides `RESPAN_BASE_URL`. |
| `is_enabled` | Set to `False` to disable tracing. |
| `is_batching_enabled` | Batch export (default: `True`); set `False` for immediate flush in tests. |

### `instrument_pydantic_ai()`

| Argument | Description |
|----------|-------------|
| `agent` | Optional. If provided, only that agent is instrumented; if `None`, all agents are instrumented globally. |
| `include_content` | Include message content in telemetry (default: `True`). |
| `include_binary_content` | Include binary content in telemetry (default: `True`). |

Traces appear in the [Respan dashboard](https://app.respan.ai). Open a trace to see the workflow → task → LLM span tree.

---

## Dev Guide

### OpenAI instrumentation dependency

This package lists `opentelemetry-instrumentation-openai` as a **mandatory** dependency.
`respan-tracing >=2.9.0` no longer bundles it, but this exporter still relies on
OpenAI instrumentation to extract token usage (`prompt_tokens`, `completion_tokens`)
as first-class metrics. Pydantic AI's own token attributes are currently mapped to
metadata fields only, so removing the OpenAI instrumentor would cause token counts
to disappear from span aggregation.

**If you want to drop the OpenAI instrumentation dependency**, you must first
refactor the token extraction so that Pydantic AI's `gen_ai.usage.*` attributes
are promoted to top-level usage metrics (not just metadata). This is a larger change.

### Setup

```bash
cd python-sdks/respan-exporter-pydantic-ai
poetry install
```

### Tests

```bash
# Unit tests (no network)
poetry run pytest tests/test_instrument.py tests/test_extraction_functions.py -v

# All tests
poetry run pytest tests/ -v

# Integration test (real gateway, requires RESPAN_API_KEY)
IS_REAL_GATEWAY_TESTING_ENABLED=1 RESPAN_API_KEY="your-key" \
  poetry run pytest tests/test_real_gateway_integration.py -v -s
```

### Run script (trace tree)

```bash
RESPAN_API_KEY="your-key" poetry run python scripts/run_real_gateway_test.py
```

---

## Further Reading

- [Pydantic AI example project](https://github.com/Nightingalelyy/respan-example-projects/tree/main/python/tracing/pydantic-ai) — runnable integration examples
- [Respan documentation](https://docs.respan.ai)
- [Pydantic AI documentation](https://ai.pydantic.dev/)
- [respan-tracing on PyPI](https://pypi.org/project/respan-tracing/)
- [OpenTelemetry Semantic Conventions for LLM spans](https://opentelemetry.io/docs/semconv/ai/llm-spans/)
