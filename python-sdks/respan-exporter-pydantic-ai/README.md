# Respan Exporter for Pydantic AI

**[respan.ai](https://respan.ai)** · **[Documentation](https://docs.respan.ai)** · **[PyPI](https://pypi.org/project/respan-exporter-pydantic-ai/)**

Instrument [Pydantic AI](https://ai.pydantic.dev/) agents with Respan: traces, spans, and metrics are sent to Respan via OpenTelemetry and standard semantic conventions. Requires [respan-tracing](https://pypi.org/project/respan-tracing/) (installed automatically).

---

## Configuration

### 1. Install

```bash
pip install respan-exporter-pydantic-ai
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Respan API key (used when `api_key` is not passed to `RespanTelemetry`) |
| `RESPAN_BASE_URL` | No | Respan API base URL (default: `https://api.respan.ai/api`) |

Example: `export RESPAN_API_KEY="your-respan-key"` so you don’t need to pass `api_key` in code.

### RespanTelemetry

Initialize once before calling `instrument_pydantic_ai()`:

- `app_name` — Application name shown in Respan.
- `api_key` — Optional if `RESPAN_API_KEY` is set.
- `base_url` — Optional; overrides `RESPAN_BASE_URL`.
- `is_enabled` — Set to `False` to disable tracing.
- `is_batching_enabled` — Batch export (default: typically `True`); set `False` for immediate flush in tests.

### instrument_pydantic_ai()

| Argument | Description |
|----------|-------------|
| `agent` | Optional. If provided, only that agent is instrumented; if `None`, all agents are instrumented globally. |
| `include_content` | Include message content in telemetry (default: `True`). |
| `include_binary_content` | Include binary content in telemetry (default: `True`). |

**Using Respan as LLM gateway**: Point your LLM client at Respan by setting `OPENAI_BASE_URL` and `OPENAI_API_KEY` from your Respan credentials in code—see Quickstart below.

---

## Quickstart

### 3. Run Script

Use your Respan API key and base URL to configure the LLM client (no separate OpenAI key needed):

```python
import os
from pydantic_ai import Agent
from respan_tracing import RespanTelemetry
from respan_exporter_pydantic_ai import instrument_pydantic_ai

# Use Respan as the LLM gateway (no separate OpenAI key needed)
respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai")
os.environ["OPENAI_BASE_URL"] = f"{respan_base_url}/api"
os.environ["OPENAI_API_KEY"] = respan_api_key

# 1. Initialize Respan (pass api_key or set RESPAN_API_KEY)
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

### 4. View Dashboard

Traces appear in the [Respan dashboard](https://app.respan.ai) (or your configured `RESPAN_BASE_URL`). Open a trace to see the workflow → task → LLM span tree.

---

## Further Reading

- **Respan:** [respan.ai](https://respan.ai), [Documentation](https://docs.respan.ai)
- **Pydantic AI:** [ai.pydantic.dev](https://ai.pydantic.dev/), [Models (OpenAI)](https://ai.pydantic.dev/models/openai/)
- **respan-tracing:** [PyPI](https://pypi.org/project/respan-tracing/), [GitHub](https://github.com/respanai/respan) — decorators (`@workflow`, `@task`), manual spans, and export options
- **OpenTelemetry:** [Semantic Conventions for LLM spans](https://opentelemetry.io/docs/semconv/ai/llm-spans/)

---

## Dev guide

### Setup

From the repo root:

```bash
cd python-sdks/respan-exporter-pydantic-ai
poetry install
# or: pip install -e ../respan-tracing -e .
```

### Unit tests

No network; validates instrumentation wiring:

```bash
poetry run pytest tests/test_instrument.py -v
```

### Integration test (real gateway)

Sends a real LLM call through the Respan gateway and checks that spans (including a trace tree) are captured. Only `RESPAN_API_KEY` is required:

```bash
IS_REAL_GATEWAY_TESTING_ENABLED=1 RESPAN_API_KEY="your-respan-key" \
  poetry run pytest tests/test_real_gateway_integration.py -v -s
```

Optional env: `RESPAN_GATEWAY_BASE_URL` or `RESPAN_BASE_URL` (default `https://api.respan.ai/api`), `RESPAN_GATEWAY_MODEL` (default `openai:gpt-4o-mini`). The test is skipped unless `IS_REAL_GATEWAY_TESTING_ENABLED=1`.

### Run script (trace tree)

Same gateway-only flow; produces a trace tree (workflow → task → LLM spans) on the Respan dashboard:

```bash
RESPAN_API_KEY="your-respan-key" poetry run python scripts/run_real_gateway_test.py
```

### All tests

```bash
poetry run pytest tests/ -v
```

Integration tests auto-skip when the required env vars are not set.
