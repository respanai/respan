# Respan Exporter for Anthropic Agent SDK

**[respan.ai](https://respan.ai)** | **[Documentation](https://respan.ai/docs)**

Exporter for Anthropic Agent SDK telemetry to Respan.

## Installation

```bash
pip install respan-exporter-anthropic-agents
```

## Quickstart

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions
from respan_exporter_anthropic_agents.respan_anthropic_agents_exporter import (
    RespanAnthropicAgentsExporter,
)

exporter = RespanAnthropicAgentsExporter()

async def main() -> None:
    options = exporter.with_options(
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="acceptEdits",
        )
    )

    async for message in exporter.query(
        prompt="Analyze this repository and summarize architecture.",
        options=options,
    ):
        print(message)

asyncio.run(main())
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. |
| `RESPAN_BASE_URL` | No | Base URL for all Respan services. Defaults to `https://api.respan.ai`. |

`RESPAN_BASE_URL` is the single base URL that controls where telemetry is exported. The exporter automatically appends `/api/v1/traces/ingest` to build the full endpoint.

### Tracing vs Inference URLs (Important)

There are two independent URL planes:

- **Tracing export URL** (Respan telemetry ingest): controlled by exporter `base_url` / `RESPAN_BASE_URL`.
- **Inference/proxy URL** (where Claude requests are sent): controlled by Anthropic SDK env/options such as `ANTHROPIC_BASE_URL`.

Using **Respan tracing + Respan gateway** together:

```python
from claude_agent_sdk import ClaudeAgentOptions
from respan_exporter_anthropic_agents import RespanAnthropicAgentsExporter

api_key = "your_respan_key"
respan_base_url = "https://api.respan.ai/api"

exporter = RespanAnthropicAgentsExporter(
    api_key=api_key,
    base_url=respan_base_url,  # tracing export
)

options = ClaudeAgentOptions(
    env={
        "ANTHROPIC_BASE_URL": f"{respan_base_url.rstrip('/')}/anthropic",  # inference proxy
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_AUTH_TOKEN": api_key,
    }
)
```

### Constructor Parameters

All configuration can also be passed directly to the constructor.

Recommended pattern (matches the runnable examples):

```python
exporter = RespanAnthropicAgentsExporter(
    api_key="your_respan_key",        # Optional; falls back to RESPAN_API_KEY
    base_url="https://api.respan.ai", # Optional; falls back to RESPAN_BASE_URL
)
```

Local gateway/proxy override:

```python
exporter = RespanAnthropicAgentsExporter(
    api_key="your_respan_key",
    base_url="http://localhost:8000/api",
)
```

Resolution order:
- `api_key`: constructor `api_key` -> `RESPAN_API_KEY`
- `endpoint`: constructor `endpoint` -> derived from constructor `base_url` -> derived from `RESPAN_BASE_URL`

In normal usage you should set `base_url` (or `RESPAN_BASE_URL`) and let the exporter derive the ingest endpoint automatically.
`endpoint` exists for internal/advanced cases and takes precedence over `base_url` if both are set.

## Examples

Runnable examples with full setup instructions:

- **Python examples root:** [python/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/anthropic-agents-sdk)
- **Python basic scripts:**
  - [hello_world_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/hello_world_test.py)
  - [wrapped_query_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/wrapped_query_test.py)
  - [tool_use_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/tool_use_test.py)
  - [gateway_test.py](https://github.com/respanai/respan-example-projects/blob/main/python/tracing/anthropic-agents-sdk/basic/gateway_test.py)
- **TypeScript examples root:** [typescript/tracing/anthropic-agents-sdk](https://github.com/respanai/respan-example-projects/tree/main/typescript/tracing/anthropic-agents-sdk)

## Dev Guide

### Running Tests

```bash
# Unit tests
python -m unittest tests.test_exporter -v

# Live integration test (opt-in, makes real API calls)
export RESPAN_API_KEY="your_respan_key"
export IS_REAL_GATEWAY_TESTING_ENABLED=1
python -m unittest tests.test_real_gateway_integration -v
```
