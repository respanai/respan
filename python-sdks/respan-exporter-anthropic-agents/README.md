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
import os
from claude_agent_sdk import ClaudeAgentOptions
from respan_exporter_anthropic_agents.respan_anthropic_agents_exporter import (
    RespanAnthropicAgentsExporter,
)

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip("/")
anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", f"{respan_base_url}/anthropic")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", respan_api_key)

exporter = RespanAnthropicAgentsExporter(
    api_key=respan_api_key,
    base_url=respan_base_url,
)

async def main() -> None:
    options = exporter.with_options(
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="acceptEdits",
            env={
                "ANTHROPIC_BASE_URL": anthropic_base_url,
                "ANTHROPIC_API_KEY": anthropic_api_key,
                "ANTHROPIC_AUTH_TOKEN": os.getenv("ANTHROPIC_AUTH_TOKEN", anthropic_api_key),
            },
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
| `RESPAN_API_KEY` | Yes | Respan API key used for telemetry export. |
| `RESPAN_BASE_URL` | No | Respan base URL for telemetry export. Defaults to `https://api.respan.ai`. |
| `ANTHROPIC_BASE_URL` | No | Inference/proxy base URL used by the Anthropic SDK. |
| `ANTHROPIC_API_KEY` | Usually | Key used by the Anthropic SDK for inference calls. |
| `ANTHROPIC_AUTH_TOKEN` | Optional | Alternate auth token used by some Anthropic client flows. |

Set both groups together when needed. `RESPAN_*` controls tracing export, while `ANTHROPIC_*` controls where model requests are sent.

```bash
# Tracing export (Respan telemetry)
RESPAN_API_KEY=your_respan_key
RESPAN_BASE_URL=https://api.respan.ai/api

# Inference/proxy routing (Anthropic SDK)
ANTHROPIC_BASE_URL=http://localhost:8000/api
ANTHROPIC_API_KEY=your_inference_key
ANTHROPIC_AUTH_TOKEN=your_inference_key
```

`RESPAN_BASE_URL` controls telemetry export only. The exporter automatically appends `/api/v1/traces/ingest` to build the full ingest endpoint.

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
