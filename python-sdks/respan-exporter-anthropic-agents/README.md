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

### Constructor Parameters

All configuration can also be passed directly to the constructor (takes priority over environment variables):

```python
exporter = RespanAnthropicAgentsExporter(
    api_key="your_respan_key",       # Overrides RESPAN_API_KEY
    base_url="https://api.respan.ai", # Overrides RESPAN_BASE_URL
    endpoint="https://custom/ingest", # Full endpoint URL (overrides base_url)
    timeout_seconds=15,
    max_retries=3,
    base_delay_seconds=1.0,
    max_delay_seconds=30.0,
)
```

## Examples

Runnable examples with full setup instructions:

- **Python:** [anthropic_agents_python_example](https://github.com/Keywords-AI/keywordsai-example-projects/tree/main/anthropic_agents_python_example)
- **TypeScript:** [anthropic_agents_typescript_example](https://github.com/Keywords-AI/keywordsai-example-projects/tree/main/anthropic_agents_typescript_example)

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
