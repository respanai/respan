# Respan Exporter for Google ADK

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-google-adk/)**

Respan exporter for Google ADK (Agent Development Kit) traces.

## Installation

```bash
pip install respan-exporter-google-adk
```

## Usage

```python
import os
from google.adk.agents import Agent
from google.adk.runners import Runner
from respan_exporter_google_adk import RespanGoogleAdkInstrumentor

# Enable Respan instrumentation — intercepts ADK's OTel spans automatically
RespanGoogleAdkInstrumentor().instrument(api_key="your-respan-api-key")

# Important: set this env var so ADK includes message content in spans
os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

agent = Agent(
    name="example_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

# Run your agent as usual — traces are sent to Respan automatically
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `RESPAN_API_KEY` | Your Respan API key | — |
| `RESPAN_BASE_URL` | Respan API base URL | `https://api.respan.ai/api` |
| `RESPAN_ENVIRONMENT` | Environment tag for traces | `production` |
| `RESPAN_CUSTOMER_IDENTIFIER` | Customer identifier for traces | — |
| `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` | Enable message content capture in ADK spans | `false` |

## How It Works

Google ADK natively emits OpenTelemetry spans with GenAI semantic convention attributes. This exporter intercepts those spans, converts them to Respan format, and sends them to the Respan ingest API.

### ADK Span Types

| ADK Span | Respan Log Type |
|---|---|
| `invocation` | `workflow` |
| `agent_run` | `agent` |
| `call_llm` | `generation` |
| `execute_tool` | `tool` |
