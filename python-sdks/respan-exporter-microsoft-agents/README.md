# Respan Microsoft Agents Exporter

Export AutoGen and Semantic Kernel traces to Respan via OpenTelemetry span interception.

## Installation

```bash
pip install respan-exporter-microsoft-agents
```

## Usage

### With AutoGen

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from respan_exporter_microsoft_agents import RespanMicrosoftAgentsInstrumentor

provider = TracerProvider()
trace.set_tracer_provider(provider)

# AutoGen runtime natively emits OTel spans
# Just add Respan interception
RespanMicrosoftAgentsInstrumentor().instrument(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    environment="production",
)
```

### With Semantic Kernel

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from respan_exporter_microsoft_agents import RespanMicrosoftAgentsInstrumentor

provider = TracerProvider()
trace.set_tracer_provider(provider)

# Semantic Kernel natively emits OTel spans
RespanMicrosoftAgentsInstrumentor().instrument(
    api_key="your-respan-api-key",
)
```

## Environment Variables

- `RESPAN_API_KEY` - API key for Respan platform
- `RESPAN_BASE_URL` - API endpoint (default: `https://api.respan.ai/api`)
- `RESPAN_ENVIRONMENT` - Environment name (default: `production`)
- `RESPAN_CUSTOMER_IDENTIFIER` - Optional customer identifier
