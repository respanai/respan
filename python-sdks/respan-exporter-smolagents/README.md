# Respan smolagents Exporter

Export smolagents traces to Respan via OpenTelemetry span interception.

## Installation

```bash
pip install respan-exporter-smolagents
```

## Usage

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from respan_exporter_smolagents import RespanSmolagentsInstrumentor

# Set up OTel provider
provider = TracerProvider()
trace.set_tracer_provider(provider)

# Instrument smolagents with OpenInference
SmolagentsInstrumentor().instrument(tracer_provider=provider)

# Add Respan interception
RespanSmolagentsInstrumentor().instrument(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    environment="production",
)
```

## Environment Variables

- `RESPAN_API_KEY` - API key for Respan platform
- `RESPAN_BASE_URL` - API endpoint (default: `https://api.respan.ai/api`)
- `RESPAN_ENVIRONMENT` - Environment name (default: `production`)
- `RESPAN_CUSTOMER_IDENTIFIER` - Optional customer identifier
