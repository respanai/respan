# Respan Exporter for CrewAI

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-crewai/)**

Export CrewAI traces to Respan via OpenTelemetry span interception.

## Installation

```bash
pip install respan-exporter-crewai
```

## Usage

```python
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from respan_exporter_crewai import RespanCrewAIInstrumentor

# Set up OTel provider
provider = TracerProvider()
trace.set_tracer_provider(provider)

# Instrument CrewAI with OpenInference
CrewAIInstrumentor().instrument(tracer_provider=provider)

# Add Respan interception
RespanCrewAIInstrumentor().instrument(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    environment="production",
)
```

## Environment Variables

- `RESPAN_API_KEY` - API key for Respan platform
- `RESPAN_BASE_URL` - API endpoint (default: `https://api.respan.ai/api`)
- `RESPAN_ENVIRONMENT` - Environment name (default: `production`)
- `RESPAN_CUSTOMER_IDENTIFIER` - Optional customer identifier
