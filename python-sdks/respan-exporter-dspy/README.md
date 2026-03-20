# Respan DSPy Exporter

Export DSPy traces to Respan via OpenTelemetry span interception.

## Installation

```bash
pip install respan-exporter-dspy
```

## Usage

```python
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from respan_exporter_dspy import RespanDSPyInstrumentor

# Set up OTel provider
provider = TracerProvider()
trace.set_tracer_provider(provider)

# Instrument DSPy with OpenInference
DSPyInstrumentor().instrument(tracer_provider=provider)

# Add Respan interception
RespanDSPyInstrumentor().instrument(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    environment="production",
)
```

## Environment Variables

- `RESPAN_API_KEY` - API key for Respan platform
- `RESPAN_BASE_URL` - API endpoint (default: `https://api.respan.ai/api`)
- `RESPAN_ENVIRONMENT` - Environment name (default: `production`)
- `RESPAN_CUSTOMER_IDENTIFIER` - Optional customer identifier
