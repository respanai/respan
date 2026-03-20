# Respan Exporter for CrewAI

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-crewai/)**

Export CrewAI traces to Respan via OpenTelemetry span interception. This exporter hooks into the OpenTelemetry span pipeline to capture CrewAI spans (crews, agents, tasks, tool calls, LLM generations) and send them to the Respan tracing endpoint.

## Features

- Automatic interception of CrewAI spans from OpenTelemetry processors
- Span deduplication to avoid duplicate exports
- Passthrough mode to forward spans to other exporters simultaneously
- Maps CrewAI span kinds (CHAIN, AGENT, TOOL, LLM) to Respan log types
- Extracts token usage, model info, input/output, and metadata from OpenInference attributes

## Installation

```bash
pip install respan-exporter-crewai
```

### Prerequisites

CrewAI must be instrumented with [OpenInference](https://github.com/Arize-ai/openinference) so that spans flow through OpenTelemetry:

```bash
pip install crewai openinference-instrumentation-crewai
```

## Quick Start

```python
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from respan_exporter_crewai import RespanCrewAIInstrumentor

# 1. Set up an OpenTelemetry TracerProvider with at least one processor
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# 2. Instrument CrewAI with OpenInference
CrewAIInstrumentor().instrument(tracer_provider=provider)

# 3. Add Respan interception (intercepts CrewAI spans from the processor)
RespanCrewAIInstrumentor().instrument(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    environment="production",
)

# 4. Run your CrewAI crew as usual -- spans are exported automatically
```

### Disabling the Exporter

```python
instrumentor = RespanCrewAIInstrumentor()
instrumentor.instrument(api_key="your-key")

# Later, to disable:
instrumentor.uninstrument()
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `RESPAN_API_KEY` | API key for Respan platform | (required) |
| `RESPAN_BASE_URL` | API endpoint | `https://api.respan.ai/api` |
| `RESPAN_ENVIRONMENT` | Environment name | `production` |
| `RESPAN_CUSTOMER_IDENTIFIER` | Optional customer/user identifier | |

## Requirements

- Python >= 3.12, < 3.14
- `crewai >= 0.80.0`
- `openinference-instrumentation-crewai >= 0.1.0`
- `opentelemetry-sdk >= 1.20.0`

## Support

- **Documentation:** https://docs.respan.ai/
- **Dashboard:** https://platform.respan.ai/
- **Issues:** [GitHub Issues](https://github.com/respanai/respan/issues)
