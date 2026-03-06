# Respan Exporter for CrewAI

**[respan.ai](https://respan.ai)** | **[Documentation](https://docs.respan.ai)** | **[PyPI](https://pypi.org/project/respan-exporter-crewai/)**

Send CrewAI workflow, agent, task, and tool spans to Respan for tracing and observability.

---

## Configuration

### 1. Install

```bash
pip install respan-exporter-crewai
```

This pulls in `openinference-instrumentation-crewai` so the instrumentor can capture OpenTelemetry spans from CrewAI.

### 2. Set Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `RESPAN_API_KEY` | Yes (or pass `api_key=` to `instrument()`) | — |
| `RESPAN_BASE_URL` | No | `https://api.respan.ai/api` |
| `RESPAN_ENDPOINT` | No | `{base_url}/v1/traces/ingest` |

---

## Quickstart

### 3. Run Script

```python
import os
from crewai import Agent, Task, Crew
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from respan_exporter_crewai.instrumentor import RespanCrewAIInstrumentor

# Set up tracing so OpenTelemetry captures spans
provider = TracerProvider()
trace.set_tracer_provider(provider)
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

RespanCrewAIInstrumentor().instrument(api_key=os.getenv("RESPAN_API_KEY"))
# With default passthrough=False, CrewAI spans are sent only to Respan (not to ConsoleSpanExporter).
# Use passthrough=True to also forward CrewAI spans to the processor above (e.g. for console debug).
CrewAIInstrumentor().instrument()

agent = Agent(
    role="Example Agent",
    goal="Provide a friendly greeting",
    backstory="You are a helpful assistant",
)
task = Task(
    description="Say hello to the user",
    expected_output="A greeting message",
    agent=agent,
)
crew = Crew(agents=[agent], tasks=[task])
crew.kickoff()
```

### 4. View Dashboard

Open [Respan](https://app.respan.ai) and check your project for the new trace (workflow → agents → tasks → tools).

---

## Further Reading

- **What gets captured:** Trace hierarchy (workflow → agents → tasks → tools), span names/inputs/outputs, token usage when available, and latency per span.
- **Gateway (optional):** Route LLM calls through Respan’s gateway for logging, cost tracking, and fallbacks — set `OPENAI_BASE_URL` and `OPENAI_API_KEY` to your Respan gateway URL and API key; see [Respan docs](https://docs.respan.ai) for gateway setup.
- **Programmatic export:** Use `RespanCrewAIExporter` for custom trace/span export (e.g. from in-memory or custom collectors).
