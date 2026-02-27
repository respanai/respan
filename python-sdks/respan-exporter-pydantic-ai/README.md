# Respan Exporter for Pydantic AI

This package provides a Respan exporter for the [Pydantic AI](https://ai.pydantic.dev/) framework. 
It seamlessly instruments Pydantic AI's agents using OpenTelemetry underneath so that all traces, spans, and metrics 
are sent to Respan using standard semantic conventions.

## Installation

```bash
pip install respan-exporter-pydantic-ai
```

## Usage

```python
from pydantic_ai.agent import Agent
from respan_tracing import RespanTelemetry
from respan_exporter_pydantic_ai import instrument_pydantic_ai

# 1. Initialize Respan Telemetry 
# (reads from RESPAN_API_KEY environment variable)
telemetry = RespanTelemetry(app_name="my-app", api_key="YOUR_RESPAN_API_KEY")

# 2. Instrument Pydantic AI
instrument_pydantic_ai()

# 3. Create and use your Agent
agent = Agent('openai:gpt-4o')

result = agent.run_sync('What is the capital of France?')
print(result.output)
```

## Instrumenting Specific Agents

If you only want to instrument specific agents instead of globally:

```python
agent = Agent('openai:gpt-4o')
instrument_pydantic_ai(agent=agent)
```
