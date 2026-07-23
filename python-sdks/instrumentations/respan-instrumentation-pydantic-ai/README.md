# respan-instrumentation-pydantic-ai

Respan instrumentation plugin for [PydanticAI](https://ai.pydantic.dev/).

This package enables PydanticAI's native OpenTelemetry emission and maps the
resulting PydanticAI attributes directly into the Respan/Traceloop conventions
used by the OTLP pipeline.

## Install

```bash
pip install respan-instrumentation-pydantic-ai
```

## Quickstart

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from respan import Respan
from respan_instrumentation_pydantic_ai import PydanticAIInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    instrumentations=[PydanticAIInstrumentor()],
)

agent = Agent(
    OpenAIModel("gpt-4o-mini", provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]))
)

result = agent.run_sync("Write a one-line haiku about tracing.")
print(result.output)

respan.flush()
```

## Notes

- By default the instrumentor enables global PydanticAI instrumentation via `Agent.instrument_all(...)`.
- Pass `agent=...` to `PydanticAIInstrumentor(...)` if you only want to instrument one agent instance.
- The plugin uses explicit `InstrumentationSettings(version=5)` to keep emitted spans on the current Pydantic AI v2 GenAI semantic conventions.
- This package does not depend on OpenInference at runtime; it consumes native PydanticAI telemetry directly.
