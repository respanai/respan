# respan-instrumentation-strands-agents

Respan instrumentation plugin for [Strands Agents](https://strandsagents.com/).

This package consumes Strands Agents' native OpenTelemetry spans and maps them
directly into the Respan span contract used by the OTLP pipeline. It does not
require OpenInference at runtime.

## Install

```bash
pip install respan-ai respan-instrumentation-strands-agents
```

Strands Agents `1.20.0` or newer is required because that release includes the
native `tools_config` telemetry surface used for canonical chat tool schemas.

## Quickstart

```python
import os

from respan import Respan
from respan_instrumentation_strands_agents import StrandsAgentsInstrumentor
from strands import Agent, tool
from strands.models.openai import OpenAIModel

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[StrandsAgentsInstrumentor()],
)

model = OpenAIModel(
    model_id="gpt-4o-mini",
    client_args={"api_key": respan_api_key, "base_url": respan_base_url},
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72F."


agent = Agent(
    name="WeatherAgent",
    model=model,
    tools=[get_weather],
    system_prompt="You are a concise weather assistant.",
)

try:
    result = agent("What is the weather in Seattle?")
    print(result)
finally:
    respan.flush()
    respan.shutdown()
```

## Notes

- Initialize `Respan(...)` before running the agent so Strands uses the active
  Respan OpenTelemetry provider.
- The instrumentor can refresh an already-created Strands tracer singleton when
  possible, which helps when an agent was constructed before activation.
- Tool definitions are enabled by default via Strands' `gen_ai_tool_definitions`
  semantic-convention opt-in. They are moved from the common-only agent span to
  canonical `llm.request.functions` on each model/chat span.
- Agent spans remain common-only by contract; model, provider, and usage fields
  belong to the child chat spans.


## Supported surfaces and privacy

Validated with released Strands Agents 1.20.0 and 1.57.2. The adapter translates
native agent, model, tool, event-loop, structured-output, and multiagent spans.
Current Strands memory telemetry is translated to task spans; older SDKs have no
memory telemetry API. The current OpenAI Responses provider is covered by an
actual controlled SSE fixture. Provider credentials, remote memory services,
experimental bidirectional audio, and live model routing are separate checks.

`TRACELOOP_TRACE_CONTENT=false` or the Respan context content opt-out disables
payload capture. A span keeps its initial opt-out even when content is enabled
later; a later opt-out also removes previously captured payloads. Normal payloads
are bounded and secrets are redacted. Known tool schemas, tool calls, and numeric
vectors remain complete when capture is enabled; use content opt-out or sampling
to control their volume.

OpenAI chat/Responses usage is read from the actual provider usage object,
including cache and reasoning details. Missing counts remain absent. Native
exceptions and statuses are preserved without inventing HTTP status codes or
error outputs. Streaming and cancellation keep the SDK's native async-generator
protocol because this adapter does not replace its iterators.

Activation is shared across compatible instrumentor instances and restores only
hooks, environment values, and native tracer state it still owns. Configure one
active OpenTelemetry provider before activation. See the paired
`python/tracing/strands-agents` example set for fixture-default runs and explicit
trace-export/live options.
