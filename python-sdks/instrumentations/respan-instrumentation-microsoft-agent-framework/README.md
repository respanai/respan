# respan-instrumentation-microsoft-agent-framework

Respan instrumentation plugin for
[Microsoft Agent Framework](https://learn.microsoft.com/agent-framework/).

Microsoft Agent Framework emits native OpenTelemetry spans for agents, tools,
chat completions, and workflows. This package registers a Respan span processor
that normalizes those native spans into the canonical Respan/OpenTelemetry span
contract before export.

## Install

```bash
pip install respan-instrumentation-microsoft-agent-framework
```

## Quickstart

```python
import os

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from respan import Respan
from respan_instrumentation_microsoft_agent_framework import (
    MicrosoftAgentFrameworkInstrumentor,
)

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    app_name="microsoft-agent-framework-example",
    instrumentations=[MicrosoftAgentFrameworkInstrumentor(capture_content=True)],
)

os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url
os.environ["OPENAI_MODEL_ID"] = os.getenv("RESPAN_MODEL", "gpt-4.1-nano")

client = OpenAIChatClient()

agent = Agent(
    client=client,
    name="trace_assistant",
    instructions="Answer concisely.",
)

result = agent.run("Say hello from Microsoft Agent Framework.")
print(result)

respan.flush()
respan.shutdown()
```

## Notes

- Initialize `Respan(...)` before running agents or workflows so Agent
  Framework uses the active Respan OpenTelemetry provider.
- `capture_content=True` asks Agent Framework to include sensitive prompt,
  completion, and tool payload data in its native spans. Set it to `False` if
  you only want metadata.
- The processor emits canonical fields such as `respan.entity.log_type`,
  `gen_ai.prompt.N.*`, `gen_ai.completion.N.*`, `llm.request.functions`, and
  `traceloop.entity.*`. It strips Agent Framework raw message/tool payload
  fields and off-contract aliases before export.
