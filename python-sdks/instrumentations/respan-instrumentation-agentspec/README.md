# respan-instrumentation-agentspec

Respan instrumentation plugin for AgentSpec (`pyagentspec`).

This package activates the upstream OpenInference AgentSpec span processor
through Respan's OpenInference translator, so AgentSpec spans are exported
through the same Respan OTLP pipeline used by `Respan`.

## Installation

```bash
pip install respan-ai respan-instrumentation-agentspec "pyagentspec[langgraph]"
```

## Usage

```python
from pyagentspec.adapters.langgraph import AgentSpecLoader
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig
from respan import Respan
from respan_instrumentation_agentspec import AgentSpecInstrumentor

respan = Respan(
    app_name="agentspec-haiku-agent",
    instrumentations=[AgentSpecInstrumentor(workflow_name="agentspec_haiku_agent")],
)

try:
    agent = Agent(
        name="haiku_assistant",
        description="A helpful assistant that writes haikus.",
        llm_config=OpenAiConfig(name="openai", model_id="gpt-4.1-nano"),
        system_prompt="You are a helpful assistant. Respond only with a haiku.",
    )

    langgraph_agent = AgentSpecLoader().load_component(agent)
    result = langgraph_agent.invoke(
        input={
            "messages": [{"role": "user", "content": "Write a haiku about tracing."}]
        }
    )

    print(result["messages"][-1].content)
finally:
    respan.shutdown()
    respan.flush()
```


## Compatibility and coverage

Validated with PyAgentSpec **26.3.1** and OpenInference AgentSpec **0.1.14**,
and with PyAgentSpec **26.1.0** and OpenInference **0.1.0**. Python 3.11–3.13
is supported. Install the runtime extra for the adapter you use; the examples
and released SDK tests exercise the LangGraph adapter.

The bridge handles synchronous and asynchronous native span events, LangGraph
agents, tool calls, completed streams, and current FlowBuilder flows with node
and tool spans. It preserves tool call IDs, zero-valued tool results, current
request history, and actual input, output, cache-read, and reasoning usage.
Model and tool failures use OTel error status; missing output or usage is not
invented. FlowBuilder connections should include explicit data edges.

Content tracing settings and `mask_sensitive_information=True` are captured
when each span starts. Suppression, the active provider's sampler, propagated
metadata, shared ownership, and failed activation cleanup are respected.
Instrumentors sharing an active native trace must use the same configuration
and provider. An externally activated AgentSpec trace is left to its owner.

On 26.1.0, native enclosing agent/flow spans are unavailable. An unfinished
async stream can therefore remain open in the bridge until the instrumentor
is deactivated; observed children are finalized when their enclosing span ends.
Stream interruption does not create a synthetic model response or token count.
The current LangGraph runtime closes those children at the agent boundary.

Other runtime adapters, external MCP services, database-backed tools, and live
provider calls are not covered by these deterministic tests. The paired examples
use the released AgentSpec/LangGraph runtime with a controlled model boundary;
optional live examples require their own provider configuration.
