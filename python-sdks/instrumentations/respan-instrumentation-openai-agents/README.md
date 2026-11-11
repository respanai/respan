# respan-instrumentation-openai-agents

Respan instrumentation plugin for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Captures current `0.20.x` agent, task, turn, tool, handoff, guardrail, Responses, Chat Completions, and streaming spans through the OTEL pipeline.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-openai-agents
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

`OPENAI_API_KEY` is required when your application calls OpenAI directly. A compatible OpenAI gateway can instead be configured through the Agents SDK's normal custom-client APIs. Hosted Responses tools such as Web Search, File Search, and Computer Use require a Responses-capable OpenAI credential and endpoint; this instrumentation does not convert those tools to Chat Completions.

## Quickstart

### 3. Run Script

```python
import asyncio
import os
from agents import Agent, Runner
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[OpenAIAgentsInstrumentor()],
)

agent = Agent(name="Assistant", instructions="You are a helpful assistant.")


async def main():
    try:
        result = await Runner.run(agent, "Hello!")
        print(result.final_output)
    finally:
        respan.flush()


asyncio.run(main())
```

Instrumentation activation is process-wide and reference-counted. Repeated `Respan` instances share one Agents tracing processor, and the SDK's previous processors are restored after the final deactivation. Auto-emitted spans use canonical JSON attributes, never `traceloop.span.kind` or the deprecated top-level tool aliases.

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Sessions

When you set no session id, the trace's `group_id` (`trace(..., group_id=...)` or `RunConfig(group_id=...)`) is sent as the run's Respan session id
(`respan.sessions.session_identifier`) on every span of the run (workflow, agent, LLM, tool), so every turn of one conversation lands in
one session with no Respan-specific code.

A session, thread or trace group you set yourself always wins:

```python
from respan import propagate_attributes

with propagate_attributes(session_identifier="my-session"):
    ...  # runs here keep "my-session"
```

An empty id or one longer than 255 characters is not sent; Respan then detects the
session on the server. Unchanged: the trace group (`respan.trace.trace_group_identifier`) is still sent on the workflow span.

## Further Reading

See the [examples/openai-agents-sdk/](https://github.com/RespanAI/respan/tree/main/examples/openai-agents-sdk) directory for runnable examples including tool use, handoffs, multi-agent workflows, guardrails, and streaming.
