# respan-instrumentation-agentscope

Respan instrumentation plugin for [AgentScope](https://docs.agentscope.io/).

This package patches AgentScope 2.x agent, chat-model, and toolkit execution
surfaces and emits Respan-compatible OpenTelemetry spans directly.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-agentscope
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key for trace export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

```python
import os

from dotenv import load_dotenv
from respan import Respan
from respan_instrumentation_agentscope import AgentScopeInstrumentor

load_dotenv()

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[AgentScopeInstrumentor()],
)

# Build and run AgentScope agents as usual.

respan.flush()
```

## What Is Captured

- Agent `reply()` and `reply_stream()` calls as `agent` spans.
- Chat model calls as `chat` spans with canonical `gen_ai.*`, `llm.*`, and
  `traceloop.*` fields.
- Toolkit `call_tool()` execution as `tool` spans.

The instrumentor does not emit off-contract shortcut attributes such as
`tools`, `tool_calls`, `model`, `prompt_tokens`, or `respan.span.tool_calls`.

## Further Reading

- [AgentScope examples in respan-example-projects](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/agentscope)
- [AgentScope documentation](https://docs.agentscope.io/)
