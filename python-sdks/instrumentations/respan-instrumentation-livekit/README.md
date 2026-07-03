# Respan LiveKit Instrumentation

Respan instrumentation plugin for [LiveKit Agents](https://docs.livekit.io/agents/).
It translates LiveKit's native `livekit-agents` LLM spans into canonical Respan
chat spans and emits Respan tool spans for `execute_function_call`.

## Installation

```bash
pip install respan-ai respan-instrumentation-livekit livekit-agents
```

## Usage

```python
import os

from livekit.agents import llm
from respan import Respan, workflow
from respan_instrumentation_livekit import LiveKitInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    app_name="livekit-example",
    instrumentations=[LiveKitInstrumentor()],
)


@workflow(name="livekit_chat.workflow")
async def run_chat(model: llm.LLM) -> str:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Say hello in one sentence.")
    response = await model.chat(chat_ctx=chat_ctx).collect()
    return response.text
```

## Captured Spans

- LiveKit `LLMStream` requests become Respan `chat` spans with request messages,
  completions, tool calls, tool definitions when available, model, provider, and
  token usage.
- LiveKit `execute_function_call` executions become Respan `tool` spans.

Use `Respan(..., customer_identifier=..., thread_identifier=..., metadata=...)`
or `respan.propagate_attributes(...)` to attach Respan attributes to LiveKit
spans.
