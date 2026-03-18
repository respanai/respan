---
name: openai-agents-sdk-integration
description: "Help users integrate the OpenAI Agents SDK with Respan tracing. Use when a user wants to set up respan-exporter-openai-agents, trace agent runs/tools/handoffs/guardrails, configure the Respan gateway, troubleshoot missing traces, or build multi-agent workflows with observability."
---

# Respan + OpenAI Agents SDK Integration Guide

## Install

```bash
pip install openai-agents respan-exporter-openai-agents
```

## Minimal Working Example

```python
import os
from agents import Agent, Runner, set_trace_processors
from respan_exporter_openai_agents import RespanTraceProcessor

# Register the Respan processor BEFORE creating agents
set_trace_processors([
    RespanTraceProcessor(
        api_key=os.getenv("RESPAN_API_KEY"),
        default_model="gpt-4o",
    )
])

agent = Agent(name="Assistant", instructions="You are helpful.")
result = Runner.run_sync(agent, "Hello!")
print(result.final_output)
```

Required env var:

```bash
export RESPAN_API_KEY="your-respan-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

View traces at https://platform.respan.ai/platform/traces

## Using the Respan Gateway (No OpenAI Key)

Route LLM calls through Respan so the user only needs `RESPAN_API_KEY` — no `OPENAI_API_KEY`:

```python
import os
from openai import AsyncOpenAI
from agents import Agent, Runner, set_default_openai_client, set_trace_processors
from respan_exporter_openai_agents import RespanTraceProcessor

RESPAN_API_KEY = os.getenv("RESPAN_API_KEY")

# 1. Route LLM calls through the gateway
client = AsyncOpenAI(api_key=RESPAN_API_KEY, base_url="https://api.respan.ai/api")
set_default_openai_client(client)

# 2. Export spans to Respan
set_trace_processors([
    RespanTraceProcessor(api_key=RESPAN_API_KEY, default_model="gpt-4o")
])

agent = Agent(name="Assistant", instructions="You are helpful.")
result = Runner.run_sync(agent, "Hello!")
```

## What Gets Traced Automatically

Once `set_trace_processors` is called, **all** of these are captured without extra code:

| SDK Event | Traced As | Auto-captured Data |
|-----------|-----------|-------------------|
| `Runner.run()` | Agent span | Agent name, tools list, handoffs list |
| LLM API call | Response span | Model, input messages, output, token counts, cost |
| `@function_tool` call | Tool span | Tool name, input arguments, output, duration |
| Agent handoff | Handoff span | Source agent, target agent |
| Input/output guardrail | Guardrail span | Guardrail name, whether it triggered |
| `with trace("name")` | Root span | Groups all nested spans under one trace |

## Examples

### Tool Calls

```python
from agents import Agent, Runner, set_trace_processors, function_tool
from respan_exporter_openai_agents import RespanTraceProcessor

set_trace_processors([RespanTraceProcessor()])

@function_tool
def get_weather(city: str) -> str:
    return f"Sunny, 72°F in {city}"

agent = Agent(
    name="Weather Agent",
    instructions="Help users check the weather.",
    tools=[get_weather],
)

result = Runner.run_sync(agent, "What's the weather in San Francisco?")
```

### Agent Handoffs

```python
billing_agent = Agent(name="Billing Agent", instructions="Handle billing questions.")

support_agent = Agent(
    name="Support Agent",
    instructions="Route billing questions to the billing agent.",
    handoffs=[billing_agent],
)

result = Runner.run_sync(support_agent, "I have a billing question")
```

### Agents as Tools (Nested Agent Runs)

```python
from agents import trace

translator = Agent(name="Translator", instructions="Translate to French.")
summarizer = Agent(name="Summarizer", instructions="Summarize concisely.")

orchestrator = Agent(
    name="Orchestrator",
    instructions="Use tools to translate and summarize.",
    tools=[
        translator.as_tool(tool_name="translate_to_french", tool_description="Translate to French"),
        summarizer.as_tool(tool_name="summarize", tool_description="Summarize text"),
    ],
)

with trace("Orchestrator workflow"):
    result = Runner.run_sync(orchestrator, "Translate and summarize: 'Hello world'")
```

### Input Guardrails

```python
from agents import (
    Agent, Runner, input_guardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
)
from pydantic import BaseModel

class SafetyCheck(BaseModel):
    is_unsafe: bool
    reasoning: str

checker = Agent(name="Safety Checker", instructions="Check if harmful.", output_type=SafetyCheck)

@input_guardrail
async def safety_guard(context, agent, input):
    result = await Runner.run(checker, input, context=context.context)
    output = result.final_output_as(SafetyCheck)
    return GuardrailFunctionOutput(output_info=output, tripwire_triggered=output.is_unsafe)

agent = Agent(name="Support", instructions="Help users.", input_guardrails=[safety_guard])

try:
    result = await Runner.run(agent, "user message")
except InputGuardrailTripwireTriggered:
    print("Blocked by guardrail")
```

### Streaming

```python
from openai.types.responses import ResponseTextDeltaEvent

agent = Agent(name="Joker", instructions="You tell jokes.")
result = Runner.run_streamed(agent, input="Tell me 3 jokes.")

async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        print(event.data.delta, end="", flush=True)
```

### Parallel Agent Runs

```python
import asyncio
from agents import trace

with trace("Parallel agents"):
    result_a, result_b = await asyncio.gather(
        Runner.run(agent_a, "Task A"),
        Runner.run(agent_b, "Task B"),
    )
```

### Multi-Turn Conversation

```python
agent = Agent(name="Chat", instructions="You are helpful.", tools=[get_weather])

result = await Runner.run(agent, "Hi!")
conversation = result.to_input_list()

result = await Runner.run(agent, conversation + [{"role": "user", "content": "Weather in Tokyo?"}])
conversation = result.to_input_list()

result = await Runner.run(agent, conversation + [{"role": "user", "content": "Thanks!"}])
```

### Named Traces (Grouping)

```python
from agents.tracing import trace

with trace("My Pipeline"):
    r1 = await Runner.run(agent_a, "step 1")
    r2 = await Runner.run(agent_b, r1.final_output)
```

## Configuration

### RespanTraceProcessor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `RESPAN_API_KEY` env | Respan API key |
| `default_model` | `str` | `None` | Fallback model name for spans that don't carry one (agent, tool, handoff, guardrail). **Always set this.** |
| `endpoint` | `str` | `https://api.respan.ai/api/openai/v1/traces/ingest` | Ingest endpoint |
| `max_queue_size` | `int` | `8192` | Max queued spans before dropping |
| `max_batch_size` | `int` | `128` | Max spans per HTTP batch |
| `schedule_delay` | `float` | `5.0` | Seconds between export cycles |
| `max_retries` | `int` | `3` | HTTP retry attempts |

### LocalSpanCollector (Self-Hosted)

For apps that process spans in-process instead of sending HTTP:

```python
from respan_exporter_openai_agents import LocalSpanCollector
from agents import set_trace_processors

collector = LocalSpanCollector(default_model="gpt-4o")
set_trace_processors([collector])

# After each agent run:
result = await Runner.run(agent, "Hello")
spans = collector.pop_trace(result.trace_id)
# spans = list of dicts, root trace at index 0, child spans after
```

## Troubleshooting

### No traces appearing

1. Ensure `set_trace_processors([...])` is called **before** any `Runner.run()`
2. The batch processor exports every ~5 seconds. If the process exits immediately, spans are lost. Add `time.sleep(5)` before exit or call `processor.force_flush()`
3. Verify `RESPAN_API_KEY` is set and valid

### Spans show "unknown model"

Only Response and Generation spans carry model names from the LLM. All other span types (agent, tool, handoff, guardrail) use `default_model` as fallback. Fix:

```python
RespanTraceProcessor(api_key=..., default_model="gpt-4o")
```

### No cost or token data

Token/cost data comes from Response and Generation spans only. If using the gateway, cost is calculated server-side. If using your own OpenAI key, ensure the model returns usage data.

### Gateway returns 401

- Check `RESPAN_API_KEY` is correct
- `base_url` must end with `/api` (e.g., `https://api.respan.ai/api`)
- Ensure your Respan account has credits or a connected provider key

### Guardrail spans missing

Guardrail spans are emitted when the guardrail function finishes. If the guardrail trips (`InputGuardrailTripwireTriggered`), spans up to the trip point are still exported. Ensure `set_trace_processors` is called before agent creation.

## Internals (For Debugging)

The package is a single file: `src/respan_exporter_openai_agents/respan_openai_agents_exporter.py`

**Data flow:** SDK emits `Trace`/`Span` objects → `convert_to_respan_log()` dispatches on `span_data` type → builds `RespanTextLogParams` → `.model_dump(mode="json")` → HTTP POST or local storage.

**Span type dispatch:** `ResponseSpanData` → response, `FunctionSpanData` → tool, `GenerationSpanData` → generation, `HandoffSpanData` → handoff, `AgentSpanData` → agent, `GuardrailSpanData` → guardrail, `CustomSpanData` → custom. Unknown types return `None`.

**Response conversion:** The SDK uses OpenAI Responses API format internally. The exporter converts to Chat Completions format (`{role, content}` messages) so the Respan UI renders clean System/User/Assistant/Tool messages.

**Token extraction:** Handles both Responses API keys (`input_tokens`/`output_tokens`) and Chat Completions keys (`prompt_tokens`/`completion_tokens`). Uses `is not None` checks to preserve `0` as valid.

**Thread safety:** `RespanTraceProcessor` uses `BatchTraceProcessor` with `queue.Queue`. `LocalSpanCollector` uses `threading.Lock`. Both are safe for concurrent agent runs.

## Links

- Docs: https://docs.respan.ai/integrations/tracing/openai-agents-sdk
- Complex example (12 scenarios): https://docs.respan.ai/integrations/tracing/openai-agents-sdk-complex-example
- Example projects: https://github.com/respanai/respan-example-projects/tree/main/python/tracing/openai-agents-sdk
- PyPI: https://pypi.org/project/respan-exporter-openai-agents/
