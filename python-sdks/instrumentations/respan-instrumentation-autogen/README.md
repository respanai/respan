# respan-instrumentation-autogen

Respan instrumentation for modern AutoGen AgentChat and the legacy `autogen`
API. The default `AutoGenInstrumentor()` wraps OpenInference's AgentChat
instrumentor. Select `AutoGenInstrumentor(api="legacy")` for legacy
`ConversableAgent`, `AssistantAgent`, `UserProxyAgent`, and `GroupChat` workflows.
Both paths translate OpenInference spans into Respan's canonical attributes.

## Configuration

### 1. Install

```bash
pip install respan-ai respan-instrumentation-autogen
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates gateway requests and trace export. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |
| `RESPAN_MODEL` | No | Defaults to `gpt-4o-mini`. |

## Quickstart

```python
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from respan import Respan
from respan_instrumentation_autogen import AutoGenInstrumentor

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
respan_model = os.getenv("RESPAN_MODEL", "gpt-4o-mini")

respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[AutoGenInstrumentor()],
)

model_client = OpenAIChatCompletionClient(
    model=respan_model,
    api_key=respan_api_key,
    base_url=respan_base_url,
)


async def main() -> None:
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )
    result = await agent.run(task="Write one sentence about recursive functions.")
    print(result.messages[-1].content)
    await model_client.close()
    respan.flush()


asyncio.run(main())
```

## Legacy `autogen` (Agent-E and auto-news)

Select the extra matching the application's installed SDK. The distributions
share the `autogen` import namespace; do not combine the two extras in one
environment. Modern AgentChat dependencies remain installed for compatibility
with existing users, but legacy mode does not activate the AgentChat patches.

```bash
# auto-news's exact pin requires Python 3.11 (pyautogen 0.2.2 requires <3.12).
pip install respan-ai 'respan-instrumentation-autogen[legacy-pyautogen]' \
  'pyautogen==0.2.2' 'openai==1.109.1' respan-instrumentation-openai

# In a separate environment, for Agent-E's autogen 0.7 API:
pip install respan-ai 'respan-instrumentation-autogen[legacy-autogen]' \
  'autogen==0.7.6' 'openai==1.109.1' respan-instrumentation-openai
```

`AutoGenInstrumentor()` continues to target modern AgentChat; select
`api="legacy"` explicitly. Legacy mode records sync and async chat,
reply, and function execution spans. Add the provider's instrumentor to record
actual LLM requests, responses, model names, and usage. A function result is
always tool content, including results containing an `assistant` role.

```python
import os

from autogen import AssistantAgent, UserProxyAgent
from respan import Respan
from respan_instrumentation_autogen import AutoGenInstrumentor
from respan_instrumentation_openai import OpenAIInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    is_auto_instrument=False,
    instrumentations=[AutoGenInstrumentor(api="legacy"), OpenAIInstrumentor()],
)
assistant = AssistantAgent(
    "assistant",
    llm_config={
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"],
        "cache_seed": None,
    },
)
user = UserProxyAgent(
    "user",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config=False,
    max_consecutive_auto_reply=0,
)
user.initiate_chat(assistant, message="Write one sentence about the news.")
respan.flush()
```

The same setup supports `await user.a_initiate_chat(...)`. Respan's runtime
propagates context into AutoGen's executor threads so provider spans stay under
their calling agent. Return values remain the SDK's own values, including
`None` from pyautogen 0.2.2 chats. The adapter also preserves function failure
results and marks their tool spans as errors.

The legacy dependency contract is `pyautogen>=0.2.2,<0.3` or
`autogen>=0.7,<0.8`. Real SDK fixtures cover pyautogen 0.2.2 (auto-news) and
autogen 0.7.6 (Agent-E's API family), with OpenAI 1.109.1 and OpenInference AG2
0.1.6. These are integration compatibility tests, not full application runs.
For example, Agent-E's separate `pydantic==2.6.2` pin must still be reconciled
with the application's Respan SDK dependency before installing the full app.

## Offline compatibility tests

Run each legacy requirements file in its own virtual environment. Use Python
3.11 for the exact pyautogen 0.2.2 fixture:

```bash
pip install -e . -r tests/requirements-legacy-pyautogen.txt
python -m pytest tests -q

# Separate environment:
pip install -e . -r tests/requirements-legacy-autogen.txt
python -m pytest tests -q
```

The fixtures execute the installed SDK's chats, GroupChat dispatch, synchronous
and asynchronous functions, error paths, suppression, and repeated activation.
The model roundtrip uses the actual legacy OpenAI wrapper and OpenAI client
with a deterministic HTTP transport; no API keys or model service are needed.

## Further Reading

See the [Respan example projects](https://github.com/respanai/respan-example-projects)
for runnable scripts.
