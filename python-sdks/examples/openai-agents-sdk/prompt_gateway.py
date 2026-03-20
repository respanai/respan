"""Prompt Gateway — Use Respan prompt management with the Agents SDK.

NOTE: The Agents SDK uses OpenAI's Responses API internally, which does
not support Respan's prompt management feature (chat completions only).
For prompt management, use the OpenAI SDK directly — see examples/openai-sdk/prompt.py.
"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import AsyncOpenAI
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, set_default_openai_client, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

client = AsyncOpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)
set_default_openai_client(client)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Be concise.",
)


async def main():
    with trace("Prompt gateway"):
        result = await Runner.run(agent, "What is prompt management and why is it useful?")
        print(result.final_output)
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
