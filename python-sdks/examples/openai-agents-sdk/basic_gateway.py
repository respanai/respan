"""Basic Gateway — Route OpenAI calls through Respan gateway with tracing."""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import AsyncOpenAI
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, set_default_openai_client

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
    result = await Runner.run(agent, "What are the benefits of using an API gateway?")
    print(result.final_output)
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
