"""Prompt Gateway — Use Respan prompt management with tracing."""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import AsyncOpenAI
from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, ModelSettings, set_default_openai_client

PROMPT_ID = os.getenv("PROMPT_ID", "your-prompt-id")

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

client = AsyncOpenAI(
    api_key=os.getenv("RESPAN_API_KEY"),
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)
set_default_openai_client(client)

agent = Agent(
    name="Prompt Agent",
    instructions="You are a helpful assistant.",
    model_settings=ModelSettings(
        extra_body={
            "prompt": {
                "prompt_id": PROMPT_ID,
                "schema_version": 2,
                "variables": {"task": "answer user questions concisely"},
                "patch": {"temperature": 0.2},
            }
        }
    ),
)


async def main():
    result = await Runner.run(agent, "What is prompt management and why is it useful?")
    print(result.final_output)
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
