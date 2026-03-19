"""Parallelization — Run multiple agents concurrently."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

spanish_agent = Agent(name="spanish_agent", instructions="Translate the user's message to Spanish")
french_agent = Agent(name="french_agent", instructions="Translate the user's message to French")
italian_agent = Agent(name="italian_agent", instructions="Translate the user's message to Italian")


async def main():
    msg = "Hello, how are you today?"
    with trace("Parallel translations"):
        results = await asyncio.gather(
            Runner.run(spanish_agent, msg),
            Runner.run(french_agent, msg),
            Runner.run(italian_agent, msg),
        )
        for r in results:
            print(f"{r.last_agent.name}: {r.final_output}")
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
