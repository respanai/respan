"""Hello World — Minimal agent with Respan tracing."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

agent = Agent(
    name="Assistant",
    instructions="You only respond in haikus.",
)


async def main():
    with trace("Hello world"):
        result = await Runner.run(agent, "Tell me about recursion in programming.")
        print(result.final_output)
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
