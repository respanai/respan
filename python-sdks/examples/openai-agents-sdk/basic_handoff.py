"""Basic Handoff — Agent routes between Spanish and English agents."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish. If the user speaks English, handoff to the English agent.",
)

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English. If the user speaks Spanish, handoff to the Spanish agent.",
)

spanish_agent.handoffs = [english_agent]
english_agent.handoffs = [spanish_agent]

triage_agent = Agent(
    name="Triage Agent",
    instructions="Detect the language and handoff to the appropriate agent.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    with trace("Basic handoff"):
        result = await Runner.run(triage_agent, "Hola, necesito ayuda con mi cuenta.")
        print(f"Handled by: {result.last_agent.name}")
        print(f"Response: {result.final_output}")
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
