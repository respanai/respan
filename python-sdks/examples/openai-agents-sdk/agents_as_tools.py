"""Agents as Tools — Orchestrator calls other agents as tools."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])

spanish_agent = Agent(name="spanish_agent", instructions="Translate the user's message to Spanish")
french_agent = Agent(name="french_agent", instructions="Translate the user's message to French")

orchestrator = Agent(
    name="orchestrator",
    instructions=(
        "You are a translation orchestrator. Use the available tools to translate "
        "the user's message to both Spanish and French. Present both translations."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translates text to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translates text to French",
        ),
    ],
)


async def main():
    with trace("Agents as tools"):
        result = await Runner.run(orchestrator, "Good morning, have a wonderful day!")
        print(result.final_output)
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
