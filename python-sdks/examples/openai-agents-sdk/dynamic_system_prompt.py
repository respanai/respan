"""Dynamic System Prompt — Agent with context-dependent instructions."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, RunContextWrapper, trace

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])


def dynamic_instructions(ctx: RunContextWrapper[str], agent: Agent) -> str:
    style = ctx.context
    if style == "haiku":
        return "You only respond in haikus."
    elif style == "pirate":
        return "You talk like a pirate. Arrr!"
    return "You are a helpful assistant."


agent = Agent(
    name="StyleAgent",
    instructions=dynamic_instructions,
)


async def main():
    for style in ["haiku", "pirate", "normal"]:
        with trace(f"Dynamic prompt ({style})"):
            result = await Runner.run(agent, "Tell me about the ocean.", context=style)
            print(f"[{style}] {result.final_output}\n")
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
