"""Lifecycle Hooks — Track agent and tool lifecycle events."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, RunHooks, RunContextWrapper, Tool, function_tool, trace
from agents.result import RunResult

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])


class LoggingHooks(RunHooks):
    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        print(f">> Agent started: {agent.name}")

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: str) -> None:
        print(f"<< Agent ended: {agent.name}")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        print(f"  >> Tool started: {tool.name}")

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
        print(f"  << Tool ended: {tool.name} -> {result[:50]}")


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny, 72F."


agent = Agent(
    name="WeatherBot",
    instructions="You help users check the weather. Use the tool to look up weather.",
    tools=[get_weather],
)


async def main():
    with trace("Lifecycle hooks example"):
        result = await Runner.run(
            agent,
            "What's the weather in San Francisco?",
            hooks=LoggingHooks(),
        )
        print(f"\nFinal: {result.final_output}")
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
