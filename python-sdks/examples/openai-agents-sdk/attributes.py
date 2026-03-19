"""Attributes — Propagate customer info and metadata to all spans."""

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, function_tool, trace

respan = Respan(
    instrumentations=[OpenAIAgentsInstrumentor()],
    # Default metadata applied to ALL spans
    metadata={"service": "chat-api", "version": "1.0.0"},
)


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny, 72F."


agent = Agent(
    name="WeatherBot",
    instructions="You help users check the weather.",
    tools=[get_weather],
)


async def handle_request(user_id: str, message: str):
    """Simulate an API request handler with per-user attributes."""
    with trace("User request"):
        # All spans within this block get customer_identifier + metadata
        with respan.propagate_attributes(
            customer_identifier=user_id,
            thread_identifier="conv_abc_123",
            metadata={"plan": "pro"},  # merged with default metadata
        ):
            result = await Runner.run(agent, message)
            print(f"[{user_id}] {result.final_output}")


async def main():
    # Simulate two different users making requests
    await handle_request("user_alice", "What's the weather in Tokyo?")
    await handle_request("user_bob", "What's the weather in London?")
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
