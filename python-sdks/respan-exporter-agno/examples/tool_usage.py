"""
Tool usage: Trace an Agno agent that calls a custom tool.

This example shows how tool invocations appear as separate spans inside
the Respan trace, letting you inspect inputs/outputs for each tool call.

Prerequisites:
    pip install respan-exporter-agno agno openai

Environment variables:
    RESPAN_API_KEY   - Your Respan API key
    OPENAI_API_KEY   - Your OpenAI API key
"""

import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from respan_exporter_agno import RespanAgnoInstrumentor

# Enable Respan instrumentation before creating any agents.
RespanAgnoInstrumentor().instrument(
    api_key=os.getenv("RESPAN_API_KEY"),
    environment="development",
)


# Define a custom tool the agent can call.
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city (stub)."""
    return f"The weather in {city} is 22°C and sunny."


# Create an agent with the tool registered.
agent = Agent(
    name="Weather Agent",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    tools=[get_weather],
    instructions=["Use the get_weather tool when asked about weather."],
)

# The agent will invoke get_weather, and both the LLM call and the tool
# call will be captured as separate spans in the Respan trace.
response = agent.run("What is the weather in San Francisco?")
print(response.content)
