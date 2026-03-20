"""
Quickstart: Instrument an Agno agent with Respan tracing.

Prerequisites:
    pip install respan-exporter-agno agno openai

Environment variables:
    RESPAN_API_KEY   - Your Respan API key
    OPENAI_API_KEY   - Your OpenAI API key
"""

import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from respan_exporter_agno import RespanAgnoInstrumentor

# Instrument Agno to send traces to Respan.
# This patches OpenTelemetry span processors so that every Agno span
# is automatically exported to the Respan tracing endpoint.
RespanAgnoInstrumentor().instrument(
    api_key=os.getenv("RESPAN_API_KEY"),
    environment="development",
)

# Create a simple Agno agent backed by GPT-4o-mini.
agent = Agent(
    name="Quickstart Agent",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
    instructions=["You are a helpful assistant."],
)

# Run the agent -- the resulting trace is sent to Respan automatically.
response = agent.run("What is OpenTelemetry in one sentence?")
print(response.content)
