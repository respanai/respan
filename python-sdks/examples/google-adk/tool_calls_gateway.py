"""Tool Calls Gateway — Trace an ADK agent with tools through the Respan gateway.

Creates an agent with a get_weather tool. LLM calls are routed through the
Respan AI gateway via LiteLLM while tool execution is captured by the
ADK instrumentor.

Prerequisites:
    pip install respan-instrumentation-google-adk google-adk litellm

Set RESPAN_API_KEY (and optionally RESPAN_BASE_URL) environment variables.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv(override=True)

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from respan import Respan
from respan_instrumentation_google_adk import GoogleAdkInstrumentor

# Point LLM calls at the Respan gateway
RESPAN_BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
os.environ["OPENAI_API_KEY"] = os.getenv("RESPAN_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = RESPAN_BASE_URL

# Enable Respan instrumentation
respan = Respan(
    instrumentations=[GoogleAdkInstrumentor()],
    environment="development",
)

os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"


def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    weather_data = {
        "Tokyo": {"temp_f": 80, "condition": "Partly cloudy"},
        "London": {"temp_f": 59, "condition": "Cloudy"},
        "New York": {"temp_f": 72, "condition": "Sunny"},
    }
    return weather_data.get(city, {"temp_f": 70, "condition": "Unknown"})


agent = Agent(
    name="weather_agent",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    instruction="You are a weather assistant. Use the get_weather tool to answer weather questions.",
    tools=[get_weather],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="tool_calls_gateway")
    session = await runner.session_service.create_session(
        app_name="tool_calls_gateway", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="What's the weather like in Tokyo and London?")],
    )

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
