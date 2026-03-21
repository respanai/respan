"""Callbacks — Track agent and tool lifecycle events with ADK callbacks.

Demonstrates before_agent_callback and after_agent_callback to log lifecycle
events, with all spans traced by Respan.

Prerequisites:
    pip install respan-instrumentation-google-adk google-adk

Set the RESPAN_API_KEY and GOOGLE_API_KEY environment variables before running.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv(override=True)

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from respan import Respan
from respan_instrumentation_google_adk import GoogleAdkInstrumentor

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


def before_agent(callback_context) -> None:
    """Called before the agent runs."""
    agent_name = callback_context.agent_name if hasattr(callback_context, "agent_name") else "unknown"
    print(f">> Agent starting: {agent_name}")
    return None


def after_agent(callback_context) -> None:
    """Called after the agent finishes."""
    agent_name = callback_context.agent_name if hasattr(callback_context, "agent_name") else "unknown"
    print(f"<< Agent finished: {agent_name}")
    return None


agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction="You are a weather assistant. Use the get_weather tool to answer weather questions.",
    tools=[get_weather],
    before_agent_callback=before_agent,
    after_agent_callback=after_agent,
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="callbacks")
    session = await runner.session_service.create_session(
        app_name="callbacks", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="What's the weather in Tokyo?")],
    )

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent: {part.text}")

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
