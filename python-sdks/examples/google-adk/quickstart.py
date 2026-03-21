"""
Quickstart: Instrument a Google ADK agent with Respan tracing.

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

# Enable Respan instrumentation
respan = Respan(instrumentations=[GoogleAdkInstrumentor(environment="development")])

# Allow ADK to include message content in spans (required for input/output capture)
os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

# Define a simple agent
agent = Agent(
    name="greeting_agent",
    model="gemini-2.5-flash",
    instruction="You are a friendly assistant. Greet the user and answer their questions.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="quickstart")
    session = await runner.session_service.create_session(
        app_name="quickstart", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="Hello! What can you do?")],
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
