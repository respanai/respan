"""
Multi-turn conversation: Trace a multi-turn chat with Respan.

This example sends multiple messages to a single agent within one session,
demonstrating how Respan captures the growing conversation context across turns.

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
respan = Respan(
    instrumentations=[GoogleAdkInstrumentor()],
    environment="development",
)

os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

agent = Agent(
    name="chat_agent",
    model="gemini-2.5-flash",
    instruction="You are a friendly assistant that remembers context from earlier in the conversation.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="multi_turn")
    session = await runner.session_service.create_session(
        app_name="multi_turn", user_id="user-1"
    )

    messages = [
        "Hi! My name is Alex.",
        "What's my name?",
        "Tell me a fun fact about the number 42.",
    ]

    for text in messages:
        message = types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )
        print(f"\nUser: {text}")
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
