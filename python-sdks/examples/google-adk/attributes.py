"""Attributes — Propagate customer info and metadata to Google ADK spans.

Demonstrates customer_identifier, environment, and propagate_attributes()
for per-user scoping of traces.

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
    instrumentations=[
        GoogleAdkInstrumentor(
            environment="development",
            customer_identifier="default-user",
        )
    ],
    metadata={"service": "chat-api", "version": "1.0.0"},
)

os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

agent = Agent(
    name="greeting_agent",
    model="gemini-2.5-flash",
    instruction="You are a friendly assistant. Greet the user and answer their questions.",
)


async def handle_request(user_id: str, text: str):
    """Simulate an API request handler with per-user attributes."""
    with respan.propagate_attributes(
        customer_identifier=user_id,
        session_identifier=f"session_{user_id}",
        metadata={"plan": "pro"},
    ):
        runner = InMemoryRunner(agent=agent, app_name="attributes")
        session = await runner.session_service.create_session(
            app_name="attributes", user_id=user_id
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text=text)],
        )

        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"[{user_id}] {part.text}")


async def main():
    await handle_request("user_alice", "Hello! What can you do?")
    await handle_request("user_bob", "Tell me a joke.")
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
