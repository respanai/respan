"""Parallel Agents — Run multiple ADK agents concurrently with asyncio.gather.

Demonstrates running multiple translation agents in parallel, with each
agent's spans independently traced by Respan.

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

respan = Respan(instrumentations=[GoogleAdkInstrumentor(environment="development")])

os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

spanish_agent = Agent(
    name="spanish_translator",
    model="gemini-2.5-flash",
    instruction="Translate the user's message to Spanish. Only output the translation.",
)

french_agent = Agent(
    name="french_translator",
    model="gemini-2.5-flash",
    instruction="Translate the user's message to French. Only output the translation.",
)

italian_agent = Agent(
    name="italian_translator",
    model="gemini-2.5-flash",
    instruction="Translate the user's message to Italian. Only output the translation.",
)


async def run_agent(agent: Agent, text: str, label: str):
    """Run a single agent and collect its output."""
    runner = InMemoryRunner(agent=agent, app_name="parallel_agents")
    session = await runner.session_service.create_session(
        app_name="parallel_agents", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=text)],
    )

    output_parts = []
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    output_parts.append(part.text)

    result = "".join(output_parts)
    print(f"{label}: {result}")
    return result


async def main():
    text = "Hello, how are you today?"
    await asyncio.gather(
        run_agent(spanish_agent, text, "Spanish"),
        run_agent(french_agent, text, "French"),
        run_agent(italian_agent, text, "Italian"),
    )
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
