"""
Multi-agent: Trace a team of Google ADK agents with Respan.

This example creates a router agent that delegates to specialist sub-agents.
All spans -- agent runs, LLM calls, and tool executions -- are exported to Respan.

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

# Enable Respan instrumentation before creating any agents
respan = Respan(instrumentations=[
    GoogleAdkInstrumentor(
        environment="development",
        customer_identifier="demo-user",
    )
])

os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

# Define specialist sub-agents
math_agent = Agent(
    name="math_agent",
    model="gemini-2.5-flash",
    instruction="You are a math tutor. Solve math problems step by step.",
)

writing_agent = Agent(
    name="writing_agent",
    model="gemini-2.5-flash",
    instruction="You are a writing assistant. Help with grammar, style, and composition.",
)

# Define the router agent that delegates to sub-agents
router_agent = Agent(
    name="router_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a helpful router. Delegate math questions to math_agent "
        "and writing questions to writing_agent. For general questions, answer directly."
    ),
    sub_agents=[math_agent, writing_agent],
)


async def main():
    runner = InMemoryRunner(agent=router_agent, app_name="multi_agent")
    session = await runner.session_service.create_session(
        app_name="multi_agent", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="What is 42 * 58?")],
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
