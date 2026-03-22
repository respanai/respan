"""Basic Gateway — Route Google ADK LLM calls through the Respan gateway.

Uses LiteLLM to point an ADK agent at the Respan AI gateway, giving you
automatic logging, fallbacks, retries, and cost tracking.

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

agent = Agent(
    name="assistant",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    instruction="You are a helpful assistant. Be concise.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="basic_gateway")
    session = await runner.session_service.create_session(
        app_name="basic_gateway", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="What are the benefits of using an API gateway?")],
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
