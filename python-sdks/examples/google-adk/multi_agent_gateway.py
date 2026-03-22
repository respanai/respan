"""Multi-Agent Gateway — Route a team of ADK agents through the Respan gateway.

Creates a router agent that delegates to specialist sub-agents. All LLM calls
are proxied through the Respan AI gateway via LiteLLM.

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
    customer_identifier="demo-user",
)

os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

GATEWAY_MODEL = LiteLlm(model="openai/gpt-4o-mini")

# Define specialist sub-agents
billing_agent = Agent(
    name="billing_agent",
    model=GATEWAY_MODEL,
    instruction="You handle billing and payment questions. Be concise and professional.",
)

technical_agent = Agent(
    name="technical_agent",
    model=GATEWAY_MODEL,
    instruction="You handle technical support questions. Be concise and helpful.",
)

# Define the router agent that delegates to sub-agents
triage_agent = Agent(
    name="triage_agent",
    model=GATEWAY_MODEL,
    instruction=(
        "You are the first point of contact. Route the user to the appropriate agent:\n"
        "- billing_agent: for payment, invoice, or subscription questions\n"
        "- technical_agent: for technical issues, bugs, or how-to questions"
    ),
    sub_agents=[billing_agent, technical_agent],
)


async def main():
    runner = InMemoryRunner(agent=triage_agent, app_name="multi_agent_gateway")
    session = await runner.session_service.create_session(
        app_name="multi_agent_gateway", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="I'm having trouble connecting to the API. Can you help?")],
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
