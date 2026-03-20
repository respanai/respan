"""
Multi-agent: Trace a team of Google ADK agents with Respan.

This example creates a router agent that delegates to specialist sub-agents.
All spans -- agent runs, LLM calls, and tool executions -- are exported to Respan.

Prerequisites:
    pip install respan-exporter-google-adk google-adk

Set the RESPAN_API_KEY and GOOGLE_API_KEY environment variables before running.
"""

import os

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from respan_exporter_google_adk import RespanGoogleAdkInstrumentor

# Enable Respan instrumentation before creating any agents
RespanGoogleAdkInstrumentor().instrument(
    api_key=os.environ.get("RESPAN_API_KEY"),
    environment="development",
    customer_identifier="demo-user",
)

os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

# Define specialist sub-agents
math_agent = Agent(
    name="math_agent",
    model="gemini-2.0-flash",
    instruction="You are a math tutor. Solve math problems step by step.",
)

writing_agent = Agent(
    name="writing_agent",
    model="gemini-2.0-flash",
    instruction="You are a writing assistant. Help with grammar, style, and composition.",
)

# Define the router agent that delegates to sub-agents
router_agent = Agent(
    name="router_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a helpful router. Delegate math questions to math_agent "
        "and writing questions to writing_agent. For general questions, answer directly."
    ),
    sub_agents=[math_agent, writing_agent],
)

# Run the multi-agent system
runner = Runner(agent=router_agent, app_name="multi_agent", session_service=InMemorySessionService())
session = runner.session_service.create_session(app_name="multi_agent", user_id="user-1")

response = runner.run(
    user_id="user-1",
    session_id=session.id,
    new_message=types.Content(parts=[types.Part(text="What is 42 * 58?")]),
)

for event in response:
    if event.content and event.content.parts:
        print(event.content.parts[0].text)
