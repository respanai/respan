"""
Quickstart: Instrument a Google ADK agent with Respan tracing.

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

# Enable Respan instrumentation -- intercepts ADK's OpenTelemetry spans automatically
RespanGoogleAdkInstrumentor().instrument(
    api_key=os.environ.get("RESPAN_API_KEY"),
    environment="development",
)

# Allow ADK to include message content in spans (required for input/output capture)
os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

# Define a simple agent
agent = Agent(
    name="greeting_agent",
    model="gemini-2.0-flash",
    instruction="You are a friendly assistant. Greet the user and answer their questions.",
)

# Run the agent
runner = Runner(agent=agent, app_name="quickstart", session_service=InMemorySessionService())
session = runner.session_service.create_session(app_name="quickstart", user_id="user-1")

response = runner.run(
    user_id="user-1",
    session_id=session.id,
    new_message=types.Content(parts=[types.Part(text="Hello! What can you do?")]),
)

for event in response:
    if event.content and event.content.parts:
        print(event.content.parts[0].text)
