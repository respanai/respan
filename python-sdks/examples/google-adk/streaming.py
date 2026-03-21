"""
Streaming: Trace a streaming Google ADK agent response with Respan.

ADK emits OpenTelemetry spans after the stream completes, so the span
hierarchy is identical to a non-streaming call. The key difference is
longer output content and higher output token counts.

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
    name="storyteller",
    model="gemini-2.5-flash",
    instruction="You are a creative storyteller. Tell vivid, detailed stories.",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.9,
        top_p=0.95,
    ),
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="streaming")
    session = await runner.session_service.create_session(
        app_name="streaming", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="Tell me a short story about a robot learning to paint.")],
    )

    print("Streaming response:")
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="", flush=True)
    print()

    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
