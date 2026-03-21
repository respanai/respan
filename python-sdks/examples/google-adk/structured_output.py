"""Structured Output — JSON mode with Google ADK's output_schema.

Demonstrates using ADK Agent's output_schema parameter to get structured
JSON responses, with all spans traced by Respan.

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

# Define a JSON schema for structured output
movie_review_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rating": {"type": "integer"},
        "summary": {"type": "string"},
        "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "rating", "summary", "pros", "cons"],
}

agent = Agent(
    name="film_critic",
    model="gemini-2.5-flash",
    instruction="You are a film critic. Rate movies 1-10. Return your review as structured JSON.",
    output_schema=movie_review_schema,
    generate_content_config=types.GenerateContentConfig(
        response_mime_type="application/json",
    ),
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="structured_output")
    session = await runner.session_service.create_session(
        app_name="structured_output", user_id="user-1"
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="Review: The Matrix")],
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
