"""
Basic CrewAI example with Respan tracing and gateway.

Traces agent runs, task executions, and tool calls automatically.
Routes LLM calls through the Respan gateway (no separate OpenAI key needed).

Prerequisites:
    pip install respan-instrumentation-crewai

Environment variables:
    RESPAN_API_KEY   - Your Respan API key (used for both tracing and gateway)
    RESPAN_BASE_URL  - Respan API endpoint (default: https://api.respan.ai)
"""

import os

from dotenv import load_dotenv

load_dotenv()

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

# Point OpenAI traffic through Respan gateway (same base URL)
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

from respan import Respan
from respan_instrumentation_crewai import CrewAIInstrumentor

# Initialize Respan BEFORE importing CrewAI
respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    instrumentations=[CrewAIInstrumentor()],
)

from crewai import Agent, Crew, Task

agent = Agent(
    role="Poet",
    goal="Write a short haiku about recursion in programming",
    backstory="You are a programmer who expresses ideas through haikus.",
    llm="gpt-4o-mini",
)

task = Task(
    description="Write a haiku about recursion in programming.",
    expected_output="A single haiku (3 lines: 5-7-5 syllables).",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
print(result.raw)

respan.flush()
