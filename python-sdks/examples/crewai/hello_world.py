"""CrewAI Hello World - simple agent with Respan tracing."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

RESPAN_BASE_URL = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
RESPAN_API_KEY = os.getenv("RESPAN_API_KEY")
if not RESPAN_API_KEY:
    raise RuntimeError("Set RESPAN_API_KEY to run this example.")

# Initialize Respan before importing CrewAI so the CrewAI patch is active.
from respan import Respan
from respan_instrumentation_crewai import CrewAIInstrumentor

respan = Respan(
    api_key=RESPAN_API_KEY,
    base_url=RESPAN_BASE_URL,
    instrumentations=[CrewAIInstrumentor()],
)

from crewai import Agent, Crew, LLM, Task

llm = LLM(
    model=os.getenv("RESPAN_MODEL", "gpt-4o-mini"),
    api_key=RESPAN_API_KEY,
    base_url=RESPAN_BASE_URL,
)

agent = Agent(
    role="Poet",
    goal="Write a short haiku about recursion in programming",
    backstory="You are a programmer who writes haikus.",
    llm=llm,
    verbose=False,
)

task = Task(
    description="Write a haiku about recursion in programming.",
    expected_output="A single haiku (3 lines: 5-7-5 syllables).",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], verbose=False)
result = crew.kickoff()
print(result.raw)
