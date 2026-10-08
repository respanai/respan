"""
CrewAI example with custom tools and Respan tracing.

Shows how tool calls are traced alongside agent and task spans.
Routes LLM calls through the Respan gateway.

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
from crewai.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Stub — replace with a real API call
    return f"Sunny, 22C in {city}"


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {
        "Paris": "2.1 million",
        "Tokyo": "13.9 million",
        "New York": "8.3 million",
    }
    return populations.get(city, "Unknown")


researcher = Agent(
    role="City Researcher",
    goal="Gather weather and population data for a given city",
    backstory="You are a researcher that collects city data using available tools.",
    tools=[get_weather, get_population],
    llm="gpt-4o-mini",
)

writer = Agent(
    role="Travel Writer",
    goal="Write a short travel summary based on research data",
    backstory="You turn raw city data into engaging travel blurbs.",
    llm="gpt-4o-mini",
)

research_task = Task(
    description="Research the weather and population of Paris.",
    expected_output="Weather and population data for Paris.",
    agent=researcher,
)

writing_task = Task(
    description="Write a 2-sentence travel blurb about Paris using the research.",
    expected_output="A short, engaging travel blurb.",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
)

result = crew.kickoff()
print(result.raw)

respan.flush()
