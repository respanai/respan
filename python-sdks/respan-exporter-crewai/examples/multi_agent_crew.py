# multi_agent_crew.py - Multi-agent CrewAI crew with Respan tracing
#
# Demonstrates tracing a crew with multiple collaborating agents.
#
# Prerequisites:
#   pip install crewai openinference-instrumentation-crewai respan-exporter-crewai
#
# Set your API key:
#   export RESPAN_API_KEY="your-respan-api-key"

from crewai import Agent, Crew, Task
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from respan_exporter_crewai import RespanCrewAIInstrumentor

# --- OpenTelemetry + Respan setup ---
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

CrewAIInstrumentor().instrument(tracer_provider=provider)

RespanCrewAIInstrumentor().instrument(
    api_key=None,  # uses RESPAN_API_KEY env var
    environment="development",
    passthrough=True,  # also forward spans to other exporters
)

# --- Define agents ---
planner = Agent(
    role="Content Planner",
    goal="Create an outline for a blog post",
    backstory="You are an experienced content strategist.",
)

writer = Agent(
    role="Content Writer",
    goal="Write a short blog post based on the provided outline",
    backstory="You are a skilled technical writer.",
)

reviewer = Agent(
    role="Editor",
    goal="Review the blog post for clarity and correctness",
    backstory="You are a meticulous editor with an eye for detail.",
)

# --- Define tasks (sequential handoff) ---
plan_task = Task(
    description="Create a three-section outline for a blog post about AI observability.",
    expected_output="A numbered outline with section titles and one-sentence summaries.",
    agent=planner,
)

write_task = Task(
    description="Write a short blog post following the outline from the planner.",
    expected_output="A blog post of roughly 200 words.",
    agent=writer,
)

review_task = Task(
    description="Review the blog post and suggest improvements.",
    expected_output="The revised blog post with inline comments.",
    agent=reviewer,
)

# --- Run the crew ---
crew = Crew(
    agents=[planner, writer, reviewer],
    tasks=[plan_task, write_task, review_task],
    verbose=True,
)

result = crew.kickoff()
print(result)
