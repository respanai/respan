# quickstart.py - Basic CrewAI instrumentation with Respan
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

# 1. Set up OpenTelemetry with a TracerProvider and at least one processor
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# 2. Instrument CrewAI with OpenInference so spans flow through OpenTelemetry
CrewAIInstrumentor().instrument(tracer_provider=provider)

# 3. Enable Respan interception to capture CrewAI spans
RespanCrewAIInstrumentor().instrument(
    api_key=None,  # uses RESPAN_API_KEY env var
    environment="development",
)

# 4. Define a simple agent and task
researcher = Agent(
    role="Researcher",
    goal="Find concise answers to questions",
    backstory="You are a helpful research assistant.",
)

task = Task(
    description="What are three benefits of OpenTelemetry tracing?",
    expected_output="A short bullet-point list.",
    agent=researcher,
)

# 5. Run the crew -- spans are automatically exported to Respan
crew = Crew(agents=[researcher], tasks=[task], verbose=True)
result = crew.kickoff()
print(result)
