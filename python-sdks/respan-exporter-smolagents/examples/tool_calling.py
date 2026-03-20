"""
tool_calling.py - smolagents tool usage with Respan tracing.

Demonstrates how custom tools are traced as tool spans in Respan
when using the smolagents exporter.

Prerequisites:
    pip install respan-exporter-smolagents smolagents openinference-instrumentation-smolagents

Environment variables:
    RESPAN_API_KEY          - Your Respan API key
    HF_TOKEN                - Hugging Face token for InferenceClient
"""

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from smolagents import CodeAgent, InferenceClientModel, tool

from respan_exporter_smolagents import RespanSmolagentsInstrumentor

# 1. Set up tracing
provider = TracerProvider()
trace.set_tracer_provider(provider)
SmolagentsInstrumentor().instrument(tracer_provider=provider)
RespanSmolagentsInstrumentor().instrument(
    environment="development",
    customer_identifier="example-user",
)


# 2. Define custom tools
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the result.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together and return the result.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


# 3. Create an agent with the tools and run a query
model = InferenceClientModel()
agent = CodeAgent(tools=[add_numbers, multiply_numbers], model=model)

result = agent.run("Add 12 and 7, then multiply the result by 3.")
print("Agent result:", result)
