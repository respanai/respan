"""
quickstart.py - Basic smolagents instrumentation with Respan.

Prerequisites:
    pip install respan-exporter-smolagents smolagents openinference-instrumentation-smolagents

Environment variables:
    RESPAN_API_KEY          - Your Respan API key
    HF_TOKEN                - Hugging Face token for InferenceClient
"""

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import CodeAgent, InferenceClientModel

from respan_exporter_smolagents import RespanSmolagentsInstrumentor

# 1. Set up OpenTelemetry tracing
provider = TracerProvider()
trace.set_tracer_provider(provider)

# 2. Instrument smolagents with OpenInference
SmolagentsInstrumentor().instrument(tracer_provider=provider)

# 3. Enable Respan interception to forward smolagents spans
RespanSmolagentsInstrumentor().instrument(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    environment="development",
)

# 4. Run a simple agent
model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model)

result = agent.run("What is 2 + 2?")
print("Agent result:", result)
