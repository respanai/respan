# quickstart.py - Basic DSPy tracing with Respan
#
# This example shows how to instrument a simple DSPy program so that
# traces are automatically exported to Respan.
#
# Prerequisites:
#   pip install respan-exporter-dspy dspy openinference-instrumentation-dspy
#
# Set your API keys before running:
#   export RESPAN_API_KEY="your-respan-api-key"
#   export OPENAI_API_KEY="your-openai-api-key"

import dspy
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from respan_exporter_dspy import RespanDSPyInstrumentor

# 1. Set up the OpenTelemetry tracer provider
provider = TracerProvider()
trace.set_tracer_provider(provider)

# 2. Instrument DSPy with OpenInference (produces OTel spans)
DSPyInstrumentor().instrument(tracer_provider=provider)

# 3. Activate Respan interception to capture DSPy spans
RespanDSPyInstrumentor().instrument(
    api_key=None,  # falls back to RESPAN_API_KEY env var
    environment="development",
)

# 4. Configure the DSPy language model
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 5. Run a simple DSPy prediction -- this will be traced automatically
predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")

print(f"Answer: {result.answer}")
