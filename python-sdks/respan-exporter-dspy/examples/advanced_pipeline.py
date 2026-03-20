# advanced_pipeline.py - Multi-step DSPy pipeline traced with Respan
#
# This example builds a DSPy module that chains two reasoning steps
# (generate a search query, then answer the question) and shows how
# every step is captured as a separate span in Respan.
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

from respan_exporter_dspy import RespanDSPyInstrumentor

# --- OpenTelemetry + Respan setup ---
provider = TracerProvider()
trace.set_tracer_provider(provider)

DSPyInstrumentor().instrument(tracer_provider=provider)

RespanDSPyInstrumentor().instrument(
    api_key=None,  # falls back to RESPAN_API_KEY env var
    environment="development",
    passthrough=False,  # set True to also forward spans to other exporters
)

# --- DSPy language model ---
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- Multi-step DSPy module ---
class ResearchAnswer(dspy.Module):
    """Two-hop pipeline: rephrase the question, then answer it."""

    def __init__(self):
        super().__init__()
        self.rephrase = dspy.ChainOfThought("question -> search_query")
        self.answer = dspy.ChainOfThought("question, search_query -> answer")

    def forward(self, question: str):
        rephrased = self.rephrase(question=question)
        return self.answer(
            question=question,
            search_query=rephrased.search_query,
        )


# --- Run the pipeline ---
pipeline = ResearchAnswer()
result = pipeline(question="Why do leaves change color in autumn?")

print(f"Search query: {result.search_query}")
print(f"Answer:       {result.answer}")
