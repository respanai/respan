"""Export deterministic Langfuse observations through an initialized Respan runtime."""

import os

from langfuse import Langfuse, observe
from respan import Respan
from respan_instrumentation_langfuse import LangfuseInstrumentor

respan = Respan(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
    instrumentations=[],
)
instrumentor = LangfuseInstrumentor()
instrumentor.instrument()
langfuse = Langfuse(public_key="pk-lf-local-example", secret_key="sk-lf-local-example")


@observe(as_type="generation")
def generate_response(prompt: str):
    """Create a generation observation without calling a model provider."""
    result = f"Generated: {prompt}"
    langfuse.update_current_generation(
        model="example-model", input=prompt, output=result
    )
    return result


try:
    print(generate_response("Write a poem"))
    langfuse.flush()
    respan.flush()
finally:
    langfuse.shutdown()
    instrumentor.uninstrument()
    respan.shutdown()
