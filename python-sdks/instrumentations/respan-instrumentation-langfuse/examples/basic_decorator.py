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


@observe()
def process_query(query: str):
    """Return a deterministic response with a Langfuse observation."""
    return f"Response to: {query}"


try:
    print(process_query("Hello World"))
    langfuse.flush()
    respan.flush()
finally:
    langfuse.shutdown()
    instrumentor.uninstrument()
    respan.shutdown()
