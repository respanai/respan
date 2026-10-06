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
def subtask(name: str):
    return f"Completed: {name}"


@observe()
def main_workflow(task: str):
    subtask("step 1")
    subtask("step 2")
    return f"Workflow done: {task}"


try:
    print(main_workflow("Process request"))
    langfuse.flush()
    respan.flush()
finally:
    langfuse.shutdown()
    instrumentor.uninstrument()
    respan.shutdown()
