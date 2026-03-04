"""Simple combined gateway + prompt + tracing example for Respan Haystack integration."""

import os
from haystack import Pipeline
from respan_exporter_haystack.connector import RespanConnector
from respan_exporter_haystack.gateway import RespanGenerator

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

# Create pipeline with gateway, prompt management, and tracing
pipeline = Pipeline()
pipeline.add_component(name="tracer", instance=RespanConnector(name="Full Stack: Gateway + Prompt + Tracing"))
pipeline.add_component(name="llm", instance=RespanGenerator(
    prompt_id="1210b368ce2f4e5599d307bc591d9b7a",
    api_key=os.getenv(key="RESPAN_API_KEY")
))

# Run with prompt variables
result = pipeline.run({
    "llm": {
        "prompt_variables": {
            "user_input": "She sells seashells by the seashore"
        }
    }
})

print("Response received successfully!")
print(f"Model: {result['llm']['meta'][0]['model']}")
print(f"Tokens: {result['llm']['meta'][0]['usage']['total_tokens']}")
print(f"\nTrace URL: {result['tracer']['trace_url']}")
