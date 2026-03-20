"""
Quickstart: Trace LlamaIndex queries with Respan.

Prerequisites:
    pip install respan-exporter-llamaindex llama-index-llms-openai

Set environment variables:
    RESPAN_API_KEY  - Your Respan API key
    OPENAI_API_KEY  - Your OpenAI API key
"""

import llama_index.core.instrumentation as instrument
from llama_index.llms.openai import OpenAI

from respan_exporter_llamaindex import RespanSpanHandler

# 1. Create the Respan span handler
handler = RespanSpanHandler(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    trace_name="quickstart-demo",
    session_identifier="session-001",
)

# 2. Register the handler with the LlamaIndex dispatcher
dispatcher = instrument.get_dispatcher()
dispatcher.add_span_handler(handler)

# 3. Use LlamaIndex as usual -- all spans are automatically traced
llm = OpenAI(model="gpt-4o-mini")
response = llm.complete("Explain what OpenTelemetry is in two sentences.")

print("Response:", response.text)
print("Trace exported to Respan!")
