"""
RAG Pipeline: Trace a retrieval-augmented generation pipeline with Respan.

Prerequisites:
    pip install respan-exporter-llamaindex llama-index-llms-openai

Set environment variables:
    RESPAN_API_KEY  - Your Respan API key
    OPENAI_API_KEY  - Your OpenAI API key
"""

import llama_index.core.instrumentation as instrument
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from respan_exporter_llamaindex import RespanSpanHandler

# 1. Set up Respan tracing
handler = RespanSpanHandler(
    api_key="your-respan-api-key",  # or set RESPAN_API_KEY env var
    trace_name="rag-pipeline",
    environment="development",
    session_identifier="rag-session-001",
)

dispatcher = instrument.get_dispatcher()
dispatcher.add_span_handler(handler)

# 2. Configure the LLM
Settings.llm = OpenAI(model="gpt-4o-mini")

# 3. Build a simple in-memory index from documents
documents = [
    Document(text="Respan provides observability for LLM applications."),
    Document(text="OpenTelemetry is an open-source observability framework."),
    Document(text="RAG combines retrieval with generation for grounded answers."),
]

index = VectorStoreIndex.from_documents(documents)

# 4. Query the index -- retrieval, synthesis, and LLM spans are all traced
query_engine = index.as_query_engine()
response = query_engine.query("What does Respan do?")

print("Answer:", response)
print("RAG trace exported to Respan!")
