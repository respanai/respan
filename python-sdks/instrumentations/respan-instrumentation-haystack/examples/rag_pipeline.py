"""
Haystack RAG pipeline example with Respan tracing and gateway.

Demonstrates a simple retrieval-augmented generation pipeline using
an in-memory document store. All pipeline components are traced.
Routes LLM calls through the Respan gateway.

Prerequisites:
    pip install respan-instrumentation-haystack

Environment variables:
    RESPAN_API_KEY   - Your Respan API key (used for both tracing and gateway)
    RESPAN_BASE_URL  - Respan API endpoint (default: https://api.respan.ai/api)
"""

import os

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from respan import Respan
from respan_instrumentation_haystack import HaystackInstrumentor


def run_rag_pipeline() -> None:
    respan_api_key = os.environ["RESPAN_API_KEY"]
    respan_base_url = os.getenv(
        "RESPAN_BASE_URL",
        "https://api.respan.ai/api",
    ).rstrip("/")

    os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    os.environ["OPENAI_API_KEY"] = respan_api_key
    os.environ["OPENAI_BASE_URL"] = respan_base_url

    respan = Respan(
        api_key=respan_api_key,
        base_url=respan_base_url,
        instrumentations=[HaystackInstrumentor()],
    )

    document_store = InMemoryDocumentStore()
    document_store.write_documents(
        [
            Document(
                content=(
                    "Python was created by Guido van Rossum and first released in 1991."
                )
            ),
            Document(
                content=(
                    "Rust is a systems programming language focused on safety and "
                    "performance."
                )
            ),
            Document(
                content=(
                    "TypeScript is a typed superset of JavaScript developed by "
                    "Microsoft."
                )
            ),
        ]
    )

    template = """
Given the following documents, answer the question.

Documents:
{% for document in documents %}
- {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

    pipeline = Pipeline()
    pipeline.add_component(
        "retriever",
        InMemoryBM25Retriever(document_store=document_store, top_k=2),
    )
    pipeline.add_component(
        "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user(template)])
    )
    pipeline.add_component(
        "generator",
        OpenAIChatGenerator(model="gpt-4o-mini"),
    )
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "generator")

    result = pipeline.run(
        {
            "retriever": {"query": "Who created Python?"},
            "prompt_builder": {"question": "Who created Python?"},
        }
    )
    print(result["generator"]["replies"][0].text)

    respan.flush()


if __name__ == "__main__":
    run_rag_pipeline()
