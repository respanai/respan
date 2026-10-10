import marqo
import pytest
from _native import server
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_marqo import MarqoInstrumentor


@pytest.fixture
def runtime():
    with server() as (url, requests):
        client = marqo.Client(url=url)
        index = client.index("docs")
        requests.clear()
        provider = TracerProvider()
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        owner = MarqoInstrumentor(tracer_provider=provider)
        owner.activate()
        yield client, index, provider, memory, owner, requests
        owner.deactivate()
        provider.shutdown()
