import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_ragas import RagasInstrumentor


@pytest.fixture
def runtime():
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    owner = RagasInstrumentor(tracer_provider=provider)
    owner.activate()
    yield provider, memory, owner
    owner.deactivate()
    provider.shutdown()
