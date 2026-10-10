import threading
from http.server import ThreadingHTTPServer

import pytest
from native_server import Handler
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_aleph_alpha import AlephAlphaInstrumentor


@pytest.fixture(scope="session")
def host():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def runtime():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AlephAlphaInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    yield provider, exporter, instrumentor
    instrumentor.deactivate()
    provider.shutdown()
