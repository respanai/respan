"""Shared fixtures for integration tests."""

import os
import pytest
from dotenv import load_dotenv

# Load .env from tests directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@pytest.fixture(autouse=True)
def reset_tracer():
    """Reset the RespanTracer singleton between tests."""
    from respan_tracing.core.tracer import RespanTracer
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def respan_api_key():
    key = os.getenv("RESPAN_API_KEY")
    if not key:
        pytest.skip("RESPAN_API_KEY not set")
    return key


@pytest.fixture
def respan_base_url():
    return os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")


def _make_trace(trace_id: str, name: str):
    """Create a concrete TraceImpl for testing."""
    from agents.tracing.traces import TraceImpl
    from agents.tracing.processor_interface import TracingProcessor

    class _NoopProcessor(TracingProcessor):
        def on_trace_start(self, trace): pass
        def on_trace_end(self, trace): pass
        def on_span_start(self, span): pass
        def on_span_end(self, span): pass
        def shutdown(self): pass
        def force_flush(self): pass

    return TraceImpl(
        name=name, trace_id=trace_id,
        group_id=None, metadata=None, processor=_NoopProcessor(),
    )
