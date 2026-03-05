"""Basic CrewAI tracing test for Respan exporter."""

import os

import pytest

pytest.importorskip("crewai")
pytest.importorskip("openai")
pytest.importorskip("openinference.instrumentation.crewai")
pytest.importorskip("opentelemetry.sdk")

from crewai import Agent, Task, Crew
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from respan_exporter_crewai.instrumentor import RespanCrewAIInstrumentor
from respan_exporter_crewai.utils import normalize_respan_base_url_for_gateway


def test_crewai_tracing_exporter_basic(monkeypatch):
    """Run an CrewAI agent and send traces to Respan."""
    
    # ... existing setup code ...
    
    monkeypatch.setenv("OPENAI_BASE_URL", _gateway_base_url())
    monkeypatch.setenv("OPENAI_API_KEY", respan_api_key)
    
    # ... rest of test ...
    
    # Cleanup instrumentor state after test
    RespanCrewAIInstrumentor().uninstrument()
    CrewAIInstrumentor().uninstrument()
