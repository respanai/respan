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


@pytest.mark.integration
def test_crewai_tracing_exporter_basic(monkeypatch):
    """Run a CrewAI agent and send traces to Respan."""
    respan_api_key = os.getenv("RESPAN_API_KEY")
    if not respan_api_key:
        pytest.skip("RESPAN_API_KEY not set")

    def _gateway_base_url() -> str:
        base_url = (
            os.getenv("RESPAN_GATEWAY_BASE_URL")
            or os.getenv("RESPAN_BASE_URL")
            or "https://api.respan.ai"
        )
        return normalize_respan_base_url_for_gateway(base_url)

    tracer_provider = trace.get_tracer_provider()
    if not isinstance(tracer_provider, TracerProvider):
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    RespanCrewAIInstrumentor().instrument(
        api_key=respan_api_key,
        endpoint=os.getenv("RESPAN_ENDPOINT"),
        base_url=os.getenv("RESPAN_BASE_URL"),
        passthrough=False,
    )
    CrewAIInstrumentor().instrument()

    monkeypatch.setenv("OPENAI_BASE_URL", _gateway_base_url())
    monkeypatch.setenv("OPENAI_API_KEY", respan_api_key)

    agent = Agent(
        role="Test Agent",
        goal="Provide a friendly greeting",
        backstory="You are a helpful test assistant",
    )
    task = Task(
        description="Say hello from Respan CrewAI exporter test",
        expected_output="A short greeting message",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    tracer_provider.force_flush()

    assert result is not None

    RespanCrewAIInstrumentor().uninstrument()
    CrewAIInstrumentor().uninstrument()
