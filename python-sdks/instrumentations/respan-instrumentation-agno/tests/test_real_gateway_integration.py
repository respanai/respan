import os

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_agno import AgnoInstrumentor
from respan_sdk.constants.llm_logging import LOG_TYPE_AGENT, LOG_TYPE_CHAT
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter

pytestmark = pytest.mark.integration
agno_agent_module = pytest.importorskip(modname="agno.agent")
agno_openai_module = pytest.importorskip(modname="agno.models.openai")
Agent = agno_agent_module.Agent
OpenAIChat = agno_openai_module.OpenAIChat


def test_real_agno_gateway_run_captures_spans(monkeypatch):
    if os.getenv(key="IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
        pytest.skip("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run.")

    respan_api_key = os.getenv(key="RESPAN_API_KEY")
    if not respan_api_key:
        pytest.skip("Set RESPAN_API_KEY to run.")

    RespanTracer.reset_instance()
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    monkeypatch.setenv(name="OPENAI_API_KEY", value=respan_api_key)
    monkeypatch.setenv(
        name="OPENAI_BASE_URL",
        value=os.getenv(
            key="RESPAN_BASE_URL",
            default="https://api.respan.ai/api",
        ),
    )

    span_exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="agno-real-gateway-test",
        api_key=None,
        is_auto_instrument=False,
        is_batching_enabled=False,
    )
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter=span_exporter),
    )

    instrumentor = AgnoInstrumentor()
    instrumentor.activate()

    try:
        agent = Agent(
            name="Gateway Test Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
        )
        result = agent.run(input='Reply with exactly "agno_gateway_ok".')
        telemetry.flush()
    finally:
        instrumentor.deactivate()
        RespanTracer.reset_instance()

    spans = span_exporter.get_finished_spans()
    assert result.content
    assert len(spans) >= 2

    span_attributes = [span.attributes or {} for span in spans]
    assert any(
        attributes.get(RESPAN_LOG_TYPE) == LOG_TYPE_AGENT
        for attributes in span_attributes
    )
    assert any(
        attributes.get(RESPAN_LOG_TYPE) == LOG_TYPE_CHAT
        for attributes in span_attributes
    )
    assert any(
        attributes.get(SpanAttributes.LLM_REQUEST_MODEL)
        for attributes in span_attributes
    )
