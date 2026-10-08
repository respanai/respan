import os

import dspy
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_dspy import DSPyInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer


@pytest.mark.integration
def test_real_gateway_dspy_spans():
    if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
        pytest.skip("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run.")
    if not os.getenv("RESPAN_API_KEY"):
        pytest.skip("Set RESPAN_API_KEY to run.")

    RespanTracer.reset_instance()
    respan_api_key = os.environ["RESPAN_API_KEY"]
    respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
    os.environ["OPENAI_API_KEY"] = respan_api_key
    os.environ["OPENAI_BASE_URL"] = respan_base_url
    os.environ["OPENAI_API_BASE"] = respan_base_url

    span_exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        api_key=respan_api_key,
        base_url=respan_base_url,
        app_name="dspy-real-gateway-test",
        is_batching_enabled=False,
    )
    instrumentor = DSPyInstrumentor()
    instrumentor.activate()
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter=span_exporter)
    )

    try:
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))
        question_answerer = dspy.Predict("question -> answer")
        question_answerer(question='Reply with exactly "dspy_ok".')
        telemetry.flush()

        spans = span_exporter.get_finished_spans()
        assert spans, "DSPy instrumentation did not produce spans."
        assert any(span.attributes.get(RESPAN_LOG_TYPE) == "chat" for span in spans), (
            f"No chat span found. Span names: {[span.name for span in spans]}"
        )
    finally:
        instrumentor.deactivate()
        RespanTracer.reset_instance()
