import os

import pytest
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from respan_instrumentation_haystack import HaystackInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter


@pytest.fixture(autouse=True)
def reset_tracer():
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.mark.integration
def test_real_gateway_pipeline_emits_haystack_spans():
    if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
        pytest.skip("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run.")

    from haystack.components.generators.chat import OpenAIChatGenerator
    from respan import Respan

    respan_api_key = os.getenv("RESPAN_API_KEY")
    if not respan_api_key:
        pytest.skip("Set RESPAN_API_KEY to run.")

    respan_base_url = os.getenv(
        "RESPAN_BASE_URL",
        "https://api.respan.ai/api",
    ).rstrip("/")
    os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    os.environ["OPENAI_API_KEY"] = respan_api_key
    os.environ["OPENAI_BASE_URL"] = respan_base_url

    span_exporter = InMemorySpanExporter()
    respan = Respan(
        api_key=respan_api_key,
        base_url=respan_base_url,
        app_name="haystack-integration-test",
        instrumentations=[HaystackInstrumentor()],
        is_batching_enabled=False,
    )
    respan.telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter)
    )

    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=[
                ChatMessage.from_user('Reply with exactly "gateway_ok": {{ question }}')
            ],
            required_variables=["question"],
        ),
    )
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
    pipeline.connect("prompt_builder", "llm")

    result = pipeline.run(
        {
            "prompt_builder": {
                "question": "Can you confirm the gateway is reachable?",
            }
        }
    )
    respan.flush()

    spans = span_exporter.get_finished_spans()
    span_attribute_keys = {
        attribute_key for span in spans for attribute_key in (span.attributes or {})
    }

    assert result["llm"]["replies"]
    assert spans
    assert RESPAN_LOG_TYPE in span_attribute_keys
