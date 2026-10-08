import os

import pytest

pytestmark = pytest.mark.integration

if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
    pytest.skip(
        "Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run.", allow_module_level=True
    )

respan_api_key = os.getenv("RESPAN_API_KEY")
if not respan_api_key:
    pytest.skip("Set RESPAN_API_KEY to run.", allow_module_level=True)

respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url

import instructor
from openai import OpenAI
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic import BaseModel
from respan_instrumentation_instructor import InstructorInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter


class UserInfo(BaseModel):
    name: str
    age: int


def test_real_instructor_gateway_pipeline_exports_spans():
    RespanTracer.reset_instance()
    span_exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="instructor-integration-test",
        api_key=respan_api_key,
        base_url=respan_base_url,
        is_batching_enabled=False,
        is_auto_instrument=False,
    )
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter)
    )

    instrumentor = InstructorInstrumentor()
    instrumentor.activate()
    try:
        client = instructor.from_openai(OpenAI())
        user_info = client.create(
            response_model=UserInfo,
            messages=[{"role": "user", "content": "Ada Lovelace is 36 years old."}],
            model="gpt-4o-mini",
        )

        telemetry.flush()
        spans = span_exporter.get_finished_spans()

        assert user_info.name
        assert spans
        assert any(
            (span.attributes or {}).get(RESPAN_LOG_TYPE) == "chat" for span in spans
        )
    finally:
        instrumentor.deactivate()
        RespanTracer.reset_instance()
