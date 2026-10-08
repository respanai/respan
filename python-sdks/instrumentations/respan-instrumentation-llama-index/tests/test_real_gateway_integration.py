import os

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from respan_instrumentation_llama_index import LlamaIndexInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter

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

llama_index_core = pytest.importorskip("llama_index.core")
llama_index_openai = pytest.importorskip("llama_index.llms.openai")
Document = llama_index_core.Document
OpenAI = llama_index_openai.OpenAI
Settings = llama_index_core.Settings
SummaryIndex = llama_index_core.SummaryIndex


def test_real_llama_index_gateway_pipeline_exports_spans():
    RespanTracer.reset_instance()
    span_exporter = InMemorySpanExporter()
    telemetry = RespanTelemetry(
        app_name="llama-index-integration-test",
        api_key=respan_api_key,
        base_url=respan_base_url,
        is_batching_enabled=False,
        is_auto_instrument=False,
    )
    telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter=span_exporter)
    )

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.activate()
    try:
        Settings.llm = OpenAI(model="gpt-4o-mini")
        index = SummaryIndex.from_documents(
            [Document(text="Respan traces LlamaIndex queries.")]
        )
        query_engine = index.as_query_engine()
        response = query_engine.query("What does Respan trace?")

        telemetry.flush()
        spans = span_exporter.get_finished_spans()

        assert str(response)
        assert spans
        assert any((span.attributes or {}).get(RESPAN_LOG_TYPE) for span in spans)
    finally:
        instrumentor.deactivate()
        RespanTracer.reset_instance()
