import os, json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")
os.environ["OPENAI_BASE_URL"] = respan_base_url
os.environ["OPENAI_API_KEY"] = respan_api_key

from pydantic_ai import Agent
from respan_tracing import RespanTelemetry, Instruments
from respan_exporter_pydantic_ai import instrument_pydantic_ai
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

class RawDump(SpanExporter):
    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            print(f"\n{'='*60}")
            print(f"SPAN: {span.name}")
            for k, v in sorted(attrs.items()):
                val = str(v)[:300]
                print(f"  {k} = {val}")
        return SpanExportResult.SUCCESS
    def shutdown(self): pass

telemetry = RespanTelemetry(
    app_name="pydantic-debug",
    api_key=respan_api_key,
    base_url=respan_base_url,
    block_instruments={Instruments.REQUESTS, Instruments.URLLIB3, Instruments.HTTPX},
)
instrument_pydantic_ai()

from respan_tracing.core.tracer import RespanTracer
RespanTracer().tracer_provider.add_span_processor(SimpleSpanProcessor(RawDump()))

agent = Agent("openai:gpt-4o", system_prompt="Be brief.")
result = agent.run_sync("What is 2+2?")
print("\nOutput:", result.output)
telemetry.flush()
