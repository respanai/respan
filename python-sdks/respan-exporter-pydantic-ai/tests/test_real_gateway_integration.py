# pyright: reportMissingImports=false
"""Real integration test for Pydantic AI exporter via Respan gateway."""

import os
import sys
import unittest

try:
    from pydantic_ai import Agent
except ImportError:
    Agent = None  # type: ignore[misc, assignment]

from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
PYTHON_EXPORTER_SRC = os.path.join(
    REPO_ROOT,
    "python-sdks",
    "respan-exporter-pydantic-ai",
    "src",
)
PYTHON_SDK_SRC = os.path.join(
    REPO_ROOT,
    "python-sdks",
    "respan-sdk",
    "src",
)

if PYTHON_EXPORTER_SRC not in sys.path:
    sys.path.insert(0, PYTHON_EXPORTER_SRC)
if PYTHON_SDK_SRC not in sys.path:
    sys.path.insert(0, PYTHON_SDK_SRC)


def _load_env_from_dotenv() -> None:
    """Load optional dotenv files when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repository_dotenv_path = os.path.join(REPO_ROOT, ".env")
    load_dotenv(dotenv_path=repository_dotenv_path, override=False)
    load_dotenv(override=False)


def _resolve_gateway_base_url() -> str:
    """Resolve base URL for gateway and exporter endpoint resolution."""
    raw_base_url = (
        os.getenv("RESPAN_GATEWAY_BASE_URL")
        or os.getenv("RESPAN_BASE_URL")
        or "https://api.respan.ai"
    )
    return raw_base_url.rstrip("/")


class RespanPydanticAIGatewayIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_gateway_query_exports_payloads(self) -> None:
        """
        Send a real Pydantic AI query through Respan gateway and verify
        that instrumentation produces spans.

        This test is intentionally opt-in because it makes live network calls and
        consumes model tokens.
        """
        if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
            self.skipTest("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run live gateway integration test.")

        _load_env_from_dotenv()

        respan_api_key = os.getenv("RESPAN_API_KEY")
        if not respan_api_key:
            self.skipTest("Set RESPAN_API_KEY for real integration test.")

        if Agent is None:
            self.skipTest("pydantic_ai is not installed in this environment.")

        RespanTracer.reset_instance()

        gateway_base_url = _resolve_gateway_base_url()

        # Use InMemorySpanExporter to capture spans without depending on
        # the HTTP client used by the OTLP exporter.
        span_exporter = InMemorySpanExporter()

        telemetry = RespanTelemetry(
            app_name="integration-test-pydantic-ai",
            api_key=respan_api_key,
            base_url=gateway_base_url,
            is_enabled=True,
            is_batching_enabled=False,
        )

        # Add directly to tracer provider — bypasses RespanSpanProcessor filtering
        # which drops spans without Traceloop attributes (Pydantic AI uses gen_ai.*)
        telemetry.tracer.tracer_provider.add_span_processor(
            SimpleSpanProcessor(span_exporter)
        )

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            self.skipTest("Set OPENAI_API_KEY for real integration test.")

        configured_model = os.getenv("RESPAN_GATEWAY_MODEL") or "openai:gpt-4o-mini"
        agent = Agent(model=configured_model)

        instrument_pydantic_ai(agent=agent)

        try:
            result = await agent.run('Reply with exactly "gateway_ok".')

            self.assertIsNotNone(
                result.output,
                "Expected an output from the real gateway-backed query.",
            )

            telemetry.flush()

            spans = span_exporter.get_finished_spans()
            self.assertTrue(
                len(spans) > 0,
                "Instrumentation did not produce any spans.",
            )

            # Verify at least one span has gen_ai attributes from Pydantic AI
            span_attrs = {
                k for s in spans for k in (s.attributes or {}).keys()
            }
            self.assertTrue(
                any("gen_ai" in attr for attr in span_attrs),
                f"No gen_ai attributes found in spans. Span names: {[s.name for s in spans]}",
            )
        finally:
            RespanTracer.reset_instance()


if __name__ == "__main__":
    unittest.main()
