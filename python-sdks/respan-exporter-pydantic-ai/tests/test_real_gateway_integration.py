# pyright: reportMissingImports=false
"""Real integration test for Pydantic AI exporter via Respan gateway."""

import os
import sys
import urllib.request
import unittest
from typing import List
from unittest.mock import patch

try:
    from pydantic_ai import Agent
except ImportError:
    Agent = None  # type: ignore[misc, assignment]

from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_tracing import RespanTelemetry
from respan_tracing.core.tracer import RespanTracer

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
        Send a real Pydantic AI query through Respan gateway and verify export upload.

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

        # Make sure tracing is clean
        RespanTracer.reset_instance()

        gateway_base_url = _resolve_gateway_base_url()

        # Patch the export mechanism to capture HTTP status codes for assertions.
        # Note: This targets urllib.request.urlopen; if the SDK switches to another
        # client (e.g. requests or httpx), this test will need to be updated.
        response_statuses: List[int] = []
        original_urlopen = urllib.request.urlopen

        def tracking_urlopen(*args, **kwargs):
            response = original_urlopen(*args, **kwargs)
            response_status = getattr(response, "status", None)
            if isinstance(response_status, int):
                response_statuses.append(response_status)
            return response

        # Initialize telemetry properly targeting the gateway
        telemetry = RespanTelemetry(
            app_name="integration-test-pydantic-ai", 
            api_key=respan_api_key,
            base_url=gateway_base_url,
            is_enabled=True,
            is_batching_enabled=False # Disable batching to ensure synchronous export for testing
        )
        
        # Set up a real Pydantic AI agent, routed via the OpenAI provider format since gateway supports it
        # You'll need OPENAI_API_KEY set or gateway equivalent if routing differently
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            self.skipTest("Set OPENAI_API_KEY for real integration test.")
            
        configured_model = os.getenv("RESPAN_GATEWAY_MODEL") or "openai:gpt-4o-mini"
        agent = Agent(model=configured_model)

        # Instrument the specific agent
        instrument_pydantic_ai(agent=agent)

        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=tracking_urlopen,
        ):
            # Use a very cheap/simple request
            result = await agent.run('Reply with exactly "gateway_ok".')

            self.assertIsNotNone(
                result.output,
                "Expected an output from the real gateway-backed query.",
            )

            # Flush telemetry to ensure export completes before assertions
            telemetry.flush()

        self.assertTrue(
            response_statuses,
            "Exporter did not make any ingest HTTP request.",
        )
        self.assertTrue(
            any(status_code < 300 for status_code in response_statuses),
            f"No successful ingest response observed. statuses={response_statuses}",
        )
        
        RespanTracer.reset_instance()

if __name__ == "__main__":
    unittest.main()
