# pyright: reportMissingImports=false
"""Real integration test for Pydantic AI exporter via Respan gateway."""

import os

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai import Agent

from respan_exporter_pydantic_ai import instrument_pydantic_ai
from respan_tracing import RespanTelemetry, workflow, task
from respan_tracing.core.tracer import RespanTracer
from respan_tracing.testing import InMemorySpanExporter

# Optional: load .env for local runs (python-dotenv not required for package)
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _load_env_from_dotenv() -> None:
    """Load optional dotenv files when python-dotenv is available."""
    if load_dotenv is None:
        return
    repository_dotenv_path = os.path.join(REPO_ROOT, ".env")
    load_dotenv(dotenv_path=repository_dotenv_path, override=False)
    load_dotenv(override=False)


def _resolve_gateway_base_url() -> str:
    """Resolve base URL for gateway and exporter endpoint resolution."""
    raw_base_url = (
        os.getenv("RESPAN_GATEWAY_BASE_URL")
        or os.getenv("RESPAN_BASE_URL")
        or "https://api.respan.ai/api"
    )
    return raw_base_url.rstrip("/")


@pytest.mark.asyncio
async def test_real_gateway_query_exports_payloads() -> None:
    """
    Send a real Pydantic AI query through Respan gateway and verify
    that instrumentation produces spans.

    This test is intentionally opt-in because it makes live network calls and
    consumes model tokens.
    """
    # Load .env first so IS_REAL_GATEWAY_TESTING_ENABLED and RESPAN_API_KEY can be read from it
    _load_env_from_dotenv()

    if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
        pytest.skip("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run live gateway integration test.")

    respan_api_key = os.getenv("RESPAN_API_KEY")
    if not respan_api_key:
        pytest.skip("Set RESPAN_API_KEY for real integration test.")

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

    # Use Respan gateway for LLM calls so only RESPAN_API_KEY is needed.
    # Point OpenAI client at Respan gateway; auth with Respan key.
    os.environ["OPENAI_BASE_URL"] = gateway_base_url
    os.environ["OPENAI_API_KEY"] = respan_api_key

    configured_model = os.getenv("RESPAN_GATEWAY_MODEL") or "openai:gpt-4o-mini"
    agent = Agent(model=configured_model)

    instrument_pydantic_ai(agent=agent)

    @task(name="agent_run")
    async def run_agent():
        return await agent.run("Reply with exactly \"gateway_ok\".")

    @workflow(name="pydantic_ai_gateway_test")
    async def run_workflow():
        return await run_agent()

    try:
        result = await run_workflow()

        assert result.output is not None, "Expected an output from the real gateway-backed query."

        telemetry.flush()

        spans = span_exporter.get_finished_spans()
        assert len(spans) > 0, "Instrumentation did not produce any spans."

        # Verify at least one span has gen_ai attributes from Pydantic AI
        span_attrs = {
            k for s in spans for k in (s.attributes or {}).keys()
        }
        assert any("gen_ai" in attr for attr in span_attrs), (
            f"No gen_ai attributes found in spans. Span names: {[s.name for s in spans]}"
        )
    finally:
        RespanTracer.reset_instance()
