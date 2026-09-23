import os
import shutil

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from respan import Respan
from respan_instrumentation_claude_agent_sdk import ClaudeAgentSDKInstrumentor
from respan_tracing.testing import InMemorySpanExporter

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_gateway_query_exports_claude_agent_spans():
    if os.getenv("IS_REAL_GATEWAY_TESTING_ENABLED") != "1":
        pytest.skip("Set IS_REAL_GATEWAY_TESTING_ENABLED=1 to run.")

    if not shutil.which("claude"):
        pytest.skip(
            "Claude Code CLI is required for the real gateway integration test."
        )

    claude_agent_sdk = pytest.importorskip("claude_agent_sdk")
    respan_api_key = os.getenv("RESPAN_API_KEY")
    if not respan_api_key:
        pytest.skip("Set RESPAN_API_KEY for the real gateway integration test.")

    respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api").rstrip(
        "/"
    )
    anthropic_base_url = f"{respan_base_url}/anthropic"

    os.environ["ANTHROPIC_API_KEY"] = respan_api_key
    os.environ["ANTHROPIC_AUTH_TOKEN"] = respan_api_key
    os.environ["ANTHROPIC_BASE_URL"] = anthropic_base_url

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

    span_exporter = InMemorySpanExporter()
    respan = Respan(
        api_key=respan_api_key,
        base_url=respan_base_url,
        app_name="claude-agent-sdk-integration-test",
        instrumentations=[ClaudeAgentSDKInstrumentor(capture_content=True)],
        is_batching_enabled=False,
    )
    respan.telemetry.tracer.tracer_provider.add_span_processor(
        SimpleSpanProcessor(span_exporter)
    )

    options = ClaudeAgentOptions(
        model=os.getenv("CLAUDE_AGENT_MODEL", "sonnet"),
        max_turns=1,
        permission_mode="bypassPermissions",
        cwd=os.getcwd(),
        env={
            "ANTHROPIC_API_KEY": respan_api_key,
            "ANTHROPIC_AUTH_TOKEN": respan_api_key,
            "ANTHROPIC_BASE_URL": anthropic_base_url,
        },
    )

    result_message = None
    async for message in claude_agent_sdk.query(
        prompt='Reply with exactly "gateway_ok".',
        options=options,
    ):
        if isinstance(message, ResultMessage):
            result_message = message

    respan.flush()

    assert result_message is not None, "Expected a ResultMessage from Claude Agent SDK."

    spans = span_exporter.get_finished_spans()
    assert spans, "Instrumentation did not emit any spans."
    assert any("gen_ai" in key for span in spans for key in (span.attributes or {})), (
        f"No gen_ai attributes found. Span names: {[span.name for span in spans]}"
    )
