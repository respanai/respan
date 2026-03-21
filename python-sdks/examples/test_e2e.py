"""End-to-end tests that hit the real Respan backend.

Requires RESPAN_API_KEY to be set (via .env or environment).
These tests send real data to the Respan API.
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from conftest import _make_trace


@pytest.fixture
def respan_api_key():
    key = os.getenv("RESPAN_API_KEY")
    if not key:
        pytest.skip("RESPAN_API_KEY not set")
    return key


@pytest.fixture
def respan_base_url():
    return os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")


class TestRealBackend:
    """Tests that send real traces to the Respan backend."""

    def test_respan_full_pipeline(self, respan_api_key, respan_base_url):
        """Full Respan pipeline: init → instrumentor → trace → flush."""
        from respan import Respan
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor

        r = Respan(
            api_key=respan_api_key,
            base_url=respan_base_url,
            instrumentations=[OpenAIAgentsInstrumentor()],
        )

        # Simulate a trace from the OpenAI Agents SDK
        trace = _make_trace("e2e-pipeline-test", "full-pipeline-test")
        processor = r._instrumentations["openai-agents"]._processor
        processor.on_trace_end(trace)

        r.flush()
        r.shutdown()

    @pytest.mark.asyncio
    async def test_openai_agents_real_run(self, respan_api_key, respan_base_url):
        """Run a real OpenAI agent and verify traces are sent to Respan."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        from respan import Respan
        from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
        from agents import Agent, Runner

        r = Respan(
            api_key=respan_api_key,
            base_url=respan_base_url,
            instrumentations=[OpenAIAgentsInstrumentor()],
        )

        agent = Agent(
            name="TestAgent",
            instructions="You only respond with the word 'hello'.",
        )
        result = await Runner.run(agent, "Say hi")
        print(f"Agent output: {result.final_output}")

        r.flush()
        r.shutdown()
