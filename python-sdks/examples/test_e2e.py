"""End-to-end tests that hit the real Respan backend.

Requires RESPAN_API_KEY to be set (via .env or environment).
These tests send real data to the Respan API.
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from conftest import _make_trace, _make_adk_spans


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

    def test_exporter_v2_sends_to_respan(self, respan_api_key, respan_base_url):
        """RespanSpanExporterV2 can POST to the real /v1/traces/ingest endpoint."""
        from respan_tracing.exporters import RespanSpanExporterV2

        endpoint = f"{respan_base_url.rstrip('/')}/v1/traces/ingest"
        exp = RespanSpanExporterV2(api_key=respan_api_key, endpoint=endpoint)

        trace = _make_trace("e2e-exporter-test", "exporter-v2-test")
        from respan_instrumentation_openai_agents import convert_to_respan_log
        data = convert_to_respan_log(trace)
        assert data is not None

        exp.export([data])
        exp.flush()
        exp.shutdown()
        # If no exception, the POST was accepted (HTTP < 300)

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


@pytest.fixture
def google_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key


class TestGoogleAdkRealBackend:
    """Tests that send Google ADK traces to the real Respan backend."""

    def test_google_adk_exporter_sends_to_respan(self, respan_api_key, respan_base_url):
        """RespanSpanExporterV2 + AdkSpanConverter -> POST to real endpoint."""
        from respan_tracing.exporters import RespanSpanExporterV2
        from respan_instrumentation_google_adk.converter import AdkSpanConverter

        endpoint = f"{respan_base_url.rstrip('/')}/v1/traces/ingest"
        exp = RespanSpanExporterV2(api_key=respan_api_key, endpoint=endpoint)

        converter = AdkSpanConverter(environment="test")
        spans = _make_adk_spans()
        payloads = converter.convert(trace_or_spans=spans)
        assert len(payloads) == 3

        exp.export(payloads)
        exp.flush()
        exp.shutdown()

    def test_google_adk_full_pipeline(self, respan_api_key, respan_base_url):
        """Full Respan pipeline with GoogleAdkInstrumentor: init -> convert -> flush."""
        from respan import Respan
        from respan_instrumentation_google_adk import GoogleAdkInstrumentor
        from respan_instrumentation_google_adk.converter import AdkSpanConverter

        r = Respan(
            api_key=respan_api_key,
            base_url=respan_base_url,
            instrumentations=[GoogleAdkInstrumentor(environment="test")],
        )

        converter = AdkSpanConverter(environment="test")
        spans = _make_adk_spans()
        payloads = converter.convert(trace_or_spans=spans)

        r.exporter.export(payloads)
        r.flush()
        r.shutdown()

    @pytest.mark.asyncio
    async def test_google_adk_real_agent_run(self, respan_api_key, respan_base_url, google_api_key):
        """Run a real ADK agent and verify traces are sent to Respan."""
        from google.adk.agents import Agent
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        from respan import Respan
        from respan_instrumentation_google_adk import GoogleAdkInstrumentor

        os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true"

        r = Respan(
            api_key=respan_api_key,
            base_url=respan_base_url,
            instrumentations=[GoogleAdkInstrumentor(environment="test")],
        )

        agent = Agent(
            name="TestAgent",
            model="gemini-2.5-flash",
            instruction="You only respond with the word 'hello'.",
        )

        runner = InMemoryRunner(agent=agent, app_name="e2e_test")
        session = await runner.session_service.create_session(
            app_name="e2e_test", user_id="test-user"
        )

        message = types.Content(
            role="user",
            parts=[types.Part(text="Say hi")],
        )

        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=message,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent output: {part.text}")

        r.flush()
        r.shutdown()
