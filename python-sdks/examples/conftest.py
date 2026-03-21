"""Shared fixtures for integration tests."""

import os
import pytest
from dotenv import load_dotenv

# Load .env from tests directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@pytest.fixture(autouse=True)
def reset_tracer():
    """Reset the RespanTracer singleton between tests."""
    from respan_tracing.core.tracer import RespanTracer
    RespanTracer.reset_instance()
    yield
    RespanTracer.reset_instance()


@pytest.fixture
def respan_api_key():
    key = os.getenv("RESPAN_API_KEY")
    if not key:
        pytest.skip("RESPAN_API_KEY not set")
    return key


@pytest.fixture
def respan_base_url():
    return os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")


def _make_trace(trace_id: str, name: str):
    """Create a concrete TraceImpl for testing."""
    from agents.tracing.traces import TraceImpl
    from agents.tracing.processor_interface import TracingProcessor

    class _NoopProcessor(TracingProcessor):
        def on_trace_start(self, trace): pass
        def on_trace_end(self, trace): pass
        def on_span_start(self, span): pass
        def on_span_end(self, span): pass
        def shutdown(self): pass
        def force_flush(self): pass

    return TraceImpl(
        name=name, trace_id=trace_id,
        group_id=None, metadata=None, processor=_NoopProcessor(),
    )


def _make_adk_spans(
    trace_id: str = "adk-trace-001",
    agent_name: str = "test_agent",
    model: str = "gemini-2.5-flash",
):
    """Create a list of 3 ADK-style span dicts (invocation -> agent_run -> call_llm).

    Uses plain dicts so no google-adk package dependency is needed in tests.
    """
    import time

    now_ms = int(time.time() * 1000)

    invocation_span = {
        "trace_id": trace_id,
        "span_id": "span-inv-001",
        "parent_id": None,
        "name": "invocation",
        "start_time": now_ms,
        "end_time": now_ms + 3000,
        "attributes": {
            "gen_ai.agent.name": agent_name,
            "gen_ai.system": "google_genai",
        },
    }

    agent_run_span = {
        "trace_id": trace_id,
        "span_id": "span-agent-001",
        "parent_id": "span-inv-001",
        "name": "agent_run",
        "start_time": now_ms + 100,
        "end_time": now_ms + 2900,
        "attributes": {
            "gen_ai.agent.name": agent_name,
            "gen_ai.system": "google_genai",
        },
    }

    call_llm_span = {
        "trace_id": trace_id,
        "span_id": "span-llm-001",
        "parent_id": "span-agent-001",
        "name": "call_llm",
        "start_time": now_ms + 200,
        "end_time": now_ms + 2800,
        "attributes": {
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.system": "google_genai",
            "gen_ai.usage.prompt_tokens": 10,
            "gen_ai.usage.completion_tokens": 25,
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Hello!",
            "gen_ai.completion.0.role": "model",
            "gen_ai.completion.0.content": "Hi there! How can I help?",
        },
    }

    return [invocation_span, agent_run_span, call_llm_span]
