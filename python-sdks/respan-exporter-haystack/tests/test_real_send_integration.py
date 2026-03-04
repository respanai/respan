"""
Real send integration tests for Respan Haystack.

These tests load API keys from the repo .env and actually send data to Respan.
They are skipped when RESPAN_API_KEY is not set (e.g. in CI without secrets).
"""

import os
from pathlib import Path

import pytest

from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from respan_exporter_haystack.connector import RespanConnector
from respan_exporter_haystack.gateway import RespanGenerator

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load .env from repo root (respan) so tests can use RESPAN_API_KEY, OPENAI_API_KEY, etc.
# Path: tests/ -> respan-exporter-haystack/ -> python-sdks/ -> respan/
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
_env_path = _repo_root / ".env"
if _env_path.exists() and load_dotenv is not None:
    load_dotenv(_env_path)


def _respan_api_key() -> str | None:
    return os.getenv("RESPAN_API_KEY")


def _openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture(scope="module")
def respan_api_key():
    """API key from .env; skip entire module if missing."""
    key = _respan_api_key()
    if not key or key.strip() in ("", "your-respan-api-key-here"):
        pytest.skip("RESPAN_API_KEY not set or placeholder in .env (required for real send tests)")
    return key


def test_real_send_trace_via_connector(respan_api_key: str):
    """
    Run a minimal pipeline with RespanConnector and verify a trace is sent.
    Uses only PromptBuilder (no LLM), so only RESPAN_API_KEY from .env is needed.
    """
    os.environ["RESPAN_API_KEY"] = respan_api_key
    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

    pipeline = Pipeline()
    pipeline.add_component("tracer", RespanConnector(name="real_send_test_trace"))
    pipeline.add_component("prompt", PromptBuilder(template="Say hello to {{name}}."))
    # RespanConnector has no inputs; run both components via run inputs (tracer gets empty dict)
    result = pipeline.run({"prompt": {"name": "integration_test"}, "tracer": {}})

    assert "tracer" in result
    assert "trace_url" in result["tracer"]
    assert result["tracer"]["trace_url"] is not None
    assert "trace_id=" in result["tracer"]["trace_url"] or "respan" in result["tracer"]["trace_url"].lower()


@pytest.mark.skipif(
    not _openai_api_key() or _openai_api_key().strip() in ("", "your-openai-api-key-here"),
    reason="OPENAI_API_KEY not set or placeholder in .env",
)
def test_real_send_gateway_request(respan_api_key: str):
    """
    Send one real request through RespanGenerator (gateway) and verify response.
    Requires both RESPAN_API_KEY and OPENAI_API_KEY in .env.
    """
    os.environ["RESPAN_API_KEY"] = respan_api_key

    pipeline = Pipeline()
    pipeline.add_component("prompt", PromptBuilder(template="Reply with one word: {{word}}."))
    pipeline.add_component("llm", RespanGenerator(model="gpt-4o-mini"))
    pipeline.connect(sender="prompt", receiver="llm")

    result = pipeline.run({"prompt": {"word": "hello"}})

    assert "llm" in result
    assert "replies" in result["llm"]
    assert len(result["llm"]["replies"]) >= 1
    reply = result["llm"]["replies"][0]
    assert getattr(reply, "content", None) or getattr(reply, "text", None) or str(reply)
