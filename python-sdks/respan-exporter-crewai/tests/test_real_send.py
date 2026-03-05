"""Real send integration test for CrewAI exporter — sends data to Respan platform."""

import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

pytest.importorskip("respan_exporter_crewai")

from respan_exporter_crewai.exporter import RespanCrewAIExporter


def _load_dotenv_if_available() -> None:
    """Load .env when present (package dir or cwd) so RESPAN_API_KEY is set."""
    if load_dotenv is None:
        return
    load_dotenv(override=False)
    _tests_dir = os.path.dirname(os.path.abspath(__file__))
    _package_root = os.path.dirname(_tests_dir)
    load_dotenv(dotenv_path=os.path.join(_package_root, ".env"), override=False)


def _trace_tree_crewai_style() -> Dict[str, Any]:
    """
    Build a trace tree (workflow → task → generation) with input, output, and usage.

    Tree shape:
    - workflow (root): input/output so trace has content
    - task: input/output
    - generation: input, output, usage (prompt_tokens, completion_tokens) for backend token calc
    """
    trace_id_hex = "a1b2c3d4e5f6789012345678abcdef01"
    root_span_id = "a1b2c3d4e5f60001"
    task_span_id = "a1b2c3d4e5f60002"
    gen_span_id = "a1b2c3d4e5f60003"
    return {
        "spans": [
            {
                "span_id": root_span_id,
                "trace_id": trace_id_hex,
                "name": "crewai_real_send_workflow",
                "parent_id": None,
                "input": "Run the crew: greet the user",
                "output": "Hello from the crew.",
            },
            {
                "span_id": task_span_id,
                "trace_id": trace_id_hex,
                "name": "crewai_real_send_task",
                "parent_id": root_span_id,
                "input": "Execute agent task: provide a greeting",
                "output": "Hello from the crew.",
            },
            {
                "span_id": gen_span_id,
                "trace_id": trace_id_hex,
                "name": "crewai_real_send_generation",
                "parent_id": task_span_id,
                "model": "gpt-4o-mini",
                "input": "Say hello in one sentence.",
                "output": "Hello from the crew.",
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 6,
                },
            },
        ]
    }


@pytest.mark.integration
def test_crewai_exporter_real_send_to_respan() -> None:
    """
    Send a CrewAI-style trace tree (workflow → task → generation) to Respan traces/ingest.

    Requirements:
    - RESPAN_API_KEY: set in environment to run (test is skipped otherwise)
    - RESPAN_BASE_URL: optional; defaults to https://api.respan.ai/api

    This test performs a real HTTP POST to Respan and asserts a 2xx response.
    """
    _load_dotenv_if_available()
    api_key = os.getenv("RESPAN_API_KEY")
    if not api_key:
        pytest.skip("RESPAN_API_KEY not set; skip real send test")
    post_results: List[Dict[str, Any]] = []
    real_post = requests.post

    def recording_post(url: str, **kwargs: Any) -> Any:
        response = real_post(url=url, **kwargs)
        post_results.append({
            "url": url,
            "status_code": response.status_code,
            "text": response.text,
        })
        return response

    exporter = RespanCrewAIExporter(
        api_key=api_key,
        base_url=os.getenv("RESPAN_BASE_URL"),
        customer_identifier="crewai_real_send_test",
    )
    trace_input = _trace_tree_crewai_style()

    with patch("respan_exporter_crewai.exporter.requests.post", side_effect=recording_post):
        payloads = exporter.export(trace_or_spans=trace_input)

    assert isinstance(payloads, list), "export should return built payloads"
    assert len(payloads) == 3, (
        "trace tree has 3 spans (workflow, task, generation); got %s" % len(payloads)
    )

    log_types = {p.get("log_type") for p in payloads}
    assert "workflow" in log_types, "root span should be workflow"
    assert "task" in log_types, "middle span should be task"
    assert "generation" in log_types, "leaf span with model should be generation"

    for p in payloads:
        assert p.get("log_method") == "tracing_integration", (
            "log_method=tracing_integration required so platform shows trace view"
        )

    generation_payloads = [p for p in payloads if p.get("log_type") == "generation"]
    assert len(generation_payloads) == 1, "expect one generation span"
    gen = generation_payloads[0]
    assert gen.get("input"), "generation should have input for backend"
    assert gen.get("output"), "generation should have output for backend"
    assert "prompt_tokens" in gen and gen.get("prompt_tokens") is not None, (
        "generation should have prompt_tokens for backend token calc"
    )
    assert "completion_tokens" in gen and gen.get("completion_tokens") is not None, (
        "generation should have completion_tokens for backend token calc"
    )
    assert gen.get("total_request_tokens") is None, (
        "do not send total_request_tokens; Respan backend calculates from prompt_tokens + completion_tokens"
    )

    assert len(post_results) == 1, "exporter should POST once to Respan"
    status_code = post_results[0]["status_code"]
    assert 200 <= status_code < 300, (
        "Respan ingest should return 2xx, got %s: %s"
        % (status_code, post_results[0]["text"])
    )
