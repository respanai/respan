import os
import threading
from typing import Any
from typing import Dict
from typing import Optional

import pytest
import requests

from respan_exporter_superagent import create_client


class DummySuperagentClient:
    async def guard(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "classification": "pass",
            "violation_types": [],
            "input_echo": kwargs.get("input"),
        }


class CapturingRequests:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def post(self, url: str, json: Any, headers: Dict[str, str], timeout: int) -> Any:
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )

        class DummyResponse:
            status_code = 200

        return DummyResponse()


class InlineThread:
    def __init__(self, *, target: Any, kwargs: Optional[Dict[str, Any]] = None, daemon: bool = False) -> None:
        self._target = target
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self) -> None:
        self._target(**self._kwargs)


@pytest.mark.asyncio
async def test_guard_call_sends_validated_log_to_respan_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    requests_capture = CapturingRequests()
    monkeypatch.setattr("respan_sdk.utils.export.requests", requests_capture)
    monkeypatch.setattr(threading, "Thread", InlineThread)

    client = create_client(
        api_key="kw_test_key",
        endpoint="https://example.respan.local/api/v1/traces/ingest",
        client=DummySuperagentClient(),
    )

    result = await client.guard(
        input="hello",
        respan_params={
            "span_workflow_name": "wf",
            "span_name": "sp",
            "customer_identifier": "user-123",
        },
    )

    assert result["classification"] == "pass"

    assert len(requests_capture.calls) == 1
    call = requests_capture.calls[0]

    assert call["url"] == "https://example.respan.local/api/v1/traces/ingest"
    assert call["headers"]["Authorization"] == "Bearer kw_test_key"

    payloads = call["json"]
    assert isinstance(payloads, list)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["span_workflow_name"] == "wf"
    assert payload["span_name"] == "sp"
    assert payload["status"] == "success"
    assert payload["customer_identifier"] == "user-123"
    assert payload["metadata"]["integration"] == "superagent"
    assert payload["metadata"]["method"] == "guard"


@pytest.mark.asyncio
async def test_disable_log_does_not_send(monkeypatch: pytest.MonkeyPatch) -> None:
    requests_capture = CapturingRequests()
    monkeypatch.setattr("respan_sdk.utils.export.requests", requests_capture)
    monkeypatch.setattr(threading, "Thread", InlineThread)

    client = create_client(
        api_key="kw_test_key",
        endpoint="https://example.respan.local/api/v1/traces/ingest",
        client=DummySuperagentClient(),
    )

    await client.guard(
        input="hello",
        respan_params={"disable_log": True},
    )

    assert requests_capture.calls == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_redact_openai_provider_sends_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Live test that exercises a real Superagent SDK call using OpenAI provider.

    Requirements:
    - SUPERAGENT_API_KEY: required by safety-agent client (usage tracking)
    - OPENAI_API_KEY: required for provider "openai/*"
    """
    if not os.getenv("SUPERAGENT_API_KEY"):
        pytest.skip("SUPERAGENT_API_KEY not set")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    try:
        from safety_agent import create_client as superagent_create_client
    except Exception:
        pytest.skip("safety-agent not installed")

    # Avoid hitting Superagent usage endpoint during test run.
    try:
        import safety_agent.client as safety_agent_client_module
    except Exception:
        safety_agent_client_module = None
    if safety_agent_client_module is not None:
        try:
            import httpx

            monkeypatch.setattr(safety_agent_client_module.httpx, "post", lambda *args, **kwargs: None)
        except Exception:
            pass

    requests_capture = CapturingRequests()
    monkeypatch.setattr("respan_sdk.utils.export.requests", requests_capture)
    monkeypatch.setattr(threading, "Thread", InlineThread)

    real_superagent_client = superagent_create_client()
    exporter_client = create_client(
        api_key="kw_test_key",
        endpoint="https://example.respan.local/api/v1/traces/ingest",
        client=real_superagent_client,
    )

    result = await exporter_client.redact(
        input="My email is john@example.com",
        model="openai/gpt-4o-mini",
        respan_params={
            "span_workflow_name": "wf_live",
            "span_name": "redact_live",
            "customer_identifier": "user-123",
        },
    )

    assert getattr(result, "redacted", None)

    assert len(requests_capture.calls) == 1
    payload = requests_capture.calls[0]["json"][0]
    assert payload["span_workflow_name"] == "wf_live"
    assert payload["span_name"] == "redact_live"
    assert payload["status"] == "success"
    assert payload["metadata"]["integration"] == "superagent"
    assert payload["metadata"]["method"] == "redact"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_guard_superagent_provider_posts_to_respan(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Live test that:
    - calls the real Superagent guard model (default superagent/guard-1.7b)
    - verifies the exporter successfully POSTs to a real Respan endpoint

    Requirements:
    - SUPERAGENT_API_KEY: required by safety-agent client
    - RESPAN_API_KEY: required by exporter
    - RESPAN_ENDPOINT: optional; defaults to https://api.respan.ai/api/v1/traces/ingest
    """
    if not os.getenv("SUPERAGENT_API_KEY"):
        pytest.skip("SUPERAGENT_API_KEY not set")
    if not os.getenv("RESPAN_API_KEY"):
        pytest.skip("RESPAN_API_KEY not set")

    try:
        from safety_agent import create_client as superagent_create_client
    except Exception:
        pytest.skip("safety-agent not installed")

    # Avoid hitting Superagent usage endpoint during test run.
    try:
        import safety_agent.client as safety_agent_client_module
    except Exception:
        safety_agent_client_module = None
    if safety_agent_client_module is not None:
        try:
            monkeypatch.setattr(safety_agent_client_module.httpx, "post", lambda *args, **kwargs: None)
        except Exception:
            pass

    # Make exporter send synchronous for determinism.
    monkeypatch.setattr(threading, "Thread", InlineThread)

    # Wrap requests.post so we still send real traffic but can assert response status.
    post_results: list[Dict[str, Any]] = []
    real_post = requests.post

    def recording_post(url: str, json: Any, headers: Dict[str, str], timeout: int) -> Any:
        response = real_post(url, json=json, headers=headers, timeout=timeout)
        post_results.append({"url": url, "status_code": response.status_code, "text": response.text})
        return response

    monkeypatch.setattr("respan_sdk.utils.export.requests.post", recording_post)

    exporter_client = create_client(
        api_key=os.environ["RESPAN_API_KEY"],
        endpoint=os.getenv("RESPAN_ENDPOINT"),
        client=superagent_create_client(),
        timeout=30,
    )

    model = os.getenv("SUPERAGENT_GUARD_MODEL", "superagent/guard-1.7b")
    try:
        result = await exporter_client.guard(
            input="Hello! Please say 'ok'.",
            model=model,
            respan_params={
                "span_workflow_name": "wf_live_guard",
                "span_name": "guard_live",
                "customer_identifier": "integration-test",
            },
        )
    except RuntimeError as exc:
        # The upstream Superagent provider/model catalog may change; treat missing models as a skip
        # so this integration test doesn't break package CI unexpectedly.
        msg = str(exc).lower()
        if "model" in msg and "not found" in msg:
            pytest.skip(f"Superagent model not available: {model} ({exc})")
        raise

    assert getattr(result, "classification", None) in {"pass", "block"}

    assert len(post_results) == 1
    assert post_results[0]["status_code"] < 300, post_results[0]

