"""Released safety-agent runtime using controlled provider HTTP responses."""

import json
import os
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import httpx
from safety_agent import create_client
from safety_agent.client import SafetyClient
from safety_agent.types import ScanResponse, ScanUsage


@contextmanager
def fixture_client(*, fail=False, before_response=None, retry=False):
    requests = []
    original = httpx.AsyncClient

    async def respond(request):
        if request.method == "GET":
            return httpx.Response(
                200, text="fixture public URL", headers={"content-type": "text/plain"}
            )
        body = json.loads(request.content)
        requests.append(body)
        if before_response is not None:
            await before_response()
        if retry and body.get("model") == "fixture-primary":
            return httpx.Response(503, json={"error": {"message": "controlled retry"}})
        if fail:
            return httpx.Response(
                401, json={"error": {"message": "controlled provider failure"}}
            )
        name = body.get("text", {}).get("format", {}).get("name", "") or body.get(
            "response_format", {}
        ).get("json_schema", {}).get("name", "")
        if name == "redact_result":
            output = {
                "redacted": "Contact <EMAIL_REDACTED>",
                "findings": ["fixture-email@example.com"],
            }
        else:
            messages = body.get("input", body.get("messages", []))
            blocked = "block" in json.dumps(messages[-1].get("content", ""))
            output = {
                "classification": "block" if blocked else "pass",
                "reasoning": "fixture result",
                "violation_types": ["prompt_injection"] if blocked else [],
                "cwe_codes": ["CWE-94"] if blocked else [],
            }
        text = json.dumps(output)
        if request.url.path.endswith("/responses"):
            payload = {
                "id": "fixture-response",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
                "usage": {"input_tokens": 0, "output_tokens": 3, "total_tokens": 3},
            }
        else:
            payload = {
                "id": "fixture-response",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 3,
                    "total_tokens": 3,
                },
            }
        return httpx.Response(200, json=payload)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        return original(*args, **kwargs)

    async def scan_boundary(self, repo, branch, model):
        return ScanResponse(
            result="fixture clean repository",
            usage=ScanUsage(
                input_tokens=0, output_tokens=0, reasoning_tokens=0, cost=0.0
            ),
        )

    async def fetched_url(url):
        from safety_agent.utils.safe_url_fetcher import PublicUrlResponse

        return PublicUrlResponse(
            final_url=url, content_type="text/plain", data=b"fixture public URL"
        )

    import safety_agent.utils.input_processor as processor

    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-provider-key"})
        )
        stack.enter_context(patch.object(httpx, "AsyncClient", factory))
        stack.enter_context(patch.object(SafetyClient, "_post_usage", lambda *a: None))
        stack.enter_context(
            patch.object(SafetyClient, "_call_daytona_scan", scan_boundary)
        )
        if hasattr(processor, "fetch_public_url"):
            stack.enter_context(
                patch.object(processor, "fetch_public_url", fetched_url)
            )
        client = create_client(api_key="fixture-superagent-key")
        yield client, requests
