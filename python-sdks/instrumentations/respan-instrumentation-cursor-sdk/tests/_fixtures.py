"""Actual released Cursor Connect transport frames, without a bridge process."""

import json

import httpx
from cursor_sdk import AsyncClient, CursorClient, LocalAgentOptions


def envelope(value, *, end=False):
    data = json.dumps(value).encode()
    return bytes([2 if end else 0]) + len(data).to_bytes(4, "big") + data


def events(*, tools=False, modern=False, status="finished", usage=None):
    rows = [
        {
            "sdkMessage": {
                "type": "status",
                "agentId": "agent-fixture",
                "runId": "run-fixture",
                "status": "running",
            }
        }
    ]
    rows.append({"interactionUpdate": {"type": "text-delta", "text": "Fixture text"}})
    if tools:
        tool = {
            "type": "tool_call",
            "agentId": "agent-fixture",
            "runId": "run-fixture",
            "callId": "current-fixture-call",
            "name": "lookup",
            "args": {"query": "fixture"},
            "status": "completed",
            "result": {
                "vectors": [float(i) / 10 for i in range(5000)],
                "zero": 0,
                "false": False,
            },
        }
        rows.append(
            {"step": {"type": "toolCall", "message": tool}}
            if modern
            else {"sdkMessage": tool}
        )
    result = {
        "runId": "run-fixture",
        "agentId": "agent-fixture",
        "status": status,
        "result": "Fixture completion" if status == "finished" else "",
        "model": {"id": "composer-fixture"},
    }
    if usage is not None:
        result["usage"] = usage
    if status == "error":
        result["error"] = {
            "message": "Controlled run failure",
            "code": "fixture-failed",
        }
    rows.append({"result": {"result": result}})
    return rows


class Fixture:
    def __init__(self, *, rows=None, error=None):
        self.rows = events() if rows is None else rows
        self.error = error
        self.requests = []

    def handler(self, request):
        method = request.url.path.rsplit("/", 1)[-1]
        self.requests.append(method)
        if method == "CreateAgent":
            return httpx.Response(
                200,
                json={"agentId": "agent-fixture", "model": {"id": "composer-fixture"}},
                request=request,
            )
        if method == "Send":
            if self.error:
                return httpx.Response(
                    self.error,
                    json={
                        "code": "unavailable",
                        "message": "Controlled transport failure",
                    },
                    request=request,
                )
            body = b"".join(envelope(row) for row in self.rows) + envelope({}, end=True)
            return httpx.Response(
                200,
                stream=httpx.ByteStream(body),
                headers={"content-type": "application/connect+json"},
                request=request,
            )
        if method == "GetUsage":
            return httpx.Response(
                200,
                json={
                    "usage": {
                        "usage": {
                            "inputTokens": 7,
                            "outputTokens": 3,
                            "totalTokens": 10,
                            "cacheReadTokens": 0,
                            "reasoningTokens": 2,
                        },
                        "cost": {"totalUsd": 0.012},
                        "runs": [],
                    }
                },
                request=request,
            )
        return httpx.Response(200, json={}, request=request)

    def client(self):
        return CursorClient(
            base_url="https://cursor-fixture.invalid",
            auth_token="synthetic-bridge-token",
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )

    def async_client(self):
        return AsyncClient(
            base_url="https://cursor-fixture.invalid",
            auth_token="synthetic-bridge-token",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )

    @staticmethod
    def agent(client):
        return client.agents.create(
            model="composer-fixture",
            api_key="synthetic-cursor-key",
            local=LocalAgentOptions(cwd="/synthetic"),
        )
