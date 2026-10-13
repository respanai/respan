"""A call answered by the Respan gateway links its span to the gateway log row.

The gateway returns the row's unique_id in ``X-Respan-Log-Id``; the row carries
the call's usage, cost and content, so the chat span carries the id and no usage
or content of its own. Driven through the real Anthropic SDK over a mocked
transport.
"""

from __future__ import annotations

import json

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic, AuthenticationError
from respan_instrumentation_anthropic import _instrumentation as instrumentation
from respan_instrumentation_anthropic import _messages as message_helpers
from respan_instrumentation_anthropic._gateway_link import (
    RESPAN_GATEWAY_LOG_ID,
    RESPAN_LOG_ID_RESPONSE_HEADER,
)

GATEWAY_LOG_ID = "9a8b7c6d5e4f40a1b2c3d4e5f6a7b8c9"
MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 4,
        "output_tokens": 2,
        "cache_read_input_tokens": 1,
    },
}
STREAM_EVENTS = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                **MESSAGE,
                "content": [],
                "usage": {"input_tokens": 4, "output_tokens": 0},
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


@pytest.fixture
def captured(monkeypatch):
    spans = []
    monkeypatch.setattr(
        message_helpers,
        "build_readable_span",
        lambda name, **kwargs: spans.append({"name": name, **kwargs}) or spans[-1],
    )
    monkeypatch.setattr(message_helpers, "inject_span", lambda span: True)
    instrumentor = instrumentation.AnthropicInstrumentor()
    instrumentor.activate()
    yield spans
    instrumentor.deactivate()


def _handler(*, log_id: str | None):
    headers = {RESPAN_LOG_ID_RESPONSE_HEADER: log_id} if log_id else {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if body["messages"][-1]["content"] == "fail":
            return httpx2.Response(
                401,
                headers=headers,
                json={
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "bad key"},
                },
            )
        if body.get("stream"):
            content = "".join(
                f"event: {name}\ndata: {json.dumps(data)}\n\n"
                for name, data in STREAM_EVENTS
            )
            return httpx2.Response(
                200,
                headers={**headers, "content-type": "text/event-stream"},
                content=content.encode(),
            )
        return httpx2.Response(200, headers=headers, json=MESSAGE)

    return handler


def _client(*, log_id: str | None) -> Anthropic:
    return Anthropic(
        api_key="test-key",
        base_url="https://gateway.invalid/api/anthropic",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(_handler(log_id=log_id))),
    )


def _create(client, content: str):
    return client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=16,
        messages=[{"role": "user", "content": content}],
    )


def _linked_dropped(attrs: dict) -> list[str]:
    return [
        key
        for key in attrs
        if key.startswith(("gen_ai.usage.", "llm.usage.", "gen_ai.prompt.", "gen_ai.completion."))
        or key in ("traceloop.entity.input", "traceloop.entity.output")
    ]


def test_gateway_response_links_create_stream_and_error(captured):
    client = _client(log_id=GATEWAY_LOG_ID)
    try:
        assert _create(client, "hello").content[0].text == "hi"
        with client.messages.stream(
            model="claude-haiku-4-5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        ) as stream:
            assert stream.get_final_text() == "hi"
        with pytest.raises(AuthenticationError):
            _create(client, "fail")
    finally:
        client.close()

    chat_spans = [span for span in captured if span["name"] == "anthropic.chat"]
    assert len(chat_spans) == 3
    for span in chat_spans:
        assert span["attributes"][RESPAN_GATEWAY_LOG_ID] == GATEWAY_LOG_ID
        assert _linked_dropped(span["attributes"]) == []


@pytest.mark.asyncio
async def test_async_gateway_response_links_span(captured):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return _handler(log_id=GATEWAY_LOG_ID)(request)

    client = AsyncAnthropic(
        api_key="test-key",
        base_url="https://gateway.invalid/api/anthropic",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        await client.close()

    assert captured[0]["attributes"][RESPAN_GATEWAY_LOG_ID] == GATEWAY_LOG_ID
    assert _linked_dropped(captured[0]["attributes"]) == []


def test_direct_provider_keeps_usage_and_nothing_leaks(captured):
    gateway = _client(log_id=GATEWAY_LOG_ID)
    direct = _client(log_id=None)
    try:
        _create(gateway, "hello")
        _create(direct, "hello")
    finally:
        gateway.close()
        direct.close()

    linked, plain = captured
    assert linked["attributes"][RESPAN_GATEWAY_LOG_ID] == GATEWAY_LOG_ID
    assert RESPAN_GATEWAY_LOG_ID not in plain["attributes"]
    assert plain["attributes"]["gen_ai.usage.prompt_tokens"] == 4


def test_deactivate_restores_process_response():
    from anthropic._base_client import AsyncAPIClient, SyncAPIClient

    sync_original = SyncAPIClient._process_response
    async_original = AsyncAPIClient._process_response
    instrumentor = instrumentation.AnthropicInstrumentor()
    instrumentor.activate()
    assert SyncAPIClient._process_response is not sync_original
    instrumentor.deactivate()
    assert SyncAPIClient._process_response is sync_original
    assert AsyncAPIClient._process_response is async_original
