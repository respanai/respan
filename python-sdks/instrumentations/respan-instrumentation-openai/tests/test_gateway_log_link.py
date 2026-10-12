"""A call answered by the Respan gateway links its span to the gateway log row.

The gateway returns the unique_id of the log row it wrote in ``X-Respan-Log-Id``.
That row already carries the call's usage and cost, so the span carries the id
(``respan.gateway.log_id``) and no usage of its own.
"""

from __future__ import annotations

import json

import httpx2
import pytest
from openai import AsyncOpenAI, AuthenticationError, OpenAI
from respan_instrumentation_openai import _instrumentation as instrumentation
from respan_instrumentation_openai import _otel_emitter as emitter
from respan_instrumentation_openai._instrumentation import OpenAIInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_GATEWAY_LOG_ID
from respan_sdk.constants.tracing_constants import RESPAN_LOG_ID_RESPONSE_HEADER

GATEWAY_LOG_ID = "4f1c1b0e5c8a4d3e9b7a6f5e4d3c2b1a"
USAGE = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
CHAT = {
    "id": "chat_1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4.1-nano",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": USAGE,
}
STREAM_CHUNKS = [
    {
        "id": "chat_stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-nano",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
    },
    {
        "id": "chat_stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-nano",
        "choices": [],
        "usage": USAGE,
    },
]


@pytest.fixture(autouse=True)
def clean_instrumentation(monkeypatch):
    instrumentation._remove_patches()
    monkeypatch.setattr(instrumentation, "_REFCOUNT", 0)
    monkeypatch.setattr(
        OpenAIInstrumentor,
        "_is_respan_tracing_enabled",
        staticmethod(lambda: True),
    )
    yield
    instrumentation._remove_patches()
    instrumentation._REFCOUNT = 0


@pytest.fixture
def captured(monkeypatch):
    spans = []
    monkeypatch.setattr(emitter, "inject_span", lambda span: spans.append(span))
    return spans


def _handler(*, log_id: str | None):
    headers = {RESPAN_LOG_ID_RESPONSE_HEADER: log_id} if log_id else {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if body["messages"][-1]["content"] == "fail":
            return httpx2.Response(
                401,
                headers=headers,
                json={"error": {"message": "bad key", "type": "invalid_request_error"}},
            )
        if body.get("stream"):
            content = "".join(f"data: {json.dumps(c)}\n\n" for c in STREAM_CHUNKS)
            return httpx2.Response(
                200,
                headers={**headers, "content-type": "text/event-stream"},
                content=(content + "data: [DONE]\n\n").encode(),
            )
        return httpx2.Response(200, headers=headers, json=CHAT)

    return handler


def _sync_client(*, log_id: str | None) -> OpenAI:
    return OpenAI(
        api_key="test-key",
        base_url="https://gateway.invalid/api",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(_handler(log_id=log_id))),
    )


def _async_client(*, log_id: str | None) -> AsyncOpenAI:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return _handler(log_id=log_id)(request)

    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://gateway.invalid/api",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _usage_keys(span) -> list[str]:
    return [
        key
        for key in span.attributes
        if key.startswith(("gen_ai.usage.", "llm.usage."))
    ]


def _chat(client, content: str, **kwargs):
    return client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": content}],
        **kwargs,
    )


def test_gateway_response_links_span_and_drops_usage(captured):
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _sync_client(log_id=GATEWAY_LOG_ID)
    try:
        chat = _chat(client, "hello")
        stream_text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in _chat(
                client, "hello", stream=True, stream_options={"include_usage": True}
            )
            if chunk.choices
        )
        with pytest.raises(AuthenticationError):
            _chat(client, "fail")
    finally:
        client.close()
        instrumentor.deactivate()

    assert chat.choices[0].message.content == "hi"
    assert stream_text == "hi"
    assert len(captured) == 3
    for span in captured:
        assert span.attributes[RESPAN_GATEWAY_LOG_ID] == GATEWAY_LOG_ID
        assert _usage_keys(span) == []
    # A link only: no content either (the server would estimate tokens from it).
    for span in captured:
        assert not [
            key
            for key in span.attributes
            if key.startswith(("gen_ai.prompt.", "gen_ai.completion."))
            or key in ("traceloop.entity.input", "traceloop.entity.output")
        ]
    assert captured[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert captured[2].attributes["status_code"] == 401


@pytest.mark.asyncio
async def test_async_gateway_response_links_span(captured):
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _async_client(log_id=GATEWAY_LOG_ID)
    try:
        await client.chat.completions.create(
            model="gpt-4.1-nano", messages=[{"role": "user", "content": "hello"}]
        )
        stream = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for _ in stream:
            pass
    finally:
        await client.close()
        instrumentor.deactivate()

    assert len(captured) == 2
    for span in captured:
        assert span.attributes[RESPAN_GATEWAY_LOG_ID] == GATEWAY_LOG_ID
        assert _usage_keys(span) == []


def test_direct_provider_response_keeps_usage(captured):
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    client = _sync_client(log_id=None)
    try:
        _chat(client, "hello")
    finally:
        client.close()
        instrumentor.deactivate()

    assert len(captured) == 1
    assert RESPAN_GATEWAY_LOG_ID not in captured[0].attributes
    assert captured[0].attributes["llm.usage.total_tokens"] == 6


def test_log_id_does_not_leak_into_the_next_call(captured):
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    gateway = _sync_client(log_id=GATEWAY_LOG_ID)
    direct = _sync_client(log_id=None)
    try:
        _chat(gateway, "hello")
        _chat(direct, "hello")
    finally:
        gateway.close()
        direct.close()
        instrumentor.deactivate()

    assert captured[0].attributes[RESPAN_GATEWAY_LOG_ID] == GATEWAY_LOG_ID
    assert RESPAN_GATEWAY_LOG_ID not in captured[1].attributes
    assert captured[1].attributes["llm.usage.total_tokens"] == 6


def test_deactivate_restores_process_response():
    from openai._base_client import AsyncAPIClient, SyncAPIClient

    sync_original = SyncAPIClient._process_response
    async_original = AsyncAPIClient._process_response
    instrumentor = OpenAIInstrumentor()
    instrumentor.activate()
    assert SyncAPIClient._process_response is not sync_original
    instrumentor.deactivate()
    assert SyncAPIClient._process_response is sync_original
    assert AsyncAPIClient._process_response is async_original
