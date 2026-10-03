"""Exercise the released Cohere client, including its generated response models."""

import asyncio
import json

import cohere
import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from respan_instrumentation_cohere import CohereInstrumentor
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


@pytest.fixture
def traced(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    instrumentor = CohereInstrumentor()
    instrumentor.activate()
    assert instrumentor._is_instrumented
    yield provider, exporter
    instrumentor.deactivate()
    provider.shutdown()


def _handler(request):
    body = json.loads(request.content)
    if request.url.path.endswith("/embed"):
        return httpx.Response(
            200,
            json={
                "id": "embed",
                "embeddings": {"float": [[0.1, 0.2]]},
                "texts": ["hello"],
                "meta": {"billed_units": {"input_tokens": 2}},
                "response_type": "embeddings_by_type",
            },
        )
    if request.url.path.endswith("/rerank"):
        return httpx.Response(
            200,
            json={
                "id": "rank",
                "results": [{"index": 0, "relevance_score": 0.99}],
                "meta": {"billed_units": {"search_units": 1}},
            },
        )
    if body["messages"][0]["content"] == "fail":
        return httpx.Response(400, json={"message": "expected fixture failure"})
    return httpx.Response(
        200,
        json={
            "id": "chat",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "tool_calls": [
                    {
                        "id": "weather-1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            "usage": {"billed_units": {"input_tokens": 2, "output_tokens": 3}},
        },
    )


@pytest.mark.skipif(
    not hasattr(cohere, "ClientV2"), reason="V2 introduced after Cohere 5.0"
)
def test_released_sync_async_chat_tools_embeddings_rerank_and_errors(traced):
    provider, exporter = traced
    client = cohere.ClientV2(
        api_key="fixture-secret",
        httpx_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        response = client.chat(
            model="command-a",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
        )
        assert response.message.content[0].text == "hello"
        client.embed(
            model="embed-v4.0",
            texts=["hello"],
            input_type="search_document",
            embedding_types=["float"],
        )
        client.rerank(model="rerank-v3.5", query="hello", documents=["hello"])
        with pytest.raises(cohere.BadRequestError):
            client.chat(
                model="command-a", messages=[{"role": "user", "content": "fail"}]
            )

        async def run():
            client = cohere.AsyncClientV2(
                api_key="fixture-secret",
                httpx_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
            )
            await client.chat(
                model="command-a", messages=[{"role": "user", "content": "hello"}]
            )

        asyncio.run(run())
    spans = [s for s in exporter.get_finished_spans() if s.name != "parent"]
    assert len(spans) == 5
    assert all(s.parent.span_id == parent.context.span_id for s in spans)
    chats = [
        s
        for s in spans
        if s.attributes[RESPAN_LOG_TYPE] == "chat"
        and s.status.status_code != trace.StatusCode.ERROR
    ]
    assert len(chats) == 2
    for span in chats:
        assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 2
        assert span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 3
        assert (
            json.loads(
                span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"]
            )[0]["function"]["name"]
            == "weather"
        )
    embedding = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "embedding")
    assert json.loads(embedding.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[
        "float_"
    ] == [[0.1, 0.2]]
    assert embedding.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 2
    assert SpanAttributes.LLM_USAGE_COMPLETION_TOKENS not in embedding.attributes
    assert "fixture-secret" not in str([dict(s.attributes) for s in spans])


def _events():
    return [
        {
            "type": "message-start",
            "id": "stream-1",
            "delta": {
                "message": {
                    "role": "assistant",
                    "content": [],
                    "tool_calls": [],
                    "tool_plan": "",
                }
            },
        },
        {
            "type": "content-start",
            "index": 0,
            "delta": {"message": {"content": {"type": "text", "text": ""}}},
        },
        {
            "type": "content-delta",
            "index": 0,
            "delta": {"message": {"content": {"text": "streamed answer"}}},
        },
        {"type": "content-end", "index": 0},
        {
            "type": "message-end",
            "delta": {
                "finish_reason": "COMPLETE",
                "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 7}},
            },
        },
    ]


def _sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.skipif(
    not hasattr(cohere, "ClientV2"), reason="V2 introduced after Cohere 5.0"
)
@pytest.mark.parametrize(
    ("asynchronous", "outcome"),
    [
        (False, "success"),
        (True, "success"),
        (False, "error"),
        (True, "error"),
        (False, "close"),
        (True, "close"),
        (True, "cancel"),
    ],
)
def test_released_streams_finish_once_and_preserve_errors(
    traced, asynchronous, outcome
):
    _, exporter = traced
    closed = []

    class SyncBody(httpx.SyncByteStream):
        def __iter__(self):
            yield _sse(_events()[:1])
            if outcome == "error":
                raise httpx.ReadError("fixture stream disconnected")
            yield _sse(_events()[1:])

        def close(self):
            closed.append(True)

    class AsyncBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _sse(_events()[:1])
            if outcome == "cancel":
                raise asyncio.CancelledError("fixture cancelled")
            if outcome == "error":
                raise httpx.ReadError("fixture stream disconnected")
            yield _sse(_events()[1:])

        async def aclose(self):
            closed.append(True)

    def handler(request):
        return httpx.Response(
            200,
            stream=AsyncBody() if asynchronous else SyncBody(),
            headers={"content-type": "text/event-stream"},
        )

    async def run_async():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as transport:
            client = cohere.AsyncClientV2(api_key="fixture", httpx_client=transport)
            stream = client.chat_stream(
                model="command-a", messages=[{"role": "user", "content": "stream"}]
            )
            if outcome == "close":
                await anext(stream)
                await stream.aclose()
            elif outcome == "cancel":
                with pytest.raises(asyncio.CancelledError, match="fixture cancelled"):
                    _ = [event async for event in stream]
            elif outcome == "error":
                with pytest.raises(
                    httpx.ReadError, match="fixture stream disconnected"
                ):
                    _ = [event async for event in stream]
            else:
                assert len([event async for event in stream]) == 5

    if asynchronous:
        asyncio.run(run_async())
    else:
        with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            client = cohere.ClientV2(api_key="fixture", httpx_client=transport)
            stream = client.chat_stream(
                model="command-a", messages=[{"role": "user", "content": "stream"}]
            )
            if outcome == "close":
                next(stream)
                stream.close()
            elif outcome == "error":
                with pytest.raises(
                    httpx.ReadError, match="fixture stream disconnected"
                ):
                    list(stream)
            else:
                assert len(list(stream)) == 5
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert closed
    if outcome == "success":
        assert (
            json.loads(
                spans[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
            )[0]["text"]
            == "streamed answer"
        )
        assert spans[0].attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 5
    elif outcome == "cancel":
        assert spans[0].status.status_code == trace.StatusCode.ERROR
        assert "fixture cancelled" in spans[0].status.description
    elif outcome == "error":
        assert spans[0].status.status_code == trace.StatusCode.ERROR
        assert "fixture stream disconnected" in spans[0].status.description


@pytest.mark.skipif(
    not hasattr(cohere, "ClientV2"), reason="V2 introduced after Cohere 5.0"
)
def test_content_switch_hides_embedding_and_rerank_payloads(traced, monkeypatch):
    _, exporter = traced
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    with httpx.Client(transport=httpx.MockTransport(_handler)) as transport:
        client = cohere.ClientV2(api_key="fixture", httpx_client=transport)
        client.embed(
            model="embed-v4.0",
            texts=["private text"],
            input_type="search_document",
            embedding_types=["float"],
        )
        client.rerank(
            model="rerank-v3.5", query="private query", documents=["private document"]
        )
    assert len(exporter.get_finished_spans()) == 2
    for span in exporter.get_finished_spans():
        assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert "private" not in str(dict(span.attributes))


@pytest.mark.skipif(
    not hasattr(cohere, "ClientV2"), reason="V2 introduced after Cohere 5.0"
)
def test_multiple_owners_and_preconstructed_upstream_are_safe(monkeypatch):
    from opentelemetry.instrumentation.cohere import CohereInstrumentor as Upstream
    from respan_instrumentation_cohere._instrumentation import (
        _compatible_instrumentor_class,
    )

    upstream = Upstream()
    compatible = _compatible_instrumentor_class(Upstream)()
    assert type(compatible) is not type(upstream)
    assert compatible.instrumentation_dependencies() == ("cohere >=5.0.0, <8",)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    first, second = CohereInstrumentor(), CohereInstrumentor()
    original = cohere.ClientV2.chat
    first.activate()
    second.activate()
    first.deactivate()
    try:
        with httpx.Client(transport=httpx.MockTransport(_handler)) as transport:
            client = cohere.ClientV2(api_key="fixture", httpx_client=transport)
            client.chat(
                model="command-a", messages=[{"role": "user", "content": "hello"}]
            )
        assert len(exporter.get_finished_spans()) == 1
    finally:
        second.deactivate()
        provider.shutdown()
    assert cohere.ClientV2.chat is original


def test_released_v1_chat_and_async_embedding(traced):
    _, exporter = traced

    def handler(request):
        if request.url.path.endswith("/embed"):
            return httpx.Response(
                200,
                json={
                    "id": "embed",
                    "embeddings": [[0.1, 0.2]],
                    "texts": ["hello"],
                    "meta": {"billed_units": {"input_tokens": 2}},
                    "response_type": "embeddings_floats",
                },
            )
        return httpx.Response(
            200,
            json={
                "response_id": "chat-v1",
                "text": "v1 answer",
                "generation_id": "generation-v1",
                "chat_history": [],
                "finish_reason": "COMPLETE",
                "meta": {"billed_units": {"input_tokens": 2, "output_tokens": 3}},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = cohere.Client(api_key="fixture", httpx_client=transport)
        assert client.chat(model="command-r", message="hello").text == "v1 answer"

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as transport:
            client = cohere.AsyncClient(api_key="fixture", httpx_client=transport)
            await client.embed(
                model="embed-english-v3.0",
                texts=["hello"],
                input_type="search_document",
            )

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    chat = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "chat")
    assert chat.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "v1 answer"
    embedding = next(s for s in spans if s.attributes[RESPAN_LOG_TYPE] == "embedding")
    assert json.loads(embedding.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        [0.1, 0.2]
    ]


@pytest.mark.skipif(
    not hasattr(cohere, "ClientV2"), reason="V2 introduced after Cohere 5.0"
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("primary", [False, True])
def test_released_stream_cleanup_preserves_primary_failures(asynchronous, primary):
    from respan_instrumentation_cohere._streaming import _guard_async, _guard_sync

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("cleanup-test").start_span("stream")

    class SyncBody(httpx.SyncByteStream):
        def __iter__(self):
            yield _sse(_events())

        def close(self):
            raise RuntimeError("transport cleanup failed")

    class AsyncBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _sse(_events())

        async def aclose(self):
            raise RuntimeError("transport cleanup failed")

    def handler(request):
        return httpx.Response(
            200,
            stream=AsyncBody() if asynchronous else SyncBody(),
            headers={"content-type": "text/event-stream"},
        )

    def process(_span, _logger, _type, response):
        yield next(response)
        if primary:
            raise ValueError("primary stream failure")

    async def aprocess(_span, _logger, _type, response):
        yield await anext(response)
        if primary:
            raise asyncio.CancelledError("primary stream cancellation")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as transport:
            client = cohere.AsyncClientV2(api_key="fixture", httpx_client=transport)
            response = client.chat_stream(
                model="command-a", messages=[{"role": "user", "content": "stream"}]
            )
            exception = asyncio.CancelledError if primary else RuntimeError
            message = (
                "primary stream cancellation" if primary else "transport cleanup failed"
            )
            with pytest.raises(exception, match=message):
                _ = [
                    x async for x in _guard_async(aprocess)(span, None, None, response)
                ]

    try:
        if asynchronous:
            asyncio.run(run())
        else:
            with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
                client = cohere.ClientV2(api_key="fixture", httpx_client=transport)
                response = client.chat_stream(
                    model="command-a", messages=[{"role": "user", "content": "stream"}]
                )
                exception = ValueError if primary else RuntimeError
                message = (
                    "primary stream failure" if primary else "transport cleanup failed"
                )
                with pytest.raises(exception, match=message):
                    list(_guard_sync(process)(span, None, None, response))
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == trace.StatusCode.ERROR
        assert ("primary stream" if primary else "transport cleanup failed") in spans[
            0
        ].status.description
    finally:
        provider.shutdown()
