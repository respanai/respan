"""Exercise released Google SDK request/response conversion through HTTP transports."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from unittest.mock import patch

import httpx
import pytest
import requests as requests_library
from google import genai
from google.genai import errors, types
from google.oauth2.credentials import Credentials
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_google_genai import GoogleGenAIInstrumentor
from respan_instrumentation_google_genai._embeddings import build_embed_content_attrs
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE


@pytest.fixture()
def capture(monkeypatch):
    spans = []
    for module in ("_otel_emitter", "_embeddings"):
        monkeypatch.setattr(
            f"respan_instrumentation_google_genai.{module}.inject_span",
            lambda span: spans.append(span),
        )
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.activate()
    try:
        yield spans
    finally:
        instrumentor.deactivate()


def client_for(handler, *, vertex=False):
    kwargs = (
        {
            "vertexai": True,
            "project": "fixture",
            "location": "us-central1",
            "credentials": Credentials(token="fixture"),
        }
        if vertex
        else {"api_key": "fixture"}
    )
    client = genai.Client(**kwargs)
    # Keep the released SDK's Models, parsers, and serializers in both versions.
    if hasattr(client._api_client, "_httpx_client"):
        client._api_client._httpx_client.close()
        client._api_client._httpx_client = httpx.Client(
            transport=httpx.MockTransport(handler)
        )
        client._api_client._async_httpx_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
    else:

        def legacy_request(_session, method, url, **kwargs):
            response = handler(httpx.Request(method, url, content=kwargs.get("data")))
            result = requests_library.Response()
            result.status_code = response.status_code
            result._content = response.content
            result._content_consumed = True
            result.headers.update(response.headers)
            return result

        client._fixture_transport = patch("requests.Session.request", legacy_request)
        client._fixture_transport.start()
    return client


def close_client(client):
    if hasattr(client, "close"):
        client.close()
    if hasattr(client, "_fixture_transport"):
        client._fixture_transport.stop()


def test_sync_embeddings_preserve_vectors_parentage_and_omit_absent_usage(capture):
    requests = []
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "embeddings": [{"values": value} for value in vectors],
                "metadata": {"billableCharacterCount": 100},
            },
        )

    client = client_for(handler)
    provider = TracerProvider()
    try:
        with provider.get_tracer("test").start_as_current_span("parent") as parent:
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=["alpha", "beta"],
                config={"output_dimensionality": 2},
            )
        assert [item.values for item in result.embeddings] == vectors
        assert len(capture) == 1
        span = capture[0]
        attrs = span.attributes
        assert attrs[RESPAN_LOG_TYPE] == "embedding"
        assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "embedding"
        assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "text-embedding-004"
        assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == [
            "alpha",
            "beta",
        ]
        assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == vectors
        assert GEN_AI_USAGE_INPUT_TOKENS not in attrs
        assert SpanAttributes.LLM_USAGE_TOTAL_TOKENS not in attrs
        assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
        assert span.parent.span_id == parent.get_span_context().span_id
        assert all(row["outputDimensionality"] == 2 for row in requests[0]["requests"])
    finally:
        close_client(client)
        provider.shutdown()


def test_async_vertex_embeddings_use_reported_input_tokens(capture):
    def handler(request):
        assert request.url.path.endswith(":predict")
        return httpx.Response(
            200,
            json={
                "predictions": [
                    {
                        "embeddings": {
                            "values": [0.1, 0.2],
                            "statistics": {"token_count": 3, "truncated": False},
                        }
                    },
                    {
                        "embeddings": {
                            "values": [0.3, 0.4],
                            "statistics": {"token_count": 4, "truncated": False},
                        }
                    },
                ]
            },
        )

    async def run():
        client = client_for(handler, vertex=True)
        try:
            result = await client.aio.models.embed_content(
                model="text-embedding-005", contents=["alpha", "beta"]
            )
            assert len(result.embeddings) == 2
        finally:
            if hasattr(client.aio, "aclose"):
                await client.aio.aclose()
            close_client(client)

    asyncio.run(run())
    assert len(capture) == 1
    attrs = capture[0].attributes
    assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 7
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


def test_embedding_error_rethrows_and_emits_error_span(capture):
    client = client_for(
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "invalid embedding fixture",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
    )
    try:
        with pytest.raises(errors.ClientError):
            client.models.embed_content(model="text-embedding-004", contents="invalid")
    finally:
        close_client(client)
    assert len(capture) == 1
    assert capture[0].status.status_code is StatusCode.ERROR
    assert capture[0].attributes[RESPAN_LOG_TYPE] == "embedding"
    assert "invalid embedding fixture" in capture[0].status.description


@pytest.mark.parametrize("counts", [[None, 2], [2.5], [True], [-1], [float("nan")]])
def test_incomplete_or_invalid_statistics_do_not_become_usage(counts):
    response = {
        "embeddings": [
            {"values": [0.1], "statistics": {"token_count": count}} for count in counts
        ]
    }
    attrs = build_embed_content_attrs(
        request_kwargs={"model": "embedding", "contents": "text"},
        response_or_chunks=response,
    )
    assert GEN_AI_USAGE_INPUT_TOKENS not in attrs


def test_released_generation_and_streaming_methods_still_emit(capture):
    def response(text, *, usage=False):
        result = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]
        }
        if usage:
            result["usageMetadata"] = {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            }
        return result

    def handler(request):
        if "streamGenerateContent" in request.url.path:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text="".join(
                    "data: " + json.dumps(response(text, usage=index == 1)) + "\n\n"
                    for index, text in enumerate(["Hello ", "world"])
                ),
            )
        return httpx.Response(200, json=response("Hello world", usage=True))

    async def run():
        client = client_for(handler)
        try:
            assert (
                client.models.generate_content(
                    model="gemini-2.0-flash", contents="hello"
                ).text
                == "Hello world"
            )
            assert (
                "".join(
                    chunk.text
                    for chunk in client.models.generate_content_stream(
                        model="gemini-2.0-flash", contents="hello"
                    )
                )
                == "Hello world"
            )
            assert (
                await client.aio.models.generate_content(
                    model="gemini-2.0-flash", contents="hello"
                )
            ).text == "Hello world"
            stream = await client.aio.models.generate_content_stream(
                model="gemini-2.0-flash", contents="hello"
            )
            assert "".join([chunk.text async for chunk in stream]) == "Hello world"
        finally:
            if hasattr(client.aio, "aclose"):
                await client.aio.aclose()
            close_client(client)

    asyncio.run(run())
    assert len(capture) == 4
    assert all(span.attributes[RESPAN_LOG_TYPE] == "chat" for span in capture)
    assert all(
        span.attributes["gen_ai.completion.0.content"] == "Hello world"
        for span in capture
    )
    assert [span.attributes[SpanAttributes.LLM_IS_STREAMING] for span in capture] == [
        False,
        True,
        False,
        True,
    ]


@pytest.mark.skipif(
    version("google-genai").startswith("1."),
    reason="Gemini Embedding 2 normalization is a current SDK feature",
)
def test_current_multimodal_embedding_retains_image_input(capture):
    client = client_for(
        lambda request: httpx.Response(
            200, json={"embeddings": [{"values": [0.1, 0.2]}]}
        )
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text="image"),
                types.Part(
                    file_data=types.FileData(
                        file_uri="gs://fixture/image.jpg", mime_type="image/jpeg"
                    )
                ),
            ],
        )
    ]
    try:
        client.models.embed_content(
            model="gemini-embedding-2-preview", contents=contents
        )
    finally:
        close_client(client)
    parsed = json.loads(capture[0].attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert parsed[0]["parts"][1]["file_data"]["file_uri"] == "gs://fixture/image.jpg"


def test_deactivate_restores_embedding_methods():
    from google.genai.models import AsyncModels, Models

    original = (Models.embed_content, AsyncModels.embed_content)
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.activate()
    assert Models.embed_content is not original[0]
    assert AsyncModels.embed_content is not original[1]
    instrumentor.deactivate()
    assert (Models.embed_content, AsyncModels.embed_content) == original


@pytest.mark.skipif(
    version("google-genai").startswith("1."), reason="Exa search is a current SDK tool"
)
def test_current_builtin_tool_definition_is_preserved():
    from respan_instrumentation_google_genai._translator import extract_tools

    config = types.GenerateContentConfig(tools=[types.Tool(exa_ai_search={})])
    assert extract_tools(config) == [{"type": "exa_ai_search", "exa_ai_search": {}}]


def test_released_tool_history_uses_canonical_roles_and_input_calls(capture):
    client = client_for(
        lambda request: httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Sunny"}]}}
                ]
            },
        )
    )
    try:
        client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text="Weather?")]),
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="weather", args={"city": "Paris"}
                            )
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="weather", response={"result": "Sunny"}
                            )
                        )
                    ],
                ),
            ],
        )
    finally:
        close_client(client)
    attrs = capture[0].attributes
    assert attrs["gen_ai.prompt.1.role"] == "assistant"
    assert attrs["gen_ai.prompt.2.role"] == "tool"
    assert (
        json.loads(attrs["gen_ai.prompt.1.tool_calls"])[0]["function"]["name"]
        == "weather"
    )
    assert "gen_ai.completion.0.tool_calls" not in attrs


@pytest.mark.parametrize("override", [False, True])
def test_embedding_content_switch_and_context_override(capture, monkeypatch, override):
    from opentelemetry import context

    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    client = client_for(
        lambda request: httpx.Response(
            200, json={"embeddings": [{"values": [0.1, 0.2]}]}
        )
    )
    token = context.attach(
        context.set_value("override_enable_content_tracing", override)
    )
    try:
        client.models.embed_content(
            model="text-embedding-004", contents="private probe text"
        )
    finally:
        context.detach(token)
        close_client(client)
    attrs = capture[0].attributes
    assert (SpanAttributes.TRACELOOP_ENTITY_INPUT in attrs) is override
    assert (SpanAttributes.TRACELOOP_ENTITY_OUTPUT in attrs) is override
    assert ("private probe text" in str(attrs)) is override
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "text-embedding-004"


def test_two_embedding_owners_release_only_the_final_patch(monkeypatch):
    from google.genai.models import AsyncModels, Models

    captured = []
    monkeypatch.setattr(
        "respan_instrumentation_google_genai._embeddings.inject_span",
        lambda span: captured.append(span),
    )
    original = (Models.embed_content, AsyncModels.embed_content)
    first, second = GoogleGenAIInstrumentor(), GoogleGenAIInstrumentor()
    client = client_for(
        lambda request: httpx.Response(200, json={"embeddings": [{"values": [0.1]}]})
    )
    first.activate()
    second.activate()
    try:
        first.deactivate()
        client.models.embed_content(model="text-embedding-004", contents="still traced")
        assert len(captured) == 1
        assert second._is_instrumented
    finally:
        first.deactivate()
        second.deactivate()
        close_client(client)
    assert (Models.embed_content, AsyncModels.embed_content) == original
