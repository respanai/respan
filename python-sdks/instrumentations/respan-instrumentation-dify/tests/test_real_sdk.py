import asyncio
import json

import dify_client.client as sdk
import pytest
import requests
from opentelemetry import context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from respan_instrumentation_dify import DifyInstrumentor
from respan_instrumentation_dify import _otel_emitter as emitter


@pytest.fixture
def captured(monkeypatch):
    spans = []
    monkeypatch.setattr(emitter, "inject_span", lambda *, span: spans.append(span))
    return spans


@pytest.fixture
def client(monkeypatch):
    payload = {
        "answer": "fixture answer",
        "model": "fixture-dify-model",
        "metadata": {"usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    }

    def content():
        value = payload.get("body", payload)
        return value.encode() if isinstance(value, str) else json.dumps(value).encode()

    def request(*args, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = content()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests, "request", request)
    instance = sdk.ChatClient("fixture-dify-key")
    if hasattr(instance, "_client"):
        import httpx

        instance._client.close()
        instance._client = httpx.Client(
            base_url="https://dify.fixture/v1",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=content())
            ),
        )
    yield instance, payload
    close = getattr(instance, "close", None)
    if close:
        close()


def call(client, **kwargs):
    return client.create_chat_message(
        inputs={}, query="fixture question", user="fixture-user", **kwargs
    )


def test_real_sdk_content_opt_out_retains_model_and_usage(
    client, captured, monkeypatch
):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    instrumentor = DifyInstrumentor()
    instrumentor.activate()
    try:
        assert call(client[0]).json()["answer"] == "fixture answer"
    finally:
        instrumentor.deactivate()
    attrs = captured[0].attributes
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "fixture-dify-model"
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 3
    assert not any(
        k.startswith(
            (
                "gen_ai.prompt.",
                "gen_ai.completion.",
                "traceloop.entity.input",
                "traceloop.entity.output",
            )
        )
        for k in attrs
    )


def test_real_stream_keeps_initiating_privacy_policy(client, captured):
    client[1].clear()
    client[1]["body"] = 'event: message\ndata: {"answer":"private output"}\n\n'
    first = DifyInstrumentor(include_content=False)
    first.activate()
    response = call(client[0], response_mode="streaming")
    first.deactivate()
    second = DifyInstrumentor(include_content=True)
    second.activate()
    try:
        assert list(response.iter_lines())
        response.close()
    finally:
        second.deactivate()
    assert len(captured) == 1
    assert "private output" not in json.dumps(dict(captured[0].attributes))


def test_real_sdk_retained_foreign_wrapper_does_not_reactivate_old_owner(
    client, captured, monkeypatch
):
    original = sdk.DifyClient._send_request
    monkeypatch.setattr(sdk.DifyClient, "_send_request", original)
    first = DifyInstrumentor()
    first.activate()
    owned = sdk.DifyClient._send_request

    def foreign(self, *args, **kwargs):
        return owned(self, *args, **kwargs)

    sdk.DifyClient._send_request = foreign
    first.deactivate()
    call(client[0])
    assert captured == []
    second = DifyInstrumentor()
    second.activate()
    try:
        call(client[0])
        assert len(captured) == 1
    finally:
        second.deactivate()
        sdk.DifyClient._send_request = original


@pytest.mark.parametrize(
    "payload,stream",
    [
        ('data: {"event":"error","message":"controlled SSE failure"}\n\n', True),
        ({"data": {"status": "failed", "error": "controlled workflow failure"}}, False),
    ],
)
def test_real_sdk_protocol_errors_mark_otel_error(client, captured, payload, stream):
    client[1].clear()
    client[1]["body"] = payload
    instrumentor = DifyInstrumentor()
    instrumentor.activate()
    try:
        result = client[0]._send_request(
            "POST", "/workflows/run", json={}, stream=stream
        )
        if stream:
            list(result.iter_lines())
            result.close()
    finally:
        instrumentor.deactivate()
    assert len(captured) == 1
    assert captured[0].status.status_code == StatusCode.ERROR
    assert "controlled" in captured[0].status.description
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in captured[0].attributes


def test_real_async_cancellation_keeps_error_and_propagates(captured):
    module = pytest.importorskip("dify_client.async_client")
    import httpx

    async def run():
        client = module.AsyncChatClient("fixture-dify-key")
        await client._client.aclose()

        async def cancel(request):
            raise asyncio.CancelledError()

        client._client = httpx.AsyncClient(
            base_url="https://dify.fixture/v1", transport=httpx.MockTransport(cancel)
        )
        instrumentor = DifyInstrumentor()
        instrumentor.activate()
        try:
            with pytest.raises(asyncio.CancelledError):
                await client.create_chat_message(
                    inputs={}, query="fixture", user="fixture"
                )
        finally:
            instrumentor.deactivate()
            await client.aclose()

    asyncio.run(run())
    assert len(captured) == 1
    assert captured[0].status.status_code == StatusCode.ERROR
    assert captured[0].status.description == "CancelledError"


def test_current_source_extended_async_client_supports_respan_params(captured):
    module = pytest.importorskip("dify_client.async_client")
    import httpx

    async def run():
        client = module.AsyncAdvancedAppClient("fixture-dify-key")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://dify.fixture/v1",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"data": []})
            ),
        )
        instrumentor = DifyInstrumentor()
        instrumentor.activate()
        try:
            result = await client.get_app_environment_variables(
                "fixture-app", respan_params={"metadata": {"scenario": "extended"}}
            )
            assert result.json() == {"data": []}
        finally:
            instrumentor.deactivate()
            await client.aclose()

    asyncio.run(run())
    assert len(captured) == 1
    assert (
        json.loads(captured[0].attributes["respan.metadata"])["scenario"] == "extended"
    )


def test_sse_failure_has_no_generated_output(client, captured):
    client[1].clear()
    client[1]["body"] = 'data: {"event":"error","message":"controlled SSE failure"}\n\n'
    instrumentor = DifyInstrumentor()
    instrumentor.activate()
    try:
        response = call(client[0], response_mode="streaming")
        list(response.iter_lines())
        response.close()
    finally:
        instrumentor.deactivate()
    assert captured[0].status.status_code == StatusCode.ERROR
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in captured[0].attributes
    assert f"{SpanAttributes.LLM_COMPLETIONS}.0.content" not in captured[0].attributes


def test_real_sdk_suppression_applies_to_blocking_and_stream(client, captured):
    instrumentor = DifyInstrumentor()
    instrumentor.activate()
    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        call(client[0])
        response = call(client[0], response_mode="streaming")
    finally:
        context.detach(token)
    try:
        list(response.iter_lines())
        response.close()
        assert captured == []
    finally:
        instrumentor.deactivate()


@pytest.mark.parametrize("method", ["close", "__exit__"])
def test_native_stream_close_failure_is_preserved(client, captured, method):
    instrumentor = DifyInstrumentor()
    instrumentor.activate()
    try:
        response = call(client[0], response_mode="streaming")

        def fail(*args):
            raise RuntimeError("native close failed")

        setattr(response._response, method, fail)
        with pytest.raises(RuntimeError, match="native close failed"):
            getattr(response, method)(
                *([] if method == "close" else [None, None, None])
            )
        assert len(captured) == 1
        assert captured[0].status.status_code == StatusCode.ERROR
    finally:
        instrumentor.deactivate()


@pytest.mark.parametrize("method", ["aclose", "__aexit__"])
def test_async_stream_close_failure_is_preserved(captured, method):
    module = pytest.importorskip("dify_client.async_client")
    import httpx

    async def run():
        client = module.AsyncChatClient("fixture-key")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url="https://dify.fixture/v1",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
        )
        instrumentor = DifyInstrumentor()
        instrumentor.activate()
        try:
            response = await client.create_chat_message(
                inputs={}, query="fixture", user="fixture", response_mode="streaming"
            )

            async def fail(*args):
                raise RuntimeError("native async close failed")

            setattr(response._response, method, fail)
            with pytest.raises(RuntimeError, match="native async close failed"):
                await getattr(response, method)(
                    *([] if method == "aclose" else [None, None, None])
                )
        finally:
            instrumentor.deactivate()
            await client.aclose()

    asyncio.run(run())
    assert len(captured) == 1
    assert captured[0].status.status_code == StatusCode.ERROR
