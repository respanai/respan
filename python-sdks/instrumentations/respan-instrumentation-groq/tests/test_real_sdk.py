"""Released Groq SDK contract checks over deterministic HTTP responses."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from types import SimpleNamespace

import httpx
import pytest
from groq import AsyncGroq, Groq, InternalServerError
from openinference.instrumentation import TraceConfig
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respan_instrumentation_groq import GroqInstrumentor
from respan_instrumentation_groq._streaming import _AsyncStreamProxy, _SyncStreamProxy
from respan_tracing.core.tracer import RespanTracer

MODEL = "fixture-groq-model"
MESSAGES = [{"role": "user", "content": "synthetic hello"}]
CALL = {
    "id": "call_weather",
    "type": "function",
    "function": {"name": "weather", "arguments": '{"city":"Tokyo"}'},
}
TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def response(request):
    body = json.loads(request.content)
    if body.get("model") == "fixture-error":
        return httpx.Response(
            503,
            json={
                "error": {"message": "synthetic unavailable", "type": "server_error"}
            },
        )
    tool = bool(body.get("tools"))
    if body.get("stream"):
        deltas = (
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_weather",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":'},
                        }
                    ],
                },
                {"tool_calls": [{"index": 0, "function": {"arguments": '"Tokyo"}'}}]},
            ]
            if tool
            else [{"role": "assistant", "content": "hello "}, {"content": "world"}]
        )
        frames = [
            {
                "id": "chat-fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            for delta in deltas
        ]
        frames.append(
            {
                "id": "chat-fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [],
                "x_groq": {
                    "id": "fixture",
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                },
            }
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
            + "data: [DONE]\n\n",
        )
    return httpx.Response(
        200,
        json={
            "id": "chat-fixture",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None if tool else "hello world",
                        "tool_calls": [CALL] if tool else None,
                    },
                    "finish_reason": "tool_calls" if tool else "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    )


@pytest.fixture
def telemetry(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(RespanTracer, "_instance", SimpleNamespace(is_enabled=True))
    adapter = GroqInstrumentor()
    adapter.activate()
    assert adapter._is_instrumented
    yield provider, exporter, adapter
    adapter.deactivate()
    provider.shutdown()


def client():
    # Explicit http_client also makes SDK 0.9 work with modern httpx.
    return Groq(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(response)),
    )


def async_client():
    return AsyncGroq(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(response)),
    )


def assert_span(span, *, content="hello world", usage=True):
    attrs = span.attributes
    assert attrs["gen_ai.system"] == "groq"
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["llm.request.type"] == "chat"
    assert attrs["respan.entity.log_type"] == "chat"
    assert attrs.get("gen_ai.completion.0.content", "") == content
    if usage:
        assert (
            attrs["gen_ai.usage.input_tokens"]
            == attrs["gen_ai.usage.prompt_tokens"]
            == 4
        )
        assert (
            attrs["gen_ai.usage.output_tokens"]
            == attrs["gen_ai.usage.completion_tokens"]
            == 2
        )
    else:
        assert "gen_ai.usage.input_tokens" not in attrs
    assert not any(
        key in attrs
        for key in (
            "tools",
            "tool_calls",
            "model",
            "respan.span.tools",
            "traceloop.span.kind",
        )
    )
    assert "Omit object" not in str(attrs)


@pytest.mark.parametrize("mode", ["normal", "raw", "streaming_response"])
@pytest.mark.parametrize("stream", [False, True])
def test_real_sync_helpers(telemetry, mode, stream):
    provider, exporter, _ = telemetry
    with (
        client() as sdk,
        provider.get_tracer("test").start_as_current_span("parent") as parent,
    ):
        resource = sdk.chat.completions
        if mode == "raw":
            response_ = resource.with_raw_response.create(
                model=MODEL, messages=MESSAGES, stream=stream
            )
            assert type(response_).__module__.startswith("groq")
            result = response_.parse()
            assert response_.parse() is result
        elif mode == "streaming_response":
            manager = resource.with_streaming_response.create(
                model=MODEL, messages=MESSAGES, stream=stream
            )
            assert not exporter.get_finished_spans()
            response_ = manager.__enter__()
            result = response_.parse()
        else:
            result = resource.create(model=MODEL, messages=MESSAGES, stream=stream)
        if stream:
            assert not exporter.get_finished_spans()
            assert len(list(result)) == 3
            result.close()
        if mode != "normal":
            response_.close()
        assert trace.get_current_span() is parent
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].parent.span_id == parent.get_span_context().span_id
        assert_span(spans[0])


@pytest.mark.parametrize("mode", ["normal", "raw", "streaming_response"])
@pytest.mark.parametrize("stream", [False, True])
def test_real_async_helpers(telemetry, mode, stream):
    provider, exporter, _ = telemetry

    async def run():
        async with async_client() as sdk:
            with provider.get_tracer("test").start_as_current_span("parent") as parent:
                resource = sdk.chat.completions
                if mode == "raw":
                    response_ = await resource.with_raw_response.create(
                        model=MODEL, messages=MESSAGES, stream=stream
                    )
                    result = await response_.parse()
                elif mode == "streaming_response":
                    manager = resource.with_streaming_response.create(
                        model=MODEL, messages=MESSAGES, stream=stream
                    )
                    assert not exporter.get_finished_spans()
                    response_ = await manager.__aenter__()
                    result = await response_.parse()
                else:
                    result = await resource.create(
                        model=MODEL, messages=MESSAGES, stream=stream
                    )
                if stream:
                    assert not exporter.get_finished_spans()
                    assert len([chunk async for chunk in result]) == 3
                    await result.close()
                if mode != "normal":
                    await response_.close()
                assert trace.get_current_span() is parent
                spans = exporter.get_finished_spans()
                assert len(spans) == 1
                assert spans[0].parent.span_id == parent.get_span_context().span_id
                assert_span(spans[0])

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_current_tool_calls_and_history(telemetry, stream):
    _, exporter, _ = telemetry
    with client() as sdk:
        result = sdk.chat.completions.create(
            model=MODEL, messages=MESSAGES, tools=[TOOL], stream=stream
        )
        if stream:
            list(result)
        sdk.chat.completions.create(
            model=MODEL,
            messages=[
                *MESSAGES,
                {"role": "assistant", "content": None, "tool_calls": [CALL]},
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "name": "weather",
                    "content": "sunny",
                },
            ],
        )
    first, second = exporter.get_finished_spans()
    assert_span(first, content="")
    assert (
        json.loads(first.attributes["gen_ai.completion.0.tool_calls"])[0]["function"]
        == CALL["function"]
    )
    assert (
        json.loads(first.attributes["llm.request.functions"])[0]["function"]["name"]
        == "weather"
    )
    assert "gen_ai.completion.0.tool_calls" not in second.attributes
    assert (
        json.loads(second.attributes["gen_ai.prompt.1.tool_calls"])[0]["id"]
        == "call_weather"
    )
    assert second.attributes["gen_ai.prompt.1.content"] == ""
    assert second.attributes["gen_ai.prompt.2.tool_call_id"] == "call_weather"


def test_multimodal_json_controls_and_no_transport_secret(telemetry):
    _, exporter, _ = telemetry
    content = [
        {"type": "text", "text": "describe synthetic image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}},
    ]
    with client() as sdk:
        sdk.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            extra_headers={"authorization": "never-export-this"},
        )
    attrs = exporter.get_finished_spans()[0].attributes
    assert json.loads(attrs["gen_ai.prompt.0.content"]) == content
    assert "never-export-this" not in str(attrs)
    assert "json_object" in attrs["traceloop.entity.input"]


def test_sdk_1x_reasoning_and_json_schema_controls(telemetry):
    if version("groq").startswith("0."):
        pytest.skip("Groq 1.x reasoning controls")
    _, exporter, _ = telemetry
    with client() as sdk:
        sdk.chat.completions.create(
            model=MODEL,
            messages=MESSAGES,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}},
            },
        )
    attrs = exporter.get_finished_spans()[0].attributes
    assert "json_schema" in attrs["traceloop.entity.input"]
    assert "low" in attrs["traceloop.entity.input"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_real_early_close_has_partial_content_without_invented_usage(
    telemetry, async_mode
):
    _, exporter, _ = telemetry
    if async_mode:

        async def run():
            async with async_client() as sdk:
                result = await sdk.chat.completions.create(
                    model=MODEL, messages=MESSAGES, stream=True
                )
                await anext(result)
                await result.close()

        asyncio.run(run())
    else:
        with client() as sdk:
            result = sdk.chat.completions.create(
                model=MODEL, messages=MESSAGES, stream=True
            )
            next(result)
            result.close()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert_span(spans[0], content="hello ", usage=False)


def test_error_and_suppression_and_reactivation(telemetry):
    _, exporter, adapter = telemetry
    with client() as sdk:
        with pytest.raises(InternalServerError) as exc:
            sdk.chat.completions.create(model="fixture-error", messages=MESSAGES)
        assert exc.value.status_code == 503
        assert exporter.get_finished_spans()[0].status.is_ok is False
        token = context.attach(
            context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            sdk.chat.completions.create(model=MODEL, messages=MESSAGES)
        finally:
            context.detach(token)
        adapter.deactivate()
        sdk.chat.completions.create(model=MODEL, messages=MESSAGES)
        assert len(exporter.get_finished_spans()) == 1
        adapter.activate()
        sdk.chat.completions.create(model=MODEL, messages=MESSAGES)
        assert len(exporter.get_finished_spans()) == 2


def test_two_owners_keep_stream_capture_until_final_deactivation(telemetry):
    _, exporter, first = telemetry
    second = GroqInstrumentor()
    second.activate()
    try:
        first.deactivate()
        with client() as sdk:
            list(
                sdk.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)
            )
        assert len(exporter.get_finished_spans()) == 1
        assert_span(exporter.get_finished_spans()[0])
    finally:
        second.deactivate()


def test_privacy_config(telemetry):
    _, exporter, adapter = telemetry
    adapter.deactivate()
    private = GroqInstrumentor(config=TraceConfig(hide_inputs=True, hide_outputs=True))
    private.activate()
    try:
        with client() as sdk:
            sdk.chat.completions.create(model=MODEL, messages=MESSAGES, tools=[TOOL])
        attrs = exporter.get_finished_spans()[0].attributes
        assert "synthetic hello" not in str(attrs)
        assert "Tokyo" not in str(attrs)
    finally:
        private.deactivate()


@pytest.mark.parametrize("async_mode", [False, True])
def test_stream_error_and_close_error_preserve_exception(async_mode):
    error = RuntimeError("synthetic stream error")
    finished = []
    if async_mode:

        async def source():
            raise error
            yield

        async def run():
            result = _AsyncStreamProxy(
                source(), lambda value, exc: finished.append(exc)
            )
            with pytest.raises(RuntimeError) as caught:
                await anext(result)
            assert caught.value is error
            await result.close()

        asyncio.run(run())
    else:

        def source():
            raise error
            yield

        result = _SyncStreamProxy(source(), lambda value, exc: finished.append(exc))
        with pytest.raises(RuntimeError) as caught:
            next(result)
        assert caught.value is error
        result.close()
    assert finished == [error]


def test_stream_cancellation():
    error = asyncio.CancelledError("synthetic cancellation")
    finished = []

    async def source():
        raise error
        yield

    async def run():
        result = _AsyncStreamProxy(source(), lambda value, exc: finished.append(exc))
        with pytest.raises(asyncio.CancelledError) as caught:
            await anext(result)
        assert caught.value is error

    asyncio.run(run())
    assert finished == [error]


def inference_response(request):
    if request.url.path.endswith("/embeddings"):
        body = json.loads(request.content)
        vector = (
            "AACAPwAAAEA=" if body.get("encoding_format") == "base64" else [1.0, 2.0]
        )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": MODEL,
                "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )
    if request.url.path.endswith("/speech"):
        return httpx.Response(
            200,
            headers={"content-type": "audio/wav"},
            stream=httpx.ByteStream(b"RIFFsynthetic-audio"),
        )
    return httpx.Response(200, json={"text": "synthetic transcript"})


@pytest.mark.parametrize(
    "operation", ["embeddings", "transcriptions", "translations", "speech"]
)
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("mode", ["normal", "raw", "streaming_response"])
def test_inference_resources_real_sdk(telemetry, operation, async_mode, mode):
    _, exporter, _ = telemetry
    parameters = {"model": MODEL}
    if operation == "embeddings":
        parameters["input"] = ["synthetic input"]
    elif operation == "speech":
        parameters.update(input="synthetic speech", voice="fixture")
    else:
        parameters["file"] = ("fixture.wav", b"synthetic upload", "audio/wav")

    def resource(sdk):
        target = (
            getattr(sdk, "embeddings", None)
            if operation == "embeddings"
            else getattr(sdk.audio, operation, None)
        )
        if target is None:
            pytest.skip(f"{operation} unavailable in Groq {version('groq')}")
        return target

    if async_mode:

        async def run():
            async with AsyncGroq(
                api_key="fixture",
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(inference_response)
                ),
            ) as sdk:
                target = resource(sdk)
                if mode == "raw":
                    response_ = await target.with_raw_response.create(**parameters)
                    result = await response_.parse()
                    await response_.close()
                elif mode == "streaming_response":
                    async with target.with_streaming_response.create(
                        **parameters
                    ) as response_:
                        assert not exporter.get_finished_spans()
                        result = (
                            b"".join([chunk async for chunk in response_.iter_bytes()])
                            if operation == "speech"
                            else await response_.parse()
                        )
                else:
                    result = await target.create(**parameters)
                    if operation == "speech":
                        result = await result.read()
                return result

        asyncio.run(run())
    else:
        with Groq(
            api_key="fixture",
            http_client=httpx.Client(transport=httpx.MockTransport(inference_response)),
        ) as sdk:
            target = resource(sdk)
            if mode == "raw":
                response_ = target.with_raw_response.create(**parameters)
                result = response_.parse()
                response_.close()
            elif mode == "streaming_response":
                with target.with_streaming_response.create(**parameters) as response_:
                    assert not exporter.get_finished_spans()
                    result = (
                        b"".join(response_.iter_bytes())
                        if operation == "speech"
                        else response_.parse()
                    )
            else:
                result = target.create(**parameters)
                if operation == "speech":
                    result = result.read()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["gen_ai.system"] == "groq"
    assert "traceloop.span.kind" not in attrs
    output = json.loads(attrs["traceloop.entity.output"])
    if operation == "embeddings":
        assert attrs["respan.entity.log_type"] == "embedding"
        assert output == [[1.0, 2.0]]
        assert attrs["gen_ai.usage.input_tokens"] == 3
    elif operation == "speech":
        assert attrs["respan.entity.log_type"] == "speech"
        assert output == {
            "encoding": "base64",
            "data": "UklGRnN5bnRoZXRpYy1hdWRpbw==",
            "truncated": False,
        }
        assert "gen_ai.usage.input_tokens" not in attrs
    else:
        assert attrs["respan.entity.log_type"] == "transcription"
        assert output == "synthetic transcript"
        assert json.loads(attrs["traceloop.entity.input"])["file"] == {
            "name": "fixture.wav"
        }
        assert "synthetic upload" not in str(attrs)


def test_native_privacy_suppression_and_foreign_patch(telemetry):
    import inspect

    from groq.resources.audio.transcriptions import Transcriptions
    from wrapt import FunctionWrapper

    _, exporter, adapter = telemetry
    adapter.deactivate()
    private = GroqInstrumentor(config=TraceConfig(hide_inputs=True, hide_outputs=True))
    private.activate()
    original = inspect.getattr_static(Transcriptions, "create")
    foreign = FunctionWrapper(
        original, lambda wrapped, instance, args, kwargs: wrapped(*args, **kwargs)
    )
    Transcriptions.create = foreign
    try:
        with Groq(
            api_key="fixture",
            http_client=httpx.Client(transport=httpx.MockTransport(inference_response)),
        ) as sdk:
            sdk.audio.transcriptions.create(
                model=MODEL, file=("fixture.wav", b"upload")
            )
            attrs = exporter.get_finished_spans()[0].attributes
            assert "traceloop.entity.input" not in attrs
            assert "traceloop.entity.output" not in attrs
            token = context.attach(
                context.set_value(context._SUPPRESS_INSTRUMENTATION_KEY, True)
            )
            try:
                sdk.audio.transcriptions.create(
                    model=MODEL, file=("fixture.wav", b"upload")
                )
            finally:
                context.detach(token)
            private.deactivate()
            assert inspect.getattr_static(Transcriptions, "create") is foreign
            sdk.audio.transcriptions.create(
                model=MODEL, file=("fixture.wav", b"upload")
            )
        assert len(exporter.get_finished_spans()) == 1
    finally:
        private.deactivate()
        Transcriptions.create = original.__wrapped__


@pytest.mark.parametrize("policy", ["context", "environment"])
def test_respan_content_policy_is_snapshotted_for_lazy_chat_and_inference(
    telemetry, monkeypatch, policy
):
    from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

    _, exporter, _ = telemetry
    token = None
    if policy == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    else:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    with client() as sdk:
        stream = sdk.chat.completions.create(
            model=MODEL, messages=MESSAGES, stream=True, tools=[TOOL]
        )
    sdk2 = Groq(
        api_key="fixture",
        http_client=httpx.Client(transport=httpx.MockTransport(inference_response)),
    )
    raw = sdk2.embeddings.with_raw_response.create(model=MODEL, input="private input")
    if token:
        context.detach(token)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    list(stream)
    raw.parse()
    raw.close()
    sdk2.close()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        attrs = span.attributes
        assert "traceloop.entity.input" not in attrs
        assert "traceloop.entity.output" not in attrs
        assert not any(
            key.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for key in attrs
        )
        assert "llm.request.functions" not in attrs
        assert attrs["gen_ai.system"] == "groq"
        assert "gen_ai.usage.input_tokens" in attrs


@pytest.mark.parametrize("async_mode", [False, True])
def test_close_errors_preserve_exception(async_mode):
    error = RuntimeError("synthetic close failure")
    finished = []
    if async_mode:

        class Source:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def close(self):
                raise error

        async def run():
            stream = _AsyncStreamProxy(
                Source(), lambda value, exc: finished.append(exc)
            )
            with pytest.raises(RuntimeError) as caught:
                await stream.close()
            assert caught.value is error

        asyncio.run(run())
    else:

        class Source:
            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

            def close(self):
                raise error

        stream = _SyncStreamProxy(Source(), lambda value, exc: finished.append(exc))
        with pytest.raises(RuntimeError) as caught:
            stream.close()
        assert caught.value is error
    assert finished == [error]


def test_stream_full_payload_contains_structured_tool_calls(telemetry):
    _, exporter, _ = telemetry
    with client() as sdk:
        list(
            sdk.chat.completions.create(
                model=MODEL, messages=MESSAGES, tools=[TOOL], stream=True
            )
        )
    payload = json.loads(
        exporter.get_finished_spans()[0].attributes["traceloop.entity.output"]
    )
    assert payload["choices"][0]["message"]["tool_calls"] == [CALL]


def test_native_embedding_base64_and_usage(telemetry):
    _, exporter, _ = telemetry
    with Groq(
        api_key="fixture",
        http_client=httpx.Client(transport=httpx.MockTransport(inference_response)),
    ) as sdk:
        sdk.embeddings.create(
            model=MODEL, input="synthetic input", encoding_format="base64"
        )
    attrs = exporter.get_finished_spans()[0].attributes
    assert json.loads(attrs["traceloop.entity.output"]) == ["AACAPwAAAEA="]
    assert attrs["gen_ai.usage.input_tokens"] == 3


def test_request_generators_are_forwarded_once(telemetry):
    _, exporter, _ = telemetry
    received = []

    def handle(request):
        received.append(json.loads(request.content))
        return response(request)

    with Groq(
        api_key="fixture",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as sdk:
        sdk.chat.completions.create(
            model=MODEL, messages=(m for m in MESSAGES), tools=(t for t in [TOOL])
        )
    assert received[0]["messages"] == MESSAGES
    assert received[0]["tools"] == [TOOL]
    assert len(exporter.get_finished_spans()) == 1


def test_granular_privacy_has_no_raw_payload_bypass(telemetry):
    _, exporter, adapter = telemetry
    adapter.deactivate()
    private = GroqInstrumentor(
        config=TraceConfig(hide_input_text=True, hide_output_text=True)
    )
    private.activate()
    try:
        with client() as sdk:
            sdk.chat.completions.create(model=MODEL, messages=MESSAGES)
        attrs = exporter.get_finished_spans()[0].attributes
        assert "synthetic hello" not in str(attrs)
        assert "hello world" not in str(attrs)
        assert attrs["gen_ai.usage.input_tokens"] == 4
    finally:
        private.deactivate()


@pytest.mark.parametrize("stream", [False, True])
def test_returned_reasoning_content_is_preserved(telemetry, stream):
    _, exporter, _ = telemetry

    def handle(request):
        result = response(request)
        if stream:
            text = result.text.replace(
                '"content": "hello "',
                '"content": "hello ", "reasoning": "fixture reasoning"',
            )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=text
            )
        payload = result.json()
        payload["choices"][0]["message"]["reasoning"] = "fixture reasoning"
        return httpx.Response(200, json=payload)

    with Groq(
        api_key="fixture",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as sdk:
        result = sdk.chat.completions.create(
            model=MODEL, messages=MESSAGES, stream=stream
        )
        if stream:
            list(result)
    attrs = exporter.get_finished_spans()[0].attributes
    assert json.loads(attrs["gen_ai.completion.0.content"]) == [
        {"type": "reasoning", "text": "fixture reasoning"},
        {"type": "text", "text": "hello world"},
    ]
