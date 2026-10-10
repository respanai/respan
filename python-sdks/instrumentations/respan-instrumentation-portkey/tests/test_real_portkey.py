from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import contextmanager

import httpx
import pytest
from _fixture import Transport
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from portkey_ai import AsyncPortkey, Portkey
from portkey_ai._vendor.openai import APIConnectionError, AuthenticationError
from portkey_ai.api_resources.apis.chat_complete import Completions
from pydantic import BaseModel
from respan_instrumentation_portkey import PortkeyInstrumentor, _instrumentation
from respan_instrumentation_portkey._streaming import Runtime
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


@contextmanager
def harness(*, capture=True, sampler=None, checkpoint=None, config=None):
    provider = TracerProvider(**({"sampler": sampler} if sampler else {}))
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = PortkeyInstrumentor(
        tracer_provider=provider,
        capture_content=capture,
        **({"config": config} if config else {}),
    )
    adapter.activate()
    transport = Transport(checkpoint)
    client = Portkey(
        api_key="fixture",
        base_url="https://portkey.test",
        http_client=httpx.Client(
            base_url="https://portkey.test", transport=httpx.MockTransport(transport)
        ),
        max_retries=0,
    )
    try:
        yield adapter, client, transport, exporter, provider
    finally:
        client.close()
        adapter.deactivate()
        provider.force_flush()
        provider.shutdown()


def chats(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("respan.entity.log_type") == "chat"
    ]


def call(client, **kwargs):
    return client.chat.completions.create(
        model=kwargs.pop("model", "fixture-model"),
        messages=kwargs.pop("messages", [{"role": "user", "content": "hello"}]),
        **kwargs,
    )


def test_native_chat_values_usage_tools_and_full_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool{i}",
                "parameters": {
                    "type": "object",
                    "properties": {f"field{j}": {"type": "string"} for j in range(120)},
                },
            },
        }
        for i in range(65)
    ]
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "historical-call",
                    "type": "function",
                    "function": {"name": "old", "arguments": '{"x":1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "historical-call", "content": "old result"},
        {"role": "user", "content": "weather"},
    ]
    with harness() as (_, client, transport, exporter, provider):
        with provider.get_tracer("test").start_as_current_span("root") as parent:
            response = call(client, tools=tools, messages=history)
        assert response.choices[0].message.tool_calls[0].id == "current-call"
        span = chats(exporter)[0]
        a = span.attributes
        assert span.parent.span_id == parent.get_span_context().span_id
        assert len(json.loads(a[SpanAttributes.LLM_REQUEST_FUNCTIONS])) == 65
        assert (
            len(
                json.loads(a[SpanAttributes.LLM_REQUEST_FUNCTIONS])[0]["function"][
                    "parameters"
                ]["properties"]
            )
            == 120
        )
        calls = json.loads(a[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
        assert (
            calls[0]["id"] == "current-call"
            and calls[0]["function"]["arguments"] == '{"city":"Tokyo"}'
        )
        assert (
            json.loads(a[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls"])[0]["id"]
            == "historical-call"
        )
        assert (
            a["gen_ai.usage.input_tokens"] == 12
            and a["gen_ai.usage.output_tokens"] == 9
        )
        assert (
            a[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4
            and a[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 2
        )
        assert SpanAttributes.TRACELOOP_SPAN_KIND not in a and "status_code" not in a
        assert transport.requests[0][1]["tools"] == tools


@pytest.mark.parametrize("mode", ["constructor", "environment", "context"])
def test_initial_privacy_bound(mode, monkeypatch):
    if mode == "environment":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if mode == "context"
        else None
    )
    try:
        with harness(capture=mode != "constructor") as (_, client, _, exporter, _):
            call(client)
            a = chats(exporter)[0].attributes
            assert not any(
                k.startswith(("gen_ai.prompt.", "gen_ai.completion.")) for k in a
            )
            assert (
                SpanAttributes.TRACELOOP_ENTITY_INPUT not in a
                and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a
            )
            assert a["gen_ai.usage.input_tokens"] == 12
    finally:
        if token is not None:
            context.detach(token)


@pytest.mark.parametrize("model", ["fixture-model", "error-401"])
def test_end_veto_before_native_context_detaches(model):
    def checkpoint(request, body):
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))

    with harness(checkpoint=checkpoint) as (_, client, _, exporter, _):
        if model == "error-401":
            with pytest.raises(AuthenticationError):
                call(client, model=model)
        else:
            call(client)
        a = chats(exporter)[0].attributes
        assert (
            SpanAttributes.TRACELOOP_ENTITY_INPUT not in a
            and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a
        )
        assert context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_suppression(key):
    with harness() as (_, client, _, exporter, _):
        token = context.attach(context.set_value(key, True))
        try:
            response = call(client)
            stream = call(client, stream=True)
            list(stream)
            assert response.choices[0].message.content == "Portkey response."
        finally:
            context.detach(token)
        assert exporter.get_finished_spans() == ()


def test_sampled_off_does_not_serialize_provider_content(monkeypatch):
    import respan_instrumentation_portkey._processor as processor

    monkeypatch.setattr(
        processor,
        "json_dumps",
        lambda *a, **k: pytest.fail("sampled-off payload serialization"),
    )
    with harness(sampler=ALWAYS_OFF) as (_, client, _, exporter, _):
        call(client)
        list(call(client, stream=True))
        client.embeddings.create(model="embed", input="hello")
        assert exporter.get_finished_spans() == ()


@pytest.mark.parametrize(
    "model,status", [("error-401", 401), ("connection-error", None)]
)
def test_native_errors_source_status_only(model, status):
    with harness() as (_, client, _, exporter, _):
        with pytest.raises(
            AuthenticationError if status else APIConnectionError
        ) as caught:
            call(client, model=model)
        span = chats(exporter)[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes.get("http.response.status_code") == status
        assert (
            "status_code" not in span.attributes
            and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        )
        assert type(caught.value).__name__ == (
            "AuthenticationError" if status else "APIConnectionError"
        )


def test_native_embedding_complete_vectors_and_usage():
    with harness() as (_, client, _, exporter, _):
        result = client.embeddings.create(model="embed", input=["hello", "world"])
        a = exporter.get_finished_spans()[0].attributes
        assert len(result.data[0].embedding) == 3072
        assert (
            json.loads(a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])[0]
            == result.data[0].embedding
        )
        assert (
            a["respan.entity.log_type"] == "embedding"
            and a["gen_ai.usage.input_tokens"] == 3
        )


def test_native_responses_and_text():
    with harness() as (_, client, _, exporter, _):
        response = client.responses.create(
            model="response-model",
            input="hello",
            tools=[
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": {"type": "object"},
                }
            ],
        )
        assert response.output[1].call_id == "response-call"
        client.completions.create(model="text-model", prompt="hello")
        response_span = chats(exporter)[0]
        a = response_span.attributes
        assert (
            a["gen_ai.usage.input_tokens"] == 13
            and a[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 3
        )
        assert (
            json.loads(a[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])[0]["id"]
            == "response-call"
        )
        assert any(
            s.attributes["respan.entity.log_type"] == "text"
            for s in exporter.get_finished_spans()
        )


def test_native_stream_tool_fragments_usage_context_and_early_close():
    with harness() as (_, client, _, exporter, provider):
        parent = provider.get_tracer("test").start_span("root")
        with trace.use_span(parent, end_on_exit=False):
            stream = call(
                client,
                stream=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        assert trace.get_current_span().get_span_context().span_id == 0
        chunks = list(stream)
        assert chunks[0].choices[0].delta.tool_calls[0].id == "stream-call"
        span = chats(exporter)[0]
        a = span.attributes
        assert span.parent.span_id == parent.get_span_context().span_id
        calls = json.loads(a[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls"])
        assert calls[0]["function"]["arguments"] == '{"city":"Tokyo"}'
        assert (
            a["gen_ai.usage.input_tokens"] == 7
            and a[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 1
        )
        early = call(client, stream=True)
        early.close()
        assert (
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in chats(exporter)[1].attributes
        )
        parent.end()


def test_stream_initial_bound_and_end_veto(monkeypatch):
    with harness() as (_, client, _, exporter, _):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        stream = call(client, stream=True)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        list(stream)
        other = call(client, stream=True)
        next(other)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        next(other)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        list(other)
        for s in chats(exporter):
            assert (
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in s.attributes
                and SpanAttributes.TRACELOOP_ENTITY_INPUT not in s.attributes
            )


@pytest.mark.asyncio
async def test_native_async_chat_stream_embedding_responses():
    with harness() as (_, _, transport, exporter, _):
        client = AsyncPortkey(
            api_key="fixture",
            base_url="https://portkey.test",
            http_client=httpx.AsyncClient(
                base_url="https://portkey.test",
                transport=httpx.MockTransport(transport),
            ),
            max_retries=0,
        )
        try:
            result = await call(client)
            assert result.choices[0].message.content == "Portkey response."
            stream = await call(client, stream=True)
            chunks = [x async for x in stream]
            assert len(chunks) == 3
            await client.embeddings.create(model="embed", input="hello")
            await client.responses.create(model="response-model", input="hello")
        finally:
            await client.close()
        assert len(exporter.get_finished_spans()) == 4


def test_native_structured_parse():
    class Weather(BaseModel):
        city: str

    with harness() as (_, client, _, exporter, _):
        method = getattr(client.chat.completions, "parse", None)
        if method is None:
            pytest.skip("released SDK lacks chat.parse")
        result = method(
            model="fixture-model",
            messages=[{"role": "user", "content": "city"}],
            response_format=Weather,
        )
        assert result.choices[0].message.parsed.city == "Tokyo"
        assert len(chats(exporter)) == 1


def test_native_responses_stream():
    with harness() as (_, client, _, exporter, _):
        stream = client.responses.create(
            model="response-model", input="hello", stream=True
        )
        assert [x.type for x in stream][-1] == "response.completed"
        a = chats(exporter)[0].attributes
        assert (
            a["gen_ai.usage.input_tokens"] == 13
            and "Responses result." in a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )


def test_shared_foreign_restore_and_mismatch():
    original = inspect.getattr_static(Completions, "create")
    with harness() as (adapter, _, _, _, provider):
        shared = PortkeyInstrumentor(tracer_provider=provider)
        shared.activate()
        adapter.deactivate()
        assert inspect.getattr_static(Completions, "create") is not original
        with pytest.raises(ValueError):
            PortkeyInstrumentor(
                tracer_provider=provider, capture_content=False
            ).activate()
        foreign = lambda self, *a, **k: "foreign"
        Completions.create = foreign
        try:
            shared.deactivate()
            assert Completions.create is foreign
        finally:
            Completions.create = original


def test_partial_activation_rollback(monkeypatch):
    original = inspect.getattr_static(Completions, "create")
    install = Runtime.install

    def fail(runtime):
        install(runtime)
        raise RuntimeError("controlled after-assignment failure")

    monkeypatch.setattr(Runtime, "install", fail)
    provider = TracerProvider()
    try:
        with pytest.raises(RuntimeError):
            PortkeyInstrumentor(tracer_provider=provider).activate()
        assert (
            inspect.getattr_static(Completions, "create") is original
            and _instrumentation._SHARED is None
        )
    finally:
        provider.shutdown()


def test_native_foreign_instrumentor_remains_active():
    from openinference.instrumentation.portkey import PortkeyInstrumentor as Native

    provider = TracerProvider()
    native = Native()
    native.instrument(tracer_provider=provider)
    original = inspect.getattr_static(Completions, "create")
    tracer = native._tracer
    try:
        adapter = PortkeyInstrumentor(tracer_provider=provider)
        adapter.activate()
        adapter.deactivate()
        assert (
            inspect.getattr_static(Completions, "create") is original
            and native._tracer is tracer
            and native.is_instrumented_by_opentelemetry
        )
    finally:
        native.uninstrument()
        provider.shutdown()


def test_deactivation_ends_open_stream_without_content_retention():
    with harness() as (adapter, client, _, exporter, _):
        stream = call(client, stream=True)
        next(stream)
        adapter.deactivate()
        assert len(chats(exporter)) == 1 and stream._operation.request == {}
        list(stream)
        assert len(chats(exporter)) == 1


def test_native_chat_stream_manager_and_final_completion():
    with harness() as (_, client, _, exporter, _):
        method = getattr(client.chat.completions, "stream", None)
        if method is None:
            pytest.skip("released SDK lacks chat.stream manager")
        with method(
            model="fixture-model", messages=[{"role": "user", "content": "hello"}]
        ) as stream:
            final = stream.get_final_completion()
            assert final.choices[0].message.content == "Portkey stream."
        a = chats(exporter)[0].attributes
        assert (
            a["gen_ai.usage.input_tokens"] == 7
            and "Portkey stream." in a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )


@pytest.mark.asyncio
async def test_native_async_parse_and_manager():
    class Weather(BaseModel):
        city: str

    with harness() as (_, _, transport, exporter, _):
        client = AsyncPortkey(
            api_key="fixture",
            base_url="https://portkey.test",
            http_client=httpx.AsyncClient(
                base_url="https://portkey.test",
                transport=httpx.MockTransport(transport),
            ),
            max_retries=0,
        )
        try:
            method = getattr(client.chat.completions, "parse", None)
            if method is None:
                pytest.skip("released SDK lacks async chat.parse")
            result = await method(
                model="fixture-model",
                messages=[{"role": "user", "content": "hello"}],
                response_format=Weather,
            )
            assert result.choices[0].message.parsed.city == "Tokyo"
            async with client.chat.completions.stream(
                model="fixture-model", messages=[{"role": "user", "content": "hello"}]
            ) as stream:
                final = await stream.get_final_completion()
                assert final.choices[0].message.content == "Portkey stream."
        finally:
            await client.close()
        assert len(chats(exporter)) == 2


def test_stream_native_http_iteration_context_and_error():
    observed = []

    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            observed.append(trace.get_current_span().get_span_context().span_id)
            yield b'data: {"id":"chunk","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("controlled SSE transport error")

        def close(self):
            observed.append(trace.get_current_span().get_span_context().span_id)

    with harness() as (_, client, _, exporter, _):
        client.openai_client._client._transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                stream=Chunks(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        )
        stream = call(client, stream=True)
        first = next(stream)
        assert first.choices[0].delta.content == "partial"
        with pytest.raises(httpx.ReadError):
            next(stream)
        span = chats(exporter)[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            observed[0] == span.context.span_id
            and "http.response.status_code" not in span.attributes
        )
        assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
        assert trace.get_current_span().get_span_context().span_id == 0


@pytest.mark.asyncio
async def test_actual_async_stream_cancel_preserves_cancel_and_context():
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"id":"chunk","object":"chat.completion.chunk","created":1,"model":"fixture-model","choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
            raise asyncio.CancelledError()

    with harness() as (_, _, _, exporter, _):
        client = AsyncPortkey(
            api_key="fixture",
            base_url="https://portkey.test",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        stream=Chunks(),
                        headers={"content-type": "text/event-stream"},
                        request=request,
                    )
                )
            ),
            max_retries=0,
        )
        try:
            stream = await call(client, stream=True)
            first = await stream.__anext__()
            assert first.choices[0].delta.content == "partial"
            with pytest.raises(asyncio.CancelledError):
                await stream.__anext__()
            span = chats(exporter)[0]
            assert span.status.status_code != StatusCode.ERROR
            assert "http.response.status_code" not in span.attributes
            assert trace.get_current_span().get_span_context().span_id == 0
        finally:
            await client.close()


def test_observed_parent_veto_is_permanent_for_child_and_pending_close(monkeypatch):
    with harness() as (adapter, client, _, exporter, _):
        parent = call(client, stream=True)
        operation = parent._operation
        with trace.use_span(operation.span, end_on_exit=False):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
            call(client)
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
            child = call(client, stream=True)
        adapter.deactivate()
        assert not operation.capture and not child._operation.capture
        for span in chats(exporter):
            assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
        parent.close()
        child.close()


def test_completed_private_parent_bound_is_inherited(monkeypatch):
    with harness() as (_, client, _, exporter, _):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        parent = call(client, stream=True)
        span = parent._operation.span
        list(parent)
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        with trace.use_span(span, end_on_exit=False):
            call(client)
        assert len(chats(exporter)) == 2
        for s in chats(exporter):
            assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in s.attributes


def test_native_delegate_partial_mutation_rollback(monkeypatch):
    native_class = _instrumentation._load_openinference_portkey_class()
    original = inspect.getattr_static(Completions, "create")
    native = native_class._instrument

    def fail(self, **kwargs):
        native(self, **kwargs)
        raise RuntimeError("controlled native partial activation")

    monkeypatch.setattr(native_class, "_instrument", fail)
    provider = TracerProvider()
    try:
        with pytest.raises(RuntimeError):
            PortkeyInstrumentor(tracer_provider=provider).activate()
        assert (
            inspect.getattr_static(Completions, "create") is original
            and not native_class().is_instrumented_by_opentelemetry
        )
    finally:
        provider.shutdown()


def test_native_traceconfig_privacy_is_preserved():
    from openinference.instrumentation import TraceConfig

    with harness(config=TraceConfig(hide_inputs=True, hide_outputs=True)) as (
        _,
        client,
        _,
        exporter,
        _,
    ):
        call(client)
        a = chats(exporter)[0].attributes
        assert (
            SpanAttributes.TRACELOOP_ENTITY_INPUT not in a
            and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a
        )
        assert a["gen_ai.usage.input_tokens"] == 12


def test_native_prompt_completion_delegation():
    with harness() as (_, client, transport, exporter, _):
        response = client.prompts.completions.create(
            prompt_id="fixture-prompt", variables={"name": "world"}
        )
        assert response.choices[0].message.content == "Portkey response."
        assert (
            len(chats(exporter)) == 1 and "fixture-prompt" in transport.requests[0][0]
        )


def test_native_foreign_tracer_field_replacement_survives_teardown():
    with harness() as (adapter, _, _, _, _):
        native = adapter._delegate
        foreign = object()
        native._tracer = foreign
        adapter.deactivate()
        assert native._tracer is foreign
        del native._tracer


@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": True, "completion_tokens": False, "total_tokens": 1}, None],
)
def test_actual_provider_usage_rejects_dto_coerced_bool_and_missing_counts(usage):
    from _fixture import chat_body

    body = chat_body()
    body["usage"] = usage
    with harness() as (_, client, _, exporter, _):
        client.openai_client._client._transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=body, request=request)
        )
        response = call(client)
        a = chats(exporter)[0].attributes
        assert (
            "gen_ai.usage.input_tokens" not in a
            and "gen_ai.usage.output_tokens" not in a
        )
        if usage is not None:
            assert (
                response.usage.prompt_tokens == 1
                and a[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 1
            )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_actual_response_failed_value_is_error_without_invented_http_or_output(
    streaming, capture
):
    from _fixture import response_body

    body = response_body()
    body.update(
        status="failed",
        error={"code": "server_error", "message": "controlled Responses failure"},
        output=[],
        usage=None,
    )

    def response(request):
        if streaming:
            event = {"type": "response.failed", "response": body, "sequence_number": 1}
            return httpx.Response(
                200,
                text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(200, json=body, request=request)

    with harness(capture=capture) as (_, client, _, exporter, _):
        client.openai_client._client._transport = httpx.MockTransport(response)
        result = client.responses.create(
            model="response-model", input="hello", stream=streaming
        )
        if streaming:
            events = list(result)
            assert events[0].response.status == "failed"
        else:
            assert result.status == "failed"
        a = chats(exporter)[0]
        assert a.status.status_code == StatusCode.ERROR
        assert (
            "http.response.status_code" not in a.attributes
            and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a.attributes
        )


def test_actual_cache_write_source_is_preserved():
    from _fixture import chat_body

    body = chat_body()
    body["usage"]["prompt_tokens_details"]["cache_write_tokens"] = 6
    with harness() as (_, client, _, exporter, _):
        client.openai_client._client._transport = httpx.MockTransport(
            lambda r: httpx.Response(200, json=body, request=r)
        )
        call(client)
        assert (
            chats(exporter)[0].attributes[
                SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS
            ]
            == 6
        )


def test_native_until_done_preserves_self_return_and_actual_snapshot():
    with harness() as (_, client, _, exporter, _):
        with client.chat.completions.stream(
            model="fixture-model", messages=[{"role": "user", "content": "hello"}]
        ) as stream:
            returned = stream.until_done()
            assert returned is stream._stream
        a = chats(exporter)[0].attributes
        assert (
            a["gen_ai.usage.input_tokens"] == 7
            and "Portkey stream." in a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
        )


@pytest.mark.parametrize("mode", ["context", "environment"])
def test_actual_delayed_sdk_stream_inherits_finished_manual_parent_veto(
    mode, monkeypatch
):
    with harness() as (_, client, _, exporter, provider):
        with provider.get_tracer("manual").start_as_current_span("manual.parent"):
            stream = call(client, stream=True)
            if mode == "context":
                context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
            else:
                monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
        assert context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
        chunks = list(stream)
        assert chunks[0].choices[0].delta.content == "Portkey "
        a = chats(exporter)[0].attributes
        assert (
            SpanAttributes.TRACELOOP_ENTITY_INPUT not in a
            and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a
        )


def test_actual_delayed_sdk_stream_parent_predates_adapter_activation(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = PortkeyInstrumentor(tracer_provider=provider)
    client = Portkey(
        api_key="fixture",
        base_url="https://portkey.test",
        http_client=httpx.Client(
            base_url="https://portkey.test", transport=httpx.MockTransport(Transport())
        ),
    )
    try:
        with provider.get_tracer("manual").start_as_current_span("preexisting.parent"):
            adapter.activate()
            stream = call(client, stream=True)
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        list(stream)
        a = chats(exporter)[0].attributes
        assert (
            SpanAttributes.TRACELOOP_ENTITY_INPUT not in a
            and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a
        )
    finally:
        client.close()
        adapter.deactivate()
        provider.shutdown()


def test_ancestor_policy_failure_preserves_native_result_and_fails_closed(monkeypatch):
    with harness() as (_, client, _, exporter, _):
        runtime = _instrumentation._SHARED["runtime"]

        def failure(*args):
            raise RuntimeError("controlled policy failure")

        monkeypatch.setattr(runtime.policy, "observe", failure)
        result = call(client)
        assert result.choices[0].message.content == "Portkey response."
        a = chats(exporter)[0].attributes
        assert (
            SpanAttributes.TRACELOOP_ENTITY_INPUT not in a
            and SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in a
        )


def test_actual150_messages_choices_preserve_contract_and_run_metadata():
    from _fixture import chat_body
    from opentelemetry.sdk.trace import SpanProcessor
    from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE, RESPAN_METADATA

    class Marker(SpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_attribute(
                RESPAN_METADATA, json.dumps({"run_id": "large-history-marker"})
            )

    body = chat_body()
    body["choices"] = [
        {
            "index": i,
            "message": {"role": "assistant", "content": f"choice{i}"},
            "finish_reason": "stop",
        }
        for i in range(150)
    ]
    messages = [{"role": "user", "content": f"message{i}"} for i in range(150)]
    with harness() as (_, client, _, exporter, provider):
        provider.add_span_processor(Marker())
        client.openai_client._client._transport = httpx.MockTransport(
            lambda r: httpx.Response(200, json=body, request=r)
        )
        result = call(client, messages=messages)
        assert len(result.choices) == 150
        a = chats(exporter)[0].attributes
        assert (
            a[RESPAN_LOG_TYPE] == "chat"
            and a[SpanAttributes.TRACELOOP_ENTITY_NAME] == "portkey.chat"
        )
        assert json.loads(a[RESPAN_METADATA])["run_id"] == "large-history-marker"
        assert (
            len(json.loads(a[SpanAttributes.TRACELOOP_ENTITY_INPUT])["messages"]) == 150
        )
        assert (
            len(json.loads(a[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])["choices"]) == 150
        )
        assert (
            a["gen_ai.usage.input_tokens"] == 12
            and a[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 21
        )


def test_actual_tool_schema_credential_property_is_structural_not_value():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "default": "synthetic secret with spaces",
                        },
                        "city": {"type": "string"},
                    },
                    "required": ["api_key"],
                },
            },
        }
    ]
    with harness() as (_, client, transport, exporter, _):
        call(client, tools=tools)
        a = chats(exporter)[0].attributes
        for data in [
            json.loads(a[SpanAttributes.LLM_REQUEST_FUNCTIONS]),
            json.loads(a[SpanAttributes.TRACELOOP_ENTITY_INPUT])["tools"],
        ]:
            node = data[0]["function"]["parameters"]["properties"]["api_key"]
            assert node["type"] == "string" and node["default"] == "<redacted>"
        assert transport.requests[0][1]["tools"] == tools


def test_unrelated_http_json_within_sdk_callback_does_not_supply_usage():
    def callback(request, body):
        httpx.Response(
            200,
            json={
                "usage": {
                    "prompt_tokens": 999,
                    "completion_tokens": 999,
                    "total_tokens": 1998,
                }
            },
            request=httpx.Request("GET", "https://other.test/info"),
        ).json()

    with harness(checkpoint=callback) as (_, client, _, exporter, _):
        call(client)
        assert chats(exporter)[0].attributes["gen_ai.usage.input_tokens"] == 12
