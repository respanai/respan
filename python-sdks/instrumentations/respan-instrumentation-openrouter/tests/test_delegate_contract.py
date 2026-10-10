"""Released SDKs through real controlled transports and recording OTel spans."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import tomllib
from openai import AsyncOpenAI, OpenAI
from openrouter import OpenRouter
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import StatusCode
from respan_instrumentation_openai import OpenAIInstrumentor
from respan_instrumentation_openrouter import OpenRouterInstrumentor, _instrumentation
from respan_instrumentation_openrouter._observer import AsyncStream, Call, SyncStream
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY

from ._transport import CALL, MODEL, TEXT, TOOL, VECTOR, reply


@pytest.fixture
def recording(monkeypatch):
    while _instrumentation._ACTIVE_BRIDGE_OWNER:
        _instrumentation._ACTIVE_BRIDGE_OWNER.deactivate()
    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    inst = OpenRouterInstrumentor()
    inst.activate()
    assert inst._is_instrumented
    yield inst, provider, exporter
    inst.deactivate()
    provider.shutdown()


def client(native, handler=reply):
    transport = httpx.Client(transport=httpx.MockTransport(handler))
    return (
        OpenRouter(
            api_key="fixture",
            client=transport,
            server_url="https://openrouter.ai/api/v1",
        )
        if native
        else OpenAI(
            api_key="fixture",
            base_url="https://openrouter.ai/api/v1",
            http_client=transport,
            max_retries=0,
        )
    )


def request(c, native, kind, **kw):
    values = {"model": MODEL, **kw}
    if kind == "chat":
        values.setdefault(
            "messages", [{"role": "user", "content": "PRIVATE fixture prompt"}]
        )
        return c.chat.send(**values) if native else c.chat.completions.create(**values)
    values.setdefault("input", "PRIVATE fixture input")
    if kind == "response":
        return c.responses.send(**values) if native else c.responses.create(**values)
    values["model"] = "openai/text-embedding-3-small"
    return (
        c.embeddings.generate(**values)
        if native
        else c.embeddings.create(**values, encoding_format="float")
    )


def close(c, native):
    c.__exit__(None, None, None) if native else c.close()


@pytest.mark.parametrize("native", [True, False], ids=["native", "compatible"])
@pytest.mark.parametrize("kind", ["chat", "response", "embedding"])
def test_released_sdk_nonstream(recording, native, kind):
    _inst, provider, exporter = recording
    c = client(native)
    with provider.get_tracer("test").start_as_current_span("parent") as parent:
        result = request(c, native, kind)
        assert result is not None
        (span,) = exporter.get_finished_spans()
        assert span.parent.span_id == parent.get_span_context().span_id
    close(c, native)
    attrs = span.attributes
    assert attrs["gen_ai.system"] == attrs["gen_ai.provider.name"] == "openrouter"
    assert attrs["respan.entity.log_type"] == (
        "embedding" if kind == "embedding" else "chat"
    )
    assert attrs["gen_ai.usage.input_tokens"] == (3 if kind == "embedding" else 7)
    assert "status_code" not in attrs and "http.response.status_code" not in attrs
    assert SpanAttributes.TRACELOOP_SPAN_KIND not in attrs
    if kind == "embedding":
        assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]) == [VECTOR]
        assert "gen_ai.usage.output_tokens" not in attrs
    else:
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 2
        assert attrs["gen_ai.usage.reasoning.output_tokens"] == 1
        assert attrs["gen_ai.completion.0.content"] == TEXT


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("kind", ["chat", "response"])
def test_stream_context_parent_and_partial_close(recording, native, kind):
    _, provider, exporter = recording
    c = client(native)
    with provider.get_tracer("test").start_as_current_span("call-parent") as parent:
        stream = request(c, native, kind, stream=True)
        parent_id = parent.get_span_context().span_id
    with stream:
        list(stream)
    spans = [
        s for s in exporter.get_finished_spans() if s.name.startswith("openrouter.")
    ]
    assert len(spans) == 1
    assert spans[0].parent.span_id == parent_id
    assert spans[0].attributes["gen_ai.request.stream"] is True
    assert spans[0].attributes["gen_ai.completion.0.content"] == TEXT
    partial = request(c, native, kind, stream=True)
    next(partial)
    partial.close()
    partial.close()
    assert (
        len(
            [
                s
                for s in exporter.get_finished_spans()
                if s.name.startswith("openrouter.")
            ]
        )
        == 2
    )
    close(c, native)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("stream", [True, False])
def test_complete_current_tool_payload(recording, native, stream):
    _, _, exporter = recording
    c = client(native)
    result = request(
        c,
        native,
        "chat",
        tools=[TOOL],
        stream=stream,
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{**CALL, "id": "historical-call"}],
            },
            {"role": "tool", "tool_call_id": "historical-call", "content": "history"},
        ],
    )
    if stream:
        with result:
            list(result)
    attrs = exporter.get_finished_spans()[0].attributes
    assert json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS]) == [TOOL]
    assert json.loads(attrs["gen_ai.completion.0.tool_calls"])[0] == CALL
    assert "historical-call" not in attrs["gen_ai.completion.0.tool_calls"]
    close(c, native)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("mode", ["env", "context", "end", "ancestor"])
def test_privacy_start_ancestor_and_end(recording, monkeypatch, native, mode):
    _inst, provider, exporter = recording
    token = None
    if mode in {"env", "ancestor"}:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "off")
    if mode == "context":
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))

    def handler(req):
        if mode == "end":
            context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        return reply(req)

    c = client(native, handler)
    if mode == "ancestor":
        with provider.get_tracer("test").start_as_current_span("private-parent"):
            monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
            request(c, native, "chat")
    else:
        request(c, native, "chat")
    if token is not None:
        context.detach(token)
    span = next(
        s for s in exporter.get_finished_spans() if s.name.startswith("openrouter.")
    )
    assert not any(
        k.startswith(("gen_ai.prompt.", "gen_ai.completion."))
        or k
        in {
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            SpanAttributes.LLM_REQUEST_FUNCTIONS,
            "error.message",
        }
        for k in span.attributes
    )
    assert "PRIVATE" not in json.dumps(dict(span.attributes))
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    close(c, native)


@pytest.mark.parametrize("native", [True, False])
def test_provider_error_source_status_no_completion(recording, native):
    _, _, exporter = recording
    c = client(native)
    with pytest.raises(Exception) as error:
        request(c, native, "chat", model="fixture/error")
    assert error.value.status_code == 429
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["http.response.status_code"] == 429
    assert "status_code" not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    close(c, native)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY]
)
def test_suppression_keeps_native_return_without_spans(recording, native, key):
    _, _, exporter = recording
    c = client(native)
    token = context.attach(context.set_value(key, True))
    try:
        result = request(c, native, "chat")
    finally:
        context.detach(token)
    assert result.choices[0].message.content == TEXT
    assert not exporter.get_finished_spans()
    close(c, native)


@pytest.mark.parametrize("native", [True, False])
def test_async_released_sdk_streams_and_vectors(recording, native):
    _, _, exporter = recording

    async def run():
        transport = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        c = (
            OpenRouter(api_key="fixture", async_client=transport)
            if native
            else AsyncOpenAI(api_key="fixture", http_client=transport, max_retries=0)
        )
        for kind in ("chat", "response"):
            kw = (
                {
                    "model": MODEL,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
                if kind == "chat"
                else {"model": MODEL, "stream": True, "input": "hi"}
            )
            stream = await (
                c.chat.send_async(**kw)
                if native and kind == "chat"
                else c.responses.send_async(**kw)
                if native
                else c.chat.completions.create(**kw)
                if kind == "chat"
                else c.responses.create(**kw)
            )
            async with stream:
                result = [v async for v in stream]
            assert result
        vector = await (
            c.embeddings.generate_async(model=MODEL, input="hi")
            if native
            else c.embeddings.create(model=MODEL, input="hi", encoding_format="float")
        )
        assert vector.data[0].embedding == VECTOR
        await transport.aclose()
        await c.__aexit__(None, None, None) if native else await c.close()

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert all(s.attributes["gen_ai.system"] == "openrouter" for s in spans)


def test_native_stream_protocol_error_identity(recording):
    inst, provider, exporter = recording
    error = RuntimeError("application mentions HTTP 404 without provider response")

    def gen():
        value = yield {"choices": [{"index": 0, "delta": {"content": "first"}}]}
        assert value == "sent"
        yield {"choices": [{"index": 0, "delta": {"content": "second"}}]}

    call = Call("chat", {"stream": True}, provider, inst._processor._policy)
    stream = SyncStream(gen(), call)
    assert next(stream)["choices"][0]["delta"]["content"] == "first"
    stream.send("sent")
    with pytest.raises(RuntimeError) as observed:
        stream.throw(error)
    assert observed.value is error
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert "http.response.status_code" not in span.attributes
    assert span.attributes["gen_ai.completion.0.content"] == "firstsecond"


def test_async_protocol_cancellation_identity(recording):
    inst, provider, exporter = recording
    error = asyncio.CancelledError()

    async def gen():
        v = yield {"choices": [{"index": 0, "delta": {"content": "first"}}]}
        assert v == "sent"
        yield {"choices": [{"index": 0, "delta": {"content": "second"}}]}

    async def run():
        call = Call("chat", {"stream": True}, provider, inst._processor._policy)
        stream = AsyncStream(gen(), call)
        await stream.__anext__()
        await stream.asend("sent")
        with pytest.raises(asyncio.CancelledError) as observed:
            await stream.athrow(error)
        assert observed.value is error

    asyncio.run(run())
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert "http.response.status_code" not in span.attributes


@pytest.mark.parametrize("order", ["before", "after"])
def test_independent_openai_lease_and_foreign_wrapper(recording, order):
    inst, _provider, exporter = recording
    independent = OpenAIInstrumentor()
    if order == "before":
        inst.deactivate()
    independent.activate()
    if order == "before":
        inst.activate()
    inst.deactivate()
    assert independent._is_instrumented
    c = client(False)
    request(c, False, "chat")
    assert exporter.get_finished_spans()
    independent.deactivate()
    close(c, False)


def test_foreign_native_wrapper_is_preserved(recording, monkeypatch):
    inst, _, _ = recording
    from openrouter.chat import Chat

    def foreign(*args, **kwargs):
        return "foreign"

    monkeypatch.setattr(Chat, "send", foreign)
    inst.deactivate()
    assert Chat.send is foreign


def test_optional_native_floor_and_released_delegate_bounds():
    deps = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())[
        "tool"
    ]["poetry"]["dependencies"]
    assert deps["openrouter"] == {"version": ">=1.3.22,<2.0.0", "optional": True}
    assert deps["respan-instrumentation-openai"] == ">=1.2.1,<2.0.0"


def test_sampling_never_injects_a_span(recording):
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    _, provider, exporter = recording
    provider.sampler = ALWAYS_OFF
    c = client(True)
    stream = request(c, True, "chat", stream=True)
    with stream:
        assert list(stream)
    assert not exporter.get_finished_spans()
    close(c, True)


def test_capture_flag_preserves_usage_and_clears_payload(recording):
    inst, _, exporter = recording
    inst.deactivate()
    private = OpenRouterInstrumentor(capture_content=False)
    private.activate()
    c = client(True)
    request(c, True, "chat", tools=[TOOL])
    attrs = exporter.get_finished_spans()[0].attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in attrs
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in attrs
    assert SpanAttributes.LLM_REQUEST_FUNCTIONS not in attrs
    assert attrs["gen_ai.usage.output_tokens"] == 5
    private.deactivate()
    close(c, True)


def test_activation_rollback_restores_owned_native_and_delegate(recording, monkeypatch):
    from openrouter.chat import Chat
    from respan_instrumentation_openrouter import _observer

    inst, provider, _ = recording
    inst.deactivate()
    original = Chat.send
    processors = provider._active_span_processor._span_processors
    install = _observer.install_native

    def fail(runtime):
        install(runtime)
        raise RuntimeError("controlled partial installation")

    monkeypatch.setattr(_observer, "install_native", fail)
    inst.activate()
    assert not inst._is_instrumented
    assert Chat.send is original
    assert provider._active_span_processor._span_processors == processors
    assert not _instrumentation.openai_instrumentation._original_methods


def test_openrouter_hostname_filter_preserves_regular_delegate(recording):
    inst, _, exporter = recording
    inst.deactivate()
    routed = OpenRouterInstrumentor(normalize_all_openai_spans=False)
    routed.activate()
    c = client(False)
    c.base_url = "https://openrouter.ai.evil.invalid/api/v1"
    request(c, False, "chat")
    assert exporter.get_finished_spans()[0].attributes["gen_ai.system"] == "openai"
    c.base_url = "https://openrouter.ai/api/v1"
    request(c, False, "chat")
    assert exporter.get_finished_spans()[1].attributes["gen_ai.system"] == "openrouter"
    routed.deactivate()
    close(c, False)


@pytest.mark.parametrize("native", [True, False])
def test_large_complete_tools_preserve_json_and_ids(recording, native):
    from ._transport import chat_payload

    _, _, exporter = recording
    arguments = json.dumps({"value": "界" * 20000})
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "parameters": {"type": "object", "description": "界" * 2000},
            },
        }
        for i in range(160)
    ]

    def handler(req):
        data = chat_payload(tools=True)
        data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            arguments
        )
        return httpx.Response(200, json=data, request=req)

    c = client(native, handler)
    request(c, native, "chat", tools=tools)
    attrs = exporter.get_finished_spans()[0].attributes
    assert len(json.loads(attrs[SpanAttributes.LLM_REQUEST_FUNCTIONS])) == 160
    call = json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]
    assert call["id"] == CALL["id"]
    assert json.loads(call["function"]["arguments"]) == json.loads(arguments)
    close(c, native)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": True, "completion_tokens": False, "total_tokens": -1},
        {"prompt_tokens": 7, "completion_tokens": 5},
    ],
)
def test_actual_body_usage_rejects_coerced_or_missing_counts(recording, native, usage):
    from ._transport import chat_payload

    _, _, exporter = recording

    def handler(req):
        data = chat_payload()
        data["usage"] = usage
        return httpx.Response(200, json=data, request=req)

    c = client(native, handler)
    if native and "total_tokens" not in usage:
        # The native SDK requires its total field; prove the same native error
        # remains visible and do not fabricate counts from a failed parse.
        from openrouter.errors import ResponseValidationError

        with pytest.raises(ResponseValidationError):
            request(c, native, "chat")
    else:
        request(c, native, "chat")
    attrs = exporter.get_finished_spans()[0].attributes
    if usage.get("prompt_tokens") is True:
        assert not any(k.startswith(("gen_ai.usage.", "llm.usage.")) for k in attrs)
    elif not native:
        assert attrs["gen_ai.usage.input_tokens"] == 7
        assert attrs["gen_ai.usage.output_tokens"] == 5
        assert "llm.usage.total_tokens" not in attrs
    close(c, native)


def test_delayed_stream_child_of_finished_private_parent(recording):
    _, provider, exporter = recording
    c = client(True)
    with provider.get_tracer("test").start_as_current_span("parent"):
        stream = request(c, True, "chat", stream=True)
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    context.detach(token)
    with stream:
        list(stream)
    span = next(
        s for s in exporter.get_finished_spans() if s.name.startswith("openrouter.")
    )
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in span.attributes
    close(c, True)


def test_deactivation_finishes_telemetry_without_closing_native_stream(recording):
    inst, _, exporter = recording
    c = client(True)
    stream = request(c, True, "chat", stream=True)
    native = stream._stream
    next(stream)
    inst.deactivate()
    assert len(exporter.get_finished_spans()) == 1
    assert not native._closed
    with stream:
        assert list(stream)
    assert len(exporter.get_finished_spans()) == 1
    close(c, True)


def test_open_parent_veto_is_inherited_after_reenable(recording):
    _, provider, exporter = recording
    c = client(True)
    with provider.get_tracer("test").start_as_current_span("parent"):
        token = context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        context.detach(token)
        request(c, True, "chat")
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    child = next(s for s in spans if s.name == "openrouter.chat")
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in child.attributes
    assert SpanAttributes.TRACELOOP_ENTITY_INPUT not in child.attributes
    close(c, True)


def test_shared_runtime_rejects_changed_provider(recording, monkeypatch):
    inst, _, _ = recording
    other = TracerProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: other)
    join = OpenRouterInstrumentor()
    join.activate()
    assert not join._is_instrumented
    assert inst._runtime.members == {inst}
    other.shutdown()


def test_large_history_choices_keep_common_fields_and_complete_entities(recording):
    from ._transport import chat_payload

    _, _, exporter = recording
    messages = [{"role": "user", "content": f"history-{i}"} for i in range(150)]

    def handler(req):
        data = chat_payload()
        data["choices"] = [
            {
                "index": i,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": f"choice-{i}"},
            }
            for i in range(150)
        ]
        return httpx.Response(200, json=data, request=req)

    c = client(False, handler)
    request(c, False, "chat", messages=messages)
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["gen_ai.system"] == "openrouter"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_INPUT]) == messages
    assert len(json.loads(attrs[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])) == 150
    close(c, False)


def test_sensitive_schema_names_preserve_nodes_but_not_credential_values(recording):
    _, _, exporter = recording
    definition = {
        "type": "function",
        "function": {
            "name": "authenticated",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_key": {
                        "type": "string",
                        "default": "private-credential-value",
                    },
                    "city": {"type": "string"},
                },
                "required": ["api_key"],
            },
        },
    }
    c = client(True)
    request(c, True, "chat", tools=[definition])
    captured = json.loads(
        exporter.get_finished_spans()[0].attributes[
            SpanAttributes.LLM_REQUEST_FUNCTIONS
        ]
    )[0]
    node = captured["function"]["parameters"]["properties"]["api_key"]
    assert node == {"type": "string", "default": "[REDACTED]"}
    assert (
        definition["function"]["parameters"]["properties"]["api_key"]["default"]
        == "private-credential-value"
    )
    close(c, True)


def test_redacted_current_arguments_with_escaped_quotes_remain_json(recording):
    from ._transport import chat_payload

    _, _, exporter = recording
    original = json.dumps({"api_key": 'private "quoted" value', "city": "Tokyo"})

    def handler(req):
        data = chat_payload(tools=True)
        data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            original
        )
        return httpx.Response(200, json=data, request=req)

    c = client(True, handler)
    result = request(c, True, "chat", tools=[TOOL])
    assert result.choices[0].message.tool_calls[0].function.arguments == original
    calls = json.loads(
        exporter.get_finished_spans()[0].attributes["gen_ai.completion.0.tool_calls"]
    )
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "api_key": "[REDACTED]",
        "city": "Tokyo",
    }
    close(c, True)


@pytest.mark.parametrize(
    "argument",
    [
        {"authorization": "Bearer synthetic-token"},
        {"content": "Bearer synthetic-token"},
        {"authorization": 'Bearer synthetic-token"quoted" value'},
        {"content": 'Bearer synthetic-token"quoted" value'},
        {"api_key": "already [REDACTED]"},
    ],
)
def test_redacted_bearer_arguments_keep_valid_json_and_native_value(
    recording, argument
):
    from ._transport import chat_payload

    _, _, exporter = recording
    original = json.dumps(argument)

    def handler(req):
        data = chat_payload(tools=True)
        data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            original
        )
        return httpx.Response(200, json=data, request=req)

    c = client(True, handler)
    result = request(c, True, "chat", tools=[TOOL])
    assert result.choices[0].message.tool_calls[0].function.arguments == original
    calls = json.loads(
        exporter.get_finished_spans()[0].attributes["gen_ai.completion.0.tool_calls"]
    )
    assert "synthetic-token" not in calls[0]["function"]["arguments"]
    assert isinstance(json.loads(calls[0]["function"]["arguments"]), dict)
    from respan_instrumentation_openrouter._processor import _redact_text

    assert (
        _redact_text(calls[0]["function"]["arguments"])
        == calls[0]["function"]["arguments"]
    )
    close(c, True)


def test_bare_assignment_redaction_is_idempotent():
    from respan_instrumentation_openrouter._processor import _redact_text

    for raw in [
        "password=synthetic",
        "api_key=synthetic",
        '"authorization":"Bearer synthetic-token"',
    ]:
        once = _redact_text(raw)
        assert _redact_text(once) == once
