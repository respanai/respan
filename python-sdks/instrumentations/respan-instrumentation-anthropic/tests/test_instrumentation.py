"""Released SDK tests use real HTTP/SSE parsing, typed responses and managers."""

import asyncio
import json

import anthropic
import pytest
from opentelemetry import context, trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import StatusCode
from respan_instrumentation_anthropic import AnthropicInstrumentor
from respan_instrumentation_anthropic import _instrumentation as module
from respan_instrumentation_anthropic._serialization import (
    json_string,
    json_value,
    redact_text,
)

try:
    import httpx2 as httpx
except ImportError:
    import httpx

REQUEST = {
    "model": "claude-controlled",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "fixture"}],
}


def message(text="complete", content=None):
    return {
        "id": "msg_controlled",
        "type": "message",
        "role": "assistant",
        "model": "claude-controlled",
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 7,
            "output_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def frames(text="complete", error=False):
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **message(),
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 7, "output_tokens": 0},
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
                "delta": {"type": "text_delta", "text": text},
            },
        ),
    ]
    if error:
        events += [
            (
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "controlled failure",
                    },
                },
            )
        ]
    else:
        events += [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 3},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode()


def client(response=None, *, asynchronous=False, handler=None):
    def send(request):
        if handler:
            return handler(request)
        return (
            httpx.Response(
                200, content=frames(), headers={"content-type": "text/event-stream"}
            )
            if json.loads(request.content).get("stream")
            else httpx.Response(200, json=response or message())
        )

    cls = anthropic.AsyncAnthropic if asynchronous else anthropic.Anthropic
    http = httpx.AsyncClient if asynchronous else httpx.Client
    return cls(
        api_key="controlled",
        http_client=http(transport=httpx.MockTransport(send)),
        max_retries=0,
    )


@pytest.fixture
def recording():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AnthropicInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    yield provider, exporter, instrumentor
    instrumentor.deactivate()
    provider.shutdown()


@pytest.mark.parametrize("beta", [False, True])
def test_native_messages_identity_and_mapping(recording, beta):
    _, exporter, _ = recording
    with client() as c:
        api = c.beta.messages if beta else c.messages
        result = api.create(**REQUEST)
    assert type(result).__module__.startswith("anthropic.types.")
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["gen_ai.usage.cache_read_input_tokens"] == 0
    assert json.loads(attrs["traceloop.entity.output"])[0]["text"] == "complete"
    assert (
        not {
            "tools",
            "tool_calls",
            "model",
            "prompt_tokens",
            "status_code",
            "traceloop.span.kind",
        }
        & attrs.keys()
    )
    assert "llm.usage.total_tokens" not in attrs


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("helper", [False, True])
def test_native_sync_stream_identity_and_no_eager_read(recording, beta, helper):
    _, exporter, _ = recording
    with client() as c:
        api = c.beta.messages if beta else c.messages
        if helper and not hasattr(api, "stream"):
            pytest.skip("beta stream helper absent at the declared minimum")
        result = api.stream(**REQUEST) if helper else api.create(**REQUEST, stream=True)
        assert type(result).__module__.startswith("anthropic")
        assert not exporter.get_finished_spans()
        with result as stream:
            events = list(stream)
            assert events
            if helper:
                assert stream.get_final_message().content[0].text == "complete"
        assert stream.response.is_closed
    assert len(exporter.get_finished_spans()) == 1
    assert (
        json.loads(
            exporter.get_finished_spans()[0].attributes["traceloop.entity.output"]
        )[0]["text"]
        == "complete"
    )


@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("helper", [False, True])
def test_native_async_stream_and_response(recording, beta, helper):
    _, exporter, _ = recording

    async def run():
        async with client(asynchronous=True) as c:
            api = c.beta.messages if beta else c.messages
            if helper and not hasattr(api, "stream"):
                pytest.skip("beta stream helper absent at the declared minimum")
            response = await api.create(**REQUEST)
            assert response.content[0].text == "complete"
            result = (
                api.stream(**REQUEST)
                if helper
                else await api.create(**REQUEST, stream=True)
            )
            assert type(result).__module__.startswith("anthropic")
            async with result as stream:
                events = [event async for event in stream]
                assert events
                if helper:
                    assert (await stream.get_final_message()).content[
                        0
                    ].text == "complete"
            assert stream.response.is_closed

    asyncio.run(run())
    assert len(exporter.get_finished_spans()) == 2
    for span in exporter.get_finished_spans():
        assert span.attributes["gen_ai.usage.output_tokens"] == 3


def test_native_tool_and_thinking_blocks_full_history(recording):
    _, exporter, _ = recording
    content = [
        {
            "type": "thinking",
            "thinking": "controlled reasoning",
            "signature": "signed-fixture",
        },
        {
            "type": "tool_use",
            "id": "toolu_controlled",
            "name": "lookup",
            "input": {
                "values": list(range(5001)),
                "zero": 0,
                "false": False,
                "empty": [],
            },
        },
    ]
    history = [
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_controlled",
                    "content": "0",
                    "is_error": False,
                }
            ],
        },
    ]
    tools = [
        {
            "name": "lookup",
            "input_schema": {
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "default": "sensitive-default"},
                    "values": {"type": "array", "items": {"type": "integer"}},
                },
            },
        }
    ]
    with client(message(content=content)) as c:
        result = c.messages.create(**{**REQUEST, "messages": history, "tools": tools})
    attrs = exporter.get_finished_spans()[0].attributes
    output = json.loads(attrs["traceloop.entity.output"])
    assert output[0]["thinking"] == "controlled reasoning"
    assert output[0]["signature"] == "signed-fixture"
    assert len(output[1]["input"]["values"]) == 5001
    assert result.content[1].input["false"] is False
    assert (
        json.loads(attrs["traceloop.entity.input"])[1]["content"][0]["tool_use_id"]
        == "toolu_controlled"
    )
    assert (
        json.loads(attrs["llm.request.functions"])[0]["function"]["parameters"][
            "properties"
        ]["api_key"]["default"]
        == "[REDACTED]"
    )
    assert (
        json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["id"]
        == "toolu_controlled"
    )
    assert (
        len(exporter.get_finished_spans()) == 1
    )  # history is not a local tool execution


@pytest.mark.parametrize(
    "key", [_SUPPRESS_INSTRUMENTATION_KEY, "suppress_language_model_instrumentation"]
)
def test_native_suppression_before_extraction(recording, key, monkeypatch):
    _, exporter, _ = recording
    monkeypatch.setattr(
        module,
        "CallState",
        lambda *args, **kwargs: pytest.fail("extracted suppressed call"),
    )
    token = context.attach(context.set_value(key, True))
    try:
        with client() as c:
            assert c.messages.create(**REQUEST).content[0].text == "complete"
    finally:
        context.detach(token)
    assert not exporter.get_finished_spans()


@pytest.mark.parametrize(
    "mode", ["initial", "late", "ancestor", "finished", "unknown", "supplied"]
)
def test_capture_vetoes_preserve_native_stream(recording, mode):
    provider, exporter, instrumentor = recording
    tracer = provider.get_tracer("test")
    ancestor = tracer.start_span("ancestor")
    if mode == "ancestor":
        ancestor.set_attribute("trace_content", False)
    if mode == "finished":
        ancestor.set_attribute("trace_content", False)
        ancestor.end()
    if mode == "unknown":
        unknown = TracerProvider().get_tracer("unknown").start_span("unobserved")
        ancestor = unknown
    if mode == "supplied":
        instrumentor.context = context.set_value("trace_content", False)
    with trace.use_span(ancestor, end_on_exit=False):
        token = (
            context.attach(context.set_value("trace_content", False))
            if mode == "initial"
            else None
        )
        try:
            with client() as c:
                stream = c.messages.create(**REQUEST, stream=True)
                next(stream)
                if mode == "late":
                    ancestor.set_attribute("trace_content", False)
                list(stream)
                stream.close()
        finally:
            if token is not None:
                context.detach(token)
    spans = [s for s in exporter.get_finished_spans() if s.name == "anthropic.chat"]
    assert len(spans) == 1
    assert "traceloop.entity.input" not in spans[0].attributes
    assert "traceloop.entity.output" not in spans[0].attributes
    if mode != "finished":
        ancestor.end()


@pytest.mark.parametrize("streaming", [False, True])
def test_real_provider_errors_have_no_invented_output(recording, streaming):
    _, exporter, _ = recording

    def send(request):
        return (
            httpx.Response(
                200,
                content=frames(error=True),
                headers={"content-type": "text/event-stream"},
            )
            if streaming
            else httpx.Response(
                404,
                json={
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": "controlled missing model",
                    },
                },
            )
        )

    with client(handler=send) as c, pytest.raises(anthropic.APIStatusError) as caught:
        if streaming:
            with c.messages.create(**REQUEST, stream=True) as stream:
                list(stream)
        else:
            c.messages.create(**REQUEST)
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == type(caught.value).__name__
    assert span.attributes["http.response.status_code"] == caught.value.status_code
    if not streaming:
        assert "traceloop.entity.output" not in span.attributes


def test_native_partial_close_does_not_drain(recording):
    _, exporter, _ = recording
    with client() as c:
        with c.messages.stream(**REQUEST) as stream:
            next(stream)
        assert stream.response.is_closed
    span = exporter.get_finished_spans()[0]
    assert "complete" not in span.attributes.get("traceloop.entity.output", "")


def test_sampling_prevents_content_hooks(monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    instrumentor = AnthropicInstrumentor(tracer_provider=provider)
    instrumentor.activate()
    import respan_instrumentation_anthropic._messages as mapping

    monkeypatch.setattr(
        mapping,
        "json_value",
        lambda *args, **kwargs: pytest.fail("sampling extracted content"),
    )
    try:
        with client() as c:
            c.messages.create(**REQUEST)
    finally:
        instrumentor.deactivate()


def test_telemetry_faults_preserve_native_cleanup(recording, monkeypatch):
    _, _exporter, _ = recording
    monkeypatch.setattr(
        module.CallState,
        "event",
        lambda *args: (_ for _ in ()).throw(RuntimeError("telemetry")),
    )
    with client() as c:
        with c.messages.stream(**REQUEST) as stream:
            assert "".join(stream.text_stream) == "complete"
        assert stream.response.is_closed


def test_lifecycle_shared_conflict_and_foreign_wrapper():
    from anthropic.resources.messages import Messages

    original = Messages.create
    provider = TracerProvider()
    first = AnthropicInstrumentor(tracer_provider=provider)
    second = AnthropicInstrumentor(tracer_provider=provider)
    conflict = AnthropicInstrumentor(tracer_provider=provider, capture_content=False)
    first.activate()
    wrapper = Messages.create
    first.activate()
    second.activate()
    conflict.activate()
    assert Messages.create is wrapper
    assert not conflict._is_instrumented
    first.deactivate()
    assert Messages.create is wrapper

    def foreign(*args, **kwargs):
        return wrapper(*args, **kwargs)

    Messages.create = foreign
    second.deactivate()
    assert Messages.create is foreign
    assert not any(
        type(p).__name__ == "PrivacyObserver"
        for p in provider._active_span_processor._span_processors
    )
    Messages.create = original


def test_unknown_object_hooks_are_never_used():
    class Unknown:
        def __getattr__(self, name):
            raise AssertionError(name)

        def __iter__(self):
            raise AssertionError("iter")

        def __str__(self):
            raise AssertionError("str")

        def model_dump(self):
            raise AssertionError("dump")

    assert json_value(Unknown()) is None


@pytest.mark.parametrize(
    "value",
    [
        '{"api_key":"two word secret","ok":false}',
        "Bearer abcdef Basic Zm9vOmJhcg==",
        "https://name:password@controlled.invalid/path?api_key=secret&ok=0",
        "api_key=secret",
    ],
)
def test_redaction_valid_idempotent(value):
    result = redact_text(value)
    assert redact_text(result) == result
    assert (
        "secret" not in result and "abcdef" not in result and "Zm9vOmJhcg" not in result
    )
    if value.startswith("{"):
        assert json.loads(result)["ok"] is False
    assert json.loads(json_string({"value": value}))["value"] == result


def test_native_managed_events(recording):
    _, exporter, _ = recording
    with client() as c:
        if not hasattr(c.beta, "sessions"):
            pytest.skip("managed session API not released at the declared minimum")
    events = [
        (
            "user.message",
            {
                "type": "user.message",
                "id": "evt_user",
                "content": [{"type": "text", "text": "controlled agent input"}],
            },
        ),
        (
            "agent.message",
            {
                "type": "agent.message",
                "id": "evt_agent",
                "content": [{"type": "text", "text": "controlled agent output"}],
            },
        ),
        (
            "span.model_request_end",
            {
                "type": "span.model_request_end",
                "id": "evt_usage",
                "model_usage": {"input_tokens": 5, "output_tokens": 2},
            },
        ),
        (
            "session.status_idle",
            {
                "type": "session.status_idle",
                "id": "evt_idle",
                "stop_reason": {"type": "end_turn"},
            },
        ),
    ]
    body = "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode()
    with client(
        handler=lambda req: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    ) as c:
        result = c.beta.sessions.events.stream("session_controlled")
        assert isinstance(result, anthropic.Stream)
        with result as stream:
            native = list(stream)
        assert all(type(e).__module__.startswith("anthropic.") for e in native)
    span = exporter.get_finished_spans()[0]
    assert span.attributes["respan.entity.log_type"] == "agent"
    assert span.attributes["gen_ai.usage.input_tokens"] == 5
    assert "controlled agent output" in span.attributes["traceloop.entity.output"]
    assert "gen_ai.request.model" not in span.attributes


@pytest.mark.parametrize("beta", [False, True])
def test_native_typed_parse_preserves_user_model(recording, beta):
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    _, exporter, _ = recording
    with client(message(text='{"text":"typed fixture"}')) as c:
        api = c.beta.messages if beta else c.messages
        if not hasattr(api, "parse"):
            pytest.skip("native parse helper not released at the declared minimum")
        response = api.parse(**REQUEST, output_format=Answer)
        assert isinstance(response.parsed_output, Answer)
        assert response.parsed_output.text == "typed fixture"
    assert len(exporter.get_finished_spans()) == 1
    assert (
        "typed fixture"
        in exporter.get_finished_spans()[0].attributes["traceloop.entity.output"]
    )


def test_startup_serialization_failure_ends_owned_span(recording, monkeypatch):
    _, exporter, _ = recording
    import respan_instrumentation_anthropic._messages as mapping

    monkeypatch.setattr(
        mapping,
        "json_value",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("serde")),
    )
    with client() as c:
        assert c.messages.create(**REQUEST).content[0].text == "complete"
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_observation_fault_keeps_native_result(
    recording, monkeypatch, asynchronous
):
    monkeypatch.setattr(
        module,
        "_observe_stream",
        lambda *args: (_ for _ in ()).throw(RuntimeError("observe")),
    )
    if asynchronous:

        async def run():
            async with client(asynchronous=True) as c:
                async with c.messages.stream(**REQUEST) as stream:
                    assert (
                        "".join([text async for text in stream.text_stream])
                        == "complete"
                    )
                assert stream.response.is_closed

        asyncio.run(run())
    else:
        with client() as c:
            with c.messages.stream(**REQUEST) as stream:
                assert "".join(stream.text_stream) == "complete"
            assert stream.response.is_closed


def test_partial_activation_restores_owned_methods(monkeypatch):
    from anthropic.resources.messages import Messages

    original = Messages.create
    instrumentor = AnthropicInstrumentor(tracer_provider=TracerProvider())
    wrapper = module._wrapper
    calls = 0

    def fail(original, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled partial activation")
        return wrapper(original, *args, **kwargs)

    monkeypatch.setattr(module, "_wrapper", fail)
    instrumentor.activate()
    assert not instrumentor._is_instrumented
    assert Messages.create is original
    assert not instrumentor._providers


def test_late_provider_registered_before_capture(recording):
    _, _, instrumentor = recording
    # Resolve a newly supplied provider, then verify its new native ancestors.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor.tracer_provider = provider
    with client() as c:
        c.messages.create(**REQUEST)
        with provider.get_tracer("late").start_as_current_span(
            "ancestor", attributes={"trace_content": False}
        ):
            c.messages.create(**REQUEST)
    calls = [s for s in exporter.get_finished_spans() if s.name == "anthropic.chat"]
    assert len(calls) == 2
    assert "traceloop.entity.output" in calls[0].attributes
    assert "traceloop.entity.output" not in calls[1].attributes


@pytest.mark.parametrize(
    "flag",
    [
        "respan_enable_content_tracing",
        "trace_content",
        "override_enable_content_tracing",
    ],
)
def test_pre_detach_veto_latches_on_native_stream(recording, flag):
    # This alias was imported before the instrumentor's runtime-boundary patch.
    from opentelemetry.context import detach

    _, exporter, _ = recording
    with client() as c, c.messages.create(**REQUEST, stream=True) as stream:
        next(stream)
        token = context.attach(context.set_value(flag, False))
        detach(token)
        list(stream)
    attrs = exporter.get_finished_spans()[0].attributes
    assert (
        "traceloop.entity.input" not in attrs and "traceloop.entity.output" not in attrs
    )
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 3


@pytest.mark.parametrize("env", ["TRACELOOP_TRACE_CONTENT", "RESPAN_TRACE_CONTENT"])
@pytest.mark.parametrize("value", ["false", "0", "off", "no"])
def test_dynamic_environment_veto(recording, monkeypatch, env, value):
    _, exporter, _ = recording
    with client() as c, c.messages.create(**REQUEST, stream=True) as stream:
        next(stream)
        monkeypatch.setenv(env, value)
        list(stream)
    assert "traceloop.entity.output" not in exporter.get_finished_spans()[0].attributes


def test_ambient_and_supplied_on_start_canonical_veto(recording):
    provider, exporter, _ = recording
    token = context.attach(context.set_value("respan_enable_content_tracing", False))
    try:
        ancestor = provider.get_tracer("test").start_span(
            "supplied", context=context.Context()
        )
    finally:
        context.detach(token)
    with trace.use_span(ancestor), client() as c:
        c.messages.create(**REQUEST)
    calls = [s for s in exporter.get_finished_spans() if s.name == "anthropic.chat"]
    assert "traceloop.entity.output" not in calls[0].attributes
    ancestor.end()


def test_finished_parent_on_end_context_veto(recording):
    provider, exporter, _ = recording
    ancestor = provider.get_tracer("test").start_span("ancestor")
    token = context.attach(context.set_value("respan_enable_content_tracing", False))
    try:
        ancestor.end()
    finally:
        context.detach(token)
    with trace.use_span(ancestor), client() as c:
        c.messages.create(**REQUEST)
    calls = [s for s in exporter.get_finished_spans() if s.name == "anthropic.chat"]
    assert "traceloop.entity.output" not in calls[0].attributes


def test_complete_canonical_history_survives_attribute_limit(recording):
    _, exporter, _ = recording
    history = [{"role": "user", "content": f"message-{n}"} for n in range(75)]
    tool = {
        "type": "web_search_20260315",
        "name": "web_search",
        "max_uses": 0,
        "allowed_domains": [],
        "cache_control": {"type": "ephemeral"},
        "input_schema": {
            "type": "object",
            "properties": {"token": {"type": "string", "default": "private-default"}},
        },
    }
    with client() as c:
        c.messages.create(**{**REQUEST, "messages": history, "tools": [tool]})
    attrs = exporter.get_finished_spans()[0].attributes
    assert len(json.loads(attrs["traceloop.entity.input"])) == 75
    function = json.loads(attrs["llm.request.functions"])[0]["function"]
    assert (
        function["type"] == tool["type"]
        and function["max_uses"] == 0
        and function["allowed_domains"] == []
    )
    assert function["parameters"]["properties"]["token"]["default"] == "[REDACTED]"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.request.model"] == "claude-controlled"
    assert "complete" in attrs["traceloop.entity.output"]


def test_escaped_json_credentials_remain_valid_and_idempotent():
    text = json.dumps(
        {"token": 'a "quoted" credential', "api_key": "line\\nsecret", "ok": False}
    )
    result = redact_text(text)
    assert json.loads(result) == {
        "token": "[REDACTED]",
        "api_key": "[REDACTED]",
        "ok": False,
    }
    assert redact_text(result) == result


def test_bodyless_ended_ancestry_is_bounded(recording):
    provider, _, instrumentor = recording
    for _ in range(4100):
        provider.get_tracer("test").start_span("ancestor").end()
    assert len(instrumentor._observer.parents) <= 4096


def test_native_request_configuration_is_preserved_and_private(recording):
    import inspect

    _, exporter, _ = recording
    with client() as c:
        optional = {
            key: value
            for key, value in (("temperature", 0.0), ("top_p", 0.5))
            if key in inspect.signature(c.beta.messages.create).parameters
        }
        c.beta.messages.create(
            **REQUEST,
            **optional,
            betas=["controlled-beta"],
            extra_headers={"authorization": "Bearer controlled-credential"},
            extra_body={
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                },
                "thinking": {"type": "disabled"},
            },
        )
    attrs = exporter.get_finished_spans()[0].attributes
    options = json.loads(attrs["respan.metadata.anthropic.request"])
    assert options["betas"] == ["controlled-beta"]
    assert options["extra_body"]["thinking"] == {"type": "disabled"}
    assert options["extra_body"]["output_config"]["format"]["schema"]["properties"][
        "text"
    ] == {"type": "string"}
    assert options["extra_headers"]["authorization"] == "[REDACTED]"
    assert attrs["gen_ai.request.max_tokens"] == 16
    for key, value in optional.items():
        assert attrs["gen_ai.request." + key] == value


@pytest.mark.parametrize("helper", [False, True])
def test_exporter_suppression_preserves_pending_native_sibling(recording, helper):
    provider, exporter, _ = recording
    with provider.get_tracer("test").start_as_current_span("parent"), client() as c:
        first = (
            c.messages.stream(**REQUEST)
            if helper
            else c.messages.create(**REQUEST, stream=True)
        )
        second = (
            c.messages.stream(**REQUEST)
            if helper
            else c.messages.create(**REQUEST, stream=True)
        )
        with first as stream:
            list(stream)
        with second as stream:
            list(stream)
    calls = [
        span for span in exporter.get_finished_spans() if span.name == "anthropic.chat"
    ]
    assert len(calls) == 2
    assert all(
        "complete" in span.attributes["traceloop.entity.output"] for span in calls
    )
