"""Native released SDK contracts at controlled HTTP/model boundaries."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
import asyncio
import json
from functools import wraps

import litellm
import pytest
from _fixtures import HISTORY, TOOLS, client, response_client
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from respan_instrumentation_litellm import LiteLLMInstrumentor, RespanLiteLLMCallback
from respan_instrumentation_litellm import _instrumentation as module
from respan_instrumentation_litellm import _translator as translator
from respan_sdk.constants.span_attributes import RESPAN_LOG_TYPE
from respan_tracing.constants.context_constants import ENABLE_CONTENT_TRACING_KEY


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    for name in module._LISTS:
        monkeypatch.setattr(litellm, name, [])
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = LiteLLMInstrumentor(tracer_provider=provider)
    plugin.activate()
    yield provider, exporter, plugin
    plugin.deactivate()
    provider.shutdown()


def spans(exporter):
    return [
        s for s in exporter.get_finished_spans() if s.attributes.get(RESPAN_LOG_TYPE)
    ]


def completion(**kwargs):
    return litellm.completion(
        model="openai/fixture-model", messages=HISTORY, client=client(**kwargs)
    )


def test_real_chat_source_usage_and_native_result(runtime):
    provider, exporter, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("caller") as parent:
        response = completion()
        assert type(response).__name__ == "ModelResponse"
        assert trace.get_current_span() is parent
    span = spans(exporter)[0]
    assert span.parent.span_id == parent.context.span_id
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 18
    assert span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 3
    assert span.attributes[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 2
    assert span.attributes["gen_ai.completion.0.content"] == "controlled response"


@pytest.mark.parametrize("mode", ["env", "context", "option"])
def test_private_start_bounds_no_serializer_and_safe_usage(runtime, monkeypatch, mode):
    _, exporter, plugin = runtime
    if mode == "option":
        plugin.deactivate()
        plugin.include_content = False
        plugin.activate()
    if mode == "env":
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    token = (
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
        if mode == "context"
        else None
    )
    monkeypatch.setattr(
        translator, "safe_json", lambda value: pytest.fail("private content serialized")
    )
    try:
        result = completion()
    finally:
        if token:
            context.detach(token)
    assert result.choices[0].message.content == "controlled response"
    attrs = spans(exporter)[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.system"] == "openai"
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


def test_native_end_context_veto_before_detach(runtime):
    _, exporter, _ = runtime
    caller = context.get_current()

    def veto():
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))

    response = completion(on_request=veto)
    assert response.choices[0].message.content == "controlled response"
    assert context.get_current() is caller
    assert "traceloop.entity.input" not in spans(exporter)[0].attributes


def test_private_finished_parent_bounds_delayed_stream(runtime, monkeypatch):
    provider, exporter, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("parent"):
        stream = litellm.completion(
            model="openai/fixture-model",
            messages=HISTORY,
            stream=True,
            stream_options={"include_usage": True},
            client=client(stream=True),
        )
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    list(stream)
    assert "traceloop.entity.output" not in spans(exporter)[0].attributes


@pytest.mark.parametrize(
    "key",
    [
        context._SUPPRESS_INSTRUMENTATION_KEY,
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    ],
)
def test_native_suppression_keeps_result(runtime, key):
    _, exporter, _ = runtime
    token = context.attach(context.set_value(key, True))
    try:
        assert completion().choices[0].message.content == "controlled response"
    finally:
        context.detach(token)
    assert not spans(exporter)


def test_always_off_does_not_serialize_and_native_sink_callbacks(monkeypatch):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    plugin = LiteLLMInstrumentor(tracer_provider=provider)
    plugin.activate()
    monkeypatch.setattr(
        translator,
        "safe_json",
        lambda value: pytest.fail("unsampled content serialized"),
    )
    try:
        assert completion().choices[0].message.content == "controlled response"
    finally:
        plugin.deactivate()
        provider.shutdown()
    assert not spans(exporter)


def test_shared_foreign_and_inert_wrapper_ownership(runtime):
    provider, exporter, first = runtime
    second = LiteLLMInstrumentor(tracer_provider=provider)
    second.activate()
    owned = litellm.completion

    @wraps(owned)
    def foreign(*args, **kwargs):
        return owned(*args, **kwargs)

    litellm.completion = foreign
    first.deactivate()
    assert completion().choices[0].message.content == "controlled response"
    second.deactivate()
    assert litellm.completion is foreign
    before = len(spans(exporter))
    assert completion().choices[0].message.content == "controlled response"
    assert len(spans(exporter)) == before
    third = LiteLLMInstrumentor(tracer_provider=provider)
    third.activate()
    try:
        completion()
        assert len(spans(exporter)) == before + 1
    finally:
        third.deactivate()
    assert litellm.completion is foreign


def test_incompatible_owner_cannot_unmask(runtime):
    provider, _, _ = runtime
    with pytest.raises(ValueError):
        LiteLLMInstrumentor(tracer_provider=provider, include_content=False).activate()


def test_install_rollback_preserves_foreign_native_hooks(monkeypatch):
    original = litellm.completion
    provider = TracerProvider()
    instrumentor = LiteLLMInstrumentor(tracer_provider=provider)
    original_patch = module._Runtime.patch

    def broken(self, owner, name, replacement):
        original_patch(self, owner, name, replacement)
        if name == "acompletion":
            raise RuntimeError("controlled partial install")

    monkeypatch.setattr(module._Runtime, "patch", broken)
    with pytest.raises(RuntimeError):
        instrumentor.activate()
    assert litellm.completion is original
    assert not any(isinstance(c, RespanLiteLLMCallback) for c in litellm.callbacks)
    assert not provider._active_span_processor._span_processors
    provider.shutdown()


def test_actual_error_has_no_invented_completion_usage_or_500(runtime):
    _, exporter, _ = runtime
    with pytest.raises(litellm.BadRequestError):
        completion(error=True)
    span = spans(exporter)[0]
    attrs = span.attributes
    assert span.status.status_code is trace.StatusCode.ERROR
    # The minimum SDK changes its final exception response to400; the retained
    # original OpenAI AuthenticationError still has the actual provider401.
    assert attrs["http.response.status_code"] == 401
    assert "traceloop.entity.output" not in attrs and not any(
        k.startswith("gen_ai.completion.") or "usage" in k for k in attrs
    )


def test_complete_tools_current_history_ids_and_vectors(runtime):
    _, exporter, _ = runtime
    result = litellm.completion(
        model="openai/fixture-model",
        messages=HISTORY,
        tools=TOOLS,
        client=client(tools=True),
    )
    assert result.choices[0].message.tool_calls[0].id == "current-source-id"
    attrs = spans(exporter)[0].attributes
    current = json.loads(attrs["gen_ai.completion.0.tool_calls"])
    history = json.loads(attrs["gen_ai.prompt.0.tool_calls"])
    assert (
        current[0]["id"] == "current-source-id" and history[0]["id"] == "historical-id"
    )
    assert current[0]["function"]["arguments"] == '{"value":2}'
    assert (
        len(
            json.loads(attrs["llm.request.functions"])[0]["function"]["parameters"][
                "properties"
            ]
        )
        == 100
    )


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": -1, "completion_tokens": False, "total_tokens": -1},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    ],
)
def test_actual_raw_usage_absent_invalid_zero(runtime, usage):
    _, exporter, _ = runtime
    completion(usage=usage)
    attrs = spans(exporter)[0].attributes
    if usage and usage["prompt_tokens"] == 0:
        assert (
            attrs["gen_ai.usage.input_tokens"] == attrs["llm.usage.total_tokens"] == 0
        )
    else:
        assert not any("usage" in k for k in attrs)


def test_native_embedding_full_provider_vector_and_usage(runtime):
    _, exporter, _ = runtime
    result = litellm.embedding(
        model="openai/fixture-embed", input=["source"], client=client()
    )
    assert len(result.data[0]["embedding"]) == 5000
    attrs = spans(exporter)[0].attributes
    assert attrs[RESPAN_LOG_TYPE] == "embedding"
    assert len(json.loads(attrs["traceloop.entity.output"])[0]) == 5000
    assert attrs["gen_ai.usage.input_tokens"] == 9


@pytest.mark.parametrize("usage_present", [True, False])
def test_native_stream_detached_context_tool_deltas_source_usage(
    runtime, usage_present
):
    provider, exporter, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("caller") as parent:
        stream = litellm.completion(
            model="openai/fixture-model",
            messages=HISTORY,
            tools=TOOLS,
            stream=True,
            stream_options={"include_usage": True},
            client=client(
                stream=True, tools=True, **({} if usage_present else {"usage": None})
            ),
        )
        assert trace.get_current_span() is parent
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
            assert trace.get_current_span() is parent
        assert chunks
    attrs = spans(exporter)[0].attributes
    assert attrs["gen_ai.is_streaming"] is True
    calls = json.loads(attrs["gen_ai.completion.0.tool_calls"])
    assert calls[0]["id"] == "current-source-id"
    assert calls[0]["function"]["arguments"] == '{"value":2}'
    if usage_present:
        assert attrs["gen_ai.usage.input_tokens"] == 11
    else:
        assert not any("usage" in k for k in attrs)


@pytest.mark.asyncio
async def test_native_async_chat_stream_embedding_values_and_context(runtime):
    provider, exporter, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("caller") as parent:
        response = await litellm.acompletion(
            model="openai/fixture-model", messages=HISTORY, client=client(async_=True)
        )
        assert response.choices[0].message.content == "controlled response"
        assert trace.get_current_span() is parent
        stream = await litellm.acompletion(
            model="openai/fixture-model",
            messages=HISTORY,
            stream=True,
            stream_options={"include_usage": True},
            client=client(async_=True, stream=True),
        )
        async for _ in stream:
            assert trace.get_current_span() is parent
        result = await litellm.aembedding(
            model="openai/fixture-embed", input=["source"], client=client(async_=True)
        )
        assert len(result.data[0]["embedding"]) == 5000
    assert len(spans(exporter)) == 3
    assert all(
        s.attributes["gen_ai.usage.input_tokens"] in (9, 11) for s in spans(exporter)
    )


@pytest.mark.parametrize("stream", [False, True])
def test_native_responses_current_calls_and_actual_usage(runtime, stream):
    provider, exporter, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("caller") as parent:
        result = litellm.responses(
            model="openai/gpt-4o-mini",
            input="source",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": TOOLS[0]["function"]["parameters"],
                }
            ],
            stream=stream,
            api_key="fixture-only",
            api_base="https://fixture.invalid/v1",
            client=response_client(stream=stream, tools=True),
        )
        if stream:
            for _ in result:
                assert trace.get_current_span() is parent
        else:
            assert result.output[0].call_id == "current-response-call-id"
    attrs = spans(exporter)[0].attributes
    assert (
        json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["id"]
        == "current-response-call-id"
    )
    assert attrs["gen_ai.usage.input_tokens"] == 11


@pytest.mark.asyncio
async def test_native_async_responses_and_stream(runtime):
    _, exporter, _ = runtime
    for stream in (False, True):
        result = await litellm.aresponses(
            model="openai/gpt-4o-mini",
            input="source",
            stream=stream,
            api_key="fixture-only",
            api_base="https://fixture.invalid/v1",
            client=response_client(async_=True, stream=stream),
        )
        if stream:
            async for _ in result:
                pass
        else:
            assert result.output[0].content[0].text == "controlled response"
    assert len(spans(exporter)) == 2


def test_policy_failure_preserves_native_value_and_cleanup(runtime, monkeypatch):
    _, exporter, plugin = runtime
    monkeypatch.setattr(
        module, "allowed", lambda: (_ for _ in ()).throw(RuntimeError("policy failure"))
    )
    assert completion().choices[0].message.content == "controlled response"
    assert not plugin.runtime.calls and not plugin.runtime.source
    assert "traceloop.entity.input" not in spans(exporter)[0].attributes


def test_telemetry_failure_preserves_native_value_and_context(runtime, monkeypatch):
    provider, _, plugin = runtime
    monkeypatch.setattr(
        translator,
        "safe_json",
        lambda value: (_ for _ in ()).throw(RuntimeError("telemetry failure")),
    )
    with provider.get_tracer("caller").start_as_current_span("caller") as caller:
        assert completion().choices[0].message.content == "controlled response"
        assert trace.get_current_span() is caller
    assert not plugin.runtime.calls and not plugin.runtime.source


def test_private_raw_usage_reader_does_not_decode_content():
    source = json.dumps(
        {
            "content": {"usage": {"prompt_tokens": 999}},
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
                "custom": {"content": "not inspected"},
            },
        }
    )
    assert translator.raw_usage(source) == {"prompt_tokens": 0, "total_tokens": 0}


@pytest.mark.parametrize("value", ["0", "off", "no", " false "])
def test_respan_native_environment_opt_out_values(runtime, monkeypatch, value):
    _, exporter, _ = runtime
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", value)
    completion()
    assert "traceloop.entity.input" not in spans(exporter)[0].attributes


def test_explicit_private_parent_context_is_a_bound(runtime):
    provider, exporter, _ = runtime
    private = context.set_value(ENABLE_CONTENT_TRACING_KEY, False)
    parent = provider.get_tracer("external").start_span("external", context=private)
    token = context.attach(trace.set_span_in_context(parent))
    try:
        completion()
    finally:
        context.detach(token)
        parent.end()
    assert "traceloop.entity.output" not in spans(exporter)[0].attributes


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_native_stream_boolean_usage_never_becomes_zero(runtime, api):
    _, exporter, _ = runtime
    invalid = {
        "prompt_tokens": False,
        "completion_tokens": False,
        "total_tokens": False,
    }
    if api == "chat":
        result = litellm.completion(
            model="openai/fixture-model",
            messages=HISTORY,
            stream=True,
            stream_options={"include_usage": True},
            client=client(stream=True, usage=invalid),
        )
    else:
        # Responses fixture currently uses its actual fixed valid usage shape.
        invalid_client = response_client(stream=True)
        handler = invalid_client.client._transport.handler

        def raw(request):
            response = handler(request)
            body = (
                response.content.replace(
                    b'"input_tokens": 11', b'"input_tokens": false'
                )
                .replace(b'"output_tokens": 7', b'"output_tokens": false')
                .replace(b'"total_tokens": 18', b'"total_tokens": false')
            )
            import httpx

            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body
            )

        invalid_client.client._transport.handler = raw
        result = litellm.responses(
            model="openai/gpt-4o-mini",
            input="source",
            stream=True,
            api_key="fixture-only",
            api_base="https://fixture.invalid/v1",
            client=invalid_client,
        )
    list(result)
    attrs = spans(exporter)[0].attributes
    assert (
        "gen_ai.usage.input_tokens" not in attrs
        and "gen_ai.usage.output_tokens" not in attrs
        and "llm.usage.total_tokens" not in attrs
    )


@pytest.mark.asyncio
async def test_native_async_early_close_and_unadvanced_close_no_fabrication(runtime):
    provider, exporter, plugin = runtime
    for advance in (True, False):
        with provider.get_tracer("caller").start_as_current_span("caller") as caller:
            result = await litellm.acompletion(
                model="openai/fixture-model",
                messages=HISTORY,
                stream=True,
                client=client(async_=True, stream=True),
            )
            if not hasattr(result.native, "aclose"):
                async for _ in result:
                    pass
                pytest.skip("Declared minimum native CustomStreamWrapper has no aclose")
            assert not hasattr(type(result), "__aenter__") and not hasattr(
                type(result), "__enter__"
            )
            if advance:
                await result.__anext__()
            await result.aclose()
            assert trace.get_current_span() is caller
    first, second = spans(exporter)
    assert first.attributes["gen_ai.completion.0.content"] == "controlled "
    assert "traceloop.entity.output" not in second.attributes and not any(
        k.startswith("gen_ai.completion.") for k in second.attributes
    )
    assert not plugin.runtime.calls and not plugin.runtime.source


def test_explicit_private_callback_bounds_shared_auto_capture(runtime):
    _, exporter, _ = runtime
    result = litellm.completion(
        model="openai/fixture-model",
        messages=HISTORY,
        client=client(),
        callbacks=[RespanLiteLLMCallback(include_content=False)],
    )
    assert result.choices[0].message.content == "controlled response"
    attrs = spans(exporter)[0].attributes
    assert (
        "traceloop.entity.input" not in attrs
        and attrs["gen_ai.usage.input_tokens"] == 11
    )


@pytest.mark.parametrize("scope", ["global", "dynamic"])
def test_native_litellm_private_source_flag_cannot_unmask(runtime, monkeypatch, scope):
    _, exporter, _ = runtime
    options = {}
    if scope == "global":
        monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    else:
        options["standard_callback_dynamic_params"] = {"turn_off_message_logging": True}
    result = litellm.completion(
        model="openai/fixture-model", messages=HISTORY, client=client(), **options
    )
    assert result.choices[0].message.content == "controlled response"
    assert "traceloop.entity.input" not in spans(exporter)[0].attributes


def test_finished_private_parent_context_veto_before_native_detach(runtime):
    provider, exporter, _ = runtime
    with provider.get_tracer("caller").start_as_current_span("parent"):
        result = litellm.completion(
            model="openai/fixture-model",
            messages=HISTORY,
            stream=True,
            stream_options={"include_usage": True},
            client=client(stream=True),
        )
        context.attach(context.set_value(ENABLE_CONTENT_TRACING_KEY, False))
    assert context.get_value(ENABLE_CONTENT_TRACING_KEY) is not False
    list(result)
    attrs = spans(exporter)[0].attributes
    assert (
        "traceloop.entity.output" not in attrs and "traceloop.entity.input" not in attrs
    )
    assert attrs["gen_ai.usage.input_tokens"] == 11


def test_owned_callback_retained_by_foreign_owner_stays_inert_after_reactivation(
    runtime,
):
    provider, exporter, first = runtime
    first.deactivate()
    private = LiteLLMInstrumentor(include_content=False, tracer_provider=provider)
    private.activate()
    old = private.runtime.callback
    private.deactivate()
    current = LiteLLMInstrumentor(tracer_provider=provider)
    current.activate()
    try:
        result = litellm.completion(
            model="openai/fixture-model",
            messages=HISTORY,
            client=client(),
            callbacks=[old],
        )
        assert result.choices[0].message.content == "controlled response"
        assert (
            len(spans(exporter)) == 1
            and spans(exporter)[0].attributes["gen_ai.completion.0.content"]
            == "controlled response"
        )
    finally:
        current.deactivate()


def test_existing_disabled_respan_setting_keeps_native_call(runtime, monkeypatch):
    _, exporter, _ = runtime
    from types import SimpleNamespace

    monkeypatch.setattr(
        module.RespanTracer, "_instance", SimpleNamespace(is_enabled=False)
    )
    assert completion().choices[0].message.content == "controlled response"
    assert not spans(exporter)


def test_native_error_survives_optional_payload_formatter_failure(runtime, monkeypatch):
    _, exporter, plugin = runtime
    monkeypatch.setattr(
        translator,
        "safe_json",
        lambda value: (_ for _ in ()).throw(RuntimeError("formatter failure")),
    )
    with pytest.raises(litellm.BadRequestError):
        completion(error=True)
    span = exporter.get_finished_spans()[0]
    assert (
        span.status.status_code is trace.StatusCode.ERROR
        and span.attributes["error.type"] == "BadRequestError"
    )
    assert not plugin.runtime.calls and not plugin.runtime.source


def test_native_large_history_and_choices_keep_common_fields_and_complete_io(runtime):
    _, exporter, _ = runtime
    import httpx

    native = client()
    handler = native._client._transport.handler

    def many(request):
        response = handler(request)
        body = response.json()
        body["choices"] = [
            {
                "index": i,
                "message": {"role": "assistant", "content": f"choice-{i}"},
                "finish_reason": "stop",
            }
            for i in range(150)
        ]
        return httpx.Response(200, json=body)

    native._client._transport.handler = many
    messages = [{"role": "user", "content": f"message-{i}"} for i in range(150)]
    result = litellm.completion(
        model="openai/fixture-model",
        messages=messages,
        client=native,
        metadata={"respan_params": {"trace_group_identifier": "large-payload-marker"}},
    )
    assert len(result.choices) == 150
    attrs = spans(exporter)[0].attributes
    assert (
        attrs[RESPAN_LOG_TYPE] == "chat"
        and attrs["gen_ai.request.model"] == "fixture-model"
        and attrs["gen_ai.system"] == "openai"
    )
    assert attrs["respan.trace.trace_group_identifier"] == "large-payload-marker"
    assert len(json.loads(attrs["traceloop.entity.input"])) == 150
    assert len(json.loads(attrs["traceloop.entity.output"])) == 150


def test_native_responses_fake_stream_preserves_requested_stream_lifecycle(runtime):
    _, exporter, _ = runtime
    native = response_client()
    result = litellm.responses(
        model="openai/fixture-model",
        input="source",
        stream=True,
        api_key="fixture-only",
        api_base="https://fixture.invalid/v1",
        client=native,
    )
    # Bare LiteLLM fakes Responses streams for models absent from its catalog,
    # making a nonstreaming HTTP request but retaining a streaming public result.
    assert not spans(exporter)
    assert list(result)
    attrs = spans(exporter)[0].attributes
    assert attrs["gen_ai.is_streaming"] is True
    assert (
        attrs["gen_ai.usage.input_tokens"] == 11 and "traceloop.entity.output" in attrs
    )


def test_actual_tool_sensitive_property_schema_kept_credentials_redacted(runtime):
    _, exporter, _ = runtime
    import copy

    import httpx

    tools = copy.deepcopy(TOOLS)
    tools[0]["function"]["parameters"]["properties"]["api_key"] = {
        "type": "string",
        "default": "fixture-default-secret",
        "examples": ["fixture-example-secret"],
    }
    native = client(tools=True)
    handler = native._client._transport.handler

    def arguments(request):
        result = handler(request)
        body = result.json()
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            json.dumps(
                {
                    "api_key": "fixture-argument-secret",
                    "content": 'Bearer fixture-token"quoted" value',
                    "basic_content": 'Basic fixture-token"quoted" value',
                }
            )
        )
        return httpx.Response(200, json=body)

    native._client._transport.handler = arguments
    response = litellm.completion(
        model="openai/fixture-model", messages=HISTORY, tools=tools, client=native
    )
    assert (
        "fixture-argument-secret"
        in response.choices[0].message.tool_calls[0].function.arguments
    )
    assert (
        json.loads(response.choices[0].message.tool_calls[0].function.arguments)[
            "content"
        ]
        == 'Bearer fixture-token"quoted" value'
    )
    attrs = spans(exporter)[0].attributes
    definition = json.loads(attrs["llm.request.functions"])[0]["function"][
        "parameters"
    ]["properties"]["api_key"]
    assert (
        definition["type"] == "string"
        and definition["default"] == "[REDACTED]"
        and definition["examples"] == ["[REDACTED]"]
    )
    assert "fixture-argument-secret" not in attrs["gen_ai.completion.0.tool_calls"]
    captured = json.loads(attrs["gen_ai.completion.0.tool_calls"])[0]["function"][
        "arguments"
    ]
    from respan_instrumentation_litellm._serialization import redact_text

    parsed = json.loads(captured)
    assert parsed["api_key"] == "[REDACTED]"
    assert parsed["content"].startswith("Bearer [REDACTED]")
    assert parsed["basic_content"].startswith("Basic [REDACTED]")
    assert redact_text(captured) == captured

    escaped = json.dumps({"password": 'fixture secret "with quote" and spaces'})
    assert json.loads(redact_text(escaped)) == {"password": "[REDACTED]"}
    assert redact_text(redact_text(escaped)) == redact_text(escaped)


@pytest.mark.asyncio
async def test_native_async_delayed_callbacks_cannot_retain_finished_calls(runtime):
    _, _, plugin = runtime
    for _ in range(3):
        await litellm.acompletion(
            model="openai/fixture-model", messages=HISTORY, client=client(async_=True)
        )
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not plugin.runtime.calls and not plugin.runtime.source
